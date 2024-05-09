"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the CC BY-NC license found in the
LICENSE.md file in the root directory of this source tree.
"""

from torch.utils.tensorboard import SummaryWriter
import random
import time
import torch
import numpy as np
from pathlib import Path

from dtlight.decision_transformer import DecisionTransformer
from dtlight.dtlight_utils.replay_buffer import ReplayBuffer
from dtlight.dtlight_utils.lamb import Lamb
from dtlight.dtlight_utils.data import create_dataloader
from dtlight.dtlight_utils.trainer import SequenceTrainer
from dtlight.dtlight_utils.logger import Logger
from dtlight.distiller import Distiller


class DTLight:
    def __init__(
        self,
        var,
        signal,
        state_dim,
        act_dim,
        action_range,
        offline_trajs=None,
        max_return=None,
        state_mean=None,
        state_std=None,
    ):
        self.var = var
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.action_range = action_range
        self.offline_trajs = offline_trajs
        self.max_return = max_return
        self.state_mean = state_mean
        self.state_std = state_std

        # Initialize by offline trajs
        self.replay_buffer = ReplayBuffer(var["replay_size"], self.offline_trajs)

        self.device = var.get("device", "cuda")
        if var["max_distill_iters"]:
            t_embed_dim = 512
            t_n_layer = 6
            t_n_head = 8
            self.teacher = DecisionTransformer(
                state_dim=self.state_dim,
                act_dim=self.act_dim,
                hidden_size=t_embed_dim,
                action_range=self.action_range,
                ordering=var["ordering"],
                max_length=var["max_seq_len"],
                eval_context_length=var["eval_context_length"],
                max_ep_len=self.var["max_episode_length"],
                stochastic_policy=var["stochastic_policy"],
                init_temperature=var["init_temperature"],
                target_entropy=-self.act_dim,
                n_layer=t_n_layer,
                n_head=t_n_head,
                n_inner=4 * t_embed_dim,
                activation_function=var["activation_function"],
                resid_pdrop=var["dropout"],
                attn_pdrop=var["dropout"],
                transformer_model=var["transformer_model"],
                adapter=var["adapter"],
            ).to(device=self.device)
            self.model = self.teacher

        else:
            self.model = DecisionTransformer(
                state_dim=self.state_dim,
                act_dim=self.act_dim,
                hidden_size=var["embed_dim"],
                action_range=self.action_range,
                ordering=var["ordering"],
                max_length=var["max_seq_len"],
                eval_context_length=var["eval_context_length"],
                max_ep_len=self.var["max_episode_length"],
                stochastic_policy=var["stochastic_policy"],
                init_temperature=var["init_temperature"],
                target_entropy=-self.act_dim,
                n_layer=var["n_layer"],
                n_head=var["n_head"],
                n_inner=4 * var["embed_dim"],
                activation_function=var["activation_function"],
                resid_pdrop=var["dropout"],
                attn_pdrop=var["dropout"],
                transformer_model=var["transformer_model"],
                adapter=var["adapter"],
            ).to(device=self.device)

        self.optimizer = Lamb(
            self.model.parameters(),
            lr=var["learning_rate"],
            weight_decay=var["weight_decay"],
            eps=1e-8,
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda steps: min((steps + 1) / var["warmup_steps"], 1)
        )

        # if var["stochastic_policy"]:
        self.log_temperature_optimizer = torch.optim.Adam(
            [self.model.log_temperature],
            lr=1e-4,
            betas=[0.9, 0.999],
        )
        # else:
        #     self.log_temperature_optimizer = None

        # Track the training progress and
        # training/evaluation/online performance in all the iterations
        self.pretrain_iter = 0
        self.distill_iter = 0
        self.online_iter = 0
        self.total_transitions_sampled = 0
        self.reward_scale = var["reward_scale"]
        self.logger = Logger(var, signal)
        self.start_time = time.time()
        self.reset_idx = True
        self.aug_trajs = []
        self.logs = {}

    def _save_model(self, path_prefix, is_pretrain_model=False, is_distill_model=False):
        to_save = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "pretrain_iter": self.pretrain_iter,
            "online_iter": self.online_iter,
            "args": self.var,
            "total_transitions_sampled": self.total_transitions_sampled,
            "np": np.random.get_state(),
            "python": random.getstate(),
            "pytorch": torch.get_rng_state(),
            "log_temperature_optimizer_state_dict": self.log_temperature_optimizer.state_dict()
            if self.model.stochastic_policy
            else None,
        }

        if is_pretrain_model:
            if self.var["max_distill_iters"]:
                model_name = "teacher_model"
            else:
                model_name = "pretrain_model"
        elif is_distill_model:
            model_name = "distill_model"
        else:
            model_name = "finetune_model"

        with open(f"{path_prefix}/{model_name}.pt", "wb") as f:
            torch.save(to_save, f)
        print(f"Model saved at {path_prefix}/{model_name}.pt\n")

    def _load_model(self, path_prefix, model_name=None, reproduce=False):
        model_name = f"{model_name}_model"
        with open(f"{path_prefix}/{model_name}.pt", "rb") as f:
            checkpoint = torch.load(f)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        # if self.var["stochastic_policy"]:
        self.log_temperature_optimizer.load_state_dict(
            checkpoint["log_temperature_optimizer_state_dict"]
        )
        if reproduce:
            self.pretrain_iter = checkpoint["pretrain_iter"]
            self.online_iter = checkpoint["online_iter"]
            self.total_transitions_sampled = checkpoint["total_transitions_sampled"]
            np.random.set_state(checkpoint["np"])
            random.setstate(checkpoint["python"])
            torch.set_rng_state(checkpoint["pytorch"])
        print(f"Model loaded at {path_prefix}/{model_name}.pt\n")

    def pretrain(self):
        if self.var["load_model"] is not None:
            self._load_model(self.logger.log_path, self.var["load_model"])
        else:
            trainer = SequenceTrainer(
                model=self.model,
                optimizer=self.optimizer,
                log_temperature_optimizer=self.log_temperature_optimizer,
                scheduler=self.scheduler,
                device=self.device,
            )
            writer = (
                SummaryWriter(self.logger.log_path)
                if self.var["log_to_tb"]
                else None
            )

            while self.pretrain_iter < self.var["max_pretrain_iters"]:
                # In every iteration, prepare the data loader
                dataloader = create_dataloader(
                    trajectories=self.offline_trajs,
                    num_iters=self.var["num_updates_per_pretrain_iter"],
                    batch_size=self.var["batch_size"],
                    max_len=self.var["max_seq_len"],
                    state_dim=self.state_dim,
                    act_dim=self.act_dim,
                    state_mean=self.state_mean,
                    state_std=self.state_std,
                    reward_scale=self.reward_scale,
                    action_range=self.action_range,
                )

                train_outputs = trainer.train_iteration(dataloader=dataloader)

                self.logs[
                    f"Pretraining logs/iter {self.pretrain_iter}"
                ] = train_outputs

                outputs = {"Pretraining time/total": time.time() - self.start_time}
                outputs.update(train_outputs)
                self.logger.log_metrics(
                    outputs,
                    iter_num=self.pretrain_iter,
                    total_transitions_sampled=self.total_transitions_sampled,
                    writer=writer,
                )

                self._save_model(
                    path_prefix=self.logger.log_path,
                    is_pretrain_model=True,
                )

                self.pretrain_iter += 1

    def distill(self):
        # Define student and set as self.model for distillation and online finetuning
        self.model = DecisionTransformer(
            state_dim=self.state_dim,
            act_dim=self.act_dim,
            hidden_size=self.var["embed_dim"],
            action_range=self.action_range,
            ordering=self.var["ordering"],
            max_length=self.var["max_seq_len"],
            eval_context_length=self.var["eval_context_length"],
            max_ep_len=self.var["max_episode_length"],
            stochastic_policy=self.var["stochastic_policy"],
            init_temperature=self.var["init_temperature"],
            target_entropy=-self.act_dim,
            n_layer=self.var["n_layer"],
            n_head=self.var["n_head"],
            n_inner=4 * self.var["embed_dim"],
            activation_function=self.var["activation_function"],
            resid_pdrop=self.var["dropout"],
            attn_pdrop=self.var["dropout"],
            transformer_model=self.var["transformer_model"],
            adapter=self.var["adapter"],
        ).to(device=self.device)
        self.optimizer = Lamb(
            self.model.parameters(),
            lr=self.var["learning_rate"],
            weight_decay=self.var["weight_decay"],
            eps=1e-8,
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda steps: min((steps + 1) / self.var["warmup_steps"], 1)
        )
        # if self.var["stochastic_policy"]:
        self.log_temperature_optimizer = torch.optim.Adam(
            [self.model.log_temperature],
            lr=1e-4,
            betas=[0.9, 0.999],
        )
        # else:
        #     self.log_temperature_optimizer = None

        # Operate knowledge distillation
        distiller = Distiller(
            var=self.var,
            teacher=self.teacher,
            student=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            log_temperature_optimizer=self.log_temperature_optimizer,
        )
        while self.distill_iter < self.var["max_distill_iters"]:
            # In every iteration, prepare the data loader
            dataloader = create_dataloader(
                trajectories=self.offline_trajs,
                num_iters=self.var["num_updates_per_distill_iter"],
                batch_size=self.var["batch_size"],
                max_len=self.var["max_seq_len"],
                state_dim=self.state_dim,
                act_dim=self.act_dim,
                state_mean=self.state_mean,
                state_std=self.state_std,
                reward_scale=self.reward_scale,
                action_range=self.action_range,
            )

            train_outputs = distiller.train(dataloader=dataloader)

            self.logs[
                f"Distill logs/iter {self.distill_iter}"
            ] = train_outputs

            self._save_model(
                    path_prefix=self.logger.log_path,
                    is_distill_model=True,
                )

            self.distill_iter += 1

    @torch.no_grad()
    def act(self, state, augment_traj=False):
        if augment_traj:
            if self.var["stochastic_policy"]:
                # More exploration for online finetuning
                use_max_prob = False
            else:
                use_max_prob = True
            scale_return = self.max_return * self.var["online_rtg_scale"]  # TODO:
        else:
            # Less exploration for evaluation
            use_max_prob = True
            scale_return = self.max_return * self.var["eval_rtg_scale"]
            self.model.eval()

        self.model.to(device=self.device)

        state_mean = torch.from_numpy(self.state_mean).to(device=self.device)
        state_std = torch.from_numpy(self.state_std).to(device=self.device)

        num_envs = 1

        if self.reset_idx:
            # We keep all the histories on the device
            # note that the latest action and reward will be "padding"
            self.states = (
                torch.from_numpy(state)
                .reshape(num_envs, self.state_dim)
                .to(device=self.device, dtype=torch.float32)
            ).reshape(num_envs, -1, self.state_dim)
            self.actions = torch.zeros(0, device=self.device, dtype=torch.float32)
            self.rewards = torch.zeros(0, device=self.device, dtype=torch.float32)

            self.target_return = torch.tensor(
                scale_return, device=self.device, dtype=torch.float32
            ).reshape(num_envs, -1, 1)
            self.timesteps = torch.tensor(
                [0] * num_envs, device=self.device, dtype=torch.long
            ).reshape(num_envs, -1)

            # episode_return, episode_length = 0.0, 0
            self.episode_return = np.zeros((num_envs, 1)).astype(float)
            self.episode_length = np.full(num_envs, np.inf)

            self.unfinished = np.ones(num_envs).astype(bool)
            self.t = 0

            self.reset_idx = False

        # the latest action and reward are "padding" since we don't know them yet
        self.actions = torch.cat(
            [
                self.actions,
                torch.zeros((num_envs, self.act_dim), device=self.device).reshape(
                    num_envs, -1, self.act_dim
                ),
            ],
            dim=1,
        )
        self.rewards = torch.cat(
            [
                self.rewards,
                torch.zeros((num_envs, 1), device=self.device).reshape(num_envs, -1, 1),
            ],
            dim=1,
        )

        state_pred, action_dist, reward_pred = self.model.get_predictions(
            (self.states.to(dtype=torch.float32) - state_mean) / state_std,
            self.actions.to(dtype=torch.float32),
            self.rewards.to(dtype=torch.float32),
            self.target_return.to(dtype=torch.float32),
            self.timesteps.to(dtype=torch.long),
            num_envs=num_envs,
        )

        # if self.model.stochastic_policy:
        # The return action is a Categorical distribution
        if use_max_prob:
            # self.action = action_dist.mean
            self.action = action_dist.probs.argmax(dim=-1)
        else:
            self.action = action_dist.sample()
        self.action = self.action.reshape(num_envs, -1, self.act_dim)[:, -1]
        self.action = self.action.clamp(*self.model.action_range)

        return self.action[0].detach().cpu().numpy()

    def observe(self, state, reward, done):
        """Update tokens (states, actions, rewards, etc.)"""
        num_envs = 1
        mode = "normal"

        reward = np.array([reward])
        self.episode_return[self.unfinished] += reward[self.unfinished].reshape(-1, 1)

        self.actions[:, -1] = self.action
        state = (
            torch.from_numpy(state)
            .to(device=self.device)
            .reshape(num_envs, -1, self.state_dim)
        )
        self.states = torch.cat([self.states, state], dim=1)
        reward = torch.from_numpy(reward).to(device=self.device).reshape(num_envs, 1)
        self.rewards[:, -1] = reward

        if mode != "delayed":
            pred_return = self.target_return[:, -1] - (reward * self.reward_scale)
        else:
            pred_return = self.target_return[:, -1]
        self.target_return = torch.cat(
            [self.target_return, pred_return.reshape(num_envs, -1, 1)], dim=1
        )

        self.timesteps = torch.cat(
            [
                self.timesteps,
                torch.ones((num_envs, 1), device=self.device, dtype=torch.long).reshape(
                    num_envs, 1
                )
                * (self.t + 1),
            ],
            dim=1,
        )
        self.t += 1

        if np.any(done):
            ind = np.where(done)[0]
            self.unfinished[ind] = False
            self.episode_length[ind] = np.minimum(self.episode_length[ind], self.t)
            self.reset_idx = True

    def _augment_trajectories(self):
        num_envs = 1

        trajs = []
        for ii in range(num_envs):
            ep_len = self.episode_length[ii].astype(int)
            terminals = np.zeros(ep_len)
            terminals[-1] = 1
            # next_observation is not used in training so we don't need to store it
            traj = {
                "observations": self.states[ii].detach().cpu().numpy()[:ep_len],
                "actions": self.actions[ii].detach().cpu().numpy()[:ep_len],
                "rewards": self.rewards[ii].detach().cpu().numpy()[:ep_len],
                "terminals": terminals,
            }
            trajs.append(traj)

        returns = self.episode_return.reshape(num_envs)
        lengths = self.episode_length.reshape(num_envs)

        self.replay_buffer.add_new_trajs(trajs)
        self.aug_trajs += trajs
        self.total_transitions_sampled += np.sum(lengths)

        return {
            "aug_traj/return": np.mean(returns),
            "aug_traj/length": np.mean(lengths),
        }

    def finetuning(self, iter=0):
        writer = (
            SummaryWriter(self.logger.log_path) if self.var["log_to_tb"] else None
        )
        outputs = {}
        augment_outputs = self._augment_trajectories()
        outputs.update(augment_outputs)

        # Disables training of all weights outside the task adapter
        if self.var["adapter"] is not None:
            self.model.transformer.train_adapter(
                self.var["adapter"]
            )  

        trainer = SequenceTrainer(
            model=self.model,
            optimizer=self.optimizer,
            log_temperature_optimizer=self.log_temperature_optimizer,
            scheduler=self.scheduler,
            device=self.device,
        )
        dataloader = create_dataloader(
            trajectories=self.replay_buffer.trajectories,  # self.replay_buffer.trajectories or self.aug_trajs
            num_iters=self.var["num_updates_per_online_iter"],
            batch_size=self.var["batch_size"],
            max_len=self.var["max_seq_len"],
            state_dim=self.state_dim,
            act_dim=self.act_dim,
            state_mean=self.state_mean,
            state_std=self.state_std,
            reward_scale=self.reward_scale,
            action_range=self.action_range,
        )

        train_outputs = trainer.train_iteration(dataloader=dataloader)

        self.logs[
            f"Finituning logs/iter {iter}"
        ] = train_outputs
        
        outputs.update(train_outputs)
        outputs["Finetuning time/total"] = time.time() - self.start_time

        # Log the metrics
        self.logger.log_metrics(
            outputs,
            iter_num=self.pretrain_iter + self.online_iter,
            total_transitions_sampled=self.total_transitions_sampled,
            writer=writer,
            verbose=False,
        )

        self._save_model(
            path_prefix=self.logger.log_path,
            is_pretrain_model=False,
        )

        self.online_iter += 1
