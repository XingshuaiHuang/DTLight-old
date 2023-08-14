"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the CC BY-NC license found in the
LICENSE.md file in the root directory of this source tree.
"""

import numpy as np
import torch
import time
from tqdm import tqdm
import torch.nn.functional as F


def stochastic_loss_fn(
        a_pred_dist,
        a_target,
        attention_mask,
        entropy_reg
    ):
    log_likelihood = a_pred_dist.log_prob(
        a_target.squeeze(-1))[attention_mask > 0].mean()
    entropy = a_pred_dist.entropy().mean()
    loss = -(log_likelihood + entropy_reg * entropy)
    return (loss, -log_likelihood, entropy)


class SequenceTrainer:
    def __init__(
        self,
        model,
        optimizer,
        log_temperature_optimizer,
        scheduler=None,
        device="cuda",
    ):
        self.model = model
        self.model.train()
        self.optimizer = optimizer
        self.log_temperature_optimizer = log_temperature_optimizer
        self.scheduler = scheduler
        self.device = device

    def compute_loss(self, trajs):
        (
            states,
            actions,
            rewards,
            dones,
            rtg,
            timesteps,
            ordering,
            padding_mask,
        ) = trajs

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)
        rtg = rtg.to(self.device)
        timesteps = timesteps.to(self.device)
        ordering = ordering.to(self.device)
        padding_mask = padding_mask.to(self.device)

        action_target = torch.clone(actions)

        _, action_preds, _, _ = self.model.forward(
            states,
            actions,
            rewards,
            rtg[:, :-1],
            timesteps,
            ordering,
            padding_mask=padding_mask,
        )

        if self.model.stochastic_policy:
            loss, nll, entropy = stochastic_loss_fn(
                action_preds,  # a_pred_dist
                action_target,
                padding_mask,
                self.model.temperature().detach(),  # no gradient taken here
            )
        else:
            # nll loss (treat action preds as logits)
            loss = F.nll_loss(
                action_preds.permute(0,2,1), 
                action_target.squeeze(-1).long()
            )
            nll = torch.tensor(0.0, device=loss.device)
            entropy = torch.tensor(0.0, device=loss.device)

        return loss, nll, entropy

    def _train_step(self, trajs):
        loss, nll, entropy = self.compute_loss(trajs)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
        self.optimizer.step()
        
        if self.model.stochastic_policy:
            self.log_temperature_optimizer.zero_grad()
            temperature_loss = (
                self.model.temperature() * (entropy - self.model.target_entropy).detach()
            )
            temperature_loss.backward()
            self.log_temperature_optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        return (
                loss.detach().cpu().item(),
                nll.detach().cpu().item(),
                entropy.detach().cpu().item(),
            )
            

    def train_iteration(
        self,
        dataloader,
    ):

        losses, nlls, entropies = [], [], []
        logs = dict()
        train_start = time.time()

        iter_bar = tqdm(enumerate(dataloader), total=len(dataloader))
        for _, trajs in iter_bar:
            loss, nll, entropy = self._train_step(trajs)
            losses.append(loss)
            nlls.append(nll)
            entropies.append(entropy)
            
            iter_bar.update()
            iter_bar.set_postfix(
                {"Loss": f"{loss:.2f}", "Avg_loss": f"{np.mean(losses):.2f}"}
            )
        iter_bar.close()

        logs["training_time"] = time.time() - train_start
        if self.model.stochastic_policy:
            logs["last_nll"] = nlls[-1]
            logs["last_entropy"] = entropies[-1]
            logs["temp_value"] = self.model.temperature().detach().cpu().item()
        logs["last_loss"] = losses[-1]
        logs["avg_loss"] = np.mean(losses)
        # logs["loss_std"] = np.std(losses)

        return logs
