import pickle
import json
import time
import os
import torch
import numpy as np

from dtlight.ma_dtlight import MADTLight
from global_utils.utils import set_seed_everywhere, NpEncoder
from env.multi_signal import MultiSignal
from summary.readXML import readXML
from configs.agent_config import agent_configs
from configs.map_config import map_configs
from configs.mdp_config import mdp_configs
from parser_offline import args_offline
from global_utils.utils import insert_data_to_json


class Experiment:
    def __init__(self, var):
        self.var = var
        self.env, obs_act = self._build_env()
        multi_offline_trajs, max_returns, state_means, state_stds = self._load_dataset(
            self.var["map"], 
            self.var["data_period"], 
            self.var["behavior_policy"], 
            self.var["config_policy"]
        )
        
        self.agent = MADTLight(
            var=self.var,
            obs_act=obs_act,
            multi_offline_trajs=multi_offline_trajs,
            max_returns=max_returns,
            state_means=state_means,
            state_stds=state_stds,
        )

        self.pretrain_iter = 0
        self.distill_iter = 0
        self.online_iter = 0
        self.logs = {'delay_history': {}, 'training_time': {}}

    def _build_env(self):
        mdp_config = mdp_configs.get(self.var["config_policy"])
        if mdp_config is not None:
            mdp_map_config = mdp_config.get(self.var["map"])
            if mdp_map_config is not None:
                mdp_config = mdp_map_config
            mdp_configs[self.var["config_policy"]] = mdp_config

        agt_config = agent_configs[self.var["config_policy"]]
        agt_map_config = agt_config.get(self.var["map"])
        if agt_map_config is not None:
            agt_config = agt_map_config

        if mdp_config is not None:
            agt_config["mdp"] = mdp_config
            management = agt_config["mdp"].get("management")
            if (
                management is not None
            ):  # Save some time and precompute the reverse mapping
                supervisors = dict()
                for manager in management:
                    workers = management[manager]
                    for worker in workers:
                        supervisors[worker] = manager
                mdp_config["supervisors"] = supervisors

        map_config = map_configs[self.var["map"]]
        route = map_config["route"]
        if route is not None:
            route = os.path.join(self.var["pwd"], route)

        # suffix = time.strftime("_%m_%d_%H_%M_%S", time.localtime(time.time()))
        policy_out = 'sto' if self.var["stochastic_policy"] else 'det'
        suffix = "_{}_{}_e{}_o{}_d{}_{}".format(
            self.var["behavior_policy"],
            policy_out,
            '~' + str(-self.var["eval_rtg_scale"]) if self.var["eval_rtg_scale"] < 0 else self.var["eval_rtg_scale"],
            '~' + str(-self.var["online_rtg_scale"]) if self.var["online_rtg_scale"] < 0 else self.var["online_rtg_scale"],
            self.var["alpha_ce"],
            'compacterp' if self.var["adapter"] == 'compacter++' else self.var["adapter"],
        )
        run_name = "DTLight" + suffix + "-tr" + str(self.var["seed"])
        env = MultiSignal(
            run_name,
            self.var["map"],
            os.path.join(self.var["pwd"], map_config["net"]),
            agt_config["state"],
            agt_config["reward"],
            route=route,
            step_length=map_config["step_length"],
            yellow_length=map_config["yellow_length"],
            step_ratio=map_config["step_ratio"],
            end_time=map_config["end_time"],
            max_distance=agt_config["max_distance"],
            lights=map_config["lights"],
            gui=self.var["gui"],
            log_dir=self.var["log_dir"],
            libsumo=self.var["libsumo"],
            warmup=map_config["warmup"],
        )
        obs_act = dict()
        for key in env.obs_shape:
            obs_act[key] = [
                env.obs_shape[key],
                len(env.phases[key]) if key in env.phases else agt_config["management_acts"],
            ]

        return env, obs_act

    def _load_dataset(self, map_name, data_period, behavior_policy, config_policy):
        if config_policy == "EMP":
            dataset_path = f"DTRL/{map_name}_{data_period}_{behavior_policy}.pkl"
        else:
            dataset_path = f"DTRL/{map_name}_{data_period}_{behavior_policy}_{config_policy}.pkl"
        with open(dataset_path, "rb") as f:
            total_trajectories = pickle.load(f)
        print(f"Loaded dataset from {dataset_path}\n")

        multi_trajectories = dict()
        max_returns = dict()
        state_means = dict()
        state_stds = dict()
        if 'results' in total_trajectories:
            print(f"\n***** {map_name} Min Dealy: {min(total_trajectories['results'].values())} *****\n")
            # print(f"\n***** {map_name} Dealy: {total_trajectories['results'].values()} *****\n")
            total_trajectories.pop('results')
        for signal in total_trajectories:
            trajectories = total_trajectories[signal]
            states, traj_lens, returns = [], [], []
            for path in trajectories:
                states.append(path["observations"])
                traj_lens.append(len(path["observations"]))
                returns.append(path["rewards"].sum())
            traj_lens, returns = np.array(traj_lens), np.array(returns)

            # used for input normalization
            states = np.concatenate(states, axis=0)
            state_mean, state_std = (
                np.mean(states, axis=0),
                np.std(states, axis=0) + 1e-6,
            )
            num_timesteps = sum(traj_lens)

            sorted_inds = np.argsort(returns)  # lowest to highest
            num_trajectories = 1
            timesteps = traj_lens[sorted_inds[-1]]
            ind = len(trajectories) - 2
            while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] < num_timesteps:
                timesteps += traj_lens[sorted_inds[ind]]
                num_trajectories += 1
                ind -= 1
            sorted_inds = sorted_inds[-num_trajectories:]
            trajectories = [trajectories[ii] for ii in sorted_inds]

            multi_trajectories[signal] = trajectories
            max_returns[signal] = np.max(returns)
            state_means[signal] = state_mean
            state_stds[signal] = state_std

        return multi_trajectories, max_returns, state_means, state_stds

    def evaluate(self, augment_traj=False):
        eval_start = time.time()
        if not augment_traj:
            print("Evaluating Policy.....\n")

        state_dict = self.env.reset()
        reward_dict = None
        done = False
        while not done:
            action_dict = self.agent.act(state_dict, augment_traj)
            state_dict, reward_dict, done, _ = self.env.step(action_dict)
            self.agent.observe(state_dict, reward_dict, done)

        if not augment_traj:
            print(f"Evaluation Time: {time.time() - eval_start:.2f}")
            min_delay, min_eps, current_delay, current_eps = readXML(
                log=self.var["log_dir"],
                plot=False,
                print_delay=False if augment_traj else True,
                save_result=False,
                task=[self.env.connection_name],
            )
            self.logs['delay_history'][current_eps] = current_delay

            return current_delay, current_eps

    def pretrain(self):
        print("\n\n***************** Pretraining *****************\n\n")
        pretrain_start = time.time()
        self.agent.pretrain()
        self.logs['training_time']['pretrain'] = time.time() - pretrain_start
        current_delay, current_eps = self.evaluate(augment_traj=False)

        return current_delay, current_eps
    
    def distill(self):
        print("\n\n***************** Kownledge Distillation *****************\n\n")
        distill_start = time.time()
        self.agent.distill()
        self.logs['training_time']['distill'] = time.time() - distill_start
        current_delay, current_eps = self.evaluate(augment_traj=False)

        return current_delay, current_eps

    def online_tuning(self):
        print("\n\n***************** Online Finetuning *****************\n\n")
        online_min_delay = np.Inf
        online_min_eps = 0

        while self.online_iter < self.var["max_online_iters"]:
            print("--- Augmenting Episode {} ---\n".format(self.online_iter + 1))
            self.evaluate(augment_traj=True)  # More exploration

            print("--- Fintuning Episode {} ---\n".format(self.online_iter + 1))
            finetuning_start = time.time()
            self.agent.finetuning(iter=self.online_iter)
            self.logs['training_time'][f'finetuning_last_iter'] = time.time() - finetuning_start

            is_last_iter = self.online_iter == self.var["max_online_iters"] - 1
            if (self.online_iter + 1) % self.var["eval_interval"] == 0 or is_last_iter:
                current_delay, current_eps = self.evaluate(augment_traj=False)
                if current_delay < online_min_delay:
                    online_min_delay = current_delay
                    online_min_eps = current_eps

            self.online_iter += 1

        return online_min_delay, online_min_eps
    
    def save_logs(self):
        exp_setting_list = ['seed', 'data_period', 'behavior_policy', 'config_policy']
        struct_list = ['transformer_model', 'adapter', 'n_layer', 'n_head', 'embed_dim']
        self.logs['exp_setting'] = {}
        self.logs['model_structure'] = {}
        for setting in exp_setting_list:
            self.logs['exp_setting'][setting] = self.var[setting]
        for struct in struct_list:
            self.logs['model_structure'][struct] = self.var[struct]

        # for signal, agent in self.agent.agents.items():
        #     self.logs[signal] = agent.logs

        log_dir = os.path.join(self.var['save_dir'], 'exp_logs')
        os.makedirs(log_dir, exist_ok=True)
        file_path = f"{log_dir}/{self.var['map']}_logs.json"
        with open(file_path, "a") as f:
            json.dump(self.logs, f, cls=NpEncoder, indent=4)
            
        print(f"Saved logs to {file_path}\n")

    def __call__(self):
        print(self.var)
        set_seed_everywhere(args.seed)

        teacher_min_delay, _ = self.pretrain()

        if self.var["max_distill_iters"]:
            pretrain_min_delay, _ = self.distill()

        if self.var["max_online_iters"]:
            online_min_delay, online_min_eps = self.online_tuning()  # TODO:

        self.save_logs()
        print("\n=*=*=*=*=*=*=* END EXPERIMENT =*=*=*=*=*=*=*=\n")

        return teacher_min_delay, pretrain_min_delay, online_min_delay, online_min_eps, self.logs


if __name__ == "__main__":
    args = args_offline()
    args.libsumo = True
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    experiment = Experiment(vars(args))
    experiment()