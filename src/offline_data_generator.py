import os
import pathlib
import multiprocessing as mp
import numpy as np
import torch
import random
import pickle
import argparse
from tqdm import tqdm
from collections import defaultdict
from datetime import datetime

from env.multi_signal import MultiSignal
from configs.agent_config import agent_configs
from configs.map_config import map_configs
from configs.mdp_config import mdp_configs
from summary.readXML import readXML


class ReplayBufferDataset():
    def __init__(self, obs_act):
        self.obs_act = obs_act
        self.offline_buffer = {}
        for signal in obs_act:
            self.offline_buffer[signal] = {
                'observations': [],
                'actions': [],
                'rewards': [],
                'next_observations': [],
                'terminals': [],
            }
        self.offline_buffer['results'] = {}

    def add(self, old_obs, act, rew, obs, done):
        for signal in self.obs_act:
            self.offline_buffer[signal]['observations'].append(old_obs[signal])
            self.offline_buffer[signal]['actions'].append(act[signal])
            self.offline_buffer[signal]['rewards'].append(rew[signal])
            self.offline_buffer[signal]['next_observations'].append(obs[signal])
            self.offline_buffer[signal]['terminals'].append(done)

    def add_result(self, result, result_eps):
        self.offline_buffer['results'][result_eps] = result

    def save(self, data_dir, file_name, save_dict=False):
        def split_into_trajectories(buffer):
            trajs = [defaultdict(list)]
            for i in tqdm(range(len(buffer['observations']))):
                trajs[-1]["observations"].append(buffer['observations'][i])
                trajs[-1]["actions"].append(buffer['actions'][i])
                trajs[-1]["rewards"].append(buffer['rewards'][i])
                trajs[-1]["next_observations"].append(buffer['next_observations'][i])
                trajs[-1]["terminals"].append(buffer['terminals'][i])
                if buffer['terminals'][i] and i + 1 < len(buffer['observations']):
                    trajs.append(defaultdict(list))
            for traj in trajs:
                for kk, vv in traj.items():
                    traj[kk] = np.array(vv)
            return trajs
            
        def save_pkl(file_path, replay_buffer):
            with open((file_path), 'wb') as f:
                pickle.dump(replay_buffer, f)
            print(f'Saved replay buffer to {file_path}\n\n')

        trajectories_arrays = {}
        for signal in self.obs_act:
            trajectories_array = split_into_trajectories(self.offline_buffer[signal])
            returns = np.array([np.sum(traj["rewards"]) for traj in trajectories_array])
            lengths = np.array([len(traj["rewards"]) for traj in trajectories_array])
            num_samples = np.sum(lengths)
            print(f"number of samples collected: {num_samples}")
            print(
                f"Trajectory returns: mean = {np.mean(returns)}, std = {np.std(returns)}, \
                    max = {np.max(returns)}, min = {np.min(returns)}"
                f"\n"
                f"Trajectory lengths: mean = {np.mean(lengths)}, std = {np.std(lengths)}, \
                    max = {np.max(lengths)}, min = {np.min(lengths)}"
            )
            trajectories_arrays[signal] = trajectories_array
        trajectories_arrays['results'] = self.offline_buffer['results']

        os.makedirs(args.data_dir, exist_ok=True)
        file_path = os.path.join(data_dir, f'{file_name}.pkl')
        save_pkl(file_path, trajectories_arrays)


def _seed_everything(seed=13):
    print(f'Setting everything to seed {seed}')
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # tf.random.set_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'


def _run_trial(args, trial):
    _seed_everything(seed=trial)
    data_dict = {'hour': 3600, 'day': 3600 * 24, 'week': 3600 * 24 * 7, 'month': 3600 * 24 * 30}

    # ----- Add neighborhood info in offline data (only for MA2C-related config policy)-----
    from configs.mdp_config import mdp_configs
    
    mdp_config = mdp_configs.get(args.config_policy)
    if mdp_config is not None:
        mdp_map_config = mdp_config.get(args.map)
        if mdp_map_config is not None:
            mdp_config = mdp_map_config
        mdp_configs[args.config_policy] = mdp_config
    if mdp_config is not None:
        management = mdp_config.get('management')
        if management is not None:    # Save some time and precompute the reverse mapping
            supervisors = dict()
            for manager in management:
                workers = management[manager]
                for worker in workers:
                    supervisors[worker] = manager
            mdp_config['supervisors'] = supervisors
    # -------------------------------------------------------------------

    agt_config = agent_configs[args.agent]
    agt_map_config = agt_config.get(args.map)
    if agt_map_config is not None:
        agt_config = agt_map_config
    alg = agt_config['agent']

    map_config = map_configs[args.map]
    # =========== Offline data specific ===========
    if 'hour' not in args.simu_time:
        map_config['net'] = map_config['net'].replace(
            '.sumocfg', '_{}_random.sumocfg'.format(args.simu_time)
            )
        map_config['start_time'] = 0
        map_config['end_time'] = map_config['start_time'] + data_dict[args.simu_time]
    # =============================================

    route = map_config['route']
    if route is not None:
        route = os.path.join(args.pwd, route)

    suffix = datetime.now().strftime("%H%M%S")
    env = MultiSignal(alg.__name__+'_'+suffix+'-tr'+str(trial),
                      args.map,
                      os.path.join(args.pwd, map_config['net']),
                      agt_config['state'],
                      agt_config['reward'],
                      route=route, step_length=map_config['step_length'], 
                      yellow_length=map_config['yellow_length'],
                      step_ratio=map_config['step_ratio'], 
                      end_time=map_config['end_time'],
                      max_distance=agt_config['max_distance'], 
                      lights=map_config['lights'], gui=args.gui,
                      log_dir=args.log_dir, libsumo=args.libsumo, 
                      warmup=map_config['warmup'],
                      offline_rl=args.config_policy)  # Make obs and rew the same as offlineRL

    agt_config['log_dir'] = os.path.join(args.log_dir, env.connection_name)
    agt_config['num_lights'] = len(env.all_ts_ids)

    # Get agent id's, observation shapes, and action sizes from env
    obs_act = dict()
    for key in env.obs_shape:
        obs_act[key] = [env.obs_shape[key], len(
            env.phases[key]) if key in env.phases else None]
    agent = alg(agt_config, obs_act, args.map, trial)

    # =========== Collect offline data ===========
    replay_buffer = ReplayBufferDataset(obs_act)
    step = 0
    for e in range(args.num_eps):
        obs, off_obs = env.reset()
        done = False
        while not done:
            step += 1
            eps_done = done
            old_off_obs = off_obs
            act = agent.act(obs)
            # Obs and rew should have the same settings as the offlineRL agent
            obs, rew, done, info, off_obs, off_rew = env.step(act)
            agent.observe(obs, rew, done, info)
            if done:
                eps_done = 1
                print('Add trajectory for episode {}'.format(e))
            replay_buffer.add(old_off_obs, act, off_rew, off_obs, eps_done)
        if 'hour' in args.simu_time:
            _, _, current_delay, current_eps= readXML(
                log=args.log_dir, plot=False, print_delay=True, save_result=False, task=[env.connection_name])
            replay_buffer.add_result(current_delay, current_eps)
    replay_buffer.save(
        data_dir=args.data_dir, 
        file_name=f'{args.map}_{args.simu_time}_{args.agent}_{args.config_policy}'
    )
    # ============================================


def main(args):
    if args.procs == 1 or args.libsumo:
        min_avg_delay = _run_trial(args, args.seed)
        return min_avg_delay
    else:
        pool = mp.Pool(processes=args.procs)
        for trial in range(1, args.trials+1):
            pool.apply_async(_run_trial, args=(args, trial))
        pool.close()
        pool.join()


def args_behavior():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, default='IDQN',
                        choices=['STOCHASTIC', 'MAXWAVE', 'MAXPRESSURE',
                                 'IDQN', 'IPPO', 'MPLight', 'MA2C', 'FMA2C',
                                 'MPLightFULL', 'FMA2CFull', 'FMA2CVAL'])
    parser.add_argument("--map", type=str, default='2lane',
                        choices=['grid4x4', 'arterial4x4', 
                                 '3lane', '2lane',
                                 'ingolstadt1', 'cologne1', 
                                 'cologne3', 'cologne8'])
    parser.add_argument("--simu_time", type=str, default='100-hour')
    parser.add_argument("--config_policy", type=str, default='MA2C',
                        help='for defining the state and reward functions used in offlineRL')
    parser.add_argument("--num_eps", type=int, default=100, help='training episode')
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--procs", type=int, default=1)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--pwd", type=str, default=str(pathlib.Path().absolute())+os.sep)
    parser.add_argument("--log_dir", type=str, default='logs/')
    parser.add_argument("--data_dir", type=str, 
                        default=str(pathlib.Path().absolute())+os.sep+'DTRL'+os.sep)
    parser.add_argument("--gui", action='store_true', default=False)
    parser.add_argument("--libsumo", action='store_true', default=False)
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = args_behavior()
    args.libsumo = True

    args.num_eps = int(args.simu_time.split('-')[0])
    
    main(args)

