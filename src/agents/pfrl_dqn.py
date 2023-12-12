from typing import Any, Sequence
import numpy as np

import torch
import torch.nn as nn

import pfrl
from pfrl import explorers, replay_buffers
from pfrl.explorer import Explorer
from pfrl.agents import DQN, DoubleDQN, IQN
from pfrl.q_functions import DiscreteActionValueHead
from pfrl.utils.contexts import evaluating

from agents.agent import IndependentAgent, Agent


def conv2d_size_out(h_in, kernel_size=2, stride=1):
    return (h_in - (kernel_size - 1) - 1) // stride + 1


class SelectItem(nn.Module):
    def __init__(self, item_index):
        super(SelectItem, self).__init__()
        self.item_index = item_index

    def forward(self, inputs):
        return inputs[self.item_index]


class Squeeze(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, inputs):
        return inputs.squeeze(self.dim)


class IDQN(IndependentAgent):
    """Learning an interpretable traffic signal control policy"""

    def __init__(self, config, obs_act, map_name, thread_number):
        super().__init__(config, obs_act, map_name, thread_number)
        print('Using {} as q_net'.format(config['q_net']))
        print('Using {} as algorithm'.format(config['algorithm']))

        for key in obs_act:
            obs_space = obs_act[key][0]  # (1, num_lanes_per_intersection, obs_dim=num_states)
            act_space = obs_act[key][1]  # int: num_phases

            if config['q_net'] == 'cnn':
                h = conv2d_size_out(obs_space[1])
                w = conv2d_size_out(obs_space[2])
                model = nn.Sequential(
                    nn.Conv2d(obs_space[0], 64, kernel_size=(2, 2)),
                    nn.ReLU(),
                    nn.Flatten(),
                    nn.Linear(h * w * 64, 64),
                    nn.ReLU(),
                    nn.Linear(64, 64),
                    nn.ReLU(),
                    nn.Linear(64, act_space),
                    DiscreteActionValueHead()
                )
            else:
                print('q_net is not defined correctly.')

            self.agents[key] = DQNAgent(config, act_space, model)


class DQNAgent(Agent):
    def __init__(self, config, act_space, model, num_agents=0):
        super().__init__()

        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters())
        replay_buffer = replay_buffers.ReplayBuffer(10000)

        if num_agents > 0:
            explorer = SharedEpsGreedy(
                config['EPS_START'],
                config['EPS_END'],
                num_agents*config['steps'],
                lambda: np.random.randint(act_space),
            )
        else:
            explorer = explorers.LinearDecayEpsilonGreedy(
                config['EPS_START'],
                config['EPS_END'],
                config['steps'],
                lambda: np.random.randint(act_space),
            )

        if num_agents > 0:
            print('USING SHAREDDQN')
            self.agent = SharedDQN(self.model, self.optimizer, replay_buffer,
                                   config['GAMMA'], explorer, gpu=self.device.index,
                                   minibatch_size=config['BATCH_SIZE'], replay_start_size=config['BATCH_SIZE'],
                                   phi=lambda x: np.asarray(x, dtype=np.float32),
                                   target_update_interval=config['TARGET_UPDATE']*num_agents, update_interval=num_agents)
        else:
            if config['algorithm'] == 'dqn':
                self.agent = DQN(self.model, self.optimizer, replay_buffer, 
                                config['GAMMA'], explorer,
                                gpu=self.device.index,
                                minibatch_size=config['BATCH_SIZE'], 
                                replay_start_size=config['BATCH_SIZE'],
                                phi=lambda x: np.asarray(x, dtype=np.float32),
                                target_update_interval=config['TARGET_UPDATE'])
            else:
                print('Optimization algorithm is not defined correctly.')

    def act(self, observation, valid_acts=None, reverse_valid=None):
        if isinstance(self.agent, SharedDQN):
            return self.agent.act(observation, valid_acts=valid_acts, reverse_valid=reverse_valid)
        else:
            return self.agent.act(observation)  # observation.shape: (1, num_lanes_per_inter, obs_dim)

    def observe(self, observation, reward, done, info):
        if isinstance(self.agent, SharedDQN):
            self.agent.observe(observation, reward, done, info)
        else:
            self.agent.observe(observation, reward, done, False)

    def save(self, path):
        if len(path) > 180:  # ** when path is too long, FileNotFoundError occurs
            path = path[: path.find('agent_cluster_')] + 'agent_cluster_multi_clusters'
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path + '.pt')


class SharedDQN(DQN):
    def __init__(self, q_function: torch.nn.Module, optimizer: torch.optim.Optimizer,
                 replay_buffer: pfrl.replay_buffer.AbstractReplayBuffer, gamma: float, explorer: Explorer,
                 gpu, minibatch_size, replay_start_size, phi, target_update_interval, update_interval):

        super().__init__(q_function, optimizer, replay_buffer, gamma, explorer,
                         gpu=gpu, minibatch_size=minibatch_size, replay_start_size=replay_start_size, phi=phi,
                         target_update_interval=target_update_interval, update_interval=update_interval)

    def act(self, obs: Any, valid_acts=None, reverse_valid=None) -> Any:
        return self.batch_act(obs, valid_acts=valid_acts, reverse_valid=reverse_valid)

    def observe(self, obs: Sequence[Any], reward: Sequence[float], done: Sequence[bool], reset: Sequence[bool]) -> None:
        self.batch_observe(obs, reward, done, reset)

    def batch_act(self, batch_obs: Sequence[Any], valid_acts=None, reverse_valid=None) -> Sequence[Any]:
        if valid_acts is None: return super(SharedDQN, self).batch_act(batch_obs)
        with torch.no_grad(), evaluating(self.model):
            batch_av = self._evaluate_model_and_update_recurrent_states(batch_obs)

            batch_qvals = batch_av.params[0].detach().cpu().numpy()
            batch_argmax = []
            for i in range(len(batch_obs)):
                batch_item = batch_qvals[i]
                max_val, max_idx = None, None
                for idx in valid_acts[i]:
                    batch_item_qval = batch_item[idx]
                    if max_val is None:
                        max_val = batch_item_qval
                        max_idx = idx
                    elif batch_item_qval > max_val:
                        max_val = batch_item_qval
                        max_idx = idx
                batch_argmax.append(max_idx)
            batch_argmax = np.asarray(batch_argmax)

        if self.training:
            batch_action = []
            for i in range(len(batch_obs)):
                av = batch_av[i : i + 1]
                greed = batch_argmax[i]
                act, greedy = self.explorer.select_action(self.t, lambda: greed, action_value=av, num_acts=len(valid_acts[i]))
                if not greedy:
                    act = reverse_valid[i][act]
                batch_action.append(act)

            self.batch_last_obs = list(batch_obs)
            self.batch_last_action = list(batch_action)
        else:
            batch_action = batch_argmax

        valid_batch_action = []
        for i in range(len(batch_action)):
            valid_batch_action.append(valid_acts[i][batch_action[i]])
        return valid_batch_action


def select_action_epsilon_greedily(epsilon, random_action_func, greedy_action_func):
    if np.random.rand() < epsilon:
        return random_action_func(), False
    else:
        return greedy_action_func(), True


class SharedEpsGreedy(explorers.LinearDecayEpsilonGreedy):

    def select_action(self, t, greedy_action_func, action_value=None, num_acts=None):
        self.epsilon = self.compute_epsilon(t)
        if num_acts is None:
            fn = self.random_action_func
        else:
            fn = lambda: np.random.randint(num_acts)
        a, greedy = select_action_epsilon_greedily(
            self.epsilon, fn, greedy_action_func
        )
        greedy_str = "greedy" if greedy else "non-greedy"
        self.logger.debug("t:%s a:%s %s", t, a, greedy_str)
        if num_acts is None:
            return a
        else:
            return a, greedy