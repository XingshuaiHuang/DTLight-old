import numpy as np
from agents.agent import SharedAgent, Agent
from configs.signal_config import signal_configs


class MAXWAVE(SharedAgent):
    def __init__(self, config, obs_act, map_name, thread_number):
        super().__init__(config, obs_act, map_name, thread_number)
        self.valid_acts = signal_configs[map_name]['valid_acts']
        self.agent = WaveAgent(signal_configs[map_name]['phase_pairs'])


class WaveAgent(Agent):
    def __init__(self, phase_pairs, epsilon_decay=False):
        super().__init__()
        self.phase_pairs = phase_pairs
        self.epsilon_decay= epsilon_decay
        self.epsilon = 1
        self.step = 0

    def act(self, observations, valid_acts=None, reverse_valid=None):
        # Linearly decay epsilon
        if self.epsilon_decay:
            self.step += 1
            self.epsilon = max(0, 1 - self.step / (360 * 100 * 0.8))
        acts = []
        for i, observation in enumerate(observations):
            if valid_acts is None:
                all_press = []
                for pair in self.phase_pairs:
                    all_press.append(observation[pair[0]] + observation[pair[1]])
                if self.epsilon_decay and np.random.random() < self.epsilon:
                    acts.append(np.random.randint(len(all_press)))
                else:
                    acts.append(np.argmax(all_press))
            else:
                max_press, max_index = None, None
                for idx in valid_acts[i]:
                    pair = self.phase_pairs[idx]
                    press = observation[pair[0]] + observation[pair[1]]
                    if max_press is None:
                        max_press = press
                        max_index = idx
                    if press > max_press:
                        max_press = press
                        max_index = idx
                if self.epsilon_decay and np.random.random() < self.epsilon:
                    acts.append(np.random.choice(list(valid_acts[i].values())))
                else:
                    acts.append(valid_acts[i][max_index])
        return acts

    def observe(self, observation, reward, done, info):
        pass

    def save(self, path):
        pass