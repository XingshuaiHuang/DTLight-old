from dtlight.dtlight import DTLight


class MADTLight():
    """Multi-agent DTLight"""
    def __init__(
        self, 
        var, 
        obs_act, 
        multi_offline_trajs, 
        max_returns,
        state_means, 
        state_stds
    ):
        self.agents = dict()
        for key in obs_act:
            obs_space = obs_act[key][0]  # (num_states, )
            num_actions = obs_act[key][1]  # int: num_phases
            state_dim = obs_space[-1]
            act_dim = 1
            action_range = [0, num_actions - 1]
            offline_trajs = multi_offline_trajs[key]
            max_return = max_returns[key]
            state_mean = state_means[key]
            state_std = state_stds[key]
            signal = key
            self.agents[key] = DTLight(
                var=var, 
                signal=signal,
                state_dim=state_dim, 
                act_dim=act_dim, 
                action_range=action_range,
                offline_trajs=offline_trajs,
                max_return=max_return,
                state_mean=state_mean,
                state_std=state_std,
            )

    def pretrain(self):  # TODO: multi-processing
        for agent_id in self.agents.keys():
            self.agents[agent_id].pretrain()
    
    def distill(self):
        for agent_id in self.agents.keys():
            self.agents[agent_id].distill()

    def act(
        self,
        state_dict, 
        augment_traj=False,
    ):

        actions = dict()
        for agent_id in state_dict.keys():
            action = self.agents[agent_id].act(
                state_dict[agent_id], 
                augment_traj
            )
            actions[agent_id] = action
            
        return actions
    
    def observe(
        self,
        state_dict, 
        reward_dict, 
        done
    ):
        for agent_id in state_dict.keys():
            self.agents[agent_id].observe(
                state_dict[agent_id],
                reward_dict[agent_id],
                done
            )

    def finetuning(self, iter):
        for agent_id in self.agents.keys():
            self.agents[agent_id].finetuning(iter)