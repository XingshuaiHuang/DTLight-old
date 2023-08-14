from env import rewards
from env import states

from agents.fixedtime import FIXEDTIME
from agents.stochastic import STOCHASTIC
from agents.maxwave import MAXWAVE
from agents.maxpressure import MAXPRESSURE, EMP
from agents.pfrl_dqn import IDQN
from agents.pfrl_ppo import IPPO
from agents.mplight import MPLight

agent_configs = {
    # *VAL configs have distance settings according to the validation scenarios
    'MAXWAVEVAL': {
        'agent': MAXWAVE,
        'state': states.wave,
        'reward': rewards.wait,
        'max_distance': 50
    },
    'MAXPRESSUREVAL': {
        'agent': MAXPRESSURE,
        'state': states.mplight,
        'reward': rewards.wait,
        'max_distance': 9999
    },
    'MPLightVAL': {
        'agent': MPLight,
        'state': states.mplight,
        'reward': rewards.pressure,
        'max_distance': 9999,
        'BATCH_SIZE': 32,
        'GAMMA': 0.99,
        'EPS_START': 1.0,
        'EPS_END': 0.0,
        'EPS_DECAY': 220,
        'TARGET_UPDATE': 500,
        'demand_shape': 1
    },

    'FIXEDTIME': {
        'agent': FIXEDTIME,
        'state': states.mplight,
        'reward': rewards.wait,
        'max_distance': 1
    },
    'STOCHASTIC': {
        'agent': STOCHASTIC,
        'state': states.mplight,
        'reward': rewards.wait,
        'max_distance': 1
    },
    'MAXWAVE': {
        'agent': MAXWAVE,
        'state': states.wave,
        'reward': rewards.wait,
        'max_distance': 50
    },
    'MAXPRESSURE': {
        'agent': MAXPRESSURE,
        'state': states.mplight,
        'reward': rewards.wait,
        'max_distance': 200
    },
    'EMP': {
        'agent': EMP,
        'state': states.mplight,
        'reward': rewards.wait,
        'max_distance': 200, 
        'epsilon_decay': True,
    },
    'IDQN': {
        'agent': IDQN,
        'algorithm': 'dqn',
        'q_net': 'cnn',
        'state': states.drq_norm,
        'reward': rewards.wait_norm,
        'max_distance': 200,
        'steps': 100*0.8*360,  
        'BATCH_SIZE': 32,
        'GAMMA': 0.99,
        'EPS_START': 1.0,  # Max epsilon
        'EPS_END': 0.0,  # Min epsilon
        'EPS_DECAY': 220,  # Epsilon decay
        'TARGET_UPDATE': 500
    },
    'IPPO': {
        'agent': IPPO,
        'state': states.drq_norm,
        'reward': rewards.wait_norm,
        'max_distance': 200,
        'lr': 2.5e-4,
        'eps': 1e-5,
        'clip_eps': 0.1,
        'update_interval': 1024,
        'minibatch_size': 256,
        'epochs': 4,
        'entropy_coef': 0.01,
        'max_grad_norm': 0.5,
    },
    'MPLight': {
        'agent': MPLight,
        'state': states.mplight,
        'reward': rewards.pressure,
        'max_distance': 200,
        'BATCH_SIZE': 32,
        'GAMMA': 0.99,
        'steps': 28800,  
        'EPS_START': 1.0,
        'EPS_END': 0.0,
        'EPS_DECAY': 220,
        'TARGET_UPDATE': 500,
        'demand_shape': 1
    },

    # *FULL configs extend state space to include obs. available to IDQN
    'MPLightFULL': {
        'agent': MPLight,
        'state': states.mplight_full,
        'reward': rewards.pressure,
        'max_distance': 200,
        'BATCH_SIZE': 32,
        'GAMMA': 0.99,
        'EPS_START': 1.0,
        'EPS_END': 0.0,
        'EPS_DECAY': 220,
        'TARGET_UPDATE': 500,
        'demand_shape': 4
    },
    'MA2C': {  # Only works for config_policy in offline_data_generator
        'state': states.ma2c,
        'reward': rewards.pressure,
        'max_distance': 200,
},
}
