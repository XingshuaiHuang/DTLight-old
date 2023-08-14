import argparse
import pathlib
import os


def args_offline():
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", type=str, default="ingolstadt1",
                        choices=['ingolstadt1', 'cologne1', 
                                 'cologne3', 'cologne8',
                                 'grid4x4', 'arterial4x4',
                                 '3lane', 'arterial1', '2lane'])
    parser.add_argument("--data_period", type=str, default="100-hour",
                        help='period of offline data collection')
    parser.add_argument("--behavior_policy", type=str, default='EMP',
                        help='behavior policy used in offline data collection')
    parser.add_argument("--config_policy", type=str, default='MA2C',
                        help='policy used for env configuration (state, reward)')
    parser.add_argument("--seed", type=int, default=13)

    # Model options
    parser.add_argument("--load_model", type=str, default=None,
                        choices=['pretrain', 'teacher', 'distill', 'finetune'])
    parser.add_argument("--max_seq_len", type=int, default=20, 
                        help='max sequence/context length of subtrajectories in training')
    parser.add_argument("--eval_context_length", type=int, default=15)
    # 0: no pos embedding others: absolute ordering
    parser.add_argument("--ordering", type=int, default=0)
    parser.add_argument("--stochastic_policy", type=bool, default=True)
    parser.add_argument("--embed_dim", type=int, default=256, 
                        help='n_embd: dimensionality of the embeddings and hidden states')
    parser.add_argument("--n_layer", type=int, default=2)
    parser.add_argument("--n_head", type=int, default=2)
    parser.add_argument("--activation_function", type=str, default="relu")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--transformer_model", type=str, default="gpt2",
                        choices=['gpt2', 'distilbert', 'trajgpt2'])
    parser.add_argument("--adapter++", type=str, default=None,
                        choices=['pfeiffer', 'houlsby', 'parallel', 'pfeiffer_inv', 'houlsby_inv', 
                                 'compacter', 'compacter++', 'prefix_tuning', 
                                 'lora', 'ia3', 'mam', 'unipelt'])

    # Evaluation options
    parser.add_argument("--eval_rtg_scale", type=int, default=0.2,
                        help='target rtg used in evaluation')
    parser.add_argument("--num_eval_episodes", type=int, default=1)

    # Shared training options
    parser.add_argument("--reward_scale", type=float, default=1)
    parser.add_argument("--init_temperature", type=float, default=0.1,
                        help='weight of exploration entropy in DT loss function')
    parser.add_argument("--batch_size", type=int, default=256,
                        help='number of sampled trajectories/subtrajectories')
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", "-wd", type=float, default=5e-4)
    parser.add_argument("--warmup_steps", type=int, default=10000)

    # Pretraining options
    parser.add_argument("--max_pretrain_iters", type=int, default=1)
    parser.add_argument("--num_updates_per_pretrain_iter", type=int, default=2000, 
                        help='total sampled training data = . * batch_size')

    # Distillation options
    parser.add_argument("--max_distill_iters", type=int, default=1)
    parser.add_argument("--num_updates_per_distill_iter", type=int, default=3000)
    parser.add_argument("--softmax_temperature", default=8, type=float, 
                        help="Temperature for the softmax temperature.")
    parser.add_argument("--alpha_ce", default=0.4, type=float, 
                        help="Linear weight for the distillation loss. Must be >=0.")
    parser.add_argument("--alpha_dt", default=1, type=float, 
                        help="Linear weight of the DecisionTransformer loss. Must be >=0.")
    parser.add_argument("--alpha_cos", default=0.0, type=float, 
                        help="Linear weight of the cosine embedding loss. Must be >=0.")

    # Finetuning options
    parser.add_argument("--max_online_iters", type=int, default=10)
    parser.add_argument("--num_updates_per_online_iter", type=int, default=300)
    parser.add_argument("--online_rtg_scale", type=int, default=0.3,
                        help='target rtg used in online.')
    parser.add_argument("--eval_interval", type=int, default=1)
    parser.add_argument("--replay_size", type=int, default=100,
                        help='number of trajectories in replay buffer')

    # Environment options
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--libsumo", type=bool, default=True,
                        help='True when in Linux system')
    parser.add_argument("--gui", action='store_true', default=False)
    parser.add_argument("--max_episode_length", type=int, default=360)
    parser.add_argument("--log_to_tb", "-w", type=bool, default=True)
    parser.add_argument("--save_dir", type=str, default="./exp/")
    parser.add_argument("--exp_name", type=str, default="default")
    parser.add_argument("--pwd", type=str, default=str(pathlib.Path().absolute())+os.sep)
    parser.add_argument("--log_dir", type=str, default='logs/')
    parser.add_argument("--data_dir", type=str, 
                        default=str(pathlib.Path().absolute())+os.sep+'DTRL'+os.sep)
    parser.add_argument("--rs_dir", type=str, 
                        default=str(pathlib.Path().absolute())+os.sep+'random_search'+os.sep)

    args = parser.parse_args()

    return args
