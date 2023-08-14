"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the CC BY-NC license found in the
LICENSE.md file in the root directory of this source tree.

Part of the code was adapted from the following: 
* https://github.com/facebookresearch/online-dt/blob/main/decision_transformer/models/decision_transformer.py
* https://github.com/kzl/decision-transformer/blob/master/gym/decision_transformer/models/decision_transformer.py
* https://github.com/denisyarats/pytorch_sac/blob/master/agent/actor.py

Both are licensed under the MIT License.
"""

import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
from torch import distributions as pyd
from transformers import (
    DistilBertConfig,
    GPT2Config,
)
from transformers.adapters import (
    DistilBertAdapterModel,
    GPT2AdapterModel,
    PfeifferConfig,
    HoulsbyConfig,
    ParallelConfig,
    PfeifferInvConfig,
    HoulsbyInvConfig,
    CompacterConfig,
    CompacterPlusPlusConfig,
    PrefixTuningConfig,
    LoRAConfig,
    IA3Config,
    MAMConfig,
    UniPELTConfig
)

from dtlight.decision_transformer.model import TrajectoryModel
from dtlight.decision_transformer.trajectory_gpt2 import TrajGPT2Model


adapter_configs = {
    "pfeiffer": PfeifferConfig(),
    "houlsby": HoulsbyConfig(),
    "parallel": ParallelConfig(),
    "pfeiffer_inv": PfeifferInvConfig(),
    "houlsby_inv": HoulsbyInvConfig(),
    "compacter": CompacterConfig(),
    "compacter++": CompacterPlusPlusConfig(),
    "prefix_tuning": PrefixTuningConfig(),
    "lora": LoRAConfig(),
    "ia3": IA3Config(),
    "mam": MAMConfig(),
    "unipelt": UniPELTConfig(),
}


class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1, 1)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2.0 * (math.log(2.0) - x - F.softplus(-2.0 * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    """
    Squashed Normal Distribution(s)

    If loc/std is of size (batch_size, sequence_length, d),
    this returns batch_size * sequence_length * d
    independent squashed univariate normal distributions.
    """

    def __init__(self, loc, std):
        self.loc = loc
        self.std = std
        self.base_dist = pyd.Normal(loc, std)

        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu

    def entropy(self, N=1):
        # sample from the distribution and then compute
        # the empirical entropy:
        x = self.rsample((N,))
        log_p = self.log_prob(x)

        # log_p: (batch_size, context_len, action_dim),
        return -log_p.mean(axis=0).sum(axis=2)

    def log_prob(self, x):
        # log_prob(x): (batch_size, context_len, action_dim)
        # sum up along the action dimensions
        # Return tensor shape: (batch_size, context_len)
        return self.log_prob(x).sum(axis=2)


class DiagGaussianActor(nn.Module):
    """torch.distributions implementation of an diagonal Gaussian policy."""

    def __init__(self, hidden_dim, act_dim, log_std_bounds=[-5.0, 2.0]):
        super().__init__()

        self.mu = torch.nn.Linear(hidden_dim, act_dim)
        self.log_std = torch.nn.Linear(hidden_dim, act_dim)
        self.log_std_bounds = log_std_bounds

        def weight_init(m):
            """Custom weight init for Conv2D and Linear layers."""
            if isinstance(m, torch.nn.Linear):
                nn.init.orthogonal_(m.weight.data)
                if hasattr(m.bias, "data"):
                    m.bias.data.fill_(0.0)

        self.apply(weight_init)

    def forward(self, obs):
        mu, log_std = self.mu(obs), self.log_std(obs)
        log_std = torch.tanh(log_std)
        # log_std is the output of tanh so it will be between [-1, 1]
        # map it to be between [log_std_min, log_std_max]
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1.0)
        std = log_std.exp()
        return SquashedNormal(mu, std), None


class CategoricalActor(nn.Module):
    """torch.distributions implementation of a categorical policy."""

    def __init__(self, hidden_dim, act_dim):
        super().__init__()

        self.logits = torch.nn.Linear(hidden_dim, act_dim)

        def weight_init(m):
            """Custom weight init for Conv2D and Linear layers."""
            if isinstance(m, torch.nn.Linear):
                nn.init.orthogonal_(m.weight.data)
                if hasattr(m.bias, "data"):
                    m.bias.data.fill_(0.0)

        self.apply(weight_init)

    def forward(self, obs):
        logits = self.logits(obs)  # can either use logits or probs in Categorical
        probs = torch.softmax(logits, dim=-1)
        return pyd.Categorical(probs=probs), logits


class DecisionTransformer(TrajectoryModel):

    """
    This model uses GPT to model (Return_1, state_1, action_1, Return_2, state_2, ...)
    """

    def __init__(
        self,
        state_dim,
        act_dim,
        hidden_size,
        action_range,
        ordering=0,
        max_length=None,
        eval_context_length=None,
        max_ep_len=360,
        stochastic_policy=True,
        init_temperature=0.1,  # used to control the degree of randomness or creativity
        target_entropy=None,
        discrete_actions=True,
        transformer_model="gpt2",
        adapter=None,
        n_layer=4,
        n_head=4,
        **kwargs
    ):
        super().__init__(state_dim, act_dim, max_length=max_length)

        if transformer_model == "trajgpt2":
            # note: the only difference between this GPT2Model and the default Huggingface version
            # is that the positional embeddings are removed (since we'll add those ourselves)
            config = GPT2Config(
                vocab_size=1,  # doesn't matter -- we don't use the vocab
                n_embd=hidden_size,
                n_layer=n_layer,
                n_head=n_head,
                **kwargs
            )
            self.transformer = TrajGPT2Model(config)
        elif transformer_model == "gpt2":
            config = GPT2Config(
                vocab_size=1,
                n_embd=hidden_size,
                n_layer=n_layer,
                n_head=n_head,
                **kwargs
            )
            self.transformer = GPT2AdapterModel(config)
        elif transformer_model == "distilbert":
            config = DistilBertConfig(
                vocab_size=1,
                dim=hidden_size,
                n_layers=n_layer,
                n_heads=n_head,
                **kwargs
            )
            self.transformer = DistilBertAdapterModel(config=config)

        if adapter is not None:
            config = adapter_configs[adapter]
            self.transformer.add_adapter(adapter, set_active=True, config=config)
            print(self.transformer.adapter_summary())

        # n_params = sum(p.numel() for p in self.transformer.parameters())
        # print("Number of Parameters for Transformer: %.2fM" % (n_params / 1e6,))

        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        if ordering:
            self.embed_ordering = nn.Embedding(max_ep_len, hidden_size)
        self.embed_return = torch.nn.Linear(1, hidden_size)
        self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
        self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)

        self.embed_ln = nn.LayerNorm(hidden_size)

        self.predict_state = torch.nn.Linear(hidden_size, self.state_dim)
        self.predict_return = torch.nn.Linear(hidden_size, 1)
        if stochastic_policy:
            if discrete_actions:
                # Output a Categorical Distribution for discrete action space
                self.predict_action = CategoricalActor(
                    hidden_size, action_range[-1] + 1
                )
            else:
                # Output a SquashedNormal Distribution
                self.predict_action = DiagGaussianActor(hidden_size, self.act_dim)
        else:
            # Output logits for discrete actions
            self.predict_action = nn.Sequential(
                nn.Linear(hidden_size, action_range[-1] + 1),
                nn.Softmax(dim=-1),
            )
        self.stochastic_policy = stochastic_policy
        self.eval_context_length = eval_context_length
        self.ordering = ordering
        self.action_range = action_range
        self.hidden_size = hidden_size

        if stochastic_policy:
            self.log_temperature = torch.tensor(np.log(init_temperature))
            self.log_temperature.requires_grad = True
            self.target_entropy = target_entropy
        else:
            self.log_temperature = None
            self.target_entropy = None

    def temperature(self):
        if self.stochastic_policy:
            return self.log_temperature.exp()
        else:
            return None

    def _input_transform(
        self, states, actions, rewards, returns_to_go, timesteps, num_envs
    ):
        # we don't care about the past rewards in this model
        # tensor shape: batch_size, seq_length, variable_dim
        states = states.reshape(num_envs, -1, self.state_dim)
        actions = actions.reshape(num_envs, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(num_envs, -1, 1)

        # tensor shape: batch_size, seq_length
        timesteps = timesteps.reshape(num_envs, -1)

        # max_length is the DT context length (should be input seq_len of the subsequence)
        # eval_context_length is how long you want to use the history for your prediction
        if self.max_length is not None:
            states = states[:, -self.eval_context_length :]
            actions = actions[:, -self.eval_context_length :]
            returns_to_go = returns_to_go[:, -self.eval_context_length :]
            timesteps = timesteps[:, -self.eval_context_length :]

            ordering = torch.tile(
                torch.arange(timesteps.shape[1], device=states.device),
                (num_envs, 1),
            )
            # pad all tokens to sequence length
            padding_mask = torch.cat(
                [
                    torch.zeros(self.max_length - states.shape[1]),
                    torch.ones(states.shape[1]),
                ]
            )
            padding_mask = padding_mask.to(
                dtype=torch.long, device=states.device
            ).reshape(1, -1)
            padding_mask = padding_mask.repeat((num_envs, 1))

            states = torch.cat(
                [
                    torch.zeros(
                        (
                            states.shape[0],
                            self.max_length - states.shape[1],
                            self.state_dim,
                        ),
                        device=states.device,
                    ),
                    states,
                ],
                dim=1,
            ).to(dtype=torch.float32)
            actions = torch.cat(
                [
                    torch.zeros(
                        (
                            actions.shape[0],
                            self.max_length - actions.shape[1],
                            self.act_dim,
                        ),
                        device=actions.device,
                    ),
                    actions,
                ],
                dim=1,
            ).to(dtype=torch.float32)
            returns_to_go = torch.cat(
                [
                    torch.zeros(
                        (
                            returns_to_go.shape[0],
                            self.max_length - returns_to_go.shape[1],
                            1,
                        ),
                        device=returns_to_go.device,
                    ),
                    returns_to_go,
                ],
                dim=1,
            ).to(dtype=torch.float32)

            timesteps = torch.cat(
                [
                    torch.zeros(
                        (timesteps.shape[0], self.max_length - timesteps.shape[1]),
                        device=timesteps.device,
                    ),
                    timesteps,
                ],
                dim=1,
            ).to(dtype=torch.long)

            ordering = torch.cat(
                [
                    torch.zeros(
                        (ordering.shape[0], self.max_length - ordering.shape[1]),
                        device=ordering.device,
                    ),
                    ordering,
                ],
                dim=1,
            ).to(dtype=torch.long)
        else:
            padding_mask = None

        return states, actions, returns_to_go, timesteps, ordering, padding_mask

    def forward(
        self,
        states,
        actions,
        rewards,
        returns_to_go,
        timesteps,
        ordering,
        padding_mask=None,
    ):
        batch_size, seq_length = states.shape[0], states.shape[1]

        if padding_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            padding_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        # embed each modality with a different head
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)

        if self.ordering:
            order_embeddings = self.embed_ordering(timesteps)
        else:
            order_embeddings = 0.0

        state_embeddings = state_embeddings + order_embeddings
        action_embeddings = action_embeddings + order_embeddings
        returns_embeddings = returns_embeddings + order_embeddings

        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = (
            torch.stack(
                (returns_embeddings, state_embeddings, action_embeddings), dim=1
            )
            .permute(0, 2, 1, 3)
            .reshape(batch_size, 3 * seq_length, self.hidden_size)
        )
        stacked_inputs = self.embed_ln(stacked_inputs)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_padding_mask = (
            torch.stack((padding_mask, padding_mask, padding_mask), dim=1)
            .permute(0, 2, 1)
            .reshape(batch_size, 3 * seq_length)
        )

        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_padding_mask,
        )
        x = transformer_outputs["last_hidden_state"]

        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)

        # get predictions
        # predict next return given state and action
        return_preds = self.predict_return(x[:, 2])
        # predict next state given state and action
        state_preds = self.predict_state(x[:, 2])
        # predict next action logit given state
        action_preds, logits = self.predict_action(x[:, 1])

        return state_preds, action_preds, return_preds, logits

    def get_predictions(
        self, states, actions, rewards, returns_to_go, timesteps, num_envs=1, **kwargs
    ):
        (
            states,
            actions,
            returns_to_go,
            timesteps,
            ordering,
            padding_mask,
        ) = self._input_transform(
            states, actions, rewards, returns_to_go, timesteps, num_envs
        )

        state_preds, action_preds, return_preds, logits = self.forward(
            states,
            actions,
            None,
            returns_to_go,
            timesteps,
            ordering,
            padding_mask=padding_mask,
            **kwargs
        )
        if self.stochastic_policy:
            return state_preds[:, -1], action_preds, return_preds[:, -1]
        else:
            return (
                state_preds[:, -1],
                action_preds[:, -1].argmax(dim=-1),
                return_preds[:, -1],
            )
