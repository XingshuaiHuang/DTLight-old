""" The distiller to distil the student.
    Adapted in part from huggingface/transformers 
    (https://github.com/huggingface/transformers/tree/main/examples/research_projects/distillation)
"""
import time

import torch
from torch import nn
from tqdm import tqdm

from transformers import get_linear_schedule_with_warmup


def stochastic_loss_fn(a_pred_dist, a_target, attention_mask, entropy_reg):
    log_likelihood = a_pred_dist.log_prob(a_target.squeeze(-1))[
        attention_mask > 0
    ].mean()
    entropy = a_pred_dist.entropy().mean()
    loss = -(log_likelihood + entropy_reg * entropy)
    return (loss, -log_likelihood, entropy)


class Distiller:
    def __init__(
        self,
        var: dict,
        student: nn.Module,
        teacher: nn.Module,
        optimizer,
        scheduler,
        log_temperature_optimizer,
    ):
        self.var = var
        self.student = student
        self.teacher = teacher
        self.optimizer = optimizer
        self.scheduler = scheduler
        # temperature for entropy loss in decisition transformer
        self.log_temperature_optimizer = log_temperature_optimizer

        # temperature for distillation loss
        self.softmax_temperature = var["softmax_temperature"]
        assert self.softmax_temperature > 0.0
        self.alpha_ce = var["alpha_ce"]
        self.alpha_dt = var["alpha_dt"]
        self.alpha_cos = var["alpha_cos"]
        self.device = var["device"]

        self.n_iter = 0
        self.n_total_iter = 0
        self.n_sequences_epoch = 0
        self.total_loss_epoch = 0
        self.last_loss = 0
        self.last_loss_ce = 0
        self.last_loss_dt = 0
        if self.alpha_cos > 0.0:
            self.last_loss_cos = 0

        self.ce_loss = nn.KLDivLoss(reduction="batchmean")
        if self.alpha_cos > 0.0:
            self.cosine_loss = nn.CosineEmbeddingLoss(reduction="mean")

    def optimize(self, loss, entropy):
        """
        Same as compute_loss in trainer.py
        """
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.student.parameters(), 0.25)
        self.optimizer.step()

        self.log_temperature_optimizer.zero_grad()
        temperature_loss = (
            self.student.temperature()
            * (entropy - self.student.target_entropy).detach()
        )
        temperature_loss.backward()
        self.log_temperature_optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

    def step(self, batch):
        """
        One optimization step: forward of student AND teacher, backward on the loss (for gradient accumulation),
        and possibly a parameter update (depending on the gradient accumulation).
        """
        (
            states,
            actions,
            rewards,
            dones,
            rtg,
            timesteps,
            ordering,
            padding_mask,
        ) = batch

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)
        rtg = rtg.to(self.device)
        timesteps = timesteps.to(self.device)
        ordering = ordering.to(self.device)
        padding_mask = padding_mask.to(self.device)

        action_target = torch.clone(actions)

        _, s_dist_pred, _, s_logits = self.student.forward(
            states,
            actions,
            rewards,
            rtg,
            timesteps,
            ordering,
            padding_mask=padding_mask,
        )

        with torch.no_grad():
            _, t_dist_pred, _, t_logits = self.teacher.forward(
                states,
                actions,
                rewards,
                rtg,
                timesteps,
                ordering,
                padding_mask=padding_mask,
            )
        assert s_logits.size() == t_logits.size()

        loss_ce = (
            self.ce_loss(
                nn.functional.log_softmax(s_logits / self.softmax_temperature, dim=-1),
                nn.functional.softmax(t_logits / self.softmax_temperature, dim=-1),
            )
            * (self.softmax_temperature) ** 2
        )
        loss = self.alpha_ce * loss_ce

        loss_dt, nll, entropy = stochastic_loss_fn(
            s_dist_pred,  # a_pred_dist
            action_target,
            padding_mask,
            self.student.temperature().detach(),  # no gradient taken here
        )
        loss += self.alpha_dt * loss_dt

        self.total_loss_epoch += loss.item()
        self.last_loss = loss.item()
        self.last_loss_ce = loss_ce.item()
        self.last_loss_dt = loss_dt.item()

        self.optimize(loss, entropy)

    def train(self, dataloader):
        """
        The real training loop.
        """
        train_start = time.time()
        logs = dict()
        self.teacher.eval()
        self.student.train()

        iter_bar = tqdm(dataloader, total=len(dataloader))
        for batch in iter_bar:
            self.step(batch)

            self.n_iter += 1
            iter_bar.update()
            iter_bar.set_postfix(
                {
                    "Loss_ce": f"{self.last_loss_ce}",
                    "Loss_dt": f"{self.last_loss_dt}",
                    "Loss": f"{self.last_loss:.2f}",
                    "Avg_loss": f"{self.total_loss_epoch/self.n_iter:.2f}",
                }
            )
        iter_bar.close()

        self.n_iter = 0
        self.total_loss_epoch = 0

        logs["training_time"] = time.time() - train_start
        logs["last_loss_ce"] = self.last_loss_ce
        logs["last_loss_dt"] = self.last_loss_dt
        # if self.alpha_cos > 0.0:
        #     logs["last_loss_cos"] = self.last_loss_cos
        logs["last_loss"] = self.last_loss
        logs["avg_loss"] = self.total_loss_epoch / len(dataloader)

        return logs
