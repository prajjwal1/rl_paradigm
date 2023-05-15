import numpy as np
import torch

from .trainer import Trainer


class SequenceTrainer(Trainer):
    def train_step(self):
        states, actions, rewards, dones, rtg, timesteps, attention_mask = next(
            iter(self.get_batch)
        )  # (self.batch_size)

        states, actions, rewards, rtg, timesteps, attention_mask = self.put_on_device(
            states, actions, rewards, rtg, timesteps, attention_mask
        )

        action_target = torch.clone(actions)

        self.optimizer.zero_grad()

        #  with torch.cuda.amp.autocast():
        state_preds, action_preds, reward_preds = self.model.forward(
            states,
            actions,
            rewards,
            rtg[:, :-1],
            timesteps,
            attention_mask=attention_mask,
        )

        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[
            attention_mask.reshape(-1) > 0
        ]

        loss = self.loss_fn(
            None,
            action_preds,
            None,
            None,
            action_target,
            None,
        )

        self.scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
        self.scaler.step(self.optimizer)  # .step()

        with torch.no_grad():
            self.diagnostics["training/action_error"] = (
                torch.mean((action_preds - action_target) ** 2).detach().cpu().item()
            )

        self.scaler.update()

        return loss.detach().cpu().item()
