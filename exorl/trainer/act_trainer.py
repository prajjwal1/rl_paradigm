import numpy as np
import torch

from .trainer import Trainer


class ActTrainer(Trainer):
    def train_step(self):
        states, actions, rewards, dones, rtg, _, attention_mask = next(
            iter(self.get_batch)
        )
        state_target, action_target, reward_target = (
            torch.clone(states),
            torch.clone(actions),
            torch.clone(rewards),
        )

        (
            states,
            actions,
            rewards,
            attention_mask,
            rtg,
            action_target,
        ) = self.put_on_device(
            states, actions, rewards, attention_mask, rtg, action_target
        )

        state_preds, action_preds, reward_preds = self.model.forward(
            states,
            actions,
            rewards,
            attention_mask=attention_mask,
            target_return=rtg[:, 0],
        )

        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)
        action_target = action_target[:, -1].reshape(-1, act_dim)

        loss = self.loss_fn(
            state_preds,
            action_preds,
            reward_preds,
            state_target,
            action_target,
            reward_target,
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().item()
