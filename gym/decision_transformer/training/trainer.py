import numpy as np
import torch
from tqdm import trange
import decision_transformer.models.utils as utils

import time


class CQLTrainer:
    def __init__(self, env, agent, replay_iter, data_args, training_args):
        self.replay_iter = replay_iter
        self.agent = agent
        self.global_step = 0
        self.env = env
        self.data_args = data_args
        self.training_args = training_args

    def train_iteration(self, num_steps, iter_num, print_logs=True):
        logs = dict()

        for idx in trange(num_steps):
            metrics = self.agent.update(self.replay_iter, self.global_step)
            self.global_step += 1
        if print_logs:
            print(metrics)

        outputs = self.eval()
        #  logs.update(outputs)
        #  for k, v in outputs.items():
        #  logs[f'evaluation/{k}'] = v
        #  print(outputs)

        return outputs

    def eval(self):
        episode_lengths = []
        rewards = []

        for idx in range(self.training_args.num_eval_episodes):
            step, episode, total_reward = 0, 0, 0
            for _ in range(1000):
                state = self.env.reset()
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(state, self.global_step, eval_mode=True)
                state, reward, done, _ = self.env.step(action)
                total_reward += reward
                step += 1

                episode += 1

                if done:
                    break
            rewards.append(total_reward)
            episode_lengths.append(step)

        avg_returns = utils.get_normalized_scores(
            self.data_args.env_name, np.mean(rewards)
        )
        print(f"Episode reward mean: {avg_returns}")
        print(f"Episode length: {np.mean(episode_lengths)}")
        print(f"Step: {self.global_step}")

        return avg_returns

    def save_model(self, path):
        pass


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        batch_size,
        get_batch,
        loss_fn,
        scheduler=None,
        eval_fns=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.get_batch = get_batch
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.eval_fns = [] if eval_fns is None else eval_fns
        self.diagnostics = dict()

        self.start_time = time.time()

    def train_iteration(self, num_steps, iter_num=0, print_logs=False):

        train_losses = []
        logs = dict()

        train_start = time.time()

        self.model.train()
        for _ in trange(num_steps):
            train_loss = self.train_step()
            train_losses.append(train_loss)
            if self.scheduler is not None:
                self.scheduler.step()

        logs["time/training"] = time.time() - train_start

        eval_start = time.time()

        self.model.eval()
        for eval_fn in self.eval_fns:
            outputs = eval_fn(self.model)
            for k, v in outputs.items():
                logs[f"evaluation/{k}"] = v

        logs["time/total"] = time.time() - self.start_time
        logs["time/evaluation"] = time.time() - eval_start
        logs["training/train_loss_mean"] = np.mean(train_losses)
        logs["training/train_loss_std"] = np.std(train_losses)

        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        if print_logs:
            print("=" * 80)
            print(f"Iteration {iter_num}")
            for k, v in logs.items():
                print(f"{k}: {v}")

        return logs

    def save_model(self, path):
        torch.save(self.model.state_dict(), path + ".pth")
        torch.save(self.optimizer.state_dict(), path + "_optim.pth")

    def train_step(self):
        states, actions, rewards, dones, attention_mask, returns = self.get_batch(
            self.batch_size
        )
        state_target, action_target, reward_target = (
            torch.clone(states),
            torch.clone(actions),
            torch.clone(rewards),
        )

        state_preds, action_preds, reward_preds = self.model.forward(
            states,
            actions,
            rewards,
            masks=None,
            attention_mask=attention_mask,
            target_return=returns,
        )

        # note: currently indexing & masking is not fully correct
        loss = self.loss_fn(
            state_preds,
            action_preds,
            reward_preds,
            state_target[:, 1:],
            action_target,
            reward_target[:, 1:],
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().item()
