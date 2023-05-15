import numpy as np
import torch
from tqdm import tqdm
import time
import os
import shutil


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
        context_len=30,
    ):
        self.model = model
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.get_batch = get_batch
        self.loss_fn = loss_fn
        self.device = "cuda"
        self.scheduler = scheduler
        self.eval_fns = [] if eval_fns is None else eval_fns
        self.diagnostics = dict()
        self.context_len = context_len
        self.scaler = torch.cuda.amp.GradScaler()

        self.start_time = time.time()

    def train_iteration(self, num_steps, iter_num=0, print_logs=False):

        train_losses = []
        logs = dict()

        train_start = time.time()

        self.model.train()

        print("Training in progress")
        for idx in tqdm(range(num_steps)):
            train_loss = self.train_step()
            #  if idx > 5:
            print(f"Train Loss at step {idx}: {train_loss}")
            train_losses.append(train_loss)
            if self.scheduler is not None:
                self.scheduler.step()

            #  if idx > 20:
            #  self.model.eval()
            #  for eval_fn in self.eval_fns:
            #  outputs = eval_fn(self.model)
            #  #  for k, v in outputs.items():
            #  #  logs[f'evaluation/{k}'] = v
            #          print(outputs)
            self.model.train()

        logs["time/training"] = time.time() - train_start

        eval_start = time.time()

        self.model.eval()
        for eval_fn in self.eval_fns:
            outputs = eval_fn(self.model)
            for k, v in outputs.items():
                logs[f"evaluation/{k}"] = v
            print(outputs)

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

    def put_on_device(self, *args):
        args = list(args)
        for idx in range(len(args)):
            args[idx] = args[idx].to(self.device)
        return args

    def save_model(self, path):
        #          if os.path.exists(path):
        #  shutil.rmtree(path)
        #          os.makedirs(path)
        torch.save(self.model.state_dict(), path + ".pth")
        torch.save(self.optimizer.state_dict(), path + "_optim.pth")

    def train_step(self):
        states, actions, rewards, dones, attention_mask, returns = next(
            iter(self.get_batch)
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
