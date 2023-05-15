import logging

# make deterministic
from transformers import HfArgumentParser
from mingpt.utils import set_seed
import numpy as np
import torch
import os
import shutil
from dataclasses import dataclass, field
from torch.nn import functional as F
from torch.utils.data import Dataset
from mingpt.model_atari import GPT, GPTConfig
from mingpt.trainer_atari import Trainer, TrainerConfig
from mingpt.utils import sample
from collections import deque
import random
from create_dataset import create_dataset


@dataclass
class DataArguments:
    data_dir_prefix: str = field(default=None)
    context_length: int = field(default=30)
    num_buffers: int = field(default=50)
    game: str = field(default="Breakout")
    trajectories_per_buffer: int = field(default=10)
    max_num_samples_per_buffer: int = field(default=100000)


@dataclass
class ModelArguments:
    model_type: str = field(default="reward_conditioned")
    n_layer: int = field(default=6)
    n_head: int = field(default=8)
    n_embed: int = field(default=128)


@dataclass
class TrainingArguments:
    seed: int = field(default=None)
    ckpt_path: str = field(default=None)
    epochs: int = field(default=5)
    num_steps: int = field(default=500000)
    batch_size: int = field(default=128)
    lr: float = field(default=6e-4)
    num_games_to_use_for_eval: int = field(default=10)


parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses()

if training_args.seed is None:
    seed = random.randint(0, 1000)
else:
    seed = training_args.seed
set_seed(seed)
training_args.seed = seed


class StateActionReturnDataset(Dataset):
    def __init__(self, data, block_size, actions, done_idxs, rtgs, timesteps):
        self.block_size = block_size
        self.vocab_size = max(actions) + 1
        self.data = data
        self.actions = actions
        self.done_idxs = done_idxs
        self.rtgs = rtgs
        self.timesteps = timesteps

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        block_size = self.block_size // 3
        done_idx = idx + block_size
        for i in self.done_idxs:
            if i > idx:  # first done_idx greater than idx
                done_idx = min(int(i), done_idx)
                break
        idx = done_idx - block_size
        states = torch.tensor(
            np.array(self.data[idx:done_idx]), dtype=torch.float32
        ).reshape(
            block_size, -1
        )  # (block_size, 4*84*84)
        states = states / 255.0
        actions = torch.tensor(self.actions[idx:done_idx], dtype=torch.long).unsqueeze(
            1
        )  # (block_size, 1)
        rtgs = torch.tensor(self.rtgs[idx:done_idx], dtype=torch.float32).unsqueeze(1)
        timesteps = torch.tensor(
            self.timesteps[idx : idx + 1], dtype=torch.int64
        ).unsqueeze(1)

        return states, actions, rtgs, timesteps


obss, actions, returns, done_idxs, rtgs, timesteps = create_dataset(
    data_args.num_buffers,
    training_args.num_steps,
    data_args.game,
    data_args.data_dir_prefix,
    data_args.trajectories_per_buffer,
    data_args.max_num_samples_per_buffer,
)

#  obss, actions, returns, done_idxs, rtgs, timesteps = create_dataset(args.num_buffers, args.num_steps, args.game, args.data_dir_prefix, args.trajectories_per_buffer)

# set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
train_dataset = StateActionReturnDataset(
    obss, data_args.context_length * 3, actions, done_idxs, rtgs, timesteps
)
# from create_dataset:
#    obss shape -> [510358, 4, 84, 84]
#    actions -> [510358]
#    returns -> [419]
#    done_idxs -> [418]
#    rtgs -> [510358]
#    timesteps -> [129106]

print("Timesteps shape from create_datase: ", timesteps.shape)
output = train_dataset[0]
sample_state, sample_action, sample_rtgs, sample_timesteps = (
    output[0],
    output[1],
    output[2],
    output[3],
)
print("State: ", sample_state.shape)
print("Action: ", sample_action.shape)
print("rtgs: ", sample_rtgs.shape)
print("timesteps: ", sample_timesteps.shape)


mconf = GPTConfig(
    train_dataset.vocab_size,
    data_args.context_length * 3,
    n_layer=model_args.n_layer,
    n_head=model_args.n_head,
    n_embd=model_args.n_embed,
    model_type=model_args.model_type,
    max_timestep=max(timesteps),
)
model = GPT(mconf)

print(f"Parameter count: {sum(param.numel() for param in model.parameters())}")

if os.path.exists(training_args.ckpt_path):
    shutil.rmtree(training_args.ckpt_path)
os.makedirs(training_args.ckpt_path)

# initialize a trainer instance and kick off training
tconf = TrainerConfig(
    max_epochs=training_args.epochs,
    batch_size=training_args.batch_size,
    learning_rate=training_args.lr,
    lr_decay=True,
    warmup_tokens=512 * 20,
    final_tokens=2 * len(train_dataset) * data_args.context_length * 3,
    num_workers=8,
    seed=training_args.seed,
    model_type=model_args.model_type,
    game=data_args.game,
    max_timestep=max(timesteps),
    ckpt_path=training_args.ckpt_path,
    num_games_to_use_for_eval=training_args.num_games_to_use_for_eval,
)
trainer = Trainer(model, train_dataset, None, tconf)

trainer.train()
