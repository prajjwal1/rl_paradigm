from replay_buffer import make_replay_loader
from pathlib import Path
import torch
import numpy as np
import os
import shutil
import json
import random
import wandb
import copy
from tqdm import trange
from transformers import HfArgumentParser
from models.decision_transformer import DecisionTransformer
from models.mlp_bc import MLPBCModel
from trainer.act_trainer import ActTrainer
from trainer.seq_trainer import SequenceTrainer
from evaluation.evaluate_episodes import evaluate_episode, evaluate_episode_rtg
from transformers import get_linear_schedule_with_warmup

from dataclasses import dataclass, field

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
import dmc


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed = random.randint(0, 1000)
set_seed(seed)


@dataclass
class DataArguments:
    replay_dir: str = field(default=None)
    context_len: int = field(default=20)
    task_name: str = field(default="hopper")
    dataset: str = field(
        default="medium"
    )  # medium, medium-replay, medium-expert, expert
    mode: str = field(default="normal")
    scale: float = field(default=1.0)
    max_ep_len: int = field(default=1000)
    discount: float = field(default=0.99)
    replay_pickled_dir: str = field(default=None)
    algorithm_name: str = field(default=None)


@dataclass
class ModelArguments:
    model_type: str = field(default="dt")  # dt, bc
    embed_dim: int = field(default=128)
    n_layer: int = field(default=6)
    n_head: int = field(default=8)
    activation_function: str = field(default="relu")


@dataclass
class TrainingArguments:
    # seed: int = field(default=None)
    max_iters: int = field(default=10)
    #  num_steps_per_epoch: int = field(default=10000)
    job_name: str = field(default="default")
    lr: float = field(default=6e-4)
    batch_size: int = field(default=6400)
    dropout: int = field(default=0.1)
    pct_traj: float = field(default=1.0)
    weight_decay: float = field(default=1e-4)
    warmup_steps: int = field(default=10)
    num_eval_episodes: int = field(default=10)
    training_steps_per_iter: int = field(default=10)
    log_to_wandb: bool = field(default=False)
    num_workers: int = field(default=32)
    ckpt_path: str = field(default=None)


def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
    return discount_cumsum


def experiment():

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    env = dmc.make(data_args.task_name)
    group_name = training_args.job_name + "_" + data_args.task_name
    # TODO: Figure out what max_ep_len and scale  needs to be

    #  data_specs = (env.observation_spec(), env.action_spec(), env.reward_spec(), env.discount_spec())
    state_dim, act_dim = env.observation_spec().shape[0], env.action_spec().shape[0]
    env_targets = [200, 300, 400, 500, 600, 700, 800]
    #  domain = get_domain(cfg.task)

    #  replay_dir = domain / domain / cfg.expl_agent / 'buffer'
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # obs, action, reward, discount, next_obs

    states, traj_lens, returns = [], [], []
    iter_cnt = 0
    trajectories = []

    state_mean = np.array([0])
    state_std = np.array([1])

    #  import time

    if data_args.replay_pickled_dir is not None:
        trajectories = np.load(data_args.replay_pickled_dir)
        print("Buffer loaded successfuly")

    else:
        print("Creating episodes: ")
        trajectories = {"observations": [], "rewards": [], "actions": []}

        examples_to_load = 10000000 * (training_args.pct_traj / 100)
        print(f"Generating {examples_to_load} examples")
        replay_dir = Path(data_args.replay_dir)
        replay_loader = make_replay_loader(
            env,
            replay_dir,
            examples_to_load,
            1,  # bs
            1,
            #  training_args.num_workers,
            data_args.discount,
        )
        #  replay_iter = iter(replay_loader)
        print(f"Length of replay loader: {len(replay_loader)}")

        observations, rewards, actions = [], [], []
        for episode in replay_loader:

            observations.append(np.array(episode["observation"]))
            rewards.append(np.array(episode["reward"]))
            actions.append(np.array(episode["action"]))

        #              trajectories['observations'].append(observations[-1])
        #  trajectories['actions'].append(actions[-1])
        #              trajectories['rewards'].append(rewards[-1])

        #  trajectories.append(episode)
        #  del episode
        #              if iter_cnt > 0 and (iter_cnt) % 100000 == 0:
        #  with open(f"/checkpoints/prajj/exorl_replay_buffer/{data_args.task_name}/{data_args.algorithm_name}/buffer_{iter_cnt}.npz", "wb") as fp:
        #  np.savez(
        #  fp,
        #  observations = np.array(observations),
        #  actions = np.array(actions),
        #  rewards = np.array(rewards)
        #  )
        #  print(f"Saved  buffer {iter_cnt}")

        trajectories["observations"] = np.array(observations)
        trajectories["rewards"] = np.array(rewards)
        trajectories["actions"] = np.array(actions)

    print(len(trajectories["observations"]))
    sum_rewards = []
    for r in trajectories["rewards"]:
        sum_rewards.append(r.sum())
    print(sum_rewards)
    exit()

    if os.path.exists(training_args.ckpt_path):
        shutil.rmtree(training_args.ckpt_path)
    os.makedirs(training_args.ckpt_path)
    #   examples_to_load = len(trajectories)
    #  while iter_cnt < examples_to_load:
    #  episode = trajectories[iter_cnt]
    #  states.append(episode['observation'])
    #  traj_lens.append(len(episode['observation']))
    #  returns.append(episode['reward'].sum())
    #  iter_cnt += 1

    #  traj_lens, returns = np.array(traj_lens), np.array(returns)

    #  #  print(f"Maxmium return: {max(returns)}")

    #  states = np.concatenate(states, axis=0)
    #  print("state_dim: ", state_dim)
    #  print(states.shape)
    #  state_mean, state_std = np.average(np.mean(states, axis=0), axis=0), np.average(np.std(states, axis=0), axis=0) + 1e-6
    #  print(f'State mean shape: , {state_mean.shape}, State Std: {state_std.shape}')

    #  num_timesteps = sum(traj_lens)
    #  num_timesteps = max(int(training_args.pct_traj*num_timesteps) / 100, 1)
    #  print(f'Trajectories: {len(traj_lens)}, Num timesteps: , {num_timesteps}')
    #  print(f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
    #  print('=' * 50)

    #  sorted_inds = np.argsort(returns)
    #  num_trajectories = 1
    #  timesteps = traj_lens[sorted_inds[-1]]
    #  ind = len(trajectories) - 2

    #  while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] <= num_timesteps:
    #  timesteps += traj_lens[sorted_inds[ind]]
    #  num_trajectories += 1
    #  ind -= 1
    #  sorted_inds = sorted_inds[-num_trajectories:]
    examples_to_load = int(
        data_args.replay_pickled_dir.split("/")[-1].split("_")[1].split(".")[0]
    )
    print(f"Examples in the buffer: {examples_to_load}")

    #  print("Maximum Return: ", returns[sorted_inds][0])
    #  #  print("Maximum reward: ", max(returns.squeeze()))

    #   p_sample = traj_lens[sorted_inds] / sum(traj_lens[sorted_inds])

    from torch.utils.data import IterableDataset

    class RSADataset(IterableDataset):
        def __init__(self, trajectories, max_len=20):
            self.observations = trajectories["observations"].squeeze(1)
            self.actions = trajectories["actions"].squeeze(1)
            self.rewards = trajectories["rewards"].squeeze(1)

            max_reward = 0
            for rew in self.rewards:
                max_reward = max(max_reward, rew.sum())

            #              self.observations = trajectories['observations'].squeeze(1)
            #  self.actions = trajectories['actions'].squeeze(1)
            #              self.rewards = trajectories['rewards'].squeeze(1)
            #  print("Actions: ", self.actions.shape)
            self.max_len = max_len

            print(self.observations.shape)
            #              print("Computing maximum reward")
            #  for reward in self.trajectories['rewards']:
            #  max_reward = max(max_reward, reward.sum())

            print("Maximum Reward: ", max_reward)

        def __len__(self):
            return len(self.observations)

        def __iter__(self):
            #              batch_inds = np.random.choice(
            #  np.arange(num_trajectories),
            #  size=self.batch_size,
            #  replace=True,
            #  p=p_sample
            #              )
            while True:
                idx = np.random.randint(0, examples_to_load)

                s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []

                #  traj = trajectories[int(sorted_inds[batch_inds[i]])]
                #  traj = trajectories[idx]

                si = random.randint(0, self.rewards[0].shape[0] - 1)

                #  if i % self._num_workers != worker_id:

                # get sequences from dataset
                s.append(
                    self.observations[idx][si : si + self.max_len].reshape(
                        1, -1, state_dim
                    )
                )
                a.append(
                    self.actions[idx][si : si + self.max_len].reshape(1, -1, act_dim)
                )
                r.append(self.rewards[idx][si : si + self.max_len].reshape(1, -1, 1))
                #    if 'terminals' in traj:
                #  d.append(traj['terminals'][si:si + max_len].reshape(1, -1))
                #  else:
                #        d.append(traj['dones'][si:si + max_len].reshape(1, -1))
                timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
                timesteps[-1][timesteps[-1] >= data_args.max_ep_len] = (
                    data_args.max_ep_len - 1
                )  # padding cutoff
                rtg.append(
                    discount_cumsum(self.rewards[idx][si:], gamma=1.0)[
                        : s[-1].shape[1] + 1
                    ].reshape(1, -1, 1)
                )
                if rtg[-1].shape[1] <= s[-1].shape[1]:
                    rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

                # padding and state + reward normalization
                tlen = s[-1].shape[1]
                s[-1] = np.concatenate(
                    [np.zeros((1, self.max_len - tlen, state_dim)), s[-1]], axis=1
                )
                #  s[-1] = (s[-1]) #  - state_mean) / state_std
                a[-1] = np.concatenate(
                    [np.ones((1, self.max_len - tlen, act_dim)) * -10.0, a[-1]], axis=1
                )
                r[-1] = np.concatenate(
                    [np.zeros((1, self.max_len - tlen, 1)), r[-1]], axis=1
                )
                #  d[-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, d[-1]], axis=1)
                rtg[-1] = (
                    np.concatenate(
                        [np.zeros((1, self.max_len - tlen, 1)), rtg[-1]], axis=1
                    )
                    / data_args.scale
                )
                timesteps[-1] = np.concatenate(
                    [np.zeros((1, self.max_len - tlen)), timesteps[-1]], axis=1
                )
                mask.append(
                    np.concatenate(
                        [np.zeros((1, self.max_len - tlen)), np.ones((1, tlen))], axis=1
                    )
                )

                s = (
                    torch.from_numpy(np.concatenate(s, axis=0))
                    .to(dtype=torch.float32)
                    .squeeze(0)
                )  # , device=device)
                a = (
                    torch.from_numpy(np.concatenate(a, axis=0))
                    .to(dtype=torch.float32)
                    .squeeze(0)
                )  # , device=device)
                r = (
                    torch.from_numpy(np.concatenate(r, axis=0))
                    .to(dtype=torch.float32)
                    .squeeze(0)
                )  # , device=device)
                #  d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=device)
                rtg = (
                    torch.from_numpy(np.concatenate(rtg, axis=0))
                    .to(dtype=torch.float32)
                    .squeeze(0)
                )  # , device=device)
                timesteps = (
                    torch.from_numpy(np.concatenate(timesteps, axis=0))
                    .to(dtype=torch.long)
                    .squeeze(0)
                )  # , device=device)
                mask = torch.from_numpy(np.concatenate(mask, axis=0)).squeeze(
                    0
                )  # .to(device=device)
                yield s, a, r, d, rtg, timesteps, mask

    dataset = RSADataset(trajectories, max_len=data_args.context_len)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=training_args.batch_size,
        pin_memory=True,
        num_workers=training_args.num_workers,
        #  worker_init_fn = _worker_init_fn
    )

    #      states, actions, rewards, dones, rtg, timesteps, mask = next(iter(dataloader))
    #      print(rtg.shape)

    #      def get_batch(batch_size=256, max_len=30):
    #  batch_inds = np.random.choice(
    #  np.arange(num_trajectories),
    #  size=batch_size,
    #  replace=True,
    #  p=p_sample
    #  )

    #  s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []

    #  for i in range(batch_size):
    #  traj = trajectories[int(sorted_inds[batch_inds[i]])]
    #  si = random.randint(0, traj['reward'].shape[0] - 1)

    #  # get sequences from dataset
    #  s.append(traj['observation'].squeeze()[si:si + max_len].reshape(1, -1, state_dim))
    #  a.append(traj['action'].squeeze()[si:si + max_len].reshape(1, -1, act_dim))
    #  r.append(traj['reward'].squeeze()[si:si + max_len].reshape(1, -1, 1))
    #  #    if 'terminals' in traj:
    #  #  d.append(traj['terminals'][si:si + max_len].reshape(1, -1))
    #  #  else:
    #  #        d.append(traj['dones'][si:si + max_len].reshape(1, -1))
    #  timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
    #  timesteps[-1][timesteps[-1] >= data_args.max_ep_len] = data_args.max_ep_len-1  # padding cutoff
    #  rtg.append(discount_cumsum(traj['reward'].squeeze()[si:], gamma=1.)[:s[-1].shape[1] + 1].reshape(1, -1, 1))
    #  if rtg[-1].shape[1] <= s[-1].shape[1]:
    #  rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

    #  # padding and state + reward normalization
    #  tlen = s[-1].shape[1]
    #  s[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1)
    #  s[-1] = (s[-1] - state_mean) / state_std
    #  a[-1] = np.concatenate([np.ones((1, max_len - tlen, act_dim)) * -10., a[-1]], axis=1)
    #  r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
    #  #  d[-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, d[-1]], axis=1)
    #  rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1) / data_args.scale
    #  timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
    #  mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))

    #  s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
    #  a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
    #  r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
    #  #  d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=device)
    #  rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
    #  timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
    #  mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)

    #  #     print(s.shape)
    #  #  print(a.shape)
    #  #  print(r.shape)
    #  #     print(timesteps.shape)

    #          return s, a, r, d, rtg, timesteps, mask

    def eval_episodes(target_rew):
        def fn(model):
            returns, lengths = [], []
            print(f"Evaluation in-progress for {target_rew}")

            for _ in trange(training_args.num_eval_episodes):
                with torch.no_grad():
                    if model_args.model_type == "dt":
                        ret, length = evaluate_episode_rtg(
                            env,
                            state_dim,
                            act_dim,
                            model,
                            max_ep_len=data_args.max_ep_len,
                            scale=data_args.scale,
                            target_return=target_rew / data_args.scale,
                            mode=data_args.mode,
                            state_mean=state_mean,
                            state_std=state_std,
                            device=device,
                        )
                    else:
                        ret, length = evaluate_episode(
                            env,
                            state_dim,
                            act_dim,
                            model,
                            max_ep_len=data_args.max_ep_len,
                            target_return=target_rew / data_args.scale,
                            mode=data_args.mode,
                            state_mean=state_mean,
                            state_std=state_std,
                            device=device,
                        )
                returns.append(ret)
                lengths.append(length)
            return {
                f"target_{target_rew}_return": sum(returns)
                / training_args.num_eval_episodes,
                #                  f'target_{target_rew}_return_std': np.std(returns),
                #  f'target_{target_rew}_length_mean': np.mean(lengths),
                #                  f'target_{target_rew}_length_std': np.std(lengths),
            }

        return fn

    if model_args.model_type == "dt":
        model = DecisionTransformer(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=data_args.context_len,
            max_ep_len=data_args.max_ep_len,
            hidden_size=model_args.embed_dim,
            n_layer=model_args.n_layer,
            n_head=model_args.n_head,
            n_inner=4 * model_args.embed_dim,
            activation_function=model_args.activation_function,
            n_positions=1024,
            resid_pdrop=training_args.dropout,
            attn_pdrop=training_args.dropout,
        )
    elif model_args.model_type == "bc":
        model = MLPBCModel(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=data_args.context_len,
            hidden_size=model_args.embed_dim,
            n_layer=model_args.n_layer,
        )
    else:
        raise NotImplementedError

    model = model.to(device=device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_args.lr,
        weight_decay=training_args.weight_decay,
    )
    #  scheduler = None
    # get_linear_schedule_with_warmup(
    #    optimizer = optimizer,
    #    num_warmup_steps = training_args.warmup_steps,
    #    num_training_steps = (training_args.training_steps_per_iter * training_args.max_iters)
    # )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda steps: min((steps + 1) / training_args.warmup_steps, 1)
    )

    if model_args.model_type == "dt":
        trainer = SequenceTrainer(
            model=model,
            optimizer=optimizer,
            batch_size=training_args.batch_size,
            get_batch=dataloader,
            scheduler=scheduler,
            loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a) ** 2),
            eval_fns=[eval_episodes(tar) for tar in env_targets],
        )
    elif model_args.model_type == "bc":
        trainer = ActTrainer(
            model=model,
            optimizer=optimizer,
            batch_size=training_args.batch_size,
            get_batch=dataloader,
            scheduler=scheduler,
            loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a) ** 2),
            eval_fns=[eval_episodes(tar) for tar in env_targets],
        )

    if training_args.log_to_wandb:
        wandb.init(
            name=training_args.job_name,
            group=group_name,
            project="decision-transformer",
            config=vars(training_args),
        )
        # wandb.watch(model)  # wandb has some bug

    results_dict = {}

    for idx in range(training_args.max_iters):
        outputs = trainer.train_iteration(
            num_steps=training_args.training_steps_per_iter,
            iter_num=idx + 1,
            print_logs=True,
        )
        results_dict[idx] = outputs
        trainer.save_model(training_args.ckpt_path + "model_" + str(idx))
        if training_args.log_to_wandb:
            wandb.log(outputs)

    with open(os.path.join(training_args.ckpt_path + "results.json"), "w") as fp:
        json.dump(results_dict, fp, indent=2)
    print(f"Results saved here : {training_args.ckpt_path}")


if __name__ == "__main__":
    experiment()
