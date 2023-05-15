import gym
import numpy as np
import torch
import wandb

import pickle
import random
import sys
from transformers import HfArgumentParser
from collections import defaultdict

from decision_transformer.evaluation.evaluate_episodes import (
    evaluate_episode,
    evaluate_episode_rtg,
)
from decision_transformer.models.decision_transformer import DecisionTransformer
from decision_transformer.models.utils import get_normalized_scores
from decision_transformer.models.mlp_bc import MLPBCModel
from decision_transformer.training.act_trainer import ActTrainer
from decision_transformer.training.seq_trainer import SequenceTrainer
from dataclasses import dataclass, field
import os
import json
import shutil


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed = random.randint(0, 1000)
set_seed(seed)


@dataclass
class DataArguments:
    context_len: int = field(default=20)
    env_name: str = field(default="hopper")
    difficulty: str = field(
        default="medium"
    )  # medium, medium-replay, medium-expert, expert
    mode: str = field(default="normal")
    scale: float = field(default=1.0)
    max_ep_len: int = field(default=1000)
    discount: float = field(default=0.99)
    replay_pickled_dir: str = field(default=None)


@dataclass
class ModelArguments:
    model_type: str = field(default="dt")  # dt, bc
    embed_dim: int = field(default=128)
    n_layer: int = field(default=3)
    n_head: int = field(default=1)
    activation_function: str = field(default="relu")


@dataclass
class TrainingArguments:
    seed: int = field(default=None)
    max_iters: int = field(default=10)
    job_name: str = field(default="default")
    lr: float = field(default=1e-4)
    batch_size: int = field(default=256)
    dropout: int = field(default=0.1)
    pct_traj: float = field(default=1.0)
    weight_decay: float = field(default=1e-4)
    warmup_steps: int = field(default=10000)
    num_eval_episodes: int = field(default=100)
    max_steps_per_iter: int = field(default=10)
    log_to_wandb: bool = field(default=False)
    num_workers: int = field(default=8)
    ckpt_path: str = field(default=None)
    reverse_rewards: bool = field(default=False)
    random_pct_traj: float = field(default=0)


def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
    return discount_cumsum


def experiment():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    env_name, dataset = data_args.env_name, data_args.difficulty
    model_type = model_args.model_type
    #  group_name = f'{exp_prefix}-{env_name}-{dataset}'
    #  exp_prefix = f'{group_name}-{random.randint(int(1e5), int(1e6) - 1)}'

    if env_name == "hopper":
        env = gym.make("Hopper-v3")
        max_ep_len = 1000
        env_targets = [3600, 1800]  # evaluation conditioning targets
        scale = 1000.0  # normalization for rewards/returns
    elif env_name == "halfcheetah":
        env = gym.make("HalfCheetah-v3")
        max_ep_len = 1000
        env_targets = [12000, 6000]
        scale = 1000.0
    elif env_name == "walker2d":
        env = gym.make("Walker2d-v3")
        max_ep_len = 1000
        env_targets = [5000, 2500]
        scale = 1000.0
    elif env_name == "reacher2d":
        from decision_transformer.envs.reacher_2d import Reacher2dEnv

        env = Reacher2dEnv()
        max_ep_len = 100
        env_targets = [76, 40]
        scale = 10.0
    elif env_name == "humanoid":
        env = gym.make("Humanoid-v2")
        max_ep_len = 1400
        env_targets = [6000]
        scale = 1000.0
    else:
        raise NotImplementedError

    if model_type == "bc":
        env_targets = env_targets[
            :1
        ]  # since BC ignores target, no need for different evaluations

    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # load dataset
    dataset_path = f"{data_args.replay_pickled_dir}/{env_name}-{dataset}-v2.pkl"

    #  from replay_buffer import OfflineReplayBuffer

    with open(dataset_path, "rb") as f:
        trajectories = pickle.load(f)

    if training_args.random_pct_traj > 0:
        random_dataset_path = f"{data_args.replay_pickled_dir}/{env_name}-random-s2.pkl"
        with open(random_dataset_path, "rb") as f:
            random_trajectories = pickle.load(f)

        rewards = []
        for random_idx in range(len(random_trajectories)):
            for k, v in random_trajectories[random_idx].items():
                random_trajectories[random_idx][k] = np.array(v)

            rewards.append(random_trajectories[random_idx]["rewards"].sum())
        random_sorted_indices = np.argsort(rewards)
        num_permissible_random_traj = int(
            len(random_trajectories) * training_args.random_pct_traj
        )
        random_sorted_indices = random_sorted_indices[-num_permissible_random_traj:]

        filtered_random_trajectories = []
        for idx in random_sorted_indices:
            filtered_random_trajectories.append(random_trajectories[idx])

        print(
            f"Adding {len(random_sorted_indices)} random trajectories to original data"
        )
        trajectories.extend(filtered_random_trajectories)

        print(
            f"Percentage of data that is optimal {100-(len(filtered_random_trajectories)/len(trajectories))*100}"
        )

    print("Length of trajectories: ", len(trajectories))
    print(trajectories[0]["observations"].shape)

    if os.path.exists(training_args.ckpt_path):
        shutil.rmtree(training_args.ckpt_path)
    os.makedirs(training_args.ckpt_path)

    # save all path information into separate lists
    mode = data_args.mode
    states, traj_lens, returns = [], [], []
    for path in trajectories:
        if mode == "delayed":  # delayed: all rewards moved to end of trajectory
            path["rewards"][-1] = path["rewards"].sum()
            path["rewards"][:-1] = 0.0
        states.append(path["observations"])
        traj_lens.append(len(path["observations"]))
        returns.append(path["rewards"].sum())

    print(f"Average length of the trajectory: {np.average(traj_lens)}")
    traj_lens, returns = np.array(traj_lens), np.array(returns)

    # used for input normalization
    states = np.concatenate(states, axis=0)

    print("States shape: ", states.shape)
    print("returns shape: ", returns.shape)
    print("traj_lens shape: ", traj_lens.shape)

    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

    num_timesteps = sum(traj_lens)

    print("=" * 50)
    print(f"Starting new experiment: {env_name} {dataset}")
    print(f"{len(traj_lens)} trajectories, {num_timesteps} timesteps found")
    print(f"Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}")
    print(f"Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}")
    print("=" * 50)

    batch_size = training_args.batch_size
    num_eval_episodes = training_args.num_eval_episodes
    pct_traj = training_args.pct_traj
    context_len = data_args.context_len

    # only train on top pct_traj trajectories (for %BC experiment)
    num_timesteps = max(int(pct_traj * num_timesteps), 1)

    if not training_args.reverse_rewards:
        sorted_inds = np.argsort(returns)  # lowest to highest
    else:
        sorted_inds = np.argsort(returns)[::-1]

    num_trajectories = 1
    timesteps = traj_lens[sorted_inds[-1]]
    ind = len(trajectories) - 2

    while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] <= num_timesteps:
        timesteps += traj_lens[sorted_inds[ind]]
        num_trajectories += 1
        ind -= 1
    sorted_inds = sorted_inds[-num_trajectories:]

    # used to reweight sampling so we sample according to timesteps instead of trajectories
    p_sample = traj_lens[sorted_inds] / sum(traj_lens[sorted_inds])

    print("Training info: ")
    print(f"Sum of returns: {sum(returns[sorted_inds])}")
    print(f"Number of trajectories: {num_trajectories}")
    print(f"Num timesteps for training: {num_timesteps}")

    def get_batch(batch_size=256, max_len=context_len):
        batch_inds = np.random.choice(
            np.arange(num_trajectories),
            size=batch_size,
            replace=True,
            p=p_sample,  # reweights so we sample according to timesteps
        )

        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        for i in range(batch_size):
            traj = trajectories[int(sorted_inds[batch_inds[i]])]
            si = random.randint(0, traj["rewards"].shape[0] - 1)

            # get sequences from dataset
            s.append(traj["observations"][si : si + max_len].reshape(1, -1, state_dim))
            a.append(traj["actions"][si : si + max_len].reshape(1, -1, act_dim))
            r.append(traj["rewards"][si : si + max_len].reshape(1, -1, 1))
            if "terminals" in traj:
                d.append(traj["terminals"][si : si + max_len].reshape(1, -1))
            else:
                d.append(traj["dones"][si : si + max_len].reshape(1, -1))
            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= max_ep_len] = (
                max_ep_len - 1
            )  # padding cutoff
            rtg.append(
                discount_cumsum(traj["rewards"][si:], gamma=1.0)[
                    : s[-1].shape[1] + 1
                ].reshape(1, -1, 1)
            )

            if rtg[-1].shape[1] <= s[-1].shape[1]:
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            # padding and state + reward normalization
            tlen = s[-1].shape[1]
            s[-1] = np.concatenate(
                [np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1
            )
            s[-1] = (s[-1] - state_mean) / state_std
            a[-1] = np.concatenate(
                [np.ones((1, max_len - tlen, act_dim)) * -10.0, a[-1]], axis=1
            )
            r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
            d[-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, d[-1]], axis=1)
            rtg[-1] = (
                np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1)
                / scale
            )
            timesteps[-1] = np.concatenate(
                [np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1
            )
            mask.append(
                np.concatenate(
                    [np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1
                )
            )

        s = torch.from_numpy(np.concatenate(s, axis=0)).to(
            dtype=torch.float32, device=device
        )
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(
            dtype=torch.float32, device=device
        )
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(
            dtype=torch.float32, device=device
        )
        d = torch.from_numpy(np.concatenate(d, axis=0)).to(
            dtype=torch.long, device=device
        )
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(
            dtype=torch.float32, device=device
        )
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(
            dtype=torch.long, device=device
        )
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)

        return s, a, r, d, rtg, timesteps, mask

    def eval_episodes(target_rew):
        def fn(model):
            returns, lengths = [], []
            for _ in range(num_eval_episodes):
                with torch.no_grad():
                    if model_type == "dt":
                        ret, length = evaluate_episode_rtg(
                            env,
                            state_dim,
                            act_dim,
                            model,
                            max_ep_len=max_ep_len,
                            scale=scale,
                            target_return=target_rew / scale,
                            mode=mode,
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
                            max_ep_len=max_ep_len,
                            target_return=target_rew / scale,
                            mode=mode,
                            state_mean=state_mean,
                            state_std=state_std,
                            device=device,
                        )
                returns.append(get_normalized_scores(env_name, ret))
                lengths.append(length)
            return {
                f"target_{target_rew}_return": np.mean(returns),
                f"target_{target_rew}_return_std": np.std(returns),
                f"target_{target_rew}_length_mean": np.mean(lengths),
                f"target_{target_rew}_length_std": np.std(lengths),
            }

        return fn

    if model_type == "dt":
        model = DecisionTransformer(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=data_args.context_len,
            max_ep_len=max_ep_len,
            hidden_size=model_args.embed_dim,
            n_layer=model_args.n_layer,
            n_head=model_args.n_head,
            n_inner=4 * model_args.embed_dim,
            activation_function=model_args.activation_function,
            n_positions=1024 * 3,
            resid_pdrop=training_args.dropout,
            attn_pdrop=training_args.dropout,
        )
    elif model_type == "bc":
        model = MLPBCModel(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=data_args.context_len,
            hidden_size=model_args.embed_dim,
            n_layer=model_args.n_layer,
        )

    model = model.to(device=device)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f"Parameters of the model: {params}")

    from tqdm import trange

    def generate_random_data_uniform(
        model, env, target_return_scaler, state_mean, state_std
    ):
        data_dict = {
            "states": [],
            "next_states": [],
            "rewards": [],
            "dones": [],
            "actions": [],
        }
        model.eval()
        model.to(device="cuda")
        state_mean = torch.from_numpy(state_mean).to(device=device)
        state_std = torch.from_numpy(state_std).to(device=device)

        for _ in trange(5000):
            state = env.reset()

            states = (
                torch.from_numpy(state)
                .reshape(1, state_dim)
                .to(device=device, dtype=torch.float32)
            )
            actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
            rewards = torch.zeros(0, device=device, dtype=torch.float32)
            target_return = torch.tensor(
                target_return_scaler, device=device, dtype=torch.float32
            ).reshape(1, 1)
            timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)
            episode_return, episode_length = 0, 0
            MAX_EP_LEN = random.randrange(0, 1000)
            SCALE = 1000
            expert_reached_success = False
            states_list, rewards_list, dones_list, actions_list, next_states_list = (
                [],
                [],
                [],
                [],
                [],
            )

            for t in range(MAX_EP_LEN):
                actions = torch.cat(
                    [actions, torch.zeros((1, act_dim), device=device)], dim=0
                )
                rewards = torch.cat([rewards, torch.zeros(1, device=device)])
                action = model.get_action(
                    (states.to(dtype=torch.float32) - state_mean) / state_std,
                    actions.to(dtype=torch.float32),
                    rewards.to(dtype=torch.float32),
                    target_return.to(dtype=torch.float32),
                    timesteps.to(dtype=torch.long),
                )
                actions[-1] = action
                action = action.detach().cpu().numpy()

                state, reward, done, _ = env.step(action)
                cur_state = (
                    torch.from_numpy(state).to(device=device).reshape(1, state_dim)
                )
                states = torch.cat([states, cur_state], dim=0)
                rewards[-1] = reward
                pred_return = target_return[0, -1] - (reward / SCALE)

                episode_return += reward

                target_return = torch.cat(
                    [target_return, pred_return.reshape(1, 1)], dim=1
                )
                timesteps = torch.cat(
                    [
                        timesteps,
                        torch.ones((1, 1), device=device, dtype=torch.long) * (t + 1),
                    ],
                    dim=1,
                )

                if done:
                    expert_reached_success = True
                    print("Expert reached done state")
                    break
            if expert_reached_success:
                continue

            print("Generating data now")
            for step_i in range(1000):
                random_action = np.random.uniform(-1, 1, act_dim)
                next_state, reward, done, _ = env.step(random_action)
                states_list.append(state)
                actions_list.append(random_action)
                dones_list.append(done)
                next_states_list.append(next_state)
                rewards_list.append(reward)
                state = next_state

                if done:
                    break

            data_dict["states"].append(states_list)
            data_dict["rewards"].append(rewards_list)
            data_dict["dones"].append(dones_list)
            data_dict["actions"].append(actions_list)
            data_dict["next_states"].append(next_states_list)
        return data_dict

    #      ckpt = torch.load(f'/checkpoints/prajj/gym_ckpt/{data_args.env_name}_medium-expert/dt_head_1_n_layer_3_context_20_pct_1_tsteps_5000_rev_False_mode_normal_random_pct_traj_0.01/1/model_9.pth')
    #  model.load_state_dict(ckpt)

    #  data_dict = generate_random_data_uniform(model, env, 12000, state_mean, state_std)
    #  with open(f"/checkpoints/prajj/gym/{data_args.env_name}-random-s2.pkl", 'wb') as fp:
    #  pickle.dump(data_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)

    #      exit()

    warmup_steps = training_args.warmup_steps
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_args.lr,
        weight_decay=training_args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda steps: min((steps + 1) / warmup_steps, 1)
    )

    if model_type == "dt":

        trainer = SequenceTrainer(
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            get_batch=get_batch,
            scheduler=scheduler,
            loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a) ** 2),
            eval_fns=[eval_episodes(tar) for tar in env_targets],
        )
    elif model_type == "bc":
        trainer = ActTrainer(
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            get_batch=get_batch,
            scheduler=scheduler,
            loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a) ** 2),
            eval_fns=[eval_episodes(tar) for tar in env_targets],
        )

    if training_args.log_to_wandb:
        group_name = training_args.job_name + "_" + data_args.task_name
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
            num_steps=training_args.max_steps_per_iter,
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
