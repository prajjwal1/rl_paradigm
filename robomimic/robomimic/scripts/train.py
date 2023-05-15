"""
The main entry point for training policies.

Args:
    config (str): path to a config json that will be used to override the default settings.
        If omitted, default settings are used. This is the preferred way to run experiments.

    algo (str): name of the algorithm to run. Only needs to be provided if @config is not
        provided.

    name (str): if provided, override the experiment name defined in the config

    dataset (str): if provided, override the dataset path defined in the config

    debug (bool): set this flag to run a quick training run for debugging purposes
"""
import argparse
import json
import numpy as np
import time
import os
import shutil
import psutil
import sys
import socket
import traceback
import pyrallis
from dataclasses import dataclass

from collections import OrderedDict, defaultdict

import torch
from torch.utils.data import DataLoader

import robomimic
import robomimic.utils.train_utils as TrainUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
from robomimic.config import config_factory
from robomimic.algo import algo_factory, RolloutPolicy
from robomimic.utils.log_utils import PrintLogger, DataLogger


@dataclass
class Config:
    config: str
    dataset: str
    experiment_name: str
    output_dir: str
    batch_size: int = 32
    num_epochs: int = 500
    rollout_n: int = 50
    debug: bool = False
    n_heads: int = 8
    n_layers: int = 6
    seq_length: int = 5
    n_embed: int = 128
    lr: float = 1e-4
    weight_decay: float = 1e-4
    dropout: float = 0.1
    warmup_steps: int = 200000
    horizon: int = 400
    num_episodes_during_eval: int = 50
    target_return: float = 1
    mode: str = "sparse"
    pct_traj: float = 1
    dt_bc_mode: bool = False
    random_timesteps: int = 0
    reward_shaping: bool = False


def train(config, device):
    """
    Train a model using the algorithm.
    """

    # first set seeds
    #  np.random.seed(config.train.seed)
    #  torch.manual_seed(config.train.seed)

    print("\n============= New Training Run with Config =============")
    print(config)
    print("")
    log_dir, ckpt_dir, video_dir = TrainUtils.get_exp_dir(
        config, auto_remove_exp_dir=True
    )

    if config.experiment.logging.terminal_output_to_txt:
        # log stdout and stderr to a text file
        logger = PrintLogger(os.path.join(log_dir, "log.txt"))
        sys.stdout = logger
        sys.stderr = logger

    # read config to set up metadata for observation modalities (e.g. detecting rgb observations)
    ObsUtils.initialize_obs_utils_with_config(config)

    # make sure the dataset exists
    dataset_path = os.path.expanduser(config.train.data)
    if not os.path.exists(dataset_path):
        raise Exception("Dataset at provided path {} not found!".format(dataset_path))

    # load basic metadata from training file
    print("\n============= Loaded Environment Metadata =============")
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=config.train.data)
    print(env_meta)

    env_meta["env_kwargs"]["reward_shaping"] = config.experiment.reward_shaping

    shape_meta = FileUtils.get_shape_metadata_from_dataset(
        dataset_path=config.train.data, all_obs_keys=config.all_obs_keys, verbose=True
    )

    if config.experiment.env is not None:
        env_meta["env_name"] = config.experiment.env
        print(
            "=" * 30
            + "\n"
            + "Replacing Env to {}\n".format(env_meta["env_name"])
            + "=" * 30
        )

    # create environment
    envs = OrderedDict()
    if config.experiment.rollout.enabled:
        # create environments for validation runs
        env_names = [env_meta["env_name"]]

        if config.experiment.additional_envs is not None:
            for name in config.experiment.additional_envs:
                env_names.append(name)
        for env_name in env_names:
            env = EnvUtils.create_env_from_metadata(
                env_meta=env_meta,
                env_name=env_name,
                render=False,
                render_offscreen=config.experiment.render_video,
                use_image_obs=shape_meta["use_images"],
            )
            envs[env.name] = env
            print(envs[env.name])

    import random
    from tqdm import trange, tqdm
    import pickle
    from robomimic.algo.dt import sample

    def generate_random_data(env, traj_bounds, policy=None, target_return=None):

        data_dict = {
            "states": [],
            "next_states": [],
            "rewards": [],
            "dones": [],
            "actions": [],
        }

        for _ in trange(2000):
            state = env.reset()
            states, next_states, rewards, dones, actions = [], [], [], [], []
            traj_len = random.randrange(traj_bounds[0], traj_bounds[1])

            for idx in range(traj_len):
                random_action = np.random.uniform(-1, 1, env.action_dimension)

                next_state, reward, done, _ = env.step(random_action)

                states.append(state)
                rewards.append(reward)
                dones.append(done)
                actions.append(random_action)
                next_states.append(next_state)
                state = next_state

                if done:
                    break

            data_dict["states"].append(states)
            data_dict["rewards"].append(rewards)
            data_dict["dones"].append(dones)
            data_dict["actions"].append(actions)
            data_dict["next_states"].append(next_states)

        return data_dict

    def generate_random_data_uniform(policy, env, target_return, traj_bounds):
        data_dict = {
            "states": [],
            "next_states": [],
            "rewards": [],
            "dones": [],
            "actions": [],
        }

        for _ in trange(5000):
            policy.start_episode()
            rtgs = [target_return]

            #############################################
            ob_dict = env.reset()

            ##############################################
            ob_dict = policy._prepare_observation(ob_dict)
            ob_dict["obs"] = ob_dict
            state = policy.policy.nets["encoder"](**ob_dict)
            state = state.unsqueeze(0)  # add batch dim
            device = policy.policy.device
            success = {k: False for k in env.is_success()}
            sampled_action = (
                sample(
                    policy.policy.nets["policy"],
                    state,
                    1,
                    temperature=1.0,
                    sample=True,
                    actions=None,
                    rtgs=torch.tensor(rtgs, dtype=torch.long)
                    .to(device)
                    .unsqueeze(0)
                    .unsqueeze(-1),
                    timesteps=torch.zeros((1, 1, 1), dtype=torch.int64).to(device),
                )
                .detach()
                .cpu()
                .numpy()
                .squeeze(0)
            )
            all_states = state
            states = []
            ############################################
            avg_rollout_traj_len = random.randrange(0, 50)
            replay_buffer_traj_avg_len = random.randrange(
                traj_bounds[0], traj_bounds[1]
            )
            states, rewards, dones, actions, next_states = [], [], [], [], []
            expert_reached_success = False

            for step_i in range(avg_rollout_traj_len):
                ob_dict, reward, done, _ = env.step(sampled_action)

                ###############################################
                actions += [sampled_action]
                ob_dict = policy._prepare_observation(ob_dict)
                ob_dict["obs"] = ob_dict
                state = policy.policy.nets["encoder"](**ob_dict)
                state = state.unsqueeze(0).to(device)
                all_states = torch.cat([all_states, state], dim=1)
                rtgs += [rtgs[-1] - reward]
                sampled_action = (
                    sample(
                        policy.policy.nets["policy"],
                        all_states,
                        1,
                        temperature=1.0,
                        sample=True,
                        actions=torch.tensor(np.array(actions), dtype=torch.float32)
                        .to(device)
                        .unsqueeze(0),  # .unsqueeze(1).unsqueeze(0),
                        rtgs=torch.tensor(rtgs, dtype=torch.long)
                        .to(device)
                        .unsqueeze(0)
                        .unsqueeze(-1),
                        timesteps=(
                            min(step_i, policy.policy.config.max_timestep)
                            * torch.ones((1, 1, 1), dtype=torch.int64).to(device)
                        ),
                    )
                    .detach()
                    .cpu()
                    .numpy()
                    .squeeze(0)
                )

                curr_success_metrics = env.is_success()
                for k in success:
                    success[k] = success[k] or curr_success_metrics[k]
                if done or success["task"]:
                    print("Success was achieved")
                    expert_reached_success = True
                    break
                #################################################
            if expert_reached_success:
                continue

            for step_i in range(replay_buffer_traj_avg_len):
                random_action = np.random.uniform(-1, 1, env.action_dimension)
                next_state, reward, done, _ = env.step(random_action)

                states.append(state)
                actions.append(random_action)
                dones.append(done)
                next_states.append(next_state)
                rewards.append(reward)
                state = next_state

                if done:
                    break

            #  print(f"Rewards: {rewards}")
            data_dict["states"].append(states)
            data_dict["rewards"].append(rewards)
            data_dict["dones"].append(dones)
            data_dict["actions"].append(actions)
            data_dict["next_states"].append(next_states)

        return data_dict

    # setup for a new training run
    data_logger = DataLogger(
        log_dir,
        log_tb=config.experiment.logging.log_tb,
    )
    model = algo_factory(
        algo_name=config.algo_name,
        config=config,
        obs_key_shapes=shape_meta["all_shapes"],
        ac_dim=shape_meta["ac_dim"],
        device=device,
    )
    print(
        f"Number of parameters in model: {sum(param.numel() for param in model.nets.parameters())}"
    )

    #  data_dict = generate_random_data(env, [102, 132]) # can
    #      data_dict = generate_random_data(env, [44, 58])   # lift

    # save the config as a json file
    with open(os.path.join(log_dir, "..", "config.json"), "w") as outfile:
        json.dump(config, outfile, indent=4)

    print("\n============= Model Summary =============")
    print(model)  # print model summary
    print("")

    # load training data
    trainset, validset = TrainUtils.load_data_for_training(
        config, obs_keys=shape_meta["all_obs_keys"]
    )
    train_sampler = trainset.get_dataset_sampler()

    print("\n============= Training Dataset =============")
    print(trainset)
    print("")

    # maybe retreve statistics for normalizing observations
    obs_normalization_stats = None
    if config.train.hdf5_normalize_obs:
        obs_normalization_stats = trainset.get_obs_normalization_stats()

    model_checkpoint_path = "/checkpoints/prajj/robomimic/output/dt/can_mg_layers_3_ctl_1_tgt_120_qual_low_dim_bc_False_mode_sparse_random_0_NEW/1/models/model_epoch_250_PickPlaceCan_success_0.82.pth"
    model.deserialize(torch.load(model_checkpoint_path)["model"])
    policy = RolloutPolicy(model, obs_normalization_stats=obs_normalization_stats)
    target_return = 120
    traj_bounds = [150, 151]
    data_dict = generate_random_data_uniform(
        policy=policy, env=env, target_return=target_return, traj_bounds=traj_bounds
    )
    #  data_dict = generate_random_data(env, [150, 151])
    with open(
        "/checkpoints/prajj/robomimic/can/mg/random_strategy_2_sparse.pkl", "wb"
    ) as fp:
        pickle.dump(data_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)
    exit()

    # initialize data loaders

    train_loader = DataLoader(
        dataset=trainset,
        sampler=train_sampler,
        batch_size=config.train.batch_size,
        shuffle=(train_sampler is None),
        num_workers=config.train.num_data_workers,
        drop_last=True,
    )

    if config.experiment.validate:
        # cap num workers for validation dataset at 1
        num_workers = min(config.train.num_data_workers, 1)
        valid_sampler = None  # validset.get_dataset_sampler()
        valid_loader = DataLoader(
            dataset=validset,
            sampler=valid_sampler,
            batch_size=config.train.batch_size,
            shuffle=(valid_sampler is None),
            num_workers=num_workers,
            drop_last=True,
        )
    else:
        valid_loader = None

    # main training loop
    best_valid_loss = None
    best_return = (
        {k: -np.inf for k in envs} if config.experiment.rollout.enabled else None
    )
    best_success_rate = (
        {k: -1.0 for k in envs} if config.experiment.rollout.enabled else None
    )
    last_ckpt_time = time.time()

    # number of learning steps per epoch (defaults to a full dataset pass)
    train_num_steps = config.experiment.epoch_every_n_steps
    valid_num_steps = config.experiment.validation_epoch_every_n_steps

    results_dict = defaultdict(dict)

    for epoch in range(1, config.train.num_epochs + 1):  # epoch numbers start at 1
        step_log = TrainUtils.run_epoch(
            model=model,
            data_loader=train_loader,
            epoch=epoch,
            num_steps=train_num_steps,
        )
        model.on_epoch_end(epoch)

        # setup checkpoint path
        epoch_ckpt_name = "model_epoch_{}".format(epoch)

        # check for recurring checkpoint saving conditions
        should_save_ckpt = False
        if config.experiment.save.enabled:
            time_check = (config.experiment.save.every_n_seconds is not None) and (
                time.time() - last_ckpt_time > config.experiment.save.every_n_seconds
            )
            epoch_check = (
                (config.experiment.save.every_n_epochs is not None)
                and (epoch > 0)
                and (epoch % config.experiment.save.every_n_epochs == 0)
            )
            epoch_list_check = epoch in config.experiment.save.epochs
            should_save_ckpt = time_check or epoch_check or epoch_list_check
        ckpt_reason = None
        if should_save_ckpt:
            last_ckpt_time = time.time()
            ckpt_reason = "time"

        if epoch > 0 and epoch % 10 == 0:
            print("Train Epoch {}".format(epoch))
            print(json.dumps(step_log, sort_keys=True, indent=4))

        for k, v in step_log.items():
            if k.startswith("Time_"):
                data_logger.record("Timing_Stats/Train_{}".format(k[5:]), v, epoch)
            else:
                data_logger.record("Train/{}".format(k), v, epoch)

        # Evaluate the model on validation set
        if epoch > 0 and epoch % 10 == 0 and config.experiment.validate:
            with torch.no_grad():
                step_log = TrainUtils.run_epoch(
                    model=model,
                    data_loader=valid_loader,
                    epoch=epoch,
                    validate=True,
                    num_steps=valid_num_steps,
                )
            for k, v in step_log.items():
                if k.startswith("Time_"):
                    data_logger.record("Timing_Stats/Valid_{}".format(k[5:]), v, epoch)
                else:
                    data_logger.record("Valid/{}".format(k), v, epoch)

            print("Validation Epoch {}".format(epoch))
            print(json.dumps(step_log, sort_keys=True, indent=4))

            # save checkpoint if achieve new best validation loss
            valid_check = "Loss" in step_log
            if valid_check and (
                best_valid_loss is None or (step_log["Loss"] <= best_valid_loss)
            ):
                if (
                    config.experiment.save.enabled
                    and config.experiment.save.on_best_validation
                ):
                    epoch_ckpt_name += "_best_validation_{}".format(best_valid_loss)
                    should_save_ckpt = True
                    ckpt_reason = "valid" if ckpt_reason is None else ckpt_reason

        # Evaluate the model by by running rollouts

        # do rollouts at fixed rate or if it's time to save a new ckpt
        video_paths = None
        rollout_check = (
            epoch % config.experiment.rollout.rate == 0
        )  # or (should_save_ckpt and ckpt_reason == "time")
        if (
            config.experiment.rollout.enabled
            and (epoch > config.experiment.rollout.warmstart)
            and rollout_check
        ):

            num_episodes = config.experiment.rollout.n

            rollout_model = RolloutPolicy(
                model, obs_normalization_stats=obs_normalization_stats
            )

            config.unlock()

            #  for tgt_return in [0.8, 1, 2, 3, 4, 5]:
            #  config.experiment.target_return = tgt_return
            all_rollout_logs, video_paths = TrainUtils.rollout_with_stats(
                config,
                policy=rollout_model,
                envs=envs,
                horizon=config.experiment.rollout.horizon,
                use_goals=config.use_goals,
                num_episodes=num_episodes,
                render=False,
                video_dir=video_dir if config.experiment.render_video else None,
                epoch=epoch,
                video_skip=config.experiment.get("video_skip", 5),
                terminate_on_success=config.experiment.rollout.terminate_on_success,
            )
            results_dict[epoch] = all_rollout_logs
            with open(config.train.output_dir + "results.json", "w") as fp:
                json.dump(results_dict, fp, indent=4)

            # summarize results from rollouts to tensorboard and terminal
            for env_name in all_rollout_logs:
                rollout_logs = all_rollout_logs[env_name]
                for k, v in rollout_logs.items():
                    if k.startswith("Time_"):
                        data_logger.record(
                            "Timing_Stats/Rollout_{}_{}".format(env_name, k[5:]),
                            v,
                            epoch,
                        )
                    else:
                        data_logger.record(
                            "Rollout/{}/{}".format(k, env_name),
                            v,
                            epoch,
                            log_stats=True,
                        )

                print(
                    "\nEpoch {} Rollouts took {}s (avg) with results:".format(
                        epoch, rollout_logs["time"]
                    )
                )
                print("Env: {}".format(env_name))
                print(json.dumps(rollout_logs, sort_keys=True, indent=4))

            # checkpoint and video saving logic
            updated_stats = TrainUtils.should_save_from_rollout_logs(
                all_rollout_logs=all_rollout_logs,
                best_return=best_return,
                best_success_rate=best_success_rate,
                epoch_ckpt_name=epoch_ckpt_name,
                save_on_best_rollout_return=config.experiment.save.on_best_rollout_return,
                save_on_best_rollout_success_rate=config.experiment.save.on_best_rollout_success_rate,
            )
            best_return = updated_stats["best_return"]
            best_success_rate = updated_stats["best_success_rate"]
            epoch_ckpt_name = updated_stats["epoch_ckpt_name"]
            should_save_ckpt = (
                config.experiment.save.enabled and updated_stats["should_save_ckpt"]
            ) or should_save_ckpt
            if updated_stats["ckpt_reason"] is not None:
                ckpt_reason = updated_stats["ckpt_reason"]

        # Only keep saved videos if the ckpt should be saved (but not because of validation score)
        should_save_video = (
            should_save_ckpt and (ckpt_reason != "valid")
        ) or config.experiment.keep_all_videos
        if video_paths is not None and not should_save_video:
            for env_name in video_paths:
                os.remove(video_paths[env_name])

        # Save model checkpoints based on conditions (success rate, validation loss, etc)
        if should_save_ckpt:
            TrainUtils.save_model(
                model=model,
                config=config,
                env_meta=env_meta,
                shape_meta=shape_meta,
                ckpt_path=os.path.join(ckpt_dir, epoch_ckpt_name + ".pth"),
                obs_normalization_stats=obs_normalization_stats,
            )

    #          # Finally, log memory usage in MB
    #  if epoch > 0 and epoch % 10 == 0:
    #  process = psutil.Process(os.getpid())
    #  mem_usage = int(process.memory_info().rss / 1000000)
    #  data_logger.record("System/RAM Usage (MB)", mem_usage, epoch)
    #              print("\nEpoch {} Memory Usage: {} MB\n".format(epoch, mem_usage))

    # terminate logging

    data_logger.close()


@pyrallis.wrap()
def main(args: Config):

    if args.config is not None:
        ext_cfg = json.load(open(args.config, "r"))
        config = config_factory(ext_cfg["algo_name"])
        config.unlock()
        config.update(ext_cfg)
        # update config with external json - this will throw errors if
        # the external config has keys not present in the base algo config
        config.experiment.name = args.experiment_name
        config.train.output_dir = args.output_dir
        config.train.data = args.dataset
        config.train.random_timesteps = args.random_timesteps
        config.experiment.reward_shaping = args.reward_shaping

        if ext_cfg["algo_name"] == "dt":
            config.algo.n_heads = args.n_heads
            config.algo.n_layers = args.n_layers
            config.algo.n_embed = args.n_embed
            #  config.algo.warmup_steps = args.warmup_steps # this is not being used currently
            config.algo.dropout = args.dropout
            config.experiment.rollout.n = args.rollout_n
            config.train.num_epochs = args.num_epochs
            config.algo.optim_params.policy.learning_rate.initial = args.lr
            config.algo.optim_params.policy.learning_rate.decay_factor = (
                args.weight_decay
            )
            config.experiment.rollout.rate = args.rollout_n
            config.experiment.rollout.horizon = args.horizon
            config.experiment.rollout.n = args.num_episodes_during_eval
            config.experiment.target_return = args.target_return
            config.train.seq_length = args.seq_length
            config.train.batch_size = args.batch_size
            config.train.mode = args.mode
            config.train.pct_traj = args.pct_traj
            config.algo.dt_bc_mode = args.dt_bc_mode

    #  else:
    #  config = config_factory(args.algo)

    #      if args.dataset is not None:
    #          config.train.data = args.dataset

    #      if args.name is not None:
    #          config.experiment.name = args.name

    # get torch device
    device = TorchUtils.get_torch_device(try_to_use_cuda=config.train.cuda)

    # maybe modify config for debugging purposes
    if args.debug:
        # shrink length of training to test whether this run is likely to crash
        config.unlock()
        config.lock_keys()

        # train and validate (if enabled) for 3 gradient steps, for 2 epochs
        config.experiment.epoch_every_n_steps = 3
        config.experiment.validation_epoch_every_n_steps = 3
        config.train.num_epochs = 2

        # if rollouts are enabled, try 2 rollouts at end of each epoch, with 10 environment steps
        config.experiment.rollout.rate = 1
        config.experiment.rollout.n = 2
        config.experiment.rollout.horizon = 10

        # send output to a temporary directory
        config.train.output_dir = "/tmp/tmp_trained_models"

    # lock config to prevent further modifications and ensure missing keys raise errors
    config.lock()

    # catch error during training and print it
    res_str = "finished run successfully!"
    try:
        train(config, device=device)
    except Exception as e:
        res_str = "run failed with error:\n{}\n\n{}".format(e, traceback.format_exc())
    print(res_str)


if __name__ == "__main__":
    # External config file that overwrites default config
    main()
