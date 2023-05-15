#!/bin/bash
#SBATCH --partition=learnai4rl
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --array=1-5
#SBATCH --account=all
#SBATCH --job-name=SE
#SBATCH --ntasks-per-node=1
#SBATCH --output=/checkpoints/prajj/slurm/dtwmed.out
#SBATCH --err=/checkpoints/prajj/slurm/dtwmed.err

############################################################

export ENV_NAME=humanoid
export DIFFICULTY=medium-expert
export CONTEXT_LEN=256

export MODEL_TYPE=dt

export MAX_STEPS_PER_ITER=5000  # 2048 -> 1250  ; 1024 -> 2500 ; 512 -> 5000

export REPLAY_PICKLED_DIR=/checkpoints/prajj/gym/
export PCT_TRAJ=1
export RANDOM_PCT_TRAJ=0

export REVERSE_REWARDS=False
export MODE=normal
############################################################
export EMBED_DIM=128
export BATCH_SIZE=512
export N_LAYER=3
export N_HEAD=1


export MAX_EP_LEN=1000
export MAX_ITERS=10
export WARMUP_STEPS=$MAX_STEPS_PER_ITER
export NUM_EVAL_EPISODES=100
export NUM_WORKERS=16
export LR=6e-4
export LOG_TO_WANDB=False

export JOB_NAME=${MODEL_TYPE}_head_${N_HEAD}_n_layer_${N_LAYER}_context_${CONTEXT_LEN}_pct_${PCT_TRAJ}_tsteps_${MAX_STEPS_PER_ITER}_rev_${REVERSE_REWARDS}_mode_${MODE}_strategy_2_random_pct_traj_${RANDOM_PCT_TRAJ}
export CKPT_PATH=/checkpoints/prajj/gym_ckpt/${ENV_NAME}_${DIFFICULTY}/${JOB_NAME}/${SLURM_ARRAY_TASK_ID}"/"

export CONDA_ENV="torch"
source $HOME/miniconda/etc/profile.d/conda.sh
cd $HOME/code/cai_research/gym

conda activate /data/home/sodhani/miniconda3/envs/hwm_tdmpc && conda activate mujoco && export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/home/prajj/.mujoco/mujoco210/bin

python3 experiment.py  \
        --env_name $ENV_NAME \
        --difficulty $DIFFICULTY \
        --max_iters $MAX_ITERS \
        --context_len $CONTEXT_LEN \
        --max_steps_per_iter $MAX_STEPS_PER_ITER \
        --max_ep_len $MAX_EP_LEN \
        --replay_pickled_dir $REPLAY_PICKLED_DIR \
        --model_type $MODEL_TYPE \
        --embed_dim $EMBED_DIM \
        --n_layer $N_LAYER \
        --ckpt_path $CKPT_PATH \
        --n_head $N_HEAD \
        --lr $LR \
        --job_name $JOB_NAME \
        --batch_size $BATCH_SIZE \
        --warmup_steps $WARMUP_STEPS \
        --pct_traj $PCT_TRAJ \
        --num_eval_episodes $NUM_EVAL_EPISODES \
        --log_to_wandb $LOG_TO_WANDB \
        --num_workers $NUM_WORKERS \
        --reverse_rewards $REVERSE_REWARDS \
        --mode $MODE \
        --random_pct_traj $RANDOM_PCT_TRAJ \
