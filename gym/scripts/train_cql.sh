#!/bin/bash
#SBATCH --partition=learnai4rl
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --array=1-5
#SBATCH --account=all
#SBATCH --job-name=cqlhu
#SBATCH --ntasks-per-node=1
#SBATCH --output=/checkpoints/prajj/slurm/cqlwmd.out
#SBATCH --err=/checkpoints/prajj/slurm/cqlwmd.err

## SBATCH OUTPUT, ERROR
## SBATCH JOB_NAME
## TASK_NAME
## NUM_STEPS
## REPLAY_PICKLED_DIR

############################################################

export ENV_NAME=humanoid
export DIFFICULTY=medium-expert
export BATCH_SIZE=2048
export MAX_STEPS_PER_ITER=1250  # 2048 -> 1250  ; 1024 -> 2500 ; 512 -> 5000
export PCT_TRAJ=1
export MODE=normal
export RANDOM_PCT_TRAJ=0
############################################################

export MAX_EP_LEN=1000
export REVERSE_REWARDS=False
export MAX_ITERS=100
export NUM_EVAL_EPISODES=100
export NUM_WORKERS=16
export REPLAY_PICKLED_DIR=/checkpoints/prajj/gym/

# export LR=6e-4
# export LOG_TO_WANDB=False

export JOB_NAME=cql_pct_${PCT_TRAJ}_tsteps_${MAX_STEPS_PER_ITER}_rev_${REVERSE_REWARDS}_mode_${MODE}_strategy_2_random_pct_traj_${RANDOM_PCT_TRAJ}
export CKPT_PATH=/checkpoints/prajj/gym_ckpt/${ENV_NAME}_${DIFFICULTY}/${JOB_NAME}/${SLURM_ARRAY_TASK_ID}"/"

source $HOME/miniconda/etc/profile.d/conda.sh
cd $HOME/code/cai_research/gym

conda activate /data/home/sodhani/miniconda3/envs/hwm_tdmpc && conda activate mujoco && export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/home/prajj/.mujoco/mujoco210/bin

python3 cql_corl.py  \
        --env_name $ENV_NAME \
        --difficulty $DIFFICULTY \
        --max_iters $MAX_ITERS \
        --max_steps_per_iter $MAX_STEPS_PER_ITER \
        --replay_pickled_dir $REPLAY_PICKLED_DIR \
        --checkpoints_path $CKPT_PATH \
        --batch_size $BATCH_SIZE \
        --pct_traj $PCT_TRAJ \
        --num_eval_episodes $NUM_EVAL_EPISODES \
        --reverse_rewards $REVERSE_REWARDS \
        --mode $MODE \
        --random_pct_traj $RANDOM_PCT_TRAJ \
