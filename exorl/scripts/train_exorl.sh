#!/bin/bash
#SBATCH --partition=hipri
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --array=1-3
#SBATCH --account=all
#SBATCH --job-name=kzl
#SBATCH --ntasks-per-node=1
#SBATCH --output=/checkpoints/prajj/slurm/kzl.out
#SBATCH --err=/checkpoints/prajj/slurm/kzl.err


## SBATCH OUTPUT, ERROR
## SBATCH JOB_NAME
## TASK_NAME
## NUM_STEPS
## REPLAY_PICKLED_DIR

############################################################

export TASK_NAME=walker_walk
export NUM_STEPS=1000000
export CONTEXT_LEN=20
export BATCH_SIZE=1024
export ALGORITHM=icm_apt

export LR=1e-4
export MODEL_TYPE=dt

export TRAINING_STEPS_PER_ITER=10000
export EMBED_DIM=128
export N_LAYER=6

export REPLAY_PICKLED_DIR=/checkpoints/prajj/exorl_replay_buffer/${TASK_NAME}/${ALGORITHM}/buffer_${NUM_STEPS}.npz
export REPLAY_DIR=/checkpoints/prajj/datasets/exorl/walker/${ALGORITHM}/buffer
export PCT_TRAJ=100
export SCALE=10
############################################################

export MAX_EP_LEN=1000
export N_HEAD=8
export MAX_ITERS=10
export WARMUP_STEPS=10000
export NUM_EVAL_EPISODES=10
export NUM_WORKERS=32
export LOG_TO_WANDB=False

export JOB_NAME=${TASK_NAME}_${ALGORITHM}_head_${N_HEAD}_n_layer_${N_LAYER}_embed_${EMBED_DIM}_context_${CONTEXT_LEN}_model_${MODEL_TYPE}_data_${NUM_STEPS}_tsteps_${TRAINING_STEPS_PER_ITER}_lr_${LR}_ws_${WARMUP_STEPS}_scale_${SCALE}
export CKPT_PATH=/checkpoints/prajj/exorl_ckpt/${TASK_NAME}/${JOB_NAME}/${SLURM_ARRAY_TASK_ID}"/"

export CONDA_ENV="torch"
export PYOPENGL_PLATFORM="egl"
source $HOME/miniconda/etc/profile.d/conda.sh
cd $HOME/code/cai_research/exorl
conda activate $CONDA_ENV
echo $SLURM_ARRAY_TASK_ID


python3 dataset.py  \
        --max_iters $MAX_ITERS \
        --context_len $CONTEXT_LEN \
        --task_name $TASK_NAME \
        --training_steps_per_iter $TRAINING_STEPS_PER_ITER \
        --max_ep_len $MAX_EP_LEN \
        --replay_pickled_dir $REPLAY_PICKLED_DIR \
        --model_type $MODEL_TYPE \
        --embed_dim $EMBED_DIM \
        --n_layer $N_LAYER \
        --ckpt_path $CKPT_PATH \
        --n_head $N_HEAD \
        --algorithm_name $ALGORITHM \
        --scale $SCALE \
        --lr $LR \
        --replay_dir $REPLAY_DIR \
        --job_name $JOB_NAME \
        --batch_size $BATCH_SIZE \
        --warmup_steps $WARMUP_STEPS \
        --pct_traj $PCT_TRAJ \
        --num_eval_episodes $NUM_EVAL_EPISODES \
        --log_to_wandb $LOG_TO_WANDB \
        --num_workers $NUM_WORKERS \
