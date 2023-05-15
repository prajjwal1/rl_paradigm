#!/bin/bash
#SBATCH --partition=learnai4rl
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --array=1-5
#SBATCH --account=all
#SBATCH --job-name=atari_context
#SBATCH --ntasks-per-node=1
#SBATCH --output=/checkpoints/prajj/slurm/scaling.out
#SBATCH --err=/checkpoints/prajj/slurm/scaling.err


export CONDA_ENV="decision-transformer-atari"
# source $HOME/miniconda/etc/profile.d/conda.sh
# zsh
cd $HOME/code/cai_research/atari
# conda activate $CONDA_ENV

echo $SLURM_ARRAY_TASK_ID

export N_HEAD=8
export N_LAYER=20
export N_EMBED=128

export CONTEXT_LENGTH=50
export NUM_BUFFERS=50
export NUM_GAMES_TO_USE_FOR_EVAL=100
export MAX_NUM_SAMPLES_PER_BUFFER=100000
export NUM_STEPS=500000
export MODEL_TYPE=reward_conditioned
export GAME="Pong"

############################
export BATCH_SIZE=16
export EPOCHS=5
export DATA_DIR_PREFIX=/fsx/prajj/datasets/atari/

export JOB_NAME=${GAME}_head_${N_HEAD}_n_layer_${N_LAYER}_embed_${N_EMBED}_context_${CONTEXT_LENGTH}_buff_${NUM_BUFFERS}_eval_${NUM_GAMES_TO_USE_FOR_EVAL}_max_samples_per_buffer_${MAX_NUM_SAMPLES_PER_BUFFER}_model_${MODEL_TYPE}_steps_${NUM_STEPS}
export CKPT_PATH=/checkpoints/prajj/${GAME}/${JOB_NAME}/

python3 run_dt_atari.py  \
--epochs $EPOCHS \
--model_type $MODEL_TYPE  \
--num_steps $NUM_STEPS \
--num_buffers $NUM_BUFFERS \
--num_games_to_use_for_eval $NUM_GAMES_TO_USE_FOR_EVAL \
--n_head $N_HEAD \
--n_layer $N_LAYER \
--max_num_samples_per_buffer $MAX_NUM_SAMPLES_PER_BUFFER \
--n_embed $N_EMBED \
--context_length $CONTEXT_LENGTH \
--game $GAME \
--batch_size $BATCH_SIZE \
--data_dir_prefix $DATA_DIR_PREFIX  \
--ckpt_path ${CKPT_PATH}${SLURM_ARRAY_TASK_ID}"/"
