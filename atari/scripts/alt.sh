#!/bin/bash
export CONDA_ENV="decision-transformer-atari"
source $HOME/miniconda/etc/profile.d/conda.sh
zsh
cd $HOME/code/cai_research/atari
conda activate $CONDA_ENV

export N_HEAD=8
export N_LAYER=24
export N_EMBED=128
export CONTEXT_LENGTH=30
export NUM_BUFFERS=1
export NUM_GAMES_TO_USE_FOR_EVAL=100
export MAX_NUM_SAMPLES_PER_BUFFER=1000000

export GAME="Pong"

############################
export JOB_NAME=${GAME}_head_${N_HEAD}_n_layer_${N_LAYER}_embed_${N_EMBED}_context_${CONTEXT_LENGTH}_buff_${NUM_BUFFERS}_eval_${NUM_GAMES_TO_USE_FOR_EVAL}_max_samples_per_buffer_${MAX_NUM_SAMPLES_PER_BUFFER}
export CKPT_PATH=/checkpoints/prajj/${GAME}/${JOB_NAME}/
export BATCH_SIZE=64
# export NUM_BUFFERS=50
export NUM_STEPS=500000
export EPOCHS=5
# export DATA_DIR_PREFIX=/fsx/chinnadhurai/atari/
export DATA_DIR_PREFIX=/fsx/prajj/datasets/atari/

for task_idx in `seq 1 5`; do
    sbatch  --job-name ${JOB_NAME}_${task_idx} \
        --partition hipri \
        --gres gpu:1 \
        --cpus-per-task=32 \
        --output ${JOB_NAME}_std.out \
        --error ${JOB_NAME}_std.err \
        --time 1440 \
        --wrap "
    #!/bin/bash
    python3 run_dt_atari.py  \
        --epochs $EPOCHS \
        --model_type 'reward_conditioned' \
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
        --ckpt_path ${CKPT_PATH}${task_idx}"/" \
"
done
wait
