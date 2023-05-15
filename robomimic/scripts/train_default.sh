#!/bin/bash
#SBATCH --partition=learnai4rl
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --array=1-5
#SBATCH --account=all
#SBATCH --job-name=can
#SBATCH --output=/checkpoints/prajj/slurm/dtliftmgdense10k.out
#SBATCH --ntasks-per-node=1
#SBATCH --err=/checkpoints/prajj/slurm/dtmgc.err

############################################################

export ALGO_NAME="dt"
export DATASET_NAME=can
export RANDOM_TIMESTEPS=0
export DATA_TYPE=mg
export MODE=dense
export SEQ_LENGTH=1
export TARGET_RETURN=120
export STRG=2

###################################################
export QUALITY=low_dim
export N_LAYERS=3
export DT_BC_MODE=False

# export WARMUP_STEPS=15000
export BATCH_SIZE=256
export NUM_EPOCHS=800
export ROLLOUT_N=50
export LR=1e-4
export WEIGHT_DECAY=0.1
export DROPOUT=0.1
export N_HEADS=1
export N_EMBED=128
export NUM_EPISODES_DURING_EVAL=50
export IMAGEIO_FFMPEG_EXE=/data/home/prajj/tools/ffmpeg-git-20220910-amd64-static/ffmpeg
export PCT_TRAJ=1

###########################################################
export CONFIG=robomimic/exps/templates/${ALGO_NAME}.json

if [ "$DATA_TYPE" == "mg" ]; then
        export DATASET=/checkpoints/prajj/robomimic/${DATASET_NAME}/${DATA_TYPE}/${QUALITY}_${MODE}.hdf5
else
        export DATASET=/checkpoints/prajj/robomimic/${DATASET_NAME}/${DATA_TYPE}/${QUALITY}.hdf5
fi

if [ "$ALGO_NAME" == "dt" ]; then
        export JOB_NAME=${DATASET_NAME}_${DATA_TYPE}_layers_${N_LAYERS}_ctl_${SEQ_LENGTH}_tgt_${TARGET_RETURN}_qual_${QUALITY}_bc_${DT_BC_MODE}_mode_${MODE}_random_${RANDOM_TIMESTEPS}_NEW
else
        export JOB_NAME=${DATASET_NAME}_${DATA_TYPE}_qual_${QUALITY}_${MODE}_random_${RANDOM_TIMESTEPS}_strg_${STRG}_NEW
fi

export OUTPUT_DIR=/checkpoints/prajj/robomimic/output/${ALGO_NAME}/${JOB_NAME}/${SLURM_ARRAY_TASK_ID}/

source $HOME/miniconda/etc/profile.d/conda.sh
cd $HOME/code/cai_research/robomimic
export PYOPENGL_PLATFORM="egl"

conda activate /data/home/sodhani/miniconda3/envs/hwm_tdmpc && conda activate mujoco && export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/home/prajj/.mujoco/mujoco210/bin

python3 robomimic/scripts/train.py \
        --config $CONFIG \
        --dataset $DATASET \
        --experiment_name $ALGO_NAME \
        --output_dir $OUTPUT_DIR \
        --batch_size $BATCH_SIZE \
        --num_epochs $NUM_EPOCHS \
        --rollout_n $ROLLOUT_N \
        --n_heads $N_HEADS \
        --n_layers $N_LAYERS \
        --n_embed $N_EMBED \
        --lr $LR \
        --seq_length $SEQ_LENGTH \
        --weight_decay $WEIGHT_DECAY \
        --dropout $DROPOUT \
        --target_return $TARGET_RETURN \
        --pct_traj $PCT_TRAJ \
        --num_episodes_during_eval $NUM_EPISODES_DURING_EVAL \
        --rollout_n $ROLLOUT_N \
        --random_timesteps $RANDOM_TIMESTEPS \
        --dt_bc_mode $DT_BC_MODE \
        --reward_shaping False\
