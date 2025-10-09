#!/bin/bash
trap 'echo "Stopping..."; pkill -P $$; exit 1' SIGINT

EXPT_ROOT=$GOAL_ROOT/logs/eval/
export PYTHONPATH=$PYTHONPATH:$GOAL_ROOT
export PYTHONPATH=$PYTHONPATH:$GOAL_ROOT/nav/astar_pycpp
export MAGNUM_LOG=quiet GLOG_minloglevel=2 HABITAT_SIM_LOG=quiet

SAVE_ROOT=$EXPT_ROOT/hm3d_transfer_mp3d_objectnav

cd $GOAL_ROOT/nav

# Read GPU list (comma-separated), max threads per GPU
IFS=',' read -ra GPUS <<< "$1"
MAX_THREADS_PER_GPU=$2

# If val parts list ($3) is empty, use default 0–10
if [ -z "$3" ]; then
    SELECTED_PARTS=(0 1 2 3 4 5 6 7 8 9 10)
else
    IFS=',' read -ra SELECTED_PARTS <<< "$3"
fi

NUM_GPUS=${#GPUS[@]}
declare -a GPU_LOAD
declare -a GPU_PIDS

# Initialize tracking arrays
for ((i = 0; i < NUM_GPUS; i++)); do
    GPU_LOAD[$i]=0
    GPU_PIDS[$i]=""
done

run_eval() {
    local part_id=$1
    local gpu_idx=$2
    local device=${GPUS[$gpu_idx]}
    local val_part="val_part_${part_id}"

    echo "Starting $val_part on GPU $device"

    CUDA_VISIBLE_DEVICES=$device python eval_goal.py \
        --exp-config $GOAL_ROOT/nav/configs/transfer_objectnav_mp3d.yaml \
        TASK_CONFIG.DATASET.DATA_PATH $GOAL_ROOT/data/datasets/objectnav/mp3d/v1/val_parts/{split}/{split}.json.gz \
        EVAL.SPLIT $val_part \
        TASK_CONFIG.SEED 100 \
        FLOW.expand_ratio 1.4 \
        FM.thr 0.30 \
        PROJECTOR.cat_pred_threshold 5.0 \
        PLANNER.change_goal_thr_up 320 \
        PLANNER.change_goal_thr_down 5 \
        GLOBAL_AGENT.seg_interval 5 \
        SCENE_SEGMENTATION.seg_pred_thr 0.7 \
        SCENE_SEGMENTATION.sem_pred_weights $GOAL_ROOT/pretrained_models/spconv_state.pth \
        FM.fm_weights $GOAL_ROOT/pretrained_models/hm3d_chatgpt.pth \
        TENSORBOARD_DIR $SAVE_ROOT/tb_seed_100_${val_part} \
        LOG_FILE $SAVE_ROOT/logs_seed_100_${val_part}.txt \
        GLOBAL_AGENT.name "PFExp" \
        GLOBAL_AGENT.smart_local_boundaries True \
        GLOBAL_AGENT.num_local_steps 1 \
        GLOBAL_AGENT.changegoal_steps 10 \
        PF_EXP_POLICY.pf_model_path $GOAL_ROOT/pretrained_models/area_potential.pth \
        PF_EXP_POLICY.pf_masking_opt "unexplored" &

    pid=$!
    GPU_LOAD[$gpu_idx]=$((GPU_LOAD[$gpu_idx]+1))
    GPU_PIDS[$gpu_idx]+="$pid "
}

wait_for_slot() {
    while true; do
        for ((gpu_idx = 0; gpu_idx < NUM_GPUS; gpu_idx++)); do
            if [ "${GPU_LOAD[$gpu_idx]}" -lt "$MAX_THREADS_PER_GPU" ]; then
                echo $gpu_idx
                return
            fi
        done

        # Check and update process completion
        for ((gpu_idx = 0; gpu_idx < NUM_GPUS; gpu_idx++)); do
            new_pids=""
            for pid in ${GPU_PIDS[$gpu_idx]}; do
                if ! kill -0 $pid 2>/dev/null; then
                    GPU_LOAD[$gpu_idx]=$((GPU_LOAD[$gpu_idx]-1))
                else
                    new_pids+="$pid "
                fi
            done
            GPU_PIDS[$gpu_idx]="$new_pids"
        done
        sleep 1
    done
}

# Dispatch tasks only for selected parts
for part_id in "${SELECTED_PARTS[@]}"; do
    gpu_idx=$(wait_for_slot)
    run_eval $part_id $gpu_idx
done

# Final wait for all remaining processes
for ((gpu_idx = 0; gpu_idx < NUM_GPUS; gpu_idx++)); do
    for pid in ${GPU_PIDS[$gpu_idx]}; do
        wait $pid
    done
done