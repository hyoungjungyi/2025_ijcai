#!/bin/bash

markets=("nasdaq" "kospi" "csi300")
data_options=("alpha158")
pred_lens=(1 5 20)
learning_rates=(0.0001 0.00005 0.00001)
gpu_cores=(4 5 6 7)
memory_threshold=9000  # 최소 필요 메모리 (MB)
check_interval=60      # GPU 체크 주기 (초)
gpu_stabilization_time=30  # GPU 할당 후 안정화 대기 시간 (초)

# GPU 메모리 확인 함수
check_gpu_memory() {
    local gpu_id=$1
    free_memory=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu_id" | awk '{print $1}')

    [[ -z "$free_memory" || "$free_memory" -lt 0 ]] && echo -1 && return
    [[ "$free_memory" -gt "$memory_threshold" ]] && echo "$free_memory" || echo -1
}

# GPU에서 실행 중인 프로세스 수 확인
check_gpu_process_count() {
    local gpu_id=$1
    process_count=$(nvidia-smi pmon -i "$gpu_id" | tail -n +3 | awk '{print $2}' | grep -v "-" | wc -l)
    echo "$process_count"
}

# 사용 가능한 GPU 찾기
find_available_gpu() {
    best_gpu=-1
    max_free_memory=0
    min_process_count=9999

    for gpu in "${gpu_cores[@]}"; do
        free_memory=$(check_gpu_memory "$gpu")
        process_count=$(check_gpu_process_count "$gpu")

        [[ "$free_memory" -eq -1 ]] && continue

        if [[ "$free_memory" -gt "$memory_threshold" && "$process_count" -lt "$min_process_count" ]]; then
            min_process_count=$process_count
            max_free_memory=$free_memory
            best_gpu=$gpu
        fi
    done

    echo "$best_gpu"
}

# 실행 중인 프로세스 추적
declare -A running_jobs

clean_finished_jobs() {
    for gpu in "${!running_jobs[@]}"; do
        if [[ ! -e /proc/${running_jobs[$gpu]} ]]; then
            unset running_jobs[$gpu]
        fi
    done
}

var=0
for experiment_count in {1..10}; do
    for market in "${markets[@]}"; do
        for data in "${data_options[@]}"; do
            for learning_rate in "${learning_rates[@]}"; do
                retry_count=0
                max_retries=10

                while [[ $retry_count -lt $max_retries ]]; do
                    clean_finished_jobs
                    best_gpu=$(find_available_gpu)

                    if [[ "$best_gpu" -ne -1 ]]; then
                        break
                    fi

                    echo "No available GPU found. Retrying in $check_interval seconds... ($((retry_count+1))/$max_retries)"
                    sleep "$check_interval"
                    retry_count=$((retry_count + 1))
                done

                if [[ "$best_gpu" -eq -1 ]]; then
                    echo "Error: No GPU available after $max_retries attempts. Skipping experiment."
                    continue
                fi

                echo "Using GPU $best_gpu for experiment (Market: $market, Data: $data, LR: $learning_rate)"
                CUDA_VISIBLE_DEVICES=$best_gpu python run.py --market "$market" --data "$data" --learning_rate "$learning_rate" --moe_train \
                    --wandb_session_name "setting_${var}_exp${experiment_count}_$(date +%s)" &

                var=$((var + 1))
                running_jobs[$best_gpu]=$!

                echo "Waiting $gpu_stabilization_time seconds for GPU $best_gpu to stabilize..."
                sleep "$gpu_stabilization_time"
            done
        done
    done
done

clean_finished_jobs
[[ ${#running_jobs[@]} -gt 0 ]] && wait
echo "All experiments completed."
