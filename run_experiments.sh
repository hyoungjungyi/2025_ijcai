#!/bin/bash

markets=("dj30" "nasdaq" "kospi" "csi300")
data_options=("alpha158")
model=('Transformer' 'Informer' 'Reformer' 'Autoformer' 'Flashformer' 'itransformer')
lr=(0.00001 0.00005)
gpu_cores=(0 1 2 3 4 5 6 7)
memory_threshold=5000  # 최소 필요 메모리 (MB)
check_interval=60      # GPU 체크 주기 (초)
max_jobs_per_gpu=2     # GPU당 동시에 실행 가능한 최대 실험 개수

# GPU 메모리 확인 함수
check_gpu_memory() {
    local gpu_id=$1
    free_memory=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu_id" | awk '{print $1}')

    # 값이 유효하지 않으면 -1 반환
    [[ -z "$free_memory" || "$free_memory" -lt 0 ]] && echo -1 && return

    # 메모리가 충분하면 GPU ID 반환, 부족하면 -1 반환
    [[ "$free_memory" -gt "$memory_threshold" ]] && echo "$free_memory" || echo -1
}

# 사용 가능한 GPU 찾기
find_available_gpu() {
    best_gpu=-1
    max_free_memory=0
    for gpu in "${gpu_cores[@]}"; do
        # 현재 GPU에서 돌고 있는 작업 수 확인
        # (running_jobs[$gpu] 내부에 공백으로 구분된 PID 목록이 있음)
        current_jobs_str="${running_jobs[$gpu]}"
        read -ra current_jobs <<< "$current_jobs_str"
        num_current_jobs=${#current_jobs[@]}

        # 이미 최대 개수(2개) 이상이면 스킵
        if [[ "$num_current_jobs" -ge "$max_jobs_per_gpu" ]]; then
            continue
        fi

        free_memory=$(check_gpu_memory "$gpu")

        # 유효한 메모리 값인지 확인
        [[ "$free_memory" -eq -1 ]] && continue

        # 메모리가 충분한 경우, 가장 여유 메모리가 큰 GPU 선택
        if [[ "$free_memory" -gt "$memory_threshold" && "$free_memory" -gt "$max_free_memory" ]]; then
            max_free_memory=$free_memory
            best_gpu=$gpu
        fi
    done
    echo "$best_gpu"
}

# 실행 중인 프로세스 추적 (GPU -> "pid1 pid2" 형태로 저장)
declare -A running_jobs

# 죽은 프로세스(종료된 작업) 정리 함수
clean_finished_jobs() {
    for gpu in "${!running_jobs[@]}"; do
        current_jobs_str="${running_jobs[$gpu]}"
        read -ra current_jobs <<< "$current_jobs_str"

        alive_jobs=()
        for pid in "${current_jobs[@]}"; do
            if [[ -e "/proc/$pid" ]]; then
                alive_jobs+=("$pid")
            fi
        done

        # 살아있는 PID만 다시 저장
        running_jobs[$gpu]="${alive_jobs[*]}"
    done
}

var=0
for experiment_count in {1..4}; do
    for market in "${markets[@]}"; do
        for model in "${model[@]}"; do
            for lr_val in "${lr[@]}"; do
                while :; do
                    clean_finished_jobs
                    best_gpu=$(find_available_gpu)

                    # 적절한 GPU가 있을 경우만 실행
                    if [[ "$best_gpu" -ne -1 ]]; then
                        # 실행 직전에 다시 한 번 메모리 체크
                        final_memory_check=$(check_gpu_memory "$best_gpu")
                        if [[ "$final_memory_check" -lt "$memory_threshold" ]]; then
                            echo "GPU $best_gpu memory dropped below threshold. Retrying..."
                            sleep "$check_interval"
                            continue
                        fi

                        echo "Using GPU $best_gpu for experiment (Market: $market, Data: $data, LR: $lr_val)"
                        CUDA_VISIBLE_DEVICES=$best_gpu python run.py --market "$market" --model "$model" --learning_rate "$lr_val"  \
                            --wandb_session_name "setting_${var}_exp${experiment_count}_$(date +%s)" &

                        pid=$!
                        # 기존에 있는 PID 목록에 추가
                        running_jobs[$best_gpu]="${running_jobs[$best_gpu]} $pid"

                        var=$((var + 1))
                        sleep 10
                        break
                    fi
                    sleep "$check_interval"
                done
            done
        done
    done
done

# 남은 프로세스들 종료될 때까지 대기
clean_finished_jobs
if [[ ${#running_jobs[@]} -gt 0 ]]; then
    for gpu in "${!running_jobs[@]}"; do
        current_jobs_str="${running_jobs[$gpu]}"
        read -ra current_jobs <<< "$current_jobs_str"
        if [[ ${#current_jobs[@]} -gt 0 ]]; then
            wait "${current_jobs[@]}"
        fi
    done
fi

echo "All experiments completed."
