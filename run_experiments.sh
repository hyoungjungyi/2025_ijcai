#!/bin/bash

# Define the values for each parameter
markets=("dj30" "nasdaq" "kospi" "csi300" "sp500")
data_options=("general" "alpha158")
pred_lens=(1 5 20)

# Define available GPU cores (0-3 in this example)
gpu_cores=(0 1 2 3)

# GPU memory threshold (in MB) to ensure enough free memory before starting a new job
memory_threshold=5000  # Set based on your model's memory requirement
check_interval=300      # Check every 5 minutes (300 seconds)

# Function to check if a GPU has enough free memory
check_gpu_memory() {
    local gpu_id=$1
    free_memory=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $gpu_id | awk '{logger.info $1}')
    if [ "$free_memory" -gt "$memory_threshold" ]; then
        return 0  # GPU has enough free memory
    else
        return 1  # GPU does not have enough free memory
    fi
}

# Keep track of running experiments
declare -A running_jobs

# Function to clean up finished jobs from tracking
clean_finished_jobs() {
    for gpu in "${!running_jobs[@]}"; do
        if ! ps -p "${running_jobs[$gpu]}" > /dev/null 2>&1; then
            echo "Job on GPU $gpu has finished."
            unset running_jobs[$gpu]
        fi
    done
}

# Iterate through all combinations of parameters
for market in "${markets[@]}"; do
    for data in "${data_options[@]}"; do
        for pred_len in "${pred_lens[@]}"; do
            while :; do
                clean_finished_jobs
                for gpu in "${gpu_cores[@]}"; do
                    # Check if the GPU is available and not already running a job
                    if [[ -z "${running_jobs[$gpu]}" ]] && check_gpu_memory $gpu; then
                        echo "Running experiment with market=$market, data=$data, pred_len=$pred_len on GPU=$gpu"

                        # Execute the Python script with the specified parameters and GPU assignment
                        CUDA_VISIBLE_DEVICES=$gpu python run.py --market "$market" --data "$data" --pred_len "$pred_len" --moe_train &
                        job_pid=$!  # Get the process ID of the launched job
                        running_jobs[$gpu]=$job_pid  # Track the job

                        echo "Started job (PID $job_pid) on GPU $gpu for market=$market, data=$data, pred_len=$pred_len"

                        sleep 5  # Short sleep before checking other jobs
                        break 2  # Exit the while loop to proceed to the next experiment
                    fi
                done
                # If no GPU is free, wait before checking again
                echo "Waiting for available GPU memory or free GPU slot..."
                sleep $check_interval
            done
        done
    done
done

# Wait for all background processes to complete
wait

echo "All experiments completed."
