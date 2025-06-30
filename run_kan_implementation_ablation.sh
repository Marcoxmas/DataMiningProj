#!/bin/bash

# Capture the start time
start_time=$(date +%s)

DATASETS=("Cora" "PubMed" "CiteSeer")

for dataset in "${DATASETS[@]}"
do    
    python kan_implementation_ablation.py --dataset "$dataset"
    status=$?

    if [ $status -eq 0 ]; then
        echo "Ablation for $dataset completed successfully"
    else
        echo "Error during experiment for $dataset"
    fi
    
    sleep 5
done

# Capture the end time and compute the duration
end_time=$(date +%s)
duration=$((end_time - start_time))

echo "All ablationn experiments completed!"
echo "Total execution time: $duration seconds"