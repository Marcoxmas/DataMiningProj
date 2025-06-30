#!/bin/bash

# Capture the start time
start_time=$(date +%s)

GNN_ARCHITECTURES=("GKAN" "GCN" "GAT" "GIN" "SAGE")

for gnn in "${GNN_ARCHITECTURES[@]}"
do    
    if [ "$gnn" == "GKAN" ]; then
        python wandb_gkan_finetune.py --nc --lp --gc
        status=$?
    else
        python wandb_gnn_finetune.py --gnn "$gnn" --nc --lp --gc
        status=$?
    fi

    if [ $status -eq 0 ]; then
        echo "Finetuning for $gnn completed successfully"
    else
        echo "Error during finetuning for $gnn"
    fi
    
    sleep 5
done

# Capture the end time and compute the duration
end_time=$(date +%s)
duration=$((end_time - start_time))

echo "All finetuning processes completed!"
echo "Total execution time: $duration seconds"
echo "Results are available in your Weights & Biases dashboard"