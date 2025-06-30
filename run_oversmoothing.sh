#!/bin/bash

# Capture the start time
start_time=$(date +%s)

DATASETS=("Cora" "PubMed" "CiteSeer")
RESIDUALS=(1 0)

for residuals in "${RESIDUALS[@]}"
do
  for dataset in "${DATASETS[@]}"
  do  
      if [ "$residuals" -eq 1 ]; then
        python oversmoothing.py --dataset "$dataset" --residuals
      else
        python oversmoothing.py --dataset "$dataset"
      fi
      
      status=$?

      if [ $status -eq 0 ]; then
          echo "Oversmoothing experiment for $dataset completed successfully"
      else
          echo "Error during experiment for $dataset"
      fi
      
      sleep 5
  done
done

# Capture the end time and compute the duration
end_time=$(date +%s)
duration=$((end_time - start_time))

echo "All oversmoothing experiments completed!"
echo "Total execution time: $duration seconds"