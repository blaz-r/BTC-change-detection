#!/bin/bash

datasets=("sysu" "levir" "gvlm" "egybcd" "clcd" "oscd96")

for dataset in "${datasets[@]}"; do
    num_workers=0
    epochs=100

    if [[ $dataset == "sysu" ]]; then
        num_workers=4
    elif [[ $dataset == "levir" ]]; then
        num_workers=4
    elif [[ $dataset == "gvlm" ]]; then
        num_workers=4
    elif [[ $dataset == "egybcd" ]]; then
        num_workers=8
    elif [[ $dataset == "clcd" ]]; then
        num_workers=8
    elif [[ $dataset == "oscd96" ]]; then
        num_workers=8
        epochs=50
    else
        echo "Unknown dataset: $dataset"
        exit 1
    fi

  sbatch ./run_slurm.sh --data.dataset "$dataset" --data.num_workers $num_workers --train.epochs $epochs
done