tasks=("NCvsPD" "NCvsProdromal" "ProdromalvsPD")
models=["Design6","ResNet18"]

for task in "${tasks[@]}"; do
    for model in "${models[@]}"; do
        python model_result.py --task "$task" --model_list "$model" --fold 1 --val_bs 16 --num_workers 4
    done
done