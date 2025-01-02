tasks=("NCvsPD" "NCvsProdromal" "ProdromalvsPD")
models=("ViT" "ResNet18" "ResNet50" "C3D" "I3D" "SlowFast" "VGG" "DenseNet121" "DenseNet264")

for task in "${tasks[@]}"; do
    for model in "${models[@]}"; do
        python main.py --task "$task" --model_name "$model" --epochs 50 --lr 0.0001 --train_bs 16 --val_bs 16 --num_workers 4
    done
done