model_name="3D_ResNet18"
tasks=("NCvsProdromal" "ProdromalvsPD" "NCvsPD")

for task in "${tasks[@]}"; do
  python main_3d.py --model_name "$model_name" --task "$task" --bs 32 --num_workers 8
done