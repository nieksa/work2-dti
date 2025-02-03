model_name="ROI_transformer"
tasks=("NCvsProdromal" "ProdromalvsPD" "NCvsPD")

for task in "${tasks[@]}"; do
  python main_roi.py --model_name "$model_name" --task "$task" --bs 32 --num_workers 8
done