model_name="Slice_design1"
tasks=("NCvsProdromal" "ProdromalvsPD" "NCvsPD")

for task in "${tasks[@]}"; do
  python main_slice.py --model_name "$model_name" --task "$task" --bs 32 --num_workers 8
done

#python main_3d.py --model_name Slice_design1 --task NCvsProdromal --bs 16 --num_workers 8