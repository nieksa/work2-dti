export CUDA_VISIBLE_DEVICES=0,1
tasks=("NCvsPD" "NCvsProdromal" "ProdromalvsPD")
for task in "${tasks[@]}"; do
  python main.py --task "$task" --bs 6 --num_workers 4
done
