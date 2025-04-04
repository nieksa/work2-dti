export CUDA_VISIBLE_DEVICES=1,2,3
#tasks=("NCvsPD" "NCvsProdromal" "ProdromalvsPD")
tasks=("NCvsPD" "ProdromalvsPD")
for task in "${tasks[@]}"; do
  python main.py --epochs 100 --lr 0.005 --model_name contrastive_model1 --task "$task" --bs 8 --num_workers 4 --debug 0 --pick_metric_name accuracy --val_start 50 --val_interval 1 --early_stop_start 50 --patience 10
done
