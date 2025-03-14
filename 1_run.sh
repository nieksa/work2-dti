export CUDA_VISIBLE_DEVICES=0,1,2,3
tasks=("NCvsPD" "NCvsProdromal" "ProdromalvsPD")
for task in "${tasks[@]}"; do
  python main.py --epochs 100 --lr 0.005 --model_name contrastive_model1 --task "$task" --bs 6 --num_workers 4 --debug False --pick_metric_name accuracy --val_start 30 --val_interval 1 --early_stop_start 30 --patience 5
done
