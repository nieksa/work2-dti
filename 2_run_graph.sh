export CUDA_VISIBLE_DEVICES=1
tasks=("NCvsPD" "NCvsProdromal" "ProdromalvsPD")
models=("graph_model_gcn" "graph_model_gat" "graph_model_graphsage")
for task in "${tasks[@]}"; do
  for model in "${models[@]}"; do
    python main_incre.py --epochs 100 --lr 0.05 --model_name "$model" --task "$task" --bs 32 --num_workers 4 --debug 0 --pick_metric_name accuracy --val_start 1 --val_interval 1 --early_stop_start 50 --patience 10
    done
done
