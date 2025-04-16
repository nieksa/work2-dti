export CUDA_VISIBLE_DEVICES=1,2,3
tasks=("NCvsPD" "NCvsProdromal" "ProdromalvsPD")
# tasks=("NCvsProdromal")
model="3D_ResNet18"
for task in "${tasks[@]}"; do
  python main.py \
  --work_type Contrastive \
  --epochs 100 --bs 8 --num_workers 4 --lr 0.005 \
  --model_name "$model" --task "$task" \
  --debug 0 \
  --pick_metric_name accuracy \
  --val_start 1 --val_interval 1 \
  --early_stop_start 50 --patience 10
done
