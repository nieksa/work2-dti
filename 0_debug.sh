export CUDA_VISIBLE_DEVICES=0,1,2,3
python main.py \
--work_type Contrastive \
--epochs 100 --bs 6 --lr 0.005  --num_workers 4\
--model_name contrastive_model2 \
--task NCvsPD  \
--debug 1 \
--pick_metric_name accuracy \
--val_start 30 --val_interval 1 \
--early_stop_start 30 --patience 5
