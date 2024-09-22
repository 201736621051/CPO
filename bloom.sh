deepspeed --num_gpus=8 lorabloom.py --deepspeed ds_config.json \
    --model_name bigscience/bloom-7b --dataset_name XSUM --domain_name news \
    --cur_epoch 0\
    --mode lora \
    --epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --generation_num_beams 2 \
    --lr 1e-6 \
    --worst_k 4

deepspeed --num_gpus=8 lorabloom.py --deepspeed ds_config.json \
    --model_name bigscience/bloom-7b --dataset_name XSUM --domain_name news \
    --cur_epoch 1\
    --mode lora \
    --epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --generation_num_beams 2 \
    --lr 2e-6 \
    --worst_k 4

deepspeed --num_gpus=8 lorabloom.py --deepspeed ds_config.json \
    --model_name bigscience/bloom-7b --dataset_name XSUM --domain_name news \
    --cur_epoch 2\
    --mode lora \
    --epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --generation_num_beams 2 \
    --lr 3e-6 \
    --worst_k 4

deepspeed --num_gpus=8 lorabloom.py --deepspeed ds_config.json \
    --model_name bigscience/bloom-7b --dataset_name XSUM --domain_name news \
    --cur_epoch 3\
    --mode lora \
    --epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --generation_num_beams 2 \
    --lr 4e-6 \
    --worst_k 4

deepspeed --num_gpus=8 lorabloom.py --deepspeed ds_config.json \
    --model_name bigscience/bloom-7b --dataset_name XSUM --domain_name news \
    --cur_epoch 4\
    --mode lora \
    --epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --generation_num_beams 2 \
    --lr 5e-6 \
    --worst_k 4

deepspeed --num_gpus=8 lorabloom.py --deepspeed ds_config.json \
    --model_name bigscience/bloom-7b --dataset_name XSUM --domain_name news \
    --cur_epoch 5\
    --mode lora \
    --epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --generation_num_beams 2 \
    --lr 6e-6 \
    --worst_k 4

deepspeed --num_gpus=8 lorabloom.py --deepspeed ds_config.json \
    --model_name bigscience/bloom-7b --dataset_name XSUM --domain_name news \
    --cur_epoch 6\
    --mode lora \
    --epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --generation_num_beams 2 \
    --lr 7e-6 \
    --worst_k 4

deepspeed --num_gpus=8 lorabloom.py --deepspeed ds_config.json \
    --model_name bigscience/bloom-7b --dataset_name XSUM --domain_name news \
    --cur_epoch 7\
    --mode lora \
    --epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --generation_num_beams 2 \
    --lr 8e-6 \
    --worst_k 4

deepspeed --num_gpus=8 lorabloom.py --deepspeed ds_config.json \
    --model_name bigscience/bloom-7b --dataset_name XSUM --domain_name news \
    --cur_epoch 8\
    --mode lora \
    --epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --generation_num_beams 2 \
    --lr 9e-6 \
    --worst_k 4

deepspeed --num_gpus=8 lorabloom.py --deepspeed ds_config.json \
    --model_name bigscience/bloom-7b --dataset_name XSUM --domain_name news \
    --cur_epoch 9\
    --mode lora \
    --epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --generation_num_beams 2 \
    --lr 1e-5 \
    --worst_k 4