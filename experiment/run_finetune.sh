# CUDA_VISIBLE_DEVICES=0 taskset 500 python3 finetune_single_thread.py --pretrain_model wav2vec2_0_original --dataset crema_d
# CUDA_VISIBLE_DEVICES=1 taskset 500 python3 finetune_single_thread.py --pretrain_model wav2vec2_0_original --dataset ravdess --learning_rate 0.0005
# CUDA_VISIBLE_DEVICES=0 taskset 500 python3 finetune_single_thread.py --pretrain_model wav2vec2_0 --dataset ravdess --learning_rate 0.0005
# CUDA_VISIBLE_DEVICES=0 taskset 500 python3 finetune_single_thread.py --pretrain_model wav2vec2_0 --dataset crema_d --num_epochs
# CUDA_VISIBLE_DEVICES=0 taskset 500 python3 finetune_single_thread.py --pretrain_model wav2vec2_0 --dataset iemocap --learning_rate 0.0002 --num_epochs 30 --downstream_model cnn

# CUDA_VISIBLE_DEVICES=0 taskset 500 python3 finetune_single_thread.py --pretrain_model wav2vec2_0_original --dataset ravdess --learning_rate 0.0005 --downstream_model cnn
# CUDA_VISIBLE_DEVICES=0 taskset 500 python3 finetune_single_thread.py --pretrain_model wav2vec2_0 --dataset ravdess --learning_rate 0.001 --downstream_model cnn --num_epochs 50
# CUDA_VISIBLE_DEVICES=0 taskset 500 python3 finetune_single_thread.py --pretrain_model wav2vec2_0_original --dataset ravdess --learning_rate 0.001 --downstream_model cnn --num_epochs 50

# CUDA_VISIBLE_DEVICES=0 taskset -c 10-30 python3 finetune_single_thread.py --pretrain_model wav2vec2_0 --dataset iemocap --learning_rate 0.001 --downstream_model cnn --num_epochs 30
# CUDA_VISIBLE_DEVICES=0 taskset -c 10-30 python3 finetune_single_thread.py --pretrain_model wav2vec2_0_original --dataset iemocap --learning_rate 0.001 --downstream_model cnn --num_epochs 30

# CUDA_VISIBLE_DEVICES=0 taskset -c 60-90 python3 finetune_single_thread.py --pretrain_model wav2vec2_0 --dataset iemocap --learning_rate 0.001 --downstream_model cnn --num_epochs 30 --num_layers 2
# CUDA_VISIBLE_DEVICES=0 taskset -c 60-90 python3 finetune_single_thread.py --pretrain_model wav2vec2_0_original --dataset crema_d --learning_rate 0.001 --downstream_model cnn --num_epochs 30 --num_layers 2
CUDA_VISIBLE_DEVICES=0 taskset -c 60-90 python3 finetune_single_thread.py --pretrain_model wav2vec2_0_original --dataset crema_d --learning_rate 0.001 --downstream_model cnn --num_epochs 30 --num_layers 3 --conv_layers 2
CUDA_VISIBLE_DEVICES=0 taskset -c 60-90 python3 finetune_single_thread.py --pretrain_model wav2vec2_0 --dataset crema_d --learning_rate 0.001 --downstream_model cnn --num_epochs 30 --num_layers 3 --conv_layers 2
# CUDA_VISIBLE_DEVICES=0 taskset -c 60-90 python3 finetune_single_thread.py --pretrain_model wav2vec2_0 --dataset msp-improv --learning_rate 0.001 --downstream_model cnn --num_epochs 30 --num_layers 2
# CUDA_VISIBLE_DEVICES=0 taskset -c 60-90 python3 finetune_single_thread.py --pretrain_model wav2vec2_0 --dataset meld --learning_rate 0.001 --downstream_model cnn --num_epochs 30 --num_layers 2
# CUDA_VISIBLE_DEVICES=0 taskset -c 60-90 python3 finetune_single_thread.py --pretrain_model wav2vec2_0 --dataset ravdess --learning_rate 0.001 --downstream_model cnn --num_epochs 30 --num_layers 2
# CUDA_VISIBLE_DEVICES=1 taskset -c 60-90 python3 finetune_single_thread.py --pretrain_model wav2vec2_0_original --dataset iemocap --learning_rate 0.001 --downstream_model cnn --num_epochs 30