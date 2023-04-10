
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

# for model_type in whisper_tiny whisper_base whisper_small; do
# for model_type in whisper_large wav2vec2_0; do
# for model_type in wav2vec2_0 wavlm_plus; do
for model_type in wav2vec2_0; do
    for dataset in iemocap crema_d; do
            CUDA_VISIBLE_DEVICES=0, python3 finetune_emotion.py --pretrain_model $model_type --dataset $dataset --learning_rate 0.0005 --num_epochs 30 --finetune_method combined --adapter_hidden_dim 128 --embedding_prompt_dim 1 --lora_rank 32
    done
done