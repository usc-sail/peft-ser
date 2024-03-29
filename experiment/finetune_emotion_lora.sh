
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

# for model_type in whisper_tiny whisper_base whisper_small; do
# for model_type in whisper_large; do
# for model_type in wav2vec2_0; do
for model_type in wavlm_plus; do
    for dataset in msp-podcast; do
        for finetune_method in lora; do
            for lora_rank in 1; do
                CUDA_VISIBLE_DEVICES=1, taskset -c 1-60 python3 finetune_emotion.py --pretrain_model $model_type --dataset $dataset --learning_rate 0.0005 --num_epochs 30 --finetune_method $finetune_method --lora_rank $lora_rank
            done
        done
    done
done