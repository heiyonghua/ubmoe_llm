home_path="/home/heiyonghua/fast_disk"

lm_eval --model_args pretrained="/data/user/user134/QA_baseqween_BiFinetune/outputs/Qwen2.5-0.5B-Instruct/Bi_Full_FNN",trust_remote_code=True \
    --tasks mmlu \
    --device cuda:0 \
    --num_fewshot 5 \
    --batch_size 1

#lm_eval --model_args pretrained="${home_path}/local_model/Qwen2.5-0.5B-Instruct" \
#    --tasks winogrande \
#    --device cuda:0 \
#    --num_fewshot 5 \
#    --batch_size 1