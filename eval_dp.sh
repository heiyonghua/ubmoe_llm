#python caculate_dp.py \
#       --model_path "/data/user/user134/local_model/Qwen2.5-0.5B-Instruct" \
#       --dataset_path "/data/user/user134/QA_baseqween_BiFinetune/data/for_dp/mmlu.json"

#python caculate_dp.py \
#       --model_path "/data/user/user134/QA_baseqween_BiFinetune/outputs/Qwen2.5-0.5B-Instruct/Bi_Full_FNN" \
#       --dataset_path "/data/user/user134/QA_baseqween_BiFinetune/data/for_dp/mmlu.json"

python caculate_dp_ubmoe.py \
       --model_path "/data/user/user134/QA_baseqween_BiFinetune/experience/output/Qwen2.5-0.5B-Instruct/ubmoe/checkpoint-1000" \
       --dataset_path "/data/user/user134/QA_baseqween_BiFinetune/data/for_dp/mmlu.json"