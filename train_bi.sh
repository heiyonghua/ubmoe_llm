home_path=""

python train_bi.py \
      --model_name "Qwen2.5-0.5B-Instruct" \
      --use_lora false \
      --training_target FNN \
      --project_path "${home_path}/QA_baseqween_BiFinetune" \
      --home_path "${home_path}" \
      --per_device_train_batch_size 8 \
      --gradient_accumulation_steps 2

python train_bi.py \
      --model_name "Qwen2.5-1.5B-Instruct" \
      --use_lora false \
      --training_target FNN \
      --project_path "${home_path}/QA_baseqween_BiFinetune" \
      --home_path "${home_path}" \
      --per_device_train_batch_size 8 \
      --gradient_accumulation_steps 2

python train_bi.py \
      --model_name "Qwen2.5-3B-Instruct" \
      --use_lora false \
      --training_target FNN \
      --project_path "${home_path}/QA_baseqween_BiFinetune" \
      --home_path "${home_path}" \
      --per_device_train_batch_size 8 \
      --gradient_accumulation_steps 2

python train_bi.py \
      --model_name "Qwen2.5-7B-Instruct" \
      --use_lora false \
      --training_target FNN \
      --project_path "${home_path}/QA_baseqween_BiFinetune" \
      --home_path "${home_path}" \
      --per_device_train_batch_size 8 \
      --gradient_accumulation_steps 2


python train_bi.py \
      --model_name "Qwen2.5-14B-Instruct" \
      --use_lora false \
      --training_target FNN \
      --project_path "${home_path}/QA_baseqween_BiFinetune" \
      --home_path "${home_path}" \
      --per_device_train_batch_size 8 \
      --gradient_accumulation_steps 2

python zhanka.py 60