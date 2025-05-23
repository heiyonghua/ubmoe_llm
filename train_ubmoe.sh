project_path="" \
model_name="Qwen2.5-0.5B-Instruct"

python train_ubmoe.py \
        --project_path "${project_path}" \
        --model_path "${project_path}/ubmoe_model/${model_name}" \
        --per_device_train_batch_size 16 \
        --gradient_accumulation_steps 2

python zhanka.py