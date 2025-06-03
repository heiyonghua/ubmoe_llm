import argparse
import swift
from swift.llm import sft_main, TrainArguments, infer_main, InferArguments
from transformers import AutoConfig
import lm_eval
if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument("--project_path",type=str)
    parser.add_argument("--model_path",type=str)
    parser.add_argument("--per_device_train_batch_size",type=int)
    parser.add_argument("--gradient_accumulation_steps",type=int)
    args=parser.parse_args().__dict__
    layer_num=AutoConfig.from_pretrained(f"{args['model_path']}",trust_remote_code=True).num_hidden_layers
    sft_main(
        TrainArguments(
            model=f"{args['model_path']}",
            dataset=[f""],
            freeze_parameters_ratio=1,
            trainable_parameters=[f"model.layers.{num}.mlp.gate" for num in range(layer_num)],
            train_type='full',
            per_device_train_batch_size=args['per_device_train_batch_size'],
            gradient_accumulation_steps=args['gradient_accumulation_steps'],
            dataloader_num_workers=16,
            dataset_num_proc=16)
    )