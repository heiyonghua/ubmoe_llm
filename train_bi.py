# -*- coding: utf-8 -*- #
import json
import os
import tqdm
import argparse
os.environ['TOKENIZERS_PARALLELISM'] = 'true'  # 控制tokenizer的并行问题
import numpy as np
import torch
from billm.config import logger
from datasets import load_dataset
from core.model.BeLLM import MyAnglE, AngleDataTokenizer
import random
from core.set_model import set_model_tokenizer
from typing import Optional
from transformers import TrainingArguments
from core.eval.eval_sts import  eval_sts
def get_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("--model_name",type=str,default="Qwen2.5-0.5B-Instruct")
    parser.add_argument("--use_lora",type=str,default="true")
    parser.add_argument("--training_target",type=str,default='ATT')
    parser.add_argument("--project_path",type=str,default="")
    parser.add_argument("--home_path",type=str,default="")

    parser.add_argument("--per_device_train_batch_size",type=int,default=64)
    parser.add_argument("--gradient_accumulation_steps",type=int,default=1)
    parser.add_argument("--learning_rate",type=float,default=3e-4)

    return parser.parse_args().__dict__
class Arg:
    def __init__(self,project_path):
        # 原始类属性
        self.dataset_name_or_path: str = f""
        self.train_subset_name: Optional[str] = None
        self.train_split_name: str = "train"
        self.valid_name_or_path: Optional[str] = None
        self.prompt_template: str = "The representative word for sentence {text} is:\""
        self.valid_subset_name: Optional[str] = None
        self.workers: int = 16
        self.loss_kwargs={
            'cosine_w': 0.0,
            'ibn_w': 1.0,
            'angle_w': 0.0,
            'cosine_tau': 20.0,
            'ibn_tau': 20.0,
            'angle_tau': 20.0,
        }
        self.pooling_strategy: str = 'mean'



def main(model, tokenizer, train_argument: TrainingArguments,use_lora:bool,project_path:str):
    args = Arg(project_path=project_path)

    if train_argument.seed is not None and train_argument.seed > 0:
        os.environ['PYTHONHASHSEED'] = str(train_argument.seed)
        random.seed(train_argument.seed)
        np.random.seed(train_argument.seed)
        torch.manual_seed(train_argument.seed)

    AnglE_model = MyAnglE(model=model,
                        tokenizer=tokenizer,
                        pooling_strategy=args.pooling_strategy,
                        train_mode=True)

    train_dataset = load_dataset(args.dataset_name_or_path, args.train_subset_name,split=f"{args.train_split_name}")

    logger.info('Processing train...')



    train_dataset = train_dataset.shuffle(train_argument.seed).map(
        AngleDataTokenizer(AnglE_model.tokenizer,
                           prompt_template=args.prompt_template,max_length=1024), num_proc=args.workers)

    train_dataset=[train_dataset[i] for i in tqdm.tqdm(range(len(train_dataset)))]
    print(train_dataset[0])
    valid_ds = None
    if valid_ds is None and args.valid_name_or_path is not None:
        logger.info('Validation detected, processing validation...')
        if os.path.exists(args.valid_name_or_path):
            valid_ds = load_dataset('json', data_files=[args.valid_name_or_path])
        else:
            valid_ds = load_dataset(args.valid_name_or_path, args.valid_subset_name)
        valid_ds = valid_ds[args.valid_subset_name or 'train'].map(
            AngleDataTokenizer(AnglE_model.tokenizer,
                               prompt_template=args.prompt_template), num_proc=args.workers
        )

    AnglE_model.fit(
        train_ds=train_dataset,
        valid_ds=valid_ds,
        loss_kwargs=args.loss_kwargs,
        args=train_argument,
    )
    if use_lora:
        print("merge")
        AnglE_model.backbone=AnglE_model.backbone.merge_and_unload()

    AnglE_model.backbone.save_pretrained(train_argument.output_dir)
    AnglE_model.tokenizer.save_pretrained(train_argument.output_dir)
    print(type(AnglE_model.backbone))
    eval_sts(model=AnglE_model.backbone.eval(),tokenizer=tokenizer,save_path=train_argument.output_dir,project_path=project_path)




if __name__ == '__main__':
    args=get_args()
    with open(f"",'r') as file:
        train_config=json.load(file)
    args['training_target'] = args['training_target'].split(",")
    args['use_lora']=True if args['use_lora'] is 'true' else False
    train_config['per_device_train_batch_size']=args['per_device_train_batch_size']
    train_config['gradient_accumulation_steps']=args['gradient_accumulation_steps']
    train_config['learning_rate']=args['learning_rate']
    train_name = f"Bi_{'Full' if not args['use_lora'] else 'Lora'}_{'_'.join(args['training_target'])}"
    train_config['output_dir']=f""

    default_lora_config={
            "r": 8,  # Lora attention dimension (the “rank”).
            "lora_alpha": 16,  # 用于Lora缩放的alpha参数
            "lora_dropout": 0.05,  # The dropout probability for Lora layers.
            "use_rslora": True,
    }
    model, tokenizer = set_model_tokenizer(model_path=f"",
                                           training_target=args['training_target'],
                                           bi_model=True,
                                           use_lora=args['use_lora'],
                                           lora_config=default_lora_config,
                                           lora_checkpoint=None
    )

    # load dataset
    train_argument = TrainingArguments(**train_config)

    main(
        model=model,
        train_argument=train_argument,
        tokenizer=tokenizer,
        use_lora=args['use_lora'],
        project_path=args['project_path']
    )