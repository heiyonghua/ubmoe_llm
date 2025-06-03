import os
import time
import torch
os.environ['TOKENIZERS_PARALLELISM'] = 'true'  # 控制tokenizer的并行问题
os.environ["CUDA_VISIBLE_DEVICES"] = device  # 控制显卡调用
import random
import wandb
from transformers import Trainer, TrainingArguments
from core.model import MyQwen2ForCausalLM
from core.Dataloder import AutoQADataLoder
from core.evaluator.myMetrics import Metric
from core.generate import generate
from core.params import train_args, global_args
from core.set_model import set_model_tokenizer
from core.utils import (set_config,
                        get_experience_model_name,zip_and_upload_to_wandb,
                        set_wandb,print_core_information,set_path_before_train,renew_global_arg)
from core.config import loraConfig


if __name__ == '__main__':
    bi=True
    method = ["debug", "eval", "train_without_save", "normal"][-1]
    set_path_before_train(global_args)
    # save files
    for arg in train_args:
        global_args=renew_global_arg(arg=arg,global_arg=global_args)
        print_core_information(arg=arg, global_arg=global_args)

        train_name = get_experience_model_name(args=arg,global_arg=global_args)
        wandb.init(project="huggingface" if method == "debug" else "修改无答案模板",
                   name=f"{train_name}_{method}" + "_new_temple",
                   dir=f"{global_args['project_path']}/WANDB")
        data_config, generate_config, model_config, dataloder_config = set_config(global_args, arg, train_name)

        print("begin to save file")
        # save_file
        artifact = wandb.Artifact(name="code", type="model")

        zip_and_upload_to_wandb(global_args["project_path"] + "/core", artifact)
        artifact.add_file(global_args["project_path"] + "/train.py")

        # set path
        global_args['train_arg']['output_dir'] = global_args['model_save'] + f"/{train_name}"

        # set  collator
        model, tokenizer, collator = set_model_tokenizer(config=model_config, model_name=model_config['model_name'],
                                                     train=True,check_point_path=arg['self_config']['checkpoint_path'])

        # load dataset
        train_dataset, dev_dataset, test_dataset = AutoQADataLoder(tokenizer=tokenizer,
                                                                   model_name=arg['self_config']['model_name'],
                                                                   method=method,
                                                                   **dataloder_config).load_dataest(**data_config)

        # print example
        chat_token_sample = collator([random.choice(train_dataset)], example=True)
        print("sample train data")
        print(f"{tokenizer.decode(chat_token_sample['input_ids'][0], skip_special_tokens=False)}")
        sft_sentence = [label if label != -100 else tokenizer.eos_token_id for label in chat_token_sample['labels'][0]]
        print("sft_sentence", f"{tokenizer.decode(sft_sentence, skip_special_tokens=False)}")

        time.sleep(sleep_hour * 3600)
        # train
        trainer = Trainer(
            args=TrainingArguments(**global_args['train_arg']),
            model=model,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            data_collator=collator,
        )
        wandb.config['method'] = method
        set_wandb(arg=arg, global_args=global_args)

        if method != "eval":
            trainer.train()

        if method in ['train_small', 'train', 'normal']:
            trainer.save_model(output_dir=global_args['train_arg']['output_dir'] + "/final")
        evaluate = Metric(toker=tokenizer)
        # generate #
        generate(eval_model=trainer.model, test_dataset=test_dataset, tokenizer=tokenizer, **generate_config,
                 collator=collator, evaluate=evaluate, model_name=model_config['model_name'])

        artifact.save()
        del model,trainer
        torch.clear_autocast_cache()
        wandb.finish()


