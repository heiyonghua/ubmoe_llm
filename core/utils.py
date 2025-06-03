import os
import random
import shutil
import wandb
from .config import loraConfig

from .config import Encoder_arg, Decoder_arg



def get_experience_model_name(args: dict, global_arg: dict) -> str:
    if args['self_config']['train_encoder']:
        name = args['self_config']['model_name']
        name += "_bi_" if len(args['addition_config_for_model']['bidirectional_interval'])!=0 else "_Uni_"
        name += format((global_arg['train_arg']['learning_rate']), ".0e")
        name += "_"
        name += str(global_arg['train_arg']['num_train_epochs'])
        name += f"_{args['addition_config_for_model']['useful_interval']}"
        pass
    else:
        name = args['self_config']['model_name']
        name += f"_{args['self_config']['dataset_list']}"
        name += f"_{args['self_config']['mask_list']}"
        name += f"_{args['self_config']['generate_mask']}"
        name += f"{'use_bidirectional_interval' if args['self_config']['use_bidirectional_interval'] else 'normal'}"
    return name


def set_config(global_args, arg, train_name):
    mask_list = arg['self_config']["mask_list"]
    assert len(mask_list) == global_args['train_arg']['num_train_epochs']

    data_config = {
        "dataset_list": arg['self_config']['dataset_list']
    }
    dataloder_config = {
        "use_cache": False,
        "mask_list": mask_list,
        "generate_mask": arg['self_config']['generate_mask'],
        "no_answer_rate_train": arg['self_config']['no_answer_rate_train'],
        "no_answer_rate_test": arg['self_config']['no_answer_rate_test'],
        "seed": global_args['train_arg']['seed'],

        "dataset_rate": arg['self_config']['dataset_rate']
    }

    generate_config = {
        "output_dir": global_args['output_dir'] + f"/{train_name}.text",
        "device": arg["self_config"]["device"]
    }

    model_config = {
        "model_name": arg["self_config"]['model_name'],

        "lora_config": global_args['lora_config'],
        "lora_path": arg['self_config']['lora_path'] if len(arg['self_config']['lora_path']) != 0 else None,

        "device": arg["self_config"]["device"],

        "addition_config_for_model": arg["addition_config_for_model"],

    }
    return data_config, generate_config, model_config, dataloder_config




def print_core_information(arg: dict, global_arg: dict):
    print("*****Basic Information*****")
    print("model name:", arg['self_config']['model_name'])
    for k in arg['addition_config_for_model'].keys():
        print(f"{k}:{arg['addition_config_for_model'][k]}")
    print("random seed:", global_arg['train_arg']['seed'])
    print("Learning rate", global_arg['train_arg']['learning_rate'])
    print("warmup_steps:", global_arg['train_arg']['warmup_steps'])
    print("per_device_train_batch_size:", global_arg['train_arg']['per_device_train_batch_size'])
    print("num_train_epochs", global_arg['train_arg']['num_train_epochs'])

    print("*****Dataset Set*****")
    print("Dataset_list:", arg['self_config']['dataset_list'])
    print("Dataset_rate", arg['self_config']['dataset_rate'])
    print("no_answer_rate_train:", arg['self_config']['no_answer_rate_train'])
    print("no_answer_rate_test:", arg['self_config']['no_answer_rate_test'])


def set_path_before_train(global_arg: dict):
    picture_path = global_arg['project_path'] + "/picture"
    text_out_path = global_arg['output_dir']
    cache_path = global_arg['cache_path']
    wandb_path = global_arg['project_path'] + "/WANDB"
    path_list = [picture_path, text_out_path, cache_path, wandb_path]
    for path in path_list:
        if not os.path.exists(path):
            os.makedirs(path)
    return 0


def renew_global_arg(arg: dict, global_arg: dict):
    global_arg['train_arg'] = Encoder_arg if arg['self_config']['train_encoder'] else Decoder_arg
    global_arg['train_arg']['seed'] = arg['self_config']['seed']
    global_arg['lora_config'] = loraConfig[arg['self_config']['model_name']]

    return global_arg

