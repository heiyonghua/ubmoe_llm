import os
import time
import torch
import random
from transformers import Trainer, TrainingArguments
from core.params import train_args, global_args
from core.set_model import set_model_tokenizer
from core.utils import (set_config,
                        get_experience_model_name,print_core_information,set_path_before_train)

import torch._dynamo
torch._dynamo.config.suppress_errors = True
if __name__ == '__main__':

    # save files
    for arg in train_args:
        print_core_information(arg=arg, global_arg=global_args)
        train_name = get_experience_model_name(arg)+"_new_bi"

        data_config, generate_config, model_config, dataloder_config = set_config(global_args, arg, train_name)


        # set path
        global_args['train_arg']['output_dir'] = global_args['model_save'] + f"/{train_name}"

        # set  collator
        model, tokenizer, collator = set_model_tokenizer(config=model_config, model_name=model_config['model_name'])



        evaluate = Metric(toker=tokenizer)
