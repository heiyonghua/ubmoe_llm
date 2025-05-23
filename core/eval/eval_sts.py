# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*- #
import os


from ..model.BeLLM import MyAnglE
device = "0"
sleep_hour = 0
os.environ['TOKENIZERS_PARALLELISM'] = 'true'  # 控制tokenizer的并行问题
os.environ["CUDA_VISIBLE_DEVICES"] = device  # 控制显卡调用
from transformers import PreTrainedModel,PreTrainedTokenizer
import os
import logging
from .SentEval import engine
# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)
import torch
import fcntl
import time
from prettytable import PrettyTable
from typing import Optional
class Arg:
    def __init__(self):
        self.prompt='The representative word for sentence {text} is:"'
        self.tokenizer_name=''
        self.pooling_strategy='mean'
        self.apply_bfloat16:Optional[bool]=None
        self.model_name_or_path:Optional[str]=None
        self.max_length=1024
        self.mode='test'
        self.task_set='sts'
        self.lora_name_or_path:Optional[str]=None
        self.pretrained_model_path:Optional[str]=None
        self.checkpoint_path:Optional[str]=None


def print_table(task_names, scores):
    tb = PrettyTable()
    tb.field_names = task_names
    tb.add_row(scores)
    return tb


def lock_and_write_file(file_path, content):
    with open(file_path, 'a') as file:
        while True:
            try:
                # Acquire an exclusive lock (non-blocking)
                fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)

                # Perform your write operations here
                file.write(content + '\n')
                file.flush()

            except IOError as e:
                print("File is locked by another process. Can't write.")
                time.sleep(1)
            finally:
                # Release the lock
                fcntl.flock(file, fcntl.LOCK_UN)
                break
def eval_sts(model:PreTrainedModel,
             tokenizer:PreTrainedTokenizer,
             save_path:str,
             project_path:str
             ):
    PATH_TO_DATA=f'{project_path}/experience/core/eval/SentEval/data'
    args=Arg()
    AnglE_model = MyAnglE(model=model.eval(),
                        tokenizer=tokenizer,
                        pooling_strategy=args.pooling_strategy,
                        train_mode=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Set up the tasks
    if args.task_set == 'sts':
        args.tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']
        if args.mode == 'dev':
            args.tasks = ['STSBenchmark-dev']
    elif args.task_set == 'transfer':
        args.tasks = ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC']
    elif args.task_set == 'full':
        args.tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']
        args.tasks += ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC']

    # Set params for SentEval
    if args.mode == 'dev' or args.mode == 'fasttest':
        # Fast mode
        params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5, 'batch_size': 32}
        params['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 32,
                                         'tenacity': 3, 'epoch_size': 2}
    elif args.mode == 'test':
        # Full mode
        params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5, 'batch_size':16}
        params['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
                                         'tenacity': 5, 'epoch_size': 4}
    else:
        raise NotImplementedError

    # SentEval prepare and batcher
    def prepare(params, samples):
        return

    def batcher(params, batch, max_length=None):
        # Handle rare token encoding issues in the dataset
        if len(batch) >= 1 and len(batch[0]) >= 1 and isinstance(batch[0][0], bytes):
            batch = [[word.decode('utf-8') for word in s] for s in batch]

        sentences = [' '.join(s) for s in batch]
        if max_length == 500:
            sentences = [tokenizer.decode(tokenizer.encode(s, add_special_tokens=False)[:max_length]) for s in sentences]
            max_length = 512

        if args.prompt is not None:
            for i, s in enumerate(sentences):
                if len(s) > 0 and s[-1] not in '.?"\'': s += '.'
                # s = s.replace('"', '\'')
                if len(s) > 0 and '?' == s[-1]: s = s[:-1] + '.'
                if not s.strip():
                    continue
                sentences[i] = {'text': s}
        return AnglE_model.encode(sentences, device=device, to_numpy=True,prompt=args.prompt)

    results = {}
    for task in args.tasks:
        se = engine.SE(params, batcher, prepare)
        result = se.eval(task)
        results[task] = result

    # Print evaluation results
    if args.mode == 'dev':
        print("------ %s ------" % (args.mode))

        task_names = []
        scores = []
        for task in ['STSBenchmark-dev']:
            task_names.append(task)
            if task in results:
                scores.append("%.2f" % (results[task]['dev']['spearman'][0] * 100))
            else:
                scores.append("0.00")
        print_table(task_names, scores)

        if args.checkpoint_path is not None:
            # evaluate checkpoints on dev
            if os.path.exists(os.path.join(args.checkpoint_path, 'dev_results')):
                max_scores = 0
                with open(os.path.join(args.checkpoint_path, 'dev_results'), 'r') as f:
                    for i in f:
                        max_scores = max(max_scores, float(i.split()[1]))
            else:
                max_scores = 0

            # save best checkpoint
            if float(scores[-1]) >= max_scores:
                import shutil
                if args.lora_name_or_path is not None:
                    shutil.copytree(args.lora_name_or_path, os.path.join(args.checkpoint_path, 'best_model'), dirs_exist_ok=True)
                else:
                    shutil.copytree(args.model_name_or_path, os.path.join(args.checkpoint_path, 'best_model'), dirs_exist_ok=True)

            # log dev results
            with open(os.path.join(args.checkpoint_path, 'dev_results'), 'a') as f:
                prefix = args.mask_embedding_sentence_template if not args.avg else 'avg'
                line = prefix + ' ' +str(scores[-1]) + ' ' + \
                    args.lora_name_or_path if args.lora_name_or_path is not None else args.model_name_or_path
                f.write( line + '\n')

        task_names = []
        scores = []
        for task in ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']:
            task_names.append(task)
            if task in results:
                scores.append("%.2f" % (results[task]['devacc']))
            else:
                scores.append("0.00")
        task_names.append("Avg.")
        scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
        print_table(task_names, scores)


    elif args.mode == 'test' or args.mode == 'fasttest':
        print("------ %s ------" % (args.mode))

        task_names = []
        scores = []
        for task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']:
            task_names.append(task)
            if task in results:
                if task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']:
                    scores.append("%.2f" % (results[task]['all']['spearman']['all'] * 100))
                else:
                    scores.append("%.2f" % (results[task]['test']['spearman'].correlation * 100))
            else:
                scores.append("0.00")
        task_names.append("Avg.")
        scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
        print({f"{name}":score for (name,score) in zip(task_names,[float(s) for s in scores])})
        with open(f"{save_path}/sts.tet",'w') as file:
            file.write(str(print_table(task_names, scores)))


