import argparse
import gc
import os
import pickle
import random
import warnings
from typing import Optional
import pandas as pd
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import PreTrainedTokenizer, AutoTokenizer
from core.model.modeling_qwen import MyQwen2Model
import json

def caculate_dp_one(attention_weight: torch.Tensor, max_n:int=100)->dict:
    dp_result={}
    #print(attention_weight.size())

    max_len=attention_weight.size()[-1]
    #attention_weight = torch.nn.functional.softmax(attention_weight)
    diagonal=torch.diag_embed(attention_weight.diagonal())
    dp_result['dp-before']=(attention_weight.tril()-diagonal).sum().sum()/max_len
    dp_result['dp-after'] = (attention_weight-attention_weight.tril()).sum().sum() / max_len
    dp_result['main']=diagonal.sum().sum() / max_len


    return dp_result

def caculate_dp(attention_weight_tuple,num_head)->dict:
    new_result={}
    new_result[f'layer']=caculate_dp_one(attention_weight_tuple.sum(dim=0)/attention_weight_tuple.size()[0]/num_head)
    return new_result


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
os.environ['TOKENIZERS_PARALLELISM'] = 'true'  # 控制tokenizer的并行问题
os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'

class Tulu_loader:
    def __init__(self, tokenizer: PreTrainedTokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.padding_token=151643

    def __call__(self, data):
        #return dict(input_ids=None)
        data = data['messages']
        #print(data)
        data = self.tokenizer.apply_chat_template(data,return_tensors='pt',return_dict=True)
        return data


    def collator(self, features: list[dict]):
        if len(features)<0:
            return dict(
            #attention_mask=torch.tensor([features[0]['attention_mask']]).to("cuda"),
            input_ids=torch.tensor([features[0]['input_ids']]).to("cuda"),
        )
        else:
            max_len=max([len(feature['input_ids']) for feature in features])
            return dict(
                #attention_mask=torch.tensor([feature['attention_mask']+[0 for _ in  range(len(feature['attention_mask']),max_len)] for feature in features]).to("cuda"),
                input_ids=torch.tensor([feature['input_ids']+[self.padding_token for _ in  range(len(feature['input_ids']),max_len)] for feature in features]).to("cuda"),
            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train embeddings script')
    parser.add_argument('--model_path', type=str, default='0', help='--model_path')
    parser.add_argument('--peft_path', type=str, default=None, help='--peft_path')
    parser.add_argument('--dataset_path', type=str, default=None, help='--peft_path')
    parser.add_argument('--seed', type=int, default=42, help='--peft_path')
    new_args = parser.parse_args()
    seed=new_args.seed
    random.seed(seed)
    model_path=new_args.model_path
    peft_path=new_args.peft_path

    model = MyQwen2Model.from_pretrained(model_path)
    num_head=len(model.layers)
    if peft_path is not None:
        model = PeftModel.from_pretrained(model, peft_path).merge_and_unload()
    model.to("cuda")
    model=model.requires_grad_(False)
    model=model.eval()

    with open(new_args.dataset_path, 'r') as f:
        dataset = json.load(f)

    dp_result: Optional[dict] = None

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    dataloader = Tulu_loader(tokenizer=tokenizer, max_len=2048)


    for data in tqdm(dataset):
        input_feature = dataloader(data)
        output = model(**input_feature.to('cuda'), output_attentions=True)
        dp_one = caculate_dp(output.attentions,num_head)
        if dp_result is None:
            dp_result = dp_one
        else:
            for layer_key in dp_one.keys():
                for iter_key in dp_one[layer_key]:
                    dp_result[layer_key][iter_key] += dp_one[layer_key][iter_key]
        del output

    dp_final = pd.DataFrame()
    # print(dp_result)
    dp_after = torch.tensor(0.0)
    dp_before = torch.tensor(0.0)
    dp_main = torch.tensor(0.0)
    for layer_key in dp_result.keys():
        dp_main += dp_result[layer_key]['main'].to("cpu")
        dp_after += dp_result[layer_key]['dp-after'].to("cpu")
        dp_before += dp_result[layer_key]['dp-before'].to("cpu")

    print(
        f"dp_main:{(dp_main / len(dp_result.keys())) / len(dataset) *100},dp_after:{(dp_after / len(dp_result.keys())) / len(dataset) *100},dp_before:{(dp_before / len(dp_result.keys())) / len(dataset) *100}")
    del model
    torch.clear_autocast_cache()
    gc.collect()