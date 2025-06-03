from typing import NamedTuple, Optional,List,Dict
import torch
from peft import LoraConfig,get_peft_model,PeftModel
from transformers import PreTrainedModel,PreTrainedTokenizer, AutoTokenizer,AutoModel



def set_model_tokenizer(
                        model_path:str,
                        training_target:List,
                        bi_model:bool,
                        use_lora:bool,
                        lora_config:Dict,
                        lora_checkpoint:Optional[str]
                        )-> tuple[PreTrainedModel,PreTrainedTokenizer]:
    if bi_model:
        model=AutoModel.from_pretrained(model_path,torch_dtype=torch.bfloat16,trust_remote_code=True,_attn_implementation='flash_attention_2')
        for layer in model.layers:
            #pass
            print(type(layer.self_attn))
            layer.self_attn.is_causal=False
    else:
        model=AutoModel.from_pretrained(model_path,torch_dtype=torch.bfloat16,trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # process training target
    if training_target is not None and len(training_target) != 0:
        if use_lora:
            if lora_checkpoint is None:
                target_modules = []
                if 'FNN' in training_target:
                    target_modules += ["gate_proj", "up_proj", "down_proj"]
                if 'FNN_GATE' in training_target:
                    target_modules += ["up_proj", "down_proj"]
                if "ATT" in training_target:
                    target_modules += ['q_proj', 'k_proj', 'v_proj', 'o_proj']
                lora_config = LoraConfig(target_modules=target_modules,**lora_config)
                lora_model = get_peft_model(model, lora_config)
                model = lora_model
                model.enable_input_require_grads()
            elif lora_checkpoint is not None:
                model = PeftModel.from_pretrained(model, lora_checkpoint, adapter_name="sft")
        else:
            model.requires_grad_(False)
            for layer in model.layers:
                if 'FNN' in training_target:
                    layer.mlp.requires_grad_(True)
                if 'ATT' in training_target:
                    layer.self_attn.requires_grad_(True)
                if 'FNN_GATE' in training_target:
                    layer.mlp.requires_grad_(True)
                    layer.mlp.gate_proj.requires_grad_(False)
        model.enable_input_require_grads()
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name)
    else:
        pass

    return model, tokenizer
