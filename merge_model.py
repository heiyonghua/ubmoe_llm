import copy
import os
import shutil
import torch
from transformers import Qwen2Config, Qwen2MoeConfig, AutoModel, Qwen2ForCausalLM, AutoTokenizer
from model.modeling_ubmoe import UBMoeForCausalLM
from model.ubmoe_config import UBMoeConfig
from transformers import AutoModelForCausalLM
if __name__ == '__main__':
    home_path=""
    project_path=f""
    local_model_path=f""
    model_name='Qwen2.5-14B-Instruct'
    setting="Bi_Lora_FNN"
    qwen_config=Qwen2Config.from_pretrained(f"{project_path}/outputs/{model_name}/{setting}")

    ubmoe_config=UBMoeConfig(
        vocab_size=qwen_config.vocab_size,
        hidden_size=qwen_config.hidden_size,
        intermediate_size=qwen_config.intermediate_size,
        num_hidden_layers=qwen_config.num_hidden_layers,
        num_attention_heads=qwen_config.num_attention_heads,
        hidden_act=qwen_config.hidden_act,
        max_position_embeddings=qwen_config.max_position_embeddings,
        initializer_range=qwen_config.initializer_range,
        rms_norm_eps=qwen_config.rms_norm_eps,
        use_cache=qwen_config.use_cache,
        tie_word_embeddings=qwen_config.tie_word_embeddings,
        rope_theta=qwen_config.rope_theta,
        use_sliding_window=qwen_config.use_sliding_window,
        sliding_window=qwen_config.sliding_window,
        max_window_layers=qwen_config.max_window_layers,
        attention_dropout=qwen_config.attention_dropout,
        decoder_sparse_step=qwen_config.decoder_start_token_id,
        moe_intermediate_size=qwen_config.intermediate_size,
        num_experts_per_tok=1,
        num_key_value_heads=qwen_config.num_key_value_heads,
        num_experts=2,
        auto_map={
            "AutoModel": "modeling_ubmoe.UBMoeForCausalLM",
            "AutoConfig":"ubmoe_config.UBMoeConfig",
            "AutoModelForCausalLM":"modeling_ubmoe.UBMoeForCausalLM"
        },
        model_type="ubmoe",
        norm_topk_prob=True
    )

    ubmoe=UBMoeForCausalLM._from_config(config=ubmoe_config,torch_dtype=torch.bfloat16)


    for name, param in ubmoe.named_parameters():
        param_size = param.numel()
        param_shape = list(param.shape)
        memory = param.element_size() * param_size / (1024 ** 2)  # 单位：MB
        print(f"{name:<60} {str(param_shape):<30} {param_size:<15} {memory:<15.2f}")

    print(ubmoe.dtype)
    #merge
    original_model=Qwen2ForCausalLM.from_pretrained(f"{local_model_path}/{model_name}",torch_dtype=torch.bfloat16)
    print(original_model.dtype)
    ubmoe.set_input_embeddings(copy.deepcopy(original_model.get_input_embeddings()))
    ubmoe.set_output_embeddings(copy.deepcopy(original_model.get_output_embeddings()))

    ubmoe.model.norm=copy.deepcopy(original_model.model.norm)
    for i,layer in enumerate(ubmoe.model.layers):
        layer.input_layernorm=copy.deepcopy(original_model.model.layers[i].input_layernorm)
        layer.post_attention_layernorm = copy.deepcopy(original_model.model.layers[i].post_attention_layernorm)
        # merge ATT
        layer.self_attn.q_proj=copy.deepcopy(original_model.model.layers[i].self_attn.q_proj)
        layer.self_attn.k_proj = copy.deepcopy(original_model.model.layers[i].self_attn.k_proj)
        layer.self_attn.v_proj = copy.deepcopy(original_model.model.layers[i].self_attn.v_proj)
        layer.self_attn.o_proj = copy.deepcopy(original_model.model.layers[i].self_attn.o_proj)
        # merge FNN
        layer.mlp.experts[0].gate_proj=copy.deepcopy(original_model.model.layers[i].mlp.gate_proj)
        layer.mlp.experts[0].up_proj=copy.deepcopy(original_model.model.layers[i].mlp.up_proj)
        layer.mlp.experts[0].down_proj = copy.deepcopy(original_model.model.layers[i].mlp.down_proj)
    del original_model

    bi_model=Qwen2ForCausalLM.from_pretrained(f"{project_path}/outputs/{model_name}/{setting}",torch_dtype=torch.bfloat16)
    #bi_model=Qwen2ForCausalLM.from_pretrained(f"{local_model_path}/{model_name}",torch_dtype=torch.bfloat16)
    for i,layer in enumerate(ubmoe.model.layers):
        layer.mlp.experts[1].gate_proj=copy.deepcopy(bi_model.model.layers[i].mlp.gate_proj)
        layer.mlp.experts[1].up_proj=copy.deepcopy(bi_model.model.layers[i].mlp.up_proj)
        layer.mlp.experts[1].down_proj = copy.deepcopy(bi_model.model.layers[i].mlp.down_proj)

    tokenizer=AutoTokenizer.from_pretrained(f"{project_path}/outputs/{model_name}/{setting}")
    os.makedirs(f"{project_path}/ubmoe_model/{model_name}",exist_ok=True)

    for name, param in ubmoe.named_parameters():
        param_size = param.numel()
        param_shape = list(param.shape)
        memory = param.element_size() * param_size / (1024 ** 2)  # 单位：MB
        print(f"{name:<60} {str(param_shape):<30} {param_size:<15} {memory:<15.2f}")

    tokenizer.save_pretrained(f"{project_path}/ubmoe_model/{model_name}")
    ubmoe.save_pretrained(f"{project_path}/ubmoe_model/{model_name}")
    shutil.copy(f'{project_path}/experience/model/modeling_ubmoe.py', f'{project_path}/ubmoe_model/{model_name}')
    shutil.copy(f'{project_path}/experience/model/ubmoe_config.py', f'{project_path}/ubmoe_model/{model_name}')



