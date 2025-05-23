
home_path=[][0]

global_args = {
    "project_path": f"",

    'model_save':   f"",

    "output_dir":   f"",

    "cache_path":   f"",

    # 如果使用lora请添加，如果不使用lora，请不要添加
    # config for trainer
    "train_arg": "",
    # config for generate
    "generate_config": {
    }
    # "group_by_length": True,
    # 消融实验
}


train_args = [
    {
        "self_config": {
            "model_name": "Qwen2.5-0.5B-Chat",

            #"checkpoint_path":"/home/heiyonghua/dist2/local_model/experience_model/Qwen1.5-0.5B_bi_3e-04_1_[0]",
            "checkpoint_path":None,
            "train_encoder":True,#是否训练编码器

            # for decoder train
            "dataset_list":["doqa_v2.1"],
            "dataset_rate": 1,

            "no_answer_rate_train":1.0,
            "no_answer_rate_test":1.0,

            "generate_mask": ['Causal'][0],
            # 混合掩码微调
            "mask_list": [['Causal']][0],
            "use_bidirectional_interval":True,

            "lora_path": "",
            "device": "cuda",

            "seed": [42, 712, 6737151][0],
        },

        "addition_config_for_model":{
            "useful_interval": [i for i in range(1)],
            "is_encoder_only": True,
            "is_embedding_model": True,
            "bidirectional_interval": [i for i in range(1)],
        }
    },
]

#train_args=bi_train_for_qwen


