# What Limits Bidirectional Model's Generative Capabilities? A Uni-Bi-Directional Mixture-of-Expert Method For Bidirectional Fine-tuning

We demonstrate that bidirectional training leads to an increase in subsequent dependency, and propose  A Uni-Bi-Directional Mixture-of-Expert Large Language Model.

[paper link](https://icml.cc/virtual/2025/poster/44255)

![img.png](image/dp.png)

# Get Started
## 🛠️ Preparation
```
pip3 install torch torchvision torchaudio

git clone https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct
git clone https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct
git clone https://huggingface.co/Qwen/Qwen2.5-3B-Instruct
git clone https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
git clone https://huggingface.co/Qwen/Qwen2.5-14B-Instruct
```
## Run 

### Calculate DP
```
python caculate_dp.py
```

### Train Embedding Model
```
python train_bi.py
```

### Train UBMOE
```
python train_ubmoe.py
```

# Citing 
```

```