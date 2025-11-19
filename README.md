<div align="center">

<p align="center">
  <img src="static/images/DiRL.jpg" alt="DiRL" width="300">
</p>

<!-- <h1>DiRL</h1> -->

<h2>An Efficient Training Framework for Diffusion Language Models</h2>

<p>
<b>Ying Zhu</b><sup>1,2,3</sup>, <b>Jiaxin Wan</b><sup>2</sup>, <b>Tianyi Liang</b><sup>2,3</sup>, <b>Xu Guo</b><sup>1,2</sup>, <b>Xiaoran Liu</b><sup>1,2,3</sup>,<br>
<b>Zengfeng Huang</b><sup>1,2</sup>, <b>Ziwei He</b><sup>2,3,†</sup>, <b>Xipeng Qiu</b><sup>1,2,3,†</sup>
</p>

<p>
<sup>1</sup>Fudan University &nbsp;&nbsp; <sup>2</sup>Shanghai Innovation Institute &nbsp;&nbsp; <sup>3</sup>OpenMoss Team
</p>

<p>
<sup>†</sup>Corresponding authors
</p>

</div>

<p align="center">
  <!-- <a href="https://arxiv.org/abs/YOUR_PAPER_ID">
    <img src="https://img.shields.io/badge/arXiv-2501.XXXXX-b31b1b.svg" alt="Paper on arXiv"/>
  </a>
  <a href="https://arxiv.org/pdf/YOUR_PAPER_ID.pdf">
    <img src="https://img.shields.io/badge/Paper-PDF-red.svg" alt="PDF"/>
  </a> -->
  <a href="https://github.com/OpenMOSS/DiRL">
    <img src="https://img.shields.io/badge/GitHub-Code-black.svg?logo=github" alt="GitHub Code"/>
  </a>
  <a href="https://huggingface.co/OpenMOSS-Team/DiRL-8B-Instruct">
    <img src="https://img.shields.io/badge/🤗%20Hugging%20Face-Model-yellow.svg" alt="Hugging Face Model"/>
  </a>
  <a href="https://huggingface.co/collections/Auraithm/dirl">
    <img src="https://img.shields.io/badge/🤗%20Hugging%20Face-Data-yellow.svg" alt="Hugging Face Data"/>
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"/>
  </a>
</p>

<p align="center">
  <img src="static/images/accuracy.png" alt="Overview" width="750">
</p>

---

## 🌟 TL;DR

We introduce **DiRL**, an open-source training framework for Diffusion Language Models (DLLMs) with SFT and RL stages. Using this framework, we train **DiRL-8B-Instruct**, achieving state-of-the-art results at the 8B scale on mathematical reasoning benchmarks, even outperforming 32B models on most tasks.

## 🌱 HighLights

- **🎯 Novel RL Algorithm:** We propose **DiPO (Discrete Diffusion Policy Optimization)**, an RL algorithm that optimizes at the generation step level for DLLMs. It achieves unbiased implementation with complete consistency between optimization objectives and training process, and integrates dynamic sampling from DAPO during rollout to filter out low-quality data.

- **🚀 Efficient Training & Inference:** We support **Accelerate** framework for distributed training and **LMDeploy** inference engine for efficient rollout, while integrate **Speed Reward** mechanism to optimize inference speed at the training level, enabling both faster training and generation without sacrificing quality.

- **🧠 SOTA Performance:** We achieve state-of-the-art results at the 8B scale among both autoregressive (AR) models and diffusion language models (DLLMs) across multiple mathematical reasoning benchmarks. Specifically, we reach **83.05%** on MATH500, **20.63%** on AIME2024, and **20.83%** on AIME2025, surpassing all 8B baselines and even outperforming the 32B Qwen2.5-32B-Instruct model on AIME benchmarks.

## 🧠 Method

We develop and release an open-source diffusion post-training framework for DLLMs, and train **DiRL-8B-Instruct** based on **SDAR-8B-Chat** through two stages: Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL). In the SFT stage, we adopt a random-masking strategy to construct the training data for model fine-tuning. In the RL stage, we design an RL algorithm -- **DiPO (Discrete Diffusion Policy Optimization)**, which optimizes at the generation step level. We achieve an unbiased implementation of RL theory, ensuring complete consistency between the optimization objective and the actual training process. Additionally, during the rollout phase, we adopt dynamic sampling from DAPO to filter out data with zero advantage standard deviation. Through this two-stage training pipeline, we successfully train **DiRL-8B-Instruct**, a high-performance diffusion language model for mathematical reasoning.

## 📊 Performance

**DiRL-8B-Instruct** achieves state-of-the-art results among DLLMs across mathematical reasoning benchmarks. Highlights include **83.05%** on MATH500 (surpassing the base model by **+11.20%**), **20.63%** on AIME2024 and **20.83%** on AIME2025 (dramatically outperforming all baselines), and **46.40%** on OlympiadBench. Our 8B model achieves performance comparable to or exceeding much larger 32B models on most benchmarks.

<p align="center">
  <img src="static/images/performance.jpg" alt="Performance Comparison" width="750">
</p>

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/OpenMOSS/DiRL.git
cd DiRL
pip install -r requirements.txt
```

If `flash-attn` installation fails, you can download the pre-built wheel file and install it manually:

```bash
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
pip install flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
```

### Download Models and Datasets

Edit `download.sh` to set your Hugging Face token and username, then run:

```bash
bash download.sh
```

### Inference

```python
from lmdeploy import pipeline, PytorchEngineConfig, GenerationConfig
from transformers import AutoTokenizer

model_path = "OpenMOSS-Team/DiRL-8B-Instruct"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Prepare prompts
prompts = [
    [{"role": "user", "content": "Solve: If x + 5 = 12, what is x?"}],
]
prompts = tokenizer.apply_chat_template(prompts, tokenize=False, add_generation_prompt=True)

# Configure backend for DLLM inference
backend_config = PytorchEngineConfig(
    dtype="float16",
    max_prefill_token_num=8192,
    cache_max_entry_count=0.8,
    dllm_block_length=4,
    dllm_denoising_steps=4,
    dllm_unmasking_strategy="low_confidence_dynamic",
    dllm_confidence_threshold=0.9,
)

# Create inference pipeline
with pipeline(model_path, backend_config=backend_config) as pipe:
    gen_config = GenerationConfig(
        top_p=1.0,
        top_k=50,
        temperature=1.0,
        do_sample=False,  # greedy decoding
        max_new_tokens=8192,
    )
    
    outputs = pipe(prompts, gen_config=gen_config)
    
    for output in outputs:
        print(output.text)
```

### Evaluation

To evaluate models on multiple benchmarks (MATH500, GSM8K, AIME2024, AIME2025, OlympiadBench):

```bash
bash examples/eval.sh
```

### Training

**Step 1: Prepare Training Data**

While the full DiRL-8B-Instruct training data is not yet released, we provide lightweight datasets for quick experimentation:
- [Light-OpenR1Math-SFT](https://huggingface.co/datasets/OpenMOSS-Team/Light-OpenR1Math-SFT): 2K SFT samples from OpenR1Math
- [Light-MATH-RL](https://huggingface.co/datasets/OpenMOSS-Team/Light-MATH-RL): 4K RL samples from MATH

> **Tip:** For initial experimentation, we recommend starting with **max_new_tokens** of 2K to reduce training time and resource requirements.

You can also create your own training datasets following the formats below:

SFT training data format:
```json
[
  {
    "prompt": "<|im_start|>user\n[question]<|im_end|>\n<|im_start|>assistant\n",
    "response": "[answer]<|im_end|><|endoftext|>"
  }
]
```

RL training data format:
```json
[
  {
    "question": "[question]",
    "ground_truth_answer": "[answer]"
  }
]
```

**Step 2: Two-Stage Training**

**Stage 1: SFT Training**

Supervised fine-tuning with random-masking strategy to adapt the base model for mathematical reasoning tasks.

```bash
bash examples/sft.sh
```

**Stage 2: RL Training**

Reinforcement learning with DiPO algorithm to optimize the model at generation step level.

```bash
bash examples/rl.sh
```

## 📋 Roadmap

- [x] Release Inference Engine and Training Framework
- [ ] Release DiRL Technical Report
- [ ] Release Training Data of DiRL-8B-Instruct
- [ ] Release Thinking Model
- [ ] Support More RL Algorithms
- [ ] More Features are working in progress


## 👏 Acknowledgement

We would like to express our gratitude to the following works ([SDAR](https://github.com/JetAstra/SDAR), [dllm-RL](https://github.com/Gen-Verse/dLLM-RL), [lmdeploy](https://github.com/InternLM/lmdeploy)) for providing important theoretical foundations and inspiration for DiRL.

## 💬 Community

Join our WeChat group to discuss DLLM training and related topics:

<p align="center">
  <img src="static/images/qr_code.jpg" alt="WeChat QR Code" width="400">
</p>


## 📧 Contact

For issues or inquiries:

- **Ying Zhu**, Shanghai Innovation Institute ([auraithm@gmail.com](mailto:auraithm@gmail.com))


## 📖 Citation

If you find our work helpful, please consider citing:

```bibtex
@misc{zhu2025dirl,
  title={DiRL: An Efficient Training Framework for Diffusion Language Models},
  author={Zhu, Ying and Wan, Jiaxin and Liang, Tianyi and Guo, Xu and Liu, Xiaoran and Huang, Zengfeng and He, Ziwei and Qiu, Xipeng},
  year={2025},
  institution={Fudan University, Shanghai Innovation Institute},
  url={https://github.com/OpenMOSS/DiRL}
}
```
