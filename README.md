# Enhancing Chain-of-Thought Reasoning in Multimodal LLMs with RLAIF and Direct Preference Optimization

Authors:  
- **Abhishek Shetty** (UC Berkeley) – [ashetty@ischool.berkeley.edu](mailto:ashetty@ischool.berkeley.edu)  
- **Annas Bin Adil** (UC Berkeley) – [ab722@ischool.berkeley.edu](mailto:ab722@ischool.berkeley.edu)  
- **Ming Cho** (UC Berkeley) – [ming.cho@berkeley.edu](mailto:ming.cho@berkeley.edu)  

---

## Abstract
Advances in Large Language Models (LLMs) have shown effectiveness of 
Chain-of-Thought (CoT) reasoning; yet, there has been little progress 
in enhancing multimodal reasoning capabilities. This paper investigates 
a resource-efficient approach to improving multimodal reasoning through 
Reinforcement Learning with AI Feedback (RLAIF) and Direct Preference 
Optimization (DPO). Using LLaVA-1.5-7B as our seed model, we generate 
and evaluate reasoning chains through an LLM judge. Evaluating our 
approach on the ScienceQA benchmark, we find: (1) zero-shot prompting 
outperforms CoT prompting across temperature settings, with our DPO tuned 
model achieving a negligible improvement in zero-shot accuracy (2) current 
DPO implementations have limitations for multimodal LLMs, as evidenced by 
our finetuned model being unable to better understand images. 

---

## Contributions
Our key contributions are:  
- **Reinforcement Learning with AI Feedback (RLAIF):** A highly scalable training methodology designed to enhance chain-of-thought reasoning in multimodal large language models (LLMs).  
- **Low-Rank Adaptation (LoRA) with Direct Preference Optimization (DPO):** A computationally efficient fine-tuning strategy that significantly reduces training overhead while improving reasoning capabilities.  
- **Multimodal Chain-of-Thought Preference Dataset:** The first dataset of its kind, now publicly available on HuggingFace to support global research efforts.  
- **DPO Shortcomings:** We found that the DPO Trainer from HuggingFace’s Transformer Reinforcement Learning library does not yield improvements for multimodal LLMs, highlighting the need for tailored optimization strategies.
---

## Background
We build on prior work:  
- **Chain-of-Thought Reasoning:** Wu et al. (2024) introduced CoT methods for text-only models.  
- **Multimodal Chain-of-Thought:** Zhang et al. (2023) implemented a two-stage reasoning process but required massive computational resources.  
- **Preference Optimization:** Wang et al. (2024) identified challenges with DPO methods in multimodal contexts, which we address using recent techniques from HuggingFace’s implementation.  

---

## Methods
We fine-tune the **LLaVA-1.5-7B** model using **DPO**, optimizing the gap between chosen and rejected responses based on reasoning quality. Our training pipeline includes:  
- **Parameter-Efficient Fine-Tuning:** Using LoRA with rank-64 targeting linear layers for scalability.  
- **Hyperparameter Tuning:** Optimization of learning rate, dropout, and LoRA parameters to balance performance and computational cost.  
- **Evaluation Metrics:** Benchmarked on the ScienceQA dataset across natural, social, and language sciences.


---

## Multimodal Chain-of-Thought Preference Dataset  

We have uploaded a multimodal chain-of-thought preference dataset to HuggingFace, available at:  
[**llava-chains-10k-individual**](https://huggingface.co/datasets/abshetty/llava-chains-10k-individual).  

This dataset contains nearly 10,000 individual examples for fine-tuning multimodal models on chain-of-thought reasoning tasks. The dataset is stored in Parquet format for efficient access and can be easily integrated into your model training pipeline. Researchers are encouraged to use this resource to experiment with multimodal chain-of-thought reasoning and explore the potential for improving multimodal LLM performance.

---

## Results
Our experiments demonstrate negligible improvements in reasoning tasks using the ScienceQA dataset:  

| **Temperature** | **Prompt**       | **Avg** | **NAT** | **SOC** | **LAN** |  
|------------------|------------------|---------|---------|---------|---------|  
| 0.4             | Zero-shot (Base) | 59.30   | 55.25   | 65.31   | 65.91   |  
| 0.4             | CoT (Base)       | 55.18   | 53.27   | 57.59   | 65.91   |  
| 0.4             | Zero-shot (DPO)  | 59.84   | 57.32   | 63.35   | 68.18   |  
| 0.4             | CoT (DPO)        | 54.64   | 52.36   | 57.98   | 59.96   |  
| 0.8             | Zero-shot (Base) | 56.67   | 54.76   | 59.55   | 59.09   |  
| 0.8             | CoT (Base)       | 45.27   | 44.91   | 45.42   | 52.27   |  
| 0.8             | Zero-shot (DPO)  | 56.92   | 54.01   | 61.13   | 63.64   |  
| 0.8             | CoT (DPO)        | 46.41   | 45.49   | 47.91   | 45.45   |  


Results show that fine-tuning via currently available DPO methods
do not improve multimodal reasoning.

---

### Repository Structure

This repository contains the following key folders, each serving a specific purpose within the project:  

- **`benchmarks/`**  
  Contains evaluation code and results for benchmarking models on 
  ScienceQA. These benchmarks help assess the performance of 
  our fine-tuned model.

- **`CoT/`**  
  Includes code and resources for generating Chain-of-Thought (CoT) 
  responses from Llava-1.5-7b-hf.

- **`DPO-train/`**  
  Holds code and configurations for Direct Preference Optimization 
  (DPO) training. This includes dataset generation from raw data,
  training pipelines and hyperparameters.

- **`model_demos/`**  
  Contains demo scripts and notebooks showcasing how to use Llava-1.5-7b-hf,
  Llama-3.2-11b-Vision and Phi-3-vision-128k-instruct.
   This folder is ideal for users who want to quickly test or visualize 
   outputs for these base models.

- **`RLAIF-scoring/`**  
  Includes code for scoring and evaluating chain-of-thought responses to 
  perform Reinforcement Learning with AI Feedback (RLAIF).

Each folder is organized to streamline experimentation, training, and evaluation processes for researchers and developers.

## How to Run the Code
### Installation
1. Clone this repository:  
   ```bash
   git clone https://github.com/ab-shetty/mm-cot-rlaif.git
   cd mm-cot-rlaif
   ```
2. Use Google Colab:  
   Transfer the notebooks to Google Colab, where environments and
   dependencies are handled via the built-in environment and `pip install`
   code blocks in the notebooks.

### Training  
To train a model using LoRA DPO:  
1. Open `DPO-train/LoRA_DPO.ipynb` in Google Colab.  
2. Follow the notebook’s step-by-step instructions to preprocess data and fine-tune the model using Direct Preference Optimization.  

### Evaluation  
To evaluate a model on ScienceQA:  
1. Open `benchmarks/benchmark.ipynb` in Google Colab.  
2. Load your trained model checkpoint as specified in the notebook and execute the evaluation cells to generate performance metrics.  

---

## Discussion  
Our resource-efficient approach to enhancing multimodal reasoning showed minimal improvements. Current DPO methods, like those from HuggingFace, are not effective for multimodal models, especially in handling image content. Researchers should avoid using these methods for fine-tuning multimodal LLMs.


---

## Future Work 
Future research should explore alternative fine-tuning methods (e.g., mDPO) and develop DPO methods suited for multimodal models. Larger, more diverse datasets and models like Llama-3.2-90b-Vision should also be considered for further improvements.

---

## Citation
If you use this code, please cite:  
```
@article{yourpaper2024,
  title={Enhancing Multimodal Reasoning in LLMs with RLAIF and Direct Preference Optimization},
  author={Shetty, Abhishek and Bin Adil, Annas and Cho, Ming},
  journal={GitHub Repository},
  year={2024}
}
```
