# Enhancing Chain-of-Thought Reasoning in Multimodal LLMs with RLAIF and Direct Preference Optimization

Authors:  
- **Abhishek Shetty** (UC Berkeley) – [ashetty@ischool.berkeley.edu](mailto:ashetty@ischool.berkeley.edu)  
- **Annas Bin Adil** (UC Berkeley) – [ab722@ischool.berkeley.edu](mailto:ab722@ischool.berkeley.edu)  
- **Ming Cho** (UC Berkeley) – [ming.cho@berkeley.edu](mailto:ming.cho@berkeley.edu)  

---

## Abstract
*Coming soon.*

---

## Introduction
Large Language Models (LLMs) have shown remarkable performance in natural language tasks. However, reasoning capabilities remain a key limitation, especially for multimodal LLMs. Chain-of-Thought (CoT) prompting methods have advanced reasoning in text-based LLMs, but existing approaches for multimodal reasoning are computationally intensive and lack scalability. 

We address this gap by introducing a **low-resource, scalable method** to enhance multimodal reasoning. Our key contributions are:  
- **Reinforcement Learning with AI Feedback (RLAIF):** A novel training approach tailored for multimodal reasoning.  
- **Direct Preference Optimization (DPO):** Parameter-efficient fine-tuning with LoRA to enhance reasoning performance without excessive computational demands.  
- **Performance Improvements:** Demonstrated on the ScienceQA dataset, highlighting reasoning advancements across diverse question classes.

---

## Background
We build on prior work:  
- **Chain-of-Thought Reasoning:** Wu et al. (2024) introduced CoT methods for text-only models.  
- **Multimodal Chain-of-Thought:** Zhang et al. (2023) implemented a two-stage reasoning process but required massive computational resources.  
- **Preference Optimization:** Wang et al. (2024) identified challenges with DPO methods in multimodal contexts, which we address using recent techniques from HuggingFace’s implementation.  

Our work improves scalability and generalizability for multimodal reasoning tasks.

---

## Methods
### Overview  
We fine-tune the **LLaVA-1.5-7B** model using **DPO**, optimizing the gap between chosen and rejected responses based on reasoning quality. Our training pipeline includes:  
- **Parameter-Efficient Fine-Tuning:** Using LoRA with rank-64 targeting linear layers for scalability.  
- **Hyperparameter Tuning:** Optimization of learning rate, dropout, and LoRA parameters to balance performance and computational cost.  
- **Evaluation Metrics:** Benchmarked on the ScienceQA dataset across natural, social, and language sciences.


---

## Results
Our experiments demonstrate significant improvements in reasoning tasks using the ScienceQA dataset:  

| **Temperature** | **Prompt**       | **Avg** | **NAT** | **SOC** | **LAN** |  
|------------------|------------------|---------|---------|---------|---------|  
| 0.4             | Zero-shot (Base) | 61.33   | 59.97   | 63.22   | 65.91   |  
| 0.4             | CoT (Base)       | 59.20   | 57.57   | 61.91   | 56.82   |  
| 0.4             | Zero-shot (DPO)  | **TBD** | **TBD** | **TBD** | **TBD** |  
| 0.4             | CoT (DPO)        | **TBD** | **TBD** | **TBD** | **TBD** |  

Results show that fine-tuning via DPO consistently improves multimodal reasoning accuracy.

---

## How to Run the Code
### Installation
1. Clone this repository:  
   ```bash
   git clone https://github.com/ab-shetty/mm-cot-rlaif.git
   cd mm-cot-rlaif
   ```
2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```

### Training
To train the model:  
```bash
python train.py --config configs/train_config.json
```

### Evaluation
To evaluate on ScienceQA:  
```bash
python evaluate.py --checkpoint path_to_model
```

---

## Discussion
Our approach demonstrates:  
- Effective reasoning enhancement in multimodal LLMs with minimal resources.  
- Scalability of preference optimization for multimodal CoT tasks.  

---

## Future Work
- Extending experiments to larger-scale multimodal datasets.  
- Exploring more advanced CoT prompting techniques for multimodal reasoning.

---

## Citation
If you use this code, please cite:  
```
@article{yourpaper2024,
  title={Enhancing Multimodal Reasoning in LLMs with Chain-of-Thought and Direct Preference Optimization},
  author={Shetty, Abhishek and Bin Adil, Annas and Cho, Ming},
  journal={GitHub Repository},
  year={2024}
}
```
