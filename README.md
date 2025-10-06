# Breaking the Single-View: Cross-Modal Graph Recasting for Vision-Augmented Recommendation
## Framework
<img width="2856" height="1052" alt="edee8d5f-e51d-42fa-ac9c-669fab0b8291" src="https://github.com/user-attachments/assets/a7033603-83e6-4ed9-a9a7-0817add1abfc" />

## Dependencies

To ensure reproducibility, we maintain **two separate virtual environments**, each corresponding to a different multimodal LLM backbone.

### 🧩 Environment 1 — Qwen2.5-VL-7B-Instruct

python==3.10  
torch==2.5.1  
torchvision==0.20.1  
torchaudio==2.5.1  
transformers==4.50.0.dev0  
accelerate==1.9.0  
bitsandbytes==0.45.2  
deepspeed==0.16.2  

### 🧠 Environment 2 — GLM-4.1V-9B-Thinking

python==3.10  
torch==2.5.1  
torchvision==0.20.1  
torchaudio==2.5.1  
transformers==4.44.2  
numpy==2.2.2  
pandas==2.2.3 


## Step  
### Stage 1: Train MITG (Multimodal Interest Transition Graph)


