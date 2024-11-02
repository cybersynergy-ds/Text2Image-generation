from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt
import torch

model_id1="dreamlike-art/dreamlike-diffusion-1.0"
model_id2="stabilityai/stable-diffusion-xl-base-1.0"

pipe= StableDiffusionPipeline.from_pretrained(
    model_id1, torch_dtype=torch.float16,use_safetensors=True
)
pipe=pipe.to("cuda")

