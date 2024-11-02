from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt
import torch

# Define the model identifiers for the two different Stable Diffusion models
model_id1 = "dreamlike-art/dreamlike-diffusion-1.0"  # A model focused on artistic styles
model_id2 = "stabilityai/stable-diffusion-xl-base-1.0"  # A base model for general-purpose image generation

# Initialize the Stable Diffusion pipeline with the first model
pipe = StableDiffusionPipeline.from_pretrained(
    model_id1,                # The identifier of the model to load
    torch_dtype=torch.float16,  # Use 16-bit floating point for efficient computation
    use_safetensors=True       # Utilize SafeTensors format for secure model loading
)

# Move the pipeline to the GPU for faster processing
pipe = pipe.to("cuda")
