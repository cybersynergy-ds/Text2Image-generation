# Import the necessary libraries
from txt2imgMODEL import pipe  # Import the image generation pipeline from a custom module
import matplotlib.pyplot as plt  # Import Matplotlib for displaying images
import torch  # Import PyTorch for tensor operations

# Define the prompt for image generation
prompt = """
cute indian boy and golden retriever dog playing along a river
"""  # A descriptive string that specifies what the image should depict

# Generate the image based on the prompt using the text-to-image pipeline
image = pipe(prompt).images[0]  # Call the pipeline with the prompt and get the first generated image

# Print the prompt to the console for reference
print("[PROMPT]: ", prompt)  # This helps the user see the text that led to the image generation

# Display the generated image using Matplotlib
plt.imshow(image)  # Render the image in the plot
plt.axis('off')  # Hide the axes for a cleaner visual presentation
plt.show()  # Show the generated image in a window
