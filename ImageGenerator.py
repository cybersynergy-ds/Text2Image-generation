from txt2imgMODEL import pipe
import matplotlib.pyplot as plt
import torch
# Define the prompt
prompt = """
cute indian boy and golder retriever dog playing along a river
"""
# Generate the image
image = pipe(prompt).images[0]
# Print the prompt
print("[PROMPT]: ", prompt)
# Display the image using Matplotlib
plt.imshow(image)
plt.axis('off')
plt.show()
