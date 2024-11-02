import matplotlib.pyplot as plt
from txt2imgMODEL import pipe


def generate_image(pipe, prompt, params):
    img = pipe(prompt, **params).images

    num_images = len(img)

    # Create subplots
    if num_images > 1:
        fig, ax = plt.subplots(nrows=1, ncols=num_images, figsize=(num_images * 5, 5))  # Adjust size as needed
        for i in range(num_images):
            ax[i].imshow(img[i])
            ax[i].axis('off')
    else:
        fig = plt.figure(figsize=(5, 5))  # Adjust size as needed
        plt.imshow(img[0])
        plt.axis('off')

    plt.tight_layout()
    plt.show()  # Display the images


prompt = """
A dreamlike, cute CAT playing the festival of colos, drapped in 
traditional Indian attire, throwing colors """

#params = {'num_inference_steps ':100}
#params = {'num_inference_steps ':100, 'width':512, 'height': int(1.5*640)}
params = {'num_inference_steps ':100, 'num_images_per_prompt':2}

generate_image(pipe, prompt, params)
