
from diffusers import StableDiffusionPipeline

def generate_image(prompt, model_name, device):
    pipeline = StableDiffusionPipeline.from_pretrained(model_name)
    pipeline = pipeline.to(device) 
    image = pipeline(prompt=prompt).images[0]

    # Save or display the image
    image.save("generated_image.png")
    image.show()

    # return the image
    return image



   

