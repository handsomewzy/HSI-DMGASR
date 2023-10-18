import numpy as np
import requests
from PIL import Image
from io import BytesIO
from diffusers import StableDiffusionUpscalePipeline
import torch

# load model and scheduler
model_id = "stabilityai/stable-diffusion-x4-upscaler"
pipeline = StableDiffusionUpscalePipeline.from_pretrained(model_id, revision="fp16", torch_dtype=torch.float16,
                                                          use_auth_token="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")
pipeline = pipeline.to("cuda")

# let's download an  image
# url = "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd2-upscale/low_res_cat.png"
# response = requests.get(url)
# low_res_img = Image.open(BytesIO(response.content)).convert("RGB")
low_res_img = Image.open("scene.gif").convert("RGB")
low_res_img = low_res_img.resize((64, 64))

# 将PIL格式转换为numpy，并且把channel换到第一个位置
imgArray = np.array(low_res_img)
imgArray = imgArray.transpose(2, 0, 1)

# 将numpy转换为tensor，并对应的增加一个维度
imgTensor = torch.tensor(imgArray)
imgTensor = imgTensor.unsqueeze(0)
print(imgTensor.shape)
low_res_img.save("lower.png")

prompt = "the mountain and the sun"

upscaled = pipeline(prompt=prompt, image=imgTensor)
print(upscaled)
upscaled_image = pipeline(prompt=prompt, image=low_res_img).images[0]

upscaled.images[0].save("upsampled1.png")
upscaled_image.save("upsampled.png")
