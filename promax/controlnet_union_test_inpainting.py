# diffusers测试ControlNet
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
sys.path.append('..')
import cv2
import copy
import torch
import random
import numpy as np
from PIL import Image
from mask import get_mask_generator
from diffusers.utils import load_image
from diffusers import EulerAncestralDiscreteScheduler, AutoencoderKL,DPMSolverMultistepScheduler
from models.controlnet_union import ControlNetModel_Union
from pipeline.pipeline_controlnet_union_inpaint_sd_xl import StableDiffusionXLControlNetUnionInpaintPipeline
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from util.PatcherUtils import modelUtils
from util.PatcherUtils import cv2_to_pil

modelutil = modelUtils()
device=torch.device('cuda:0')

eulera_scheduler = DPMSolverMultistepScheduler.from_pretrained("C:/NVME_SYNC/Projects/MEED/PanoramaFiller/models/SDModels/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b", subfolder="scheduler")
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
# Note you should set the model and the config to the promax version manually, default is not the promax version. 
from huggingface_hub import snapshot_download
#snapshot_download(repo_id="xinsir/controlnet-union-sdxl-1.0", local_dir='controlnet-union-sdxl-1.0')
# you should make a new dir controlnet-union-sdxl-1.0-promax and mv the promax config and promax model into it and rename the promax config and the promax model.
controlnet_model = ControlNetModel_Union.from_pretrained("C:/NVME_SYNC/Projects/MEED/PanoramaFiller/models/ControlnetModels/models--xinsir--controlnet-union-sdxl-1.0/snapshots/60ccbb0afea4fb991abea616e2b2ee455869b84b/promax", torch_dtype=torch.float16, use_safetensors=True)


pipe = StableDiffusionXLControlNetUnionInpaintPipeline.from_pretrained(
    "C:/NVME_SYNC/Projects/MEED/PanoramaFiller/models/SDModels/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b", controlnet=controlnet_model, 
    vae=vae,
    torch_dtype=torch.float16,
    # scheduler=ddim_scheduler,
    scheduler=eulera_scheduler,
    use_safetensors=True,
    variant="fp16",
)


pipe = pipe.to(device)

def HWC3(x):
    print(f"Input dtype: {x.dtype}, shape: {x.shape}")
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    print(f"Input shape: {x.shape}")
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        print(f"Color shape: {color.shape}, Alpha shape: {alpha.shape}")
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        print(f"Output dtype: {y.dtype}, shape: {y.shape}")
        return y
mask_gen_kwargs = {
            "irregular_proba": 1,
            "irregular_kwargs": {
                "max_angle": 4,
                "max_len": 200 * 4,
                "max_width": 100 * 4,
                "max_times": 1,
                "min_times": 1
            },
            "box_proba": 1,
            "box_kwargs": {
                "margin": 10,
                "bbox_min_size": 30 * 4,
                "bbox_max_size": 150 * 4,
                "max_times": 1,
                "min_times": 1
            },
        }

mask_gen = get_mask_generator(kind='mixed', kwargs=mask_gen_kwargs)


prompt = "A finnish hockey player mural painted with spray paint"
#prompt = "A highly detailed, photorealistic view of the Sahara Desert seen through a large, destroyed window in an abandoned, graffiti-covered industrial warehouse. The desert landscape should feature expansive sand dunes under a clear sky. Add stark sunlight casting shadows on the dunes. The overall atmosphere should be gritty, with the contrast of the harsh outdoor environment against the decayed, urban ruin of the building interior. Tags: photograph+, highly realistic++, grimy, urban decay, desert landscape+, hdr, 4k"
##prompt = "A highly detailed, photorealistic view of the Sahara Desert seen through a large, destroyed window "
#prompt = "A rough grimy exposed view of the sahara desert filled with sand dunes and an oasis in the distance revealed trough a mystical protal visible in the red curtain, 4k, high quality, sharp details"
#prompt = "A wall of paintings featuing abstract versions of clowsn in different artstyles"
negative_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'


seed = random.randint(0, 2147483647)
seed = 12434554223
# The original image you want to repaint.
original_img = cv2.imread("E:/Datasets/Research_Datasets/Test_Datasets/imageMaskPair/1024/City4.png", cv2.IMREAD_UNCHANGED)
original_mask = cv2.imread("E:/Datasets/Research_Datasets/Test_Datasets/imageMaskPair/1024/mask-Buffer2.png", cv2.IMREAD_UNCHANGED)


test = False
# # inpainting support any mask shape
# # where you want to repaint, the mask image should be a binary image, with value 0 or 255.
mask_img = cv2.imread("C:/Users/deata/Downloads/env360-cropped-mask copy.png") 

height, width, _  = original_img.shape
ratio = np.sqrt(1024. * 1024. / (width * height))
W, H = int(width * ratio) // 8 * 8, int(height * ratio) // 8 * 8
if test:

    original_img = cv2.resize(original_img, (W, H))
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

    import copy
    controlnet_img = copy.deepcopy(original_img)
    controlnet_img = np.transpose(controlnet_img, (2, 0, 1))
    mask_img = np.transpose(mask_img, (2, 0, 1))
    mask = mask_gen(controlnet_img)
    #_, mask_img = cv2.threshold(mask_img, 127, 255, cv2.THRESH_BINARY)
    controlnet_img = np.transpose(controlnet_img, (1, 2, 0))
    mask_img = np.transpose(mask_img, (1, 2, 0))
    mask = np.transpose(mask, (1, 2, 0))
    #_, mask = cv2.threshold(mask_img, 127, 255, cv2.THRESH_BINARY)
    print("controlnet_img dimensions:", controlnet_img.shape)
    print("mask_img dimensions:", mask_img.shape)
    controlnet_img[mask_img.squeeze() > 0.0] = 0
    mask2 = Image.fromarray(mask_img)
    mask = HWC3(mask_img)

    controlnet_img = Image.fromarray(controlnet_img)
    
    mask = Image.fromarray(mask)
    mask_img = cv2.imread("C:/Users/deata/Downloads/env360-cropped-mask copy.png") 
    original_img = Image.fromarray(original_img)
    

controlnet_img,mask = modelutil.generate_mask(original_mask)
original_img = cv2_to_pil(original_img)

mask.save("mask.png")
original_img.save("original_img.webp")
controlnet_img.save("control_inpainting.webp")

width, height = W, H

# 0 -- openpose
# 1 -- depth
# 2 -- hed/pidi/scribble/ted
# 3 -- canny/lineart/anime_lineart/mlsd
# 4 -- normal
# 5 -- segment
# 6 -- tile
# 7 -- repaint
print("Peak Memory:", torch.cuda.max_memory_allocated() / 1024 / 1024, "MB")
images = pipe(prompt=[prompt]*1,
            image=original_img,
            mask_image=mask,
            control_image_list=[0, 0, 0, 0, 0, 0, 0, controlnet_img], 
            negative_prompt=[negative_prompt]*1,
            # generator=generator,
            #guidance_scale=8.0,
            num_images_per_prompt=1,
            width=width, 
            height=height,
            num_inference_steps=30,
            union_control=True,
            union_control_type=torch.Tensor([0, 0, 0, 0, 0, 0, 0, 1]),
            ).images


for i, img in enumerate(images):
    img.save(f"./inpainting_{i}.png")
    img.show()
#images[0].save(f"./inpainting.png")

