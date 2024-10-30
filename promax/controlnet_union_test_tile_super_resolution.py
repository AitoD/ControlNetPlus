# diffusers测试ControlNet
from datetime import datetime
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
sys.path.append('..')
import cv2
from guided_filter import FastGuidedFilter
import time
import torch
import random
import numpy as np
from PIL import Image, ImageDraw
from diffusers.utils import load_image
from diffusers import EulerAncestralDiscreteScheduler, AutoencoderKL
from models.controlnet_union import ControlNetModel_Union
from pipeline.pipeline_controlnet_union_sd_xl_img2img import StableDiffusionXLControlNetUnionImg2ImgPipeline
from pipeline.pipeline_controlnet_union_sd_xl import StableDiffusionXLControlNetUnionPipeline

device=torch.device('cuda:0')

eulera_scheduler = EulerAncestralDiscreteScheduler.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="scheduler")
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
# Note you should set the model and the config to the promax version manually, default is not the promax version. 
from huggingface_hub import snapshot_download
snapshot_download(repo_id="xinsir/controlnet-union-sdxl-1.0", local_dir='controlnet-union-sdxl-1.0')
# you should make a new dir controlnet-union-sdxl-1.0-promax and mv the promax config and promax model into it and rename the promax config and the promax model.
controlnet_model = ControlNetModel_Union.from_pretrained("C:/NVME_SYNC/Projects/MEED/PanoramaFiller/models/ControlnetModels/models--xinsir--controlnet-union-sdxl-1.0/snapshots/60ccbb0afea4fb991abea616e2b2ee455869b84b/promax", torch_dtype=torch.float16, use_safetensors=True)


pipe = StableDiffusionXLControlNetUnionImg2ImgPipeline.from_pretrained(
    "C:/NVME_SYNC/Projects/MEED/PanoramaFiller/models/SDModels/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b", controlnet=controlnet_model, 
    vae=vae,
    torch_dtype=torch.float16,
    # scheduler=ddim_scheduler,
    scheduler=eulera_scheduler,
    use_safetensors=True,
    variant="fp16",
)

pipe = pipe.to(device)

def apply_gaussian_blur(image_np, ksize=5, sigmaX=1.0):
    if ksize % 2 == 0:
        ksize += 1  # ksize must be odd
    blurred_image = cv2.GaussianBlur(image_np, (ksize, ksize), sigmaX=sigmaX)
    return blurred_image

def apply_guided_filter(image_np, radius, eps, scale):
    filter = FastGuidedFilter(image_np, radius, eps, scale)
    return filter.filter(image_np)


prompt = "Visible ground made up of just broccoli, with clear blue sky."
negative_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'

seed = 2147483647 #random.randint(0, 2147483647)
generator = torch.Generator('cuda').manual_seed(seed)

imgPath = r"C:\Users\deata\Downloads\returned_01_wild_hydrant.png_result_image.png"
base_name = os.path.basename(imgPath)

# Split the base name into filename and extension
file_name, file_extension = os.path.splitext(base_name)
controlnet_img = cv2.imread(imgPath)
height, width, _  = controlnet_img.shape
ratio = np.sqrt(1024. * 1024. / (width * height))
# 3 * 3 upscale correspond to 16 * 3 multiply, 2 * 2 correspond to 16 * 2 multiply and so on.
W, H = int(width * ratio) // 48 * 48, int(height * ratio) // 48 * 48
controlnet_img = cv2.resize(controlnet_img, (W, H))
#controlnet_img = apply_gaussian_blur(controlnet_img, ksize=int(1), sigmaX=1 / 2)   
controlnet_img = cv2.cvtColor(controlnet_img, cv2.COLOR_BGR2RGB)
controlnet_img = Image.fromarray(controlnet_img)


overlap = 16  # Adjust overlap as needed for seamless stitching

# Calculate target tile dimensions including overlap
target_width = (W // 3) + overlap * 2
target_height = (H // 3) + overlap * 2
tile_width = W // 3
tile_height = H // 3

# Store tiles and their overlap information
tiles = []
crops_coords_list = []

# Loop through the 3x3 grid to crop and expand each tile
for i in range(3):
    for j in range(3):
        # Initial crop boundaries without overlap
        left = j * tile_width
        top = i * tile_height
        right = left + tile_width
        bottom = top + tile_height

        # Crop the base tile without overlap
        cropped_image = controlnet_img.crop((left, top, right, bottom))

        # Expand boundaries to add overlap, depending on tile position
        expand_left = overlap if j > 0 else 0
        expand_top = overlap if i > 0 else 0
        expand_right = overlap if j < 2 else 0
        expand_bottom = overlap if i < 2 else 0

        # Adjust the cropping coordinates for overlap
        left = max(left - expand_left, 0)
        top = max(top - expand_top, 0)
        right = min(right + expand_right, W)
        bottom = min(bottom + expand_bottom, H)

        # Re-crop the image with expanded boundaries to include overlap
        expanded_image = controlnet_img.crop((left, top, right, bottom))

        # Calculate scaling factor to resize to (W, H)
        expanded_width, expanded_height = expanded_image.size
        scale_x = W / expanded_width
        scale_y = H / expanded_height

        # Resize to model-required dimensions
        resized_image = expanded_image.resize((W, H))

        # Adjust overlaps to match resized dimensions
        left_overlap = int(expand_left * scale_x)
        top_overlap = int(expand_top * scale_y)
        right_overlap = int(expand_right * scale_x)
        bottom_overlap = int(expand_bottom * scale_y)

        # Store the resized image and overlap info
        tiles.append({
            'image': resized_image,
            'left_overlap': left_overlap,
            'top_overlap': top_overlap,
            'right_overlap': right_overlap,
            'bottom_overlap': bottom_overlap,
        })

        # Store the original crop coordinates for reference
        crops_coords_list.append((left, top))
        
        
base_file_path = r"C:\Users\deata\Downloads"

# 0 -- openpose
# 1 -- depth
# 2 -- hed/pidi/scribble/ted
# 3 -- canny/lineart/anime_lineart/mlsd
# 4 -- normal
# 5 -- segment
# 6 -- tile
# 7 -- repaint
'''
new_width, new_height = W, H
images  = pipe(prompt=[prompt]*1,
                image_list=[0, 0, 0, 0, 0, 0, controlnet_img, 0], 
                negative_prompt=[negative_prompt]*1,
                generator=generator,
                guidance_scale= 7,
                width=new_width, 
                height=new_height,
                num_inference_steps=55,
                union_control=True,
                union_control_type=torch.Tensor([0, 0, 0, 0, 0, 0, 1, 0]),
            ).images

images[0].save(base_file_path + f"\\{base_name}_result_image.png")
'''
result_images = []
for sub_tile, crops_coords in zip(tiles, crops_coords_list):
    sub_img = sub_tile['image']
    print(f"Processing tile at coordinates: {crops_coords} image size: {sub_img.size}")
    new_width, new_height = W, H
    out = pipe(prompt=[prompt]*1,
                image=sub_img, 
                control_image_list=[0, 0, 0, 0, 0, 0, sub_img, 0],
                negative_prompt=[negative_prompt]*1,
                generator=generator,
                guidance_scale= 7,
                width=new_width, 
                height=new_height,
                num_inference_steps=55,
                crops_coords_top_left=crops_coords,
                target_size=(W, H),
                original_size=(W * 2, H * 2),
                union_control=True,
                union_control_type=torch.Tensor([0, 0, 0, 0, 0, 0, 1, 0]),
                strength=1.0,
            )
    result_images.append(out.images[0])
    #out.images[0].save(rf"C:\Users\deata\Downloads\{base_name}_result_image_{len(result_images)}.png")
    
effective_tile_width = W
effective_tile_height = H
final_image_width = W * 3
final_image_height = H * 3
assembled_image = Image.new('RGB', (final_image_width, final_image_height))
# Initial assembly without overlaps (to serve as base layer)
for idx, tile_info in enumerate(tiles):
    row = idx // 3
    col = idx % 3

    # Remove overlap by cropping each tile back to the effective dimensions
    result_img = result_images[idx]
    left_overlap = tile_info['left_overlap']
    top_overlap = tile_info['top_overlap']
    right_overlap = tile_info['right_overlap']
    bottom_overlap = tile_info['bottom_overlap']
    
    cropped_tile = result_img.crop((
        left_overlap,
        top_overlap,
        result_img.width - right_overlap,
        result_img.height - bottom_overlap
    ))
    cropped_tile = cropped_tile.resize((effective_tile_width, effective_tile_height))
    # Calculate position in the final assembled image
    pos_x = col * effective_tile_width
    pos_y = row * effective_tile_height
    assembled_image.paste(cropped_tile, (pos_x, pos_y))



final_image_with_fades = assembled_image.copy()  # Start with the assembled base image

# Original dimensions with overlap
original_tile_width = W // 3 + 2 * overlap
scaling_factor_x = W / original_tile_width
scaling_factor_y = H / original_tile_width

# Loop over each tile to create and apply faded overlaps with consistent scaling
for idx, tile_info in enumerate(tiles):
    row = idx // 3
    col = idx % 3
    if idx > 0:
        break
    result_img = result_images[idx]
    base_pos_x = col * effective_tile_width
    base_pos_y = row * effective_tile_height

    # Left overlap
    if tile_info['left_overlap'] > 0:
        left_overlap_region = result_img.crop((0, 0, tile_info['left_overlap'], result_img.height))
        # Scale uniformly based on width
        left_overlap_region = left_overlap_region.resize((int(tile_info['left_overlap'] * scaling_factor_x), H), Image.LANCZOS)

        # Mask for fading
        mask = Image.new("L", left_overlap_region.size, 255)
        draw = ImageDraw.Draw(mask)
        for x in range(left_overlap_region.width):
            alpha = int(255 * (1 - 0.5 * (x / left_overlap_region.width)))
            draw.line([(x, 0), (x, left_overlap_region.height)], fill=alpha)
        left_overlap_region.putalpha(mask)

        # Position the faded left overlap exactly to the left of the main tile
        final_image_with_fades.paste(left_overlap_region, (base_pos_x - left_overlap_region.width, base_pos_y), mask=left_overlap_region)

    # Right overlap
    if tile_info['right_overlap'] > 0:
        right_overlap_region = result_img.crop((result_img.width - tile_info['right_overlap'], 0, result_img.width, result_img.height))
        right_overlap_region = right_overlap_region.resize((int(tile_info['right_overlap'] * scaling_factor_x), H), Image.LANCZOS)

        # Mask for fading
        mask = Image.new("L", right_overlap_region.size, 255)
        draw = ImageDraw.Draw(mask)
        for x in range(right_overlap_region.width):
            alpha = int(255 * (1 - 0.5 * (x / right_overlap_region.width)))
            draw.line([(x, 0), (x, right_overlap_region.height)], fill=alpha)
        right_overlap_region.putalpha(mask)

        # Position the faded right overlap
        final_image_with_fades.paste(right_overlap_region, (base_pos_x + effective_tile_width, base_pos_y), mask=right_overlap_region)

    # Top overlap
    if tile_info['top_overlap'] > 0:
        top_overlap_region = result_img.crop((0, 0, result_img.width, tile_info['top_overlap']))
        top_overlap_region = top_overlap_region.resize((W, int(tile_info['top_overlap'] * scaling_factor_y)), Image.LANCZOS)

        # Mask for fading
        mask = Image.new("L", top_overlap_region.size, 255)
        draw = ImageDraw.Draw(mask)
        for y in range(top_overlap_region.height):
            alpha = int(255 * (1 - 0.5 * (y / top_overlap_region.height)))
            draw.line([(0, y), (top_overlap_region.width, y)], fill=alpha)
        top_overlap_region.putalpha(mask)

        # Position the faded top overlap
        final_image_with_fades.paste(top_overlap_region, (base_pos_x, base_pos_y - top_overlap_region.height), mask=top_overlap_region)

    # Bottom overlap
    if tile_info['bottom_overlap'] > 0:
        bottom_overlap_region = result_img.crop((0, result_img.height - tile_info['bottom_overlap'], result_img.width, result_img.height))
        bottom_overlap_region = bottom_overlap_region.resize((W, int(tile_info['bottom_overlap'] * scaling_factor_y)), Image.LANCZOS)

        # Mask for fading
        mask = Image.new("L", bottom_overlap_region.size, 255)
        draw = ImageDraw.Draw(mask)
        for y in range(bottom_overlap_region.height):
            alpha = int(255 * (1 - 0.5 * (y / bottom_overlap_region.height)))
            draw.line([(0, y), (bottom_overlap_region.width, y)], fill=alpha)
        bottom_overlap_region.putalpha(mask)

        # Position the faded bottom overlap
        final_image_with_fades.paste(bottom_overlap_region, (base_pos_x, base_pos_y + effective_tile_height), mask=bottom_overlap_region)

    # Save each cropped tile for inspection
    #cropped_tile.save(rf"C:\Users\deata\Downloads\{base_name}_scaled_cropped_tile_{idx}.png")
    
    


filepath = os.path.join(base_file_path, base_name + file_extension)
# Check if the file already exists
if os.path.exists(filepath):
    # Get the file name and extension
    file_name, file_extension = os.path.splitext(filepath)
    
    # Generate a new file name by appending a timestamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    new_file_path = f"{file_name}_{timestamp}{file_extension}"
else:
    # Use the base file path if the file does not exist
    new_file_path = filepath

# Save the image with the new file name
assembled_image.save(new_file_path)

filepath = os.path.join(base_file_path, base_name + "_Crosfade_" + file_extension)
# Check if the file already exists
if os.path.exists(filepath):
    # Get the file name and extension
    file_name, file_extension = os.path.splitext(filepath)
    
    # Generate a new file name by appending a timestamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    new_file_path = f"{file_name}_{timestamp}{file_extension}"
else:
    # Use the base file path if the file does not exist
    new_file_path = filepath
    
final_image_with_fades.save(new_file_path)

# Print the file path to verify
print(f"Image saved to: {new_file_path}")