import os
import numpy as np 
from rich.progress import Console
from torchvision import transforms
from copy import deepcopy
from PIL import Image
from typing_extensions import Literal
from lang_sam import LangSAM
import utils

import torch, random
from diffusers.schedulers import DDIMScheduler, DDIMInverseScheduler
from diffusers.models.attention_processor import AttnProcessor
from diffusers import StableDiffusionControlNetPipeline, StableDiffusionControlNetInpaintPipeline, ControlNetModel, UNet2DConditionModel
from diffusers.schedulers import DDIMScheduler, DDIMInverseScheduler

CONSOLE = Console(width=120)


class GaussCtrlPipeline():
    def __init__(
        self, device,
        image_path = "",
        depth_path = "",
        mask_path = "",
        output_folder = "",
        diffusion_ckpt = 'CompVis/stable-diffusion-v1-4',
        edit_prompt = "",
        reverse_prompt = "",
        ref_view_num = 4,
        num_inference_steps = 20,
        guidance_scale = 5,
        chunk_size = 5,
        test_mode: Literal["test", "val", "inference"] = "val",
    ):
        self.test_mode = test_mode
        self.langsam = LangSAM()
        self.device = device
        self.image_path = image_path
        self.depth_path = depth_path
        self.mask_path = mask_path
        self.output_folder = output_folder
        self.diffusion_ckpt = diffusion_ckpt
        self.ref_view_num = ref_view_num
        self.chunk_size = chunk_size
        
        self.edit_prompt = edit_prompt
        self.reverse_prompt = reverse_prompt
        self.pipe_device = 'cuda:0'
        self.ddim_scheduler = DDIMScheduler.from_pretrained(self.diffusion_ckpt, subfolder="scheduler")
        self.ddim_inverser = DDIMInverseScheduler.from_pretrained(self.diffusion_ckpt, subfolder="scheduler")
        
        controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth")
        self.pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            self.diffusion_ckpt, controlnet=controlnet
        ).to(self.device).to(torch.float16)
        self.pipe.to(self.pipe_device)

        # added_prompt = 'best quality, extremely detailed'
        # self.positive_prompt = self.edit_prompt + ', ' + added_prompt
        # self.positive_reverse_prompt = self.reverse_prompt + ', ' + added_prompt
        self.positive_prompt = self.edit_prompt
        self.negative_prompts = ''

        valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp")
        image_files = sorted([f for f in os.listdir(image_path) if f.lower().endswith(valid_extensions)])
        depth_files = sorted([f for f in os.listdir(depth_path) if f.lower().endswith(valid_extensions)])
        mask_files = sorted([f for f in os.listdir(mask_path) if f.lower().endswith(valid_extensions)])
        self.load_images_from_folder(image_files, depth_files, mask_files)
        
        random.seed(13789)
        self.ref_indices = [1, 2, 3, 4] 
        self.num_ref_views = len(self.ref_indices)

        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.controlnet_conditioning_scale = 1.0
        self.eta = 0.0
        self.chunk_size = self.chunk_size
    
    def load_images_from_folder(self, image_files, depth_files, mask_files):
        """
        Load and preprocess RGB, depth, and mask images into self.train_data.
        Each image gets corresponding depth, mask, and latent features (z_0_image).
        """

        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ])

        self.train_data = []

        for idx, (img_fname, depth_fname, mask_fname) in enumerate(zip(image_files, depth_files, mask_files)):
            img_path = os.path.join(self.image_path, img_fname)
            depth_path = os.path.join(self.depth_path, depth_fname)
            mask_path = os.path.join(self.mask_path, mask_fname)

            img = Image.open(img_path).convert("RGB")
            depth_img = Image.open(depth_path).convert("F")
            mask_img = Image.open(mask_path).convert("L")

            img_tensor = transform(img)
            depth_tensor = transform(depth_img)
            mask_tensor = transform(mask_img)
            mask_tensor = (mask_tensor > 0.5).float()

            z_0_latent = self.image2latent(img_tensor.permute(1, 2, 0)).detach()
            img_tensor.unsqueeze_(0).detach()

            # BCHW
            self.train_data.append({
                "image_idx": idx,
                "image": img_tensor,
                "depth_image": depth_tensor.numpy(),
                "mask_image": mask_tensor.numpy(),
                "z_0_image": z_0_latent,
            })

    def edit_images(self):
        '''Edit images with ControlNet and AttnAlign''' 
        # Set up ControlNet and AttnAlign
        self.pipe.scheduler = self.ddim_scheduler
        self.pipe.unet.set_attn_processor(
                        processor=utils.CrossViewAttnProcessor(self_attn_coeff=0.6,
                        unet_chunk_size=2))
        self.pipe.controlnet.set_attn_processor(
                        processor=utils.CrossViewAttnProcessor(self_attn_coeff=0,
                        unet_chunk_size=2)) 
        CONSOLE.print("Done Resetting Attention Processor", style="bold blue")
        
        print("#############################")
        CONSOLE.print("Start Editing: ", style="bold yellow")
        CONSOLE.print(f"Reference views are {[j+1 for j in self.ref_indices]}", style="bold yellow")
        print("#############################")

        ref_image_list = []
        for ref_idx in self.ref_indices:
            ref_data = deepcopy(self.train_data[ref_idx]) 
            ref_image = ref_data['image']
            ref_image_list.append(ref_image)
            
        ref_images = np.concatenate(ref_image_list, axis=0)
        ref_image_torch = torch.from_numpy(ref_images.copy()).to(torch.float16).to(self.pipe_device)

        # Edit images in chunk
        for idx in range(0, len(self.train_data), self.chunk_size): 
            chunked_data = self.train_data[idx: idx+self.chunk_size]
            
            indices = [current_data['image_idx'] for current_data in chunked_data]
            CONSOLE.print(f"Generating view: {indices}", style="bold yellow")

            images = [current_data['image'] for current_data in chunked_data] # list of np array
            depth_images = [self.depth2disparity(current_data['depth_image']) for current_data in chunked_data]
            mask_images = [current_data['mask_image'] for current_data in chunked_data] 

            imgs = np.concatenate(images, axis=0)
            images_torch = torch.from_numpy(imgs.copy()).to(torch.float16).to(self.pipe_device)
            images_chunk = torch.concatenate((ref_image_torch, images_torch), dim=0)

            disparities = np.concatenate(depth_images, axis=0)
            disparities_torch = torch.from_numpy(disparities.copy()).to(torch.float16).to(self.pipe_device)

            masks = np.concatenate(mask_images, axis=0)
            masks_torch = torch.from_numpy(masks.copy()).to(torch.float16).to(self.pipe_device).unsqueeze(0)
            
            chunk_edited = self.pipe(
                                prompt=[self.positive_prompt] * (self.num_ref_views+len(chunked_data)),
                                negative_prompt=[self.negative_prompts] * (self.num_ref_views+len(chunked_data)),
                                image=images_chunk,
                                control_image=disparities_torch,
                                mask_image=masks_torch,
                                num_inference_steps=self.num_inference_steps,
                                guidance_scale=self.guidance_scale,
                                controlnet_conditioning_scale=self.controlnet_conditioning_scale,
                                eta=self.eta,
                                output_type='pt',
                            ).images[self.num_ref_views:]

            chunk_edited = chunk_edited.cpu() 

            # Insert edited images back to train data for training
            for local_idx, edited_image in enumerate(chunk_edited):
                global_idx = indices[local_idx]

                bg_cntrl_edited_image = edited_image
                img_np = bg_cntrl_edited_image.to(torch.float16).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
                img_pil = Image.fromarray((img_np * 255).astype("uint8"))
                img_pil.save(os.path.join(self.output_folder, f"edited_{global_idx:04d}.png"))

        print("#############################")
        CONSOLE.print("Done Editing", style="bold yellow")
        print("#############################")

    @torch.no_grad()
    def depth2disparity(self, depth):
        """
        Args: depth numpy array [1 512 512]
        Return: disparity
        """
        disparity = 1 / (depth + 1e-5)
        disparity_map = disparity / np.max(disparity) # 0.00233~1
        disparity_map = np.concatenate([disparity_map, disparity_map, disparity_map], axis=0)
        return disparity_map[None]

    def image2latent(self, image):
        """Encode images to latents"""
        image = image * 2 - 1
        image = image.permute(2, 0, 1).unsqueeze(0) # torch.Size([1, 3, 512, 512]) -1~1
        image = image.to(dtype=self.pipe.vae.dtype, device=self.pipe.device)
        latents = self.pipe.vae.encode(image)['latent_dist'].mean
        latents = latents * 0.18215
        return latents

    def forward(self):
        pass

    
if __name__ == "__main__":
    image_folder = "/root/nam/lama/output/4"
    depth_folder = "/root/nam/lama/output/4_depth"
    mask_folder = "/root/nam/lama/output/4_mask"
    output_folder = "./output"
    os.makedirs(output_folder, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipeline = GaussCtrlPipeline(
        device=device,
        image_path=image_folder,
        depth_path=depth_folder,
        mask_path=mask_folder,
        output_folder=output_folder,
        diffusion_ckpt='stable-diffusion-v1-5/stable-diffusion-v1-5',
        edit_prompt="Concrete outdoor steps with no objects, clean and clear surface, natural shadows, and surrounding plants. No box or obstruction.",
        reverse_prompt="",
        ref_view_num=4,
        num_inference_steps=20,
        guidance_scale=5,
        chunk_size=1,
        test_mode="val"
    )

    pipeline.edit_images()
 