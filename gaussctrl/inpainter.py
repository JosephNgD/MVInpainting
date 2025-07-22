import os
import numpy as np 
from rich.progress import Console
from copy import deepcopy
from PIL import Image
from typing_extensions import Literal
from lang_sam import LangSAM
import utils

import torch, random
from diffusers.schedulers import DDIMScheduler, DDIMInverseScheduler
from diffusers.models.attention_processor import AttnProcessor
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UNet2DConditionModel
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
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(self.diffusion_ckpt, controlnet=controlnet).to(self.device).to(torch.float16)
        self.pipe.to(self.pipe_device)

        added_prompt = 'best quality, extremely detailed'
        self.positive_prompt = self.edit_prompt + ', ' + added_prompt
        self.positive_reverse_prompt = self.reverse_prompt + ', ' + added_prompt
        # self.negative_prompts = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'
        self.negative_prompts = ''
        

        valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp")
        image_files = sorted([f for f in os.listdir(image_path) if f.lower().endswith(valid_extensions)])
        depth_files = sorted([f for f in os.listdir(depth_path) if f.lower().endswith(valid_extensions)])
        mask_files = sorted([f for f in os.listdir(mask_path) if f.lower().endswith(valid_extensions)])
        self.load_images_from_folder(image_files, depth_files, mask_files)
        num_images = len(image_files)
        view_num = num_images
        # anchors = [(view_num * i) // self.ref_view_num for i in range(self.ref_view_num)]
        anchors = [(view_num * i) // self.ref_view_num for i in range(self.ref_view_num)] + [view_num]
        
        random.seed(13789)
        # self.ref_indices = [random.randint(anchor, anchors[idx+1]) for idx, anchor in enumerate(anchors[:-1])] 
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
        from torchvision import transforms

        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),  # [C, H, W]
        ])

        depth_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),  # [1, H, W]
        ])

        mask_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),  # [1, H, W]
        ])

        self.train_data = []

        for idx, (img_fname, depth_fname, mask_fname) in enumerate(zip(image_files, depth_files, mask_files)):
            img_path = os.path.join(self.image_path, img_fname)
            depth_path = os.path.join(self.depth_path, depth_fname)
            mask_path = os.path.join(self.mask_path, mask_fname)

            # Load and preprocess RGB image
            img = Image.open(img_path).convert("RGB")
            img_tensor = transform(img)  # [3, 512, 512]

            # Load depth
            if depth_path.endswith(".npy"):
                depth_array = np.load(depth_path)
                if depth_array.ndim == 2:
                    depth_array = depth_array[None]  # [1, H, W]
                depth_tensor = torch.from_numpy(depth_array).float()
                depth_tensor = torch.nn.functional.interpolate(
                    depth_tensor[None], size=(512, 512), mode="bilinear", align_corners=False
                )[0]
            else:
                depth_img = Image.open(depth_path).convert("F")
                depth_tensor = depth_transform(depth_img)

            # Load and process mask
            mask_img = Image.open(mask_path).convert("L")
            mask_tensor = mask_transform(mask_img)
            mask_tensor = (mask_tensor > 0.5).float()  # binarize to 0 or 1

            # Compute z_0_image using the VAE
            z_0_latent = self.image2latent(img_tensor.permute(1, 2, 0))  # [H, W, C]
            z_0_numpy = z_0_latent.detach().to(torch.float16).cpu().numpy()


            self.train_data.append({
                "image_idx": idx,
                "unedited_image": img_tensor,
                "depth_image": depth_tensor.numpy(),
                "z_0_image": z_0_numpy,
                "mask_image": mask_tensor.numpy(),
                "image": img_tensor,
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
        ref_disparity_list = []
        ref_z0_list = []
        for ref_idx in self.ref_indices:
            ref_data = deepcopy(self.train_data[ref_idx]) 
            # ref_data = (self.train_data[ref_idx]) 
            ref_disparity = self.depth2disparity(ref_data['depth_image']) 
            ref_z0 = ref_data['z_0_image']
            ref_disparity_list.append(ref_disparity)
            ref_z0_list.append(ref_z0) 
            
        ref_disparities = np.concatenate(ref_disparity_list, axis=0)
        ref_z0s = np.concatenate(ref_z0_list, axis=0)
        ref_disparity_torch = torch.from_numpy(ref_disparities.copy()).to(torch.float16).to(self.pipe_device)
        ref_z0_torch = torch.from_numpy(ref_z0s.copy()).to(torch.float16).to(self.pipe_device)

        # Edit images in chunk
        for idx in range(0, len(self.train_data), self.chunk_size): 
            chunked_data = self.train_data[idx: idx+self.chunk_size]
            
            indices = [current_data['image_idx'] for current_data in chunked_data]
            # indices = [0]
            mask_images = [current_data['mask_image'] for current_data in chunked_data if 'mask_image' in current_data.keys()] 
            unedited_images = [current_data['unedited_image'] for current_data in chunked_data]
            CONSOLE.print(f"Generating view: {indices}", style="bold yellow")

            depth_images = [self.depth2disparity(current_data['depth_image']) for current_data in chunked_data]
            disparities = np.concatenate(depth_images, axis=0)
            disparities_torch = torch.from_numpy(disparities.copy()).to(torch.float16).to(self.pipe_device)

            z_0_images = [current_data['z_0_image'] for current_data in chunked_data] # list of np array
            z0s = np.concatenate(z_0_images, axis=0)
            latents_torch = torch.from_numpy(z0s.copy()).to(torch.float16).to(self.pipe_device)

            disp_ctrl_chunk = torch.concatenate((ref_disparity_torch, disparities_torch), dim=0)
            latents_chunk = torch.concatenate((ref_z0_torch, latents_torch), dim=0)
            
            chunk_edited = self.pipe(
                                prompt=[self.positive_prompt] * (self.num_ref_views+len(chunked_data)),
                                negative_prompt=[self.negative_prompts] * (self.num_ref_views+len(chunked_data)),
                                latents=latents_chunk,
                                image=disp_ctrl_chunk,
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
                # if mask_images != []:
                #     # mask = torch.from_numpy(mask_images[local_idx])
                #     # mask = mask.expand(3, -1, -1)
                #     # bg_mask = 1 - mask
                #     import torch.nn.functional as F

                #     # Assumes mask is [1, 512, 512]
                #     mask = torch.from_numpy(mask_images[local_idx])
                #     mask = mask.float()

                #     # Expand and blur the mask to smooth edges
                #     mask = F.avg_pool2d(mask[None], kernel_size=15, stride=1, padding=7)[0]  # Softens mask edges
                #     mask = mask.expand(3, -1, -1)  # [3, 512, 512]
                #     bg_mask = 1 - mask

                #     # unedited_image = unedited_images[local_idx].permute(2, 0, 1)
                #     unedited_image = unedited_images[local_idx].unsqueeze(0)
                #     breakpoint()
                #     bg_cntrl_edited_image = edited_image * mask[None] + unedited_image * bg_mask[None] 
                #     bg_cntrl_edited_image = bg_cntrl_edited_image.squeeze(0)

                # Save edited image to train_data
                # self.train_data[global_idx]["image"] = bg_cntrl_edited_image.permute(1, 2, 0).to(torch.float32)  # [512, 512, 3]

                # Save edited image to output folder as PNG
                # img_np = bg_cntrl_edited_image.permute(1, 2, 0).clamp(0, 1).cpu().numpy()  # [H, W, C]
                breakpoint()
                img_np = bg_cntrl_edited_image.to(torch.float16).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
                img_pil = Image.fromarray((img_np * 255).astype("uint8"))
                img_pil.save(os.path.join(self.output_folder, f"edited_{global_idx:04d}.png"))
                breakpoint()

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
        diffusion_ckpt='CompVis/stable-diffusion-v1-4',
        edit_prompt="No object",
        reverse_prompt="",
        ref_view_num=4,
        num_inference_steps=5,
        guidance_scale=5,
        chunk_size=1,
        test_mode="val"
    )

    pipeline.edit_images()
 