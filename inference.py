#Copyright 2024-2025 Adobe Inc.

import os, sys
import argparse
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from pytorch_lightning.utilities import rank_zero_only
from src.utils.train_util import instantiate_from_config
from torchvision.transforms import v2
import torch 
import numpy as np
from PIL import Image
from datetime import datetime
import torch.nn.functional as F

@rank_zero_only
def rank_zero_print(*args):
    print(*args)

def unscale_image(image):
    image = image / 0.5 * 0.8
    return image


def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        default=None,
        help="resume from checkpoint",
    )
    parser.add_argument(
        "-b",
        "--base",
        type=str,
        default="base_config.yaml",
        help="path to base configs",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,
        help="random seed",
    )

    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="logs",
        help="directory for logging data",
    )
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        default="",
        help="image prompt name",
    )

    parser.add_argument(
        "--background",
        type=bool,
        default=True,
        help="image prompt name",
    )
    return parser



def make_mask(image):
    #given an image, turn white color to its alpha channel
    idx = torch.nonzero((image > 240).all(dim=2))
    alpha = torch.ones_like(image[:,:,0]) * 255
    alpha[idx[:,0], idx[:,1]] = 0
    image = torch.cat([image, alpha.unsqueeze(2)], dim=2)
    return image



def attach_gray_center_square(
    img: torch.Tensor,
    alpha_channel: torch.Tensor,
    ratio: float = 0.8,
    gray_value: float = 0.5,
    margin: float = 0.65  # The foreground's max dimension will be margin * square_side
) -> torch.Tensor:
    # Check that the image has 3 channels.
    assert img.ndim == 3 and img.shape[0] == 3, "Expected input of shape [3, H, W]."
    
    # Create an output image with a white background.
    output = torch.ones_like(img)
    _, H, W = img.shape

    # Compute the dimensions and location of the centered square.
    square_side = int(min(H, W) * ratio)
    left = (W - square_side) // 2
    top = (H - square_side) // 2
    right = left + square_side
    bottom = top + square_side

    # Fill the centered square with gray.
    for c in range(3):
        output[c, top:bottom, left:right] = gray_value

    # --- Extract the foreground bounding box from the alpha channel ---
    fg_mask = (alpha_channel != 0)
    if fg_mask.sum() == 0:
        # If no foreground, return the white image with a gray center.
        return output

    ys, xs = torch.where(fg_mask)
    y_min, y_max = int(ys.min()), int(ys.max())
    x_min, x_max = int(xs.min()), int(xs.max())
    fg_height = y_max - y_min + 1
    fg_width = x_max - x_min + 1

    # --- Compute the scale factor so that the foreground's maximum dimension equals margin * square_side ---
    scale_factor = (margin * square_side) / max(fg_width, fg_height)

    # --- Extract and rescale the foreground crop and its alpha mask ---
    fg_crop = img[:, y_min:y_max+1, x_min:x_max+1]
    fg_alpha_crop = alpha_channel[y_min:y_max+1, x_min:x_max+1]

    new_fg_width = max(1, int(fg_width * scale_factor))
    new_fg_height = max(1, int(fg_height * scale_factor))
    
    fg_crop_scaled = F.interpolate(fg_crop.unsqueeze(0), size=(new_fg_height, new_fg_width),
                                   mode='bilinear', align_corners=False).squeeze(0)
    fg_alpha_scaled = F.interpolate(fg_alpha_crop.unsqueeze(0).unsqueeze(0).float(),
                                    size=(new_fg_height, new_fg_width),
                                    mode='nearest').squeeze(0).squeeze(0)
    
    # --- Determine placement: center the (scaled) foreground within the gray square ---
    center_x = left + square_side // 2
    center_y = top + square_side // 2
    paste_left = center_x - new_fg_width // 2
    paste_top = center_y - new_fg_height // 2

    # --- Composite the foreground over the gray square ---
    for c in range(3):
        region = output[c, paste_top:paste_top+new_fg_height, paste_left:paste_left+new_fg_width]
        region[fg_alpha_scaled > 0] = fg_crop_scaled[c][fg_alpha_scaled > 0]

    return output




def load_im( path, color=[1.0,1.0,1.0]):
    pil_img = Image.open(path)


    image = np.asarray(pil_img, dtype=np.float32) / 255.
    alpha = image[:, :, 3:]
    image = image[:, :, :3] * alpha + color * (1 - alpha)

    image = torch.from_numpy(image).permute(2, 0, 1).contiguous().float()
    alpha  = torch.from_numpy(alpha).permute(2, 0, 1).contiguous().float()
    image = v2.functional.resize(image, 512, interpolation=3, antialias=True).clamp(0, 1)
    alpha = v2.functional.resize(alpha, 512, interpolation=3, antialias=True).clamp(0, 1)
    if BACKGROUND:
        image = attach_gray_center_square(image, alpha[0],ratio=0.7)
    
    image = image.contiguous().float()
    image = image.to("cuda")
    return image
    
if __name__ == "__main__":
    # add cwd for convenience and to make classes in this file available when
    # running as `python main.py`
    sys.path.append(os.getcwd())

    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    exp_name = opt.base.split("/")[-1].split(".")[0]
    prompt = opt.prompt + datetime.now().strftime("-%Y%m%d-%H%M%S")
    name = opt.prompt
    BACKGROUND = opt.background
    

    logdir = os.getcwd()
    no_ext_name = name.split(".")[0]
    outputexpdir = os.path.join(logdir, "mvoutput",exp_name)
    if not os.path.exists(outputexpdir):
        os.makedirs(outputexpdir,exist_ok=True)
    outputdir = os.path.join(logdir, "mvoutput",exp_name,no_ext_name)
    if not os.path.exists(outputdir):
        os.makedirs(outputdir,exist_ok=True)
    checkpointdir = os.path.join(logdir, "checkpoints",exp_name)
    seed_everything(opt.seed)

    # init configs
    config = OmegaConf.load(opt.base)
    lightning_config = config.lightning
    

    
    # model
    model = instantiate_from_config(config.model)

    print("Loading model from checkpoint")
    model = model.__class__.load_from_checkpoint(os.path.join(checkpointdir,opt.resume), **config.model.params)
    model.pipeline = model.pipeline.to("cuda")
    model.logdir = logdir
    model.eval()
    


    data_path = f"./selected_input/{name}"



    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    img = load_im(data_path)
    img = v2.functional.to_pil_image(img)
    img.save(os.path.join(outputdir,f"{no_ext_name}_input.png"))


    latent = model.pipeline(img, num_inference_steps=75, output_type='latent').images
    image = unscale_image(model.pipeline.vae.decode(latent / model.pipeline.vae.config.scaling_factor, return_dict=False)[0])   # [-1, 1]
    image = (image * 0.5 + 0.5).clamp(0, 1)

    image = torch.from_numpy((image[0].permute((1,2,0)).cpu().clone().detach().numpy() * 255).astype(np.uint8))

    index = [0,1,3,4,5,6]
    image_list = []
    for i in range(3):
        for j in range(2):  # There are 3 columns
            sub_image = image[i*320:(i+1)*320, j*320:(j+1)*320, :]
            image_list.append(sub_image)

    for i in range(len(image_list)):
        sub_image = image_list[i]
        sub_image = make_mask(sub_image)
        sub_image_pil = Image.fromarray(sub_image.numpy())
        sub_image_pil.save(os.path.join(outputdir, f"normals_{no_ext_name}_{index[i]}.png"))

### python inference.py --base config/example_run.yaml --resume example.ckpt --prompt ear2
