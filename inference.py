# This project uses Stable Diffusion, a model developed by Stability AI and released under the CreativeML Open RAIL-M license.

import os
from tqdm.auto import tqdm
from tqdm import tqdm
import numpy as np
import torch
import pywt
import torch.utils.checkpoint
import json
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline
from config import parse_args
from utils_model import load_weights
from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextConfig
import random
from model import model_types


def generate_and_save_seeds(num_seeds=2, seed_range=(1, 1000000), output_file="generated_seeds.json"):
    """Generate random seeds and save them to a JSON file."""
    # random.seed(42)  # For reproducibility of seed generation
    seeds = random.sample(range(seed_range[0], seed_range[1]), num_seeds)

    with open(output_file, "w") as f:
        json.dump({"seeds": seeds}, f)

    return seeds


def load_seeds(input_file="generated_seeds.json"):
    """Load previously generated seeds from JSON file."""
    with open(input_file, "r") as f:
        data = json.load(f)
    return data["seeds"]


def plot_and_save_grid(pipe, seed, prompt, modification, save_dir, scale=7.5, condition=None):
    '''Generate and save grid of images with modifications'''
    factors = [0.25, 0.3, 0.35]
    gen = torch.Generator(device=pipe.device)
    gen.manual_seed(seed)
    for factor in factors:
        def modify_h_space(module, input, output, step=[0]):
            change_o = modification[:, step[0], :, :, :].squeeze()
            change = factor * change_o.to(output.device)  # Ensure same device
            step[0] += 1
            return output + change

        with torch.no_grad(), pipe.unet.mid_block.register_forward_hook(modify_h_space):
            out = pipe(prompt=prompt, generator=gen, guidance_scale=scale,
                       controlnet_cond=condition)
            image = out.images[0]
            if factor == factors[-1]:
                image.save(os.path.join(save_dir, f'seed_{seed}.jpg'))


def extract_h_spaces(pipe, prompt, seed):
    '''Extract h-space representations'''
    h_space = []

    def get_h_space(module, input, output):
        h_space[-1].append(output.cpu())

    with torch.no_grad(), pipe.unet.mid_block.register_forward_hook(get_h_space):
        h_space.append([])
        gen = torch.Generator("cuda").manual_seed(seed)
        out = pipe(prompt=prompt, generator=gen, guidance_scale=7.5)

    h_space = torch.cat([torch.stack(x, dim=1) for x in h_space])
    return h_space.numpy(), out.images[0]


def compute_dwt_and_reconstruct_ll(data, n_components=51):
    """Compute DWT and reconstruct using only LL subband"""
    batch_size = data.shape[0]
    reconstructed_data = np.zeros_like(data[:n_components])

    for i in tqdm(range(batch_size)):
        sample = data[i]
        reconstructed_channels = []

        for channel in range(sample.shape[0]):
            coeffs = pywt.dwt2(sample[channel], 'db1')

            if i < n_components:
                new_coeffs = (coeffs[0], (np.zeros_like(coeffs[1][0]),
                                          np.zeros_like(coeffs[1][1]),
                                          np.zeros_like(coeffs[1][2])))
                reconstructed = pywt.idwt2(new_coeffs, 'db1')
                reconstructed_channels.append(reconstructed)

        if i < n_components:
            reconstructed_data[i] = np.stack(reconstructed_channels)

    return {'reconstructed': reconstructed_data}


def main():
    args = parse_args()
    base_output_dir = args.output_dir + '/' + args.image_dir

    # Create three main directories
    dirs = {
        'original': os.path.join(base_output_dir, 'original'),
        'dwt_modified': os.path.join(base_output_dir, 'dwt_modified')
    }

    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    # Generate or load seeds
    seeds_file = os.path.join(args.output_dir, "generated_seeds.json")
    if not os.path.exists(seeds_file):
        seeds = generate_and_save_seeds(output_file=seeds_file)
        print(f"Generated and saved {len(seeds)} new seeds")
    else:
        seeds = load_seeds(seeds_file)
        print(f"Loaded {len(seeds)} existing seeds")

    device = 'cuda'

    # Load the text encoder configuration
    text_config_dict = {
        "architectures": ["CLIPTextModel"],
        "attention_dropout": 0.0,
        "bos_token_id": 0,
        "dropout": 0.0,
        "eos_token_id": 2,
        "hidden_act": "quick_gelu",
        "hidden_size": 768,
        "initializer_factor": 1.0,
        "intermediate_size": 3072,
        "layer_norm_eps": 1e-05,
        "max_position_embeddings": 77,
        "model_type": "clip_text_model",
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "pad_token_id": 1,
        "torch_dtype": "float32",
        "transformers_version": "4.25.1",
        "vocab_size": 49408
    }

    # Create text config and remove id2label if present
    text_config = CLIPTextConfig.from_dict(text_config_dict)
    if hasattr(text_config, "id2label"):
        delattr(text_config, "id2label")

    # Initialize components with proper config
    tokenizer = CLIPTokenizer.from_pretrained('CompVis/stable-diffusion-v1-4', subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(
        'CompVis/stable-diffusion-v1-4',
        subfolder="text_encoder",
        config=text_config
    )

    # Initialize pipeline with custom components
    pipe = StableDiffusionPipeline.from_pretrained(
        'CompVis/stable-diffusion-v1-4',
        text_encoder=text_encoder,
        tokenizer=tokenizer,
    ).to(device)
    mlp = model_types[args.model_type](resolution=args.resolution // 64)
    mlp = mlp.to(device)
    pipe.unet.set_controlnet(mlp)
    pipe.unet = load_weights(pipe.unet, 'unet.pth')
    pipe.unet = pipe.unet.to(device)

    # Load concept dictionary and set condition
    concept_dict = json.load(open("concept_dict.json", "r"))
    condition = torch.zeros(1, 100, device=device)
    condition[:, concept_dict['female']] = 1

    # Define prompts
    prompt = "a photo of a doctor in the hospital"
    guidance_scale = 7.5
    pipe.safety_checker = None

    # Make sure the controlnet weights are on the same device
    for param in pipe.unet.parameters():
        param_device = param.device
        if param_device != device:
            param.data = param.data.to(device)

    # Also ensure all MLP parameters are on the correct device
    for param in mlp.parameters():
        param.data = param.data.to(device)

    # Process each seed
    for seed in tqdm(seeds, desc="Processing seeds"):
        # 1. Generate original image
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)
        out_original = pipe(prompt=prompt, generator=gen, guidance_scale=guidance_scale)
        out_original.images[0].save(os.path.join(dirs['original'], f'seed_{seed}.jpg'))


        # 2. Generate DWT modified versions
        h_out, _ = extract_h_spaces(pipe, prompt, seed)
        results = compute_dwt_and_reconstruct_ll(h_out, n_components=51)
        filtered_h_space = results['reconstructed']
        filtered_h_space_tensor = torch.tensor(filtered_h_space).float().to(device)

        plot_and_save_grid(
            pipe=pipe,
            seed=seed,
            prompt=prompt,
            modification=filtered_h_space_tensor,
            save_dir=dirs['dwt_modified'],
            scale=guidance_scale,
            condition=condition
        )


if __name__ == "__main__":
    main()
