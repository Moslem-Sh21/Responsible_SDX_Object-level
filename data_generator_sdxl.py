# This project uses Stable Diffusion XL, a model developed by Stability AI

import os
import argparse
import json
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from diffusers import StableDiffusionXLPipeline, DiffusionPipeline
from trace import trace
from utils import set_seed, auto_device, cache_dir


class DAAMDataCreatorXL:
    def __init__(self,
                 prompt: str,
                 concepts: List[str],
                 output_dir: str,
                 num_samples: int = 10,
                 guidance_scale: float = 7.5,
                 num_inference_steps: int = 50,
                 height: int = 1024,  # SDXL native resolution
                 width: int = 1024,   # SDXL native resolution
                 low_memory: bool = False,
                 base_seed: Optional[int] = 42,
                 model_path: str = "stabilityai/stable-diffusion-xl-base-1.0",
                 use_refiner: bool = False,
                 refiner_model_path: str = "stabilityai/stable-diffusion-xl-refiner-1.0",
                 high_noise_frac: float = 0.8):
        """
        Initialize DAAM Data Creator for SDXL generating multiple images and their attention heatmaps.

        Args:
            prompt: The text prompt to generate images from.
            concepts: List of words to generate heatmaps for.
            output_dir: Directory to save the outputs.
            num_samples: Number of samples to generate.
            guidance_scale: Guidance scale for the diffusion model.
            num_inference_steps: Number of denoising steps.
            height: Height of the generated images (1024 for SDXL).
            width: Width of the generated images (1024 for SDXL).
            low_memory: Whether to use low memory mode.
            base_seed: Starting seed for reproducibility.
            model_path: Path to the pretrained SDXL base model.
            use_refiner: Whether to use SDXL refiner.
            refiner_model_path: Path to the SDXL refiner model.
            high_noise_frac: Fraction of noise steps for base model when using refiner.
        """
        self.prompt = prompt
        self.concepts = concepts
        self.output_dir = Path(output_dir)
        self.num_samples = num_samples
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.height = height
        self.width = width
        self.low_memory = low_memory
        self.base_seed = base_seed
        self.model_path = model_path
        self.use_refiner = use_refiner
        self.refiner_model_path = refiner_model_path
        self.high_noise_frac = high_noise_frac

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir = self.output_dir
        self.images_dir.mkdir(exist_ok=True)

        if len(concepts) > 0:
            self.first_concept_dir = self.images_dir
        else:
            self.first_concept_dir = None

        if len(concepts) > 1:
            self.second_concept_dir = self.images_dir
        else:
            self.second_concept_dir = None

        # Set up metadata
        self.metadata = {
            "prompt": prompt,
            "concepts": concepts,
            "samples": [],
        }

        # Set up the SDXL model
        print(f"Loading SDXL base model from {model_path}...")
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            use_safetensors=True,
            variant="fp16" if torch.cuda.is_available() else None
        )
        self.pipe = auto_device(self.pipe)
        
        # Disable safety checker for research purposes
        self.pipe.safety_checker = None
        
        # Load refiner if requested
        if self.use_refiner:
            print(f"Loading SDXL refiner model from {refiner_model_path}...")
            self.refiner = DiffusionPipeline.from_pretrained(
                refiner_model_path,
                text_encoder_2=self.pipe.text_encoder_2,
                vae=self.pipe.vae,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                use_safetensors=True,
                variant="fp16" if torch.cuda.is_available() else None
            )
            self.refiner = auto_device(self.refiner)
        else:
            self.refiner = None

    def save_clean_heatmap(self, heatmap_data: torch.Tensor, save_path: Path, cmap: str = 'jet'):
        """
        Save a clean heatmap image without borders, axes, or titles.

        Args:
            heatmap_data: The heatmap tensor data
            save_path: Path to save the heatmap image
            cmap: Colormap to use for visualization
        """
        # Create figure with no padding or margins
        plt.figure(figsize=(10, 10))
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)

        # Plot heatmap with no axes
        plt.imshow(heatmap_data.cpu().numpy(), cmap=cmap)
        plt.axis('off')

        # Save with tight bounding box and no padding
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=100)
        plt.close()

    def generate_sample(self, sample_idx: int):
        """
        Generate a single sample (image + heatmaps) using SDXL.

        Args:
            sample_idx: Index of the sample to generate.

        Returns:
            Dict containing metadata for the generated sample.
        """
        # Use deterministic seed based on sample index
        seed = self.base_seed + sample_idx if self.base_seed is not None else None
        generator = set_seed(seed) if seed is not None else None

        print(f"\nGenerating sample {sample_idx + 1}/{self.num_samples}")
        print(f"Prompt: '{self.prompt}', Seed: {seed}")

        # Create a tracer to track cross-attention during generation
        with trace(self.pipe, low_memory=self.low_memory) as tracer:
            # Generate the image with SDXL
            if self.use_refiner and self.refiner is not None:
                # Use base + refiner pipeline
                image = self.pipe(
                    prompt=self.prompt,
                    height=self.height,
                    width=self.width,
                    num_inference_steps=self.num_inference_steps,
                    guidance_scale=self.guidance_scale,
                    generator=generator,
                    denoising_end=self.high_noise_frac,
                    output_type="latent",
                ).images[0]
                
                # Refine the image
                image = self.refiner(
                    prompt=self.prompt,
                    image=image,
                    num_inference_steps=self.num_inference_steps,
                    denoising_start=self.high_noise_frac,
                    generator=generator,
                ).images[0]
            else:
                # Use base model only
                image = self.pipe(
                    prompt=self.prompt,
                    height=self.height,
                    width=self.width,
                    num_inference_steps=self.num_inference_steps,
                    guidance_scale=self.guidance_scale,
                    generator=generator,
                ).images[0]

            # Save the generated image
            image_path = self.images_dir / f"image_{sample_idx:04d}.jpg"
            image.save(image_path)
            print(f"Saved image to {image_path}")

            # Compute the global heat map for visualization
            global_heat_map = tracer.compute_global_heat_map()

            # Process each concept and generate heatmap
            concept_data = {}
            for concept_idx, concept in enumerate(self.concepts):
                try:
                    # Compute word-specific heat map
                    word_heat_map = global_heat_map.compute_word_heat_map(concept)

                    # Determine which directory to use based on concept index
                    if concept_idx == 0 and self.first_concept_dir is not None:  # First concept
                        concept_dir = self.first_concept_dir
                    elif concept_idx == 1 and self.second_concept_dir is not None:  # Second concept
                        concept_dir = self.second_concept_dir
                    else:  # Other concepts
                        concept_dir = self.other_viz_dir / concept
                        concept_dir.mkdir(exist_ok=True)

                    # Save the raw heatmap (clean version without borders, axes, or titles)
                    heatmap_data = word_heat_map.expand_as(image)
                    heatmap_path = concept_dir / f"{concept}_{sample_idx:04d}.jpg"

                    # Save clean heatmap image
                    self.save_clean_heatmap(heatmap_data, heatmap_path)

                    concept_data[concept] = {
                        "heatmap_path": str(heatmap_path.relative_to(self.output_dir)),
                    }

                    print(f"Saved heatmaps for concept '{concept}'")

                except ValueError as e:
                    print(f"Could not generate heatmap for concept '{concept}': {e}")
                    concept_data[concept] = {"error": str(e)}

        # Record metadata for this sample
        sample_metadata = {
            "index": sample_idx,
            "seed": seed,
            "image_path": str(image_path.relative_to(self.output_dir)),
            "concepts": concept_data,
        }

        return sample_metadata

    def run(self):
        """
        Generate all samples and save metadata.
        """
        print(f"Starting generation of {self.num_samples} samples for prompt: '{self.prompt}'")
        print(f"Tracking heatmaps for concepts: {', '.join(self.concepts)}")
        print(f"Output directory: {self.output_dir}")
        print(f"Using SDXL with resolution: {self.width}x{self.height}")
        if self.use_refiner:
            print("Using SDXL refiner for enhanced quality")

        # Generate all samples with progress bar
        for i in tqdm(range(self.num_samples), desc="Generating samples"):
            sample_metadata = self.generate_sample(i)
            self.metadata["samples"].append(sample_metadata)

            # Save metadata after each sample (in case of interruptions)
            with open(self.output_dir / "metadata.json", "w") as f:
                json.dump(self.metadata, f, indent=2)

        print(f"\nGeneration completed successfully.")
        print(f"Generated {self.num_samples} samples.")
        print(f"Images saved to: {self.images_dir}")

        # Print locations of concept heatmaps
        if self.first_concept_dir is not None:
            print(f"Heatmaps for '{self.concepts[0]}' saved to: {self.first_concept_dir}")

        if self.second_concept_dir is not None and len(self.concepts) > 1:
            print(f"Heatmaps for '{self.concepts[1]}' saved to: {self.second_concept_dir}")

        print(f"Metadata saved to: {self.output_dir / 'metadata.json'}")


def repeat_ntimes(x, n):
    return [item for item in x for i in range(n)]

def parse_args():
    parser = argparse.ArgumentParser(description="Generate images and DAAM heatmaps for SDXL")
    parser.add_argument("--prompt", type=str, default="a photo of a woman", help="Text prompt for image generation")
    parser.add_argument("--concepts", nargs="+", default=["woman"], help="Concepts to generate heatmaps for")
    parser.add_argument("--output_dir", type=str, default="./datasets", help="Output directory")
    parser.add_argument("--num_samples", type=int, default=2000, help="Number of samples to generate")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Guidance scale")
    parser.add_argument("--steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--low_memory", action="store_true", help="Use low memory mode")
    parser.add_argument("--height", type=int, default=1024, help="Image height (SDXL native: 1024)")
    parser.add_argument("--width", type=int, default=1024, help="Image width (SDXL native: 1024)")
    parser.add_argument("--model_path", type=str, default="stabilityai/stable-diffusion-xl-base-1.0",
                        help="Path to pretrained SDXL base model")
    parser.add_argument("--use_refiner", action="store_true", help="Use SDXL refiner model")
    parser.add_argument("--refiner_model_path", type=str, default="stabilityai/stable-diffusion-xl-refiner-1.0",
                        help="Path to SDXL refiner model")
    parser.add_argument("--high_noise_frac", type=float, default=0.8, 
                        help="Fraction of noise steps for base model when using refiner")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    concept_dict = ["female", "male"]
    concept_dict = {c: i for i, c in enumerate(concept_dict)}
    image_prompt = [
        "a photo of a person",
    ]

    input_prompt_and_target_concept = [
        [
            ["a photo of a person", ["female"]],
        ],
    ]

    image_prompt = repeat_ntimes(image_prompt, args.num_samples)
    input_prompt_and_target_concept = repeat_ntimes(input_prompt_and_target_concept, args.num_samples)

    os.makedirs(args.output_dir, exist_ok=True)
    json.dump(input_prompt_and_target_concept, open(args.output_dir + "/labels.json", "w"))
    json.dump(concept_dict, open(args.output_dir + "/concept_dict.json", "w"))

    creator = DAAMDataCreatorXL(
        prompt=args.prompt,
        concepts=args.concepts,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.steps,
        height=args.height,
        width=args.width,
        low_memory=args.low_memory,
        base_seed=args.seed,
        model_path=args.model_path,
        use_refiner=args.use_refiner,
        refiner_model_path=args.refiner_model_path,
        high_noise_frac=args.high_noise_frac
    )

    creator.run()