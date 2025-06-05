import glob
import os
import json
from PIL import Image
import numpy as np
import torch
import pandas as pd


def int_to_onehot(x, n):
    if not isinstance(x, list):
        x = [x]
    assert isinstance(x[0], int)
    x = torch.tensor(x).long()
    v = torch.zeros(n)
    v[x] = 1.
    return v


random_select = lambda l: l[np.random.choice(len(l))]
top_select = lambda l: l[0]


class TrainingDataset(torch.utils.data.Dataset):
    def __init__(self, image_folder, transform, tokenizer, max_concept_length, select):
        # Find all image files with the specified naming pattern
        image_paths = glob.glob(os.path.join(image_folder, "image_*.jpg"))
        if not image_paths:
            image_paths = glob.glob(os.path.join(image_folder, "**", "image_*.jpg"), recursive=True)

        # Sort images by their numerical ID
        def get_image_id(path):
            basename = os.path.basename(path)
            # Extract the ID from image_XXXX.jpg format
            if 'image_' in basename:
                try:
                    id_part = basename.split('image_')[1].split('.')[0]
                    return int(id_part)
                except (ValueError, IndexError):
                    return float('inf')  # Place at the end if parsing fails
            return float('inf')

        # Sort images by their ID
        image_paths = sorted(image_paths, key=get_image_id)

        self.image_paths = image_paths
        self.transform = transform
        self.tokenizer = tokenizer
        self.concept_dict = json.load(open(os.path.join(image_folder, 'concept_dict.json'), 'r'))
        self.max_concept_length = max_concept_length

        # Select method setup
        if select == "top":
            self.select_method = top_select
        elif select == "random":
            self.select_method = random_select
        else:
            raise NotImplementedError(f"Unknown select method: {select}")

        self.labels = json.load(open(os.path.join(image_folder, 'labels.json'), 'r'))

        # Map each image to its corresponding heatmap
        self.heatmap_paths = []

        # Collect all possible heatmap paths first
        all_heatmap_paths = glob.glob(os.path.join(image_folder, "woman_*.jpg"))
        if not all_heatmap_paths:
            all_heatmap_paths = glob.glob(os.path.join(image_folder, "**", "woman_*.jpg"), recursive=True)

        # Also check for other extensions if needed
        all_heatmap_paths.extend(glob.glob(os.path.join(image_folder, "woman_*.png")))

        # Create a dictionary for quick lookup: ID -> heatmap_path
        heatmap_dict = {}
        for hmap_path in all_heatmap_paths:
            basename = os.path.basename(hmap_path)
            if 'woman_' in basename:
                try:
                    id_part = basename.split('woman_')[1].split('.')[0]
                    heatmap_dict[id_part] = hmap_path
                except (ValueError, IndexError):
                    continue

        # Match each image with its corresponding heatmap
        for img_path in self.image_paths:
            basename = os.path.basename(img_path)
            try:
                # Extract ID from image_XXXX.jpg
                id_part = basename.split('image_')[1].split('.')[0]

                # Look up the heatmap with the same ID
                if id_part in heatmap_dict:
                    self.heatmap_paths.append(heatmap_dict[id_part])
                else:
                    print(f"Warning: No heatmap found for {img_path} (ID: {id_part})")
                    self.heatmap_paths.append(None)
            except (ValueError, IndexError):
                print(f"Warning: Could not parse ID for {img_path}")
                self.heatmap_paths.append(None)

        # Verify we have the same number of images and heatmaps
        if len(self.image_paths) != len(self.heatmap_paths):
            print(
                f"Warning: Number of images ({len(self.image_paths)}) doesn't match number of heatmaps ({len(self.heatmap_paths)})")

    def __getitem__(self, index):
        # Get input prompt and target concept
        input_prompt, target_concept = self.select_method(self.labels[index])
        
        # Handle dual tokenization for SDXL
        if isinstance(self.tokenizer, tuple):
            # If tokenizer is a tuple of two tokenizers
            tokenizer_1, tokenizer_2 = self.tokenizer
            input_prompt_tokens_1 = tokenizer_1([input_prompt])[0]
            input_prompt_tokens_2 = tokenizer_2([input_prompt])[0]
            input_prompt_tokens = (input_prompt_tokens_1, input_prompt_tokens_2)
        else:
            # Single tokenizer function that returns tuple for both encoders
            input_prompt_tokens = self.tokenizer([input_prompt])
        
        # Handle heatmap prompt tokenization
        input_prompt_heatmap = ""
        if isinstance(self.tokenizer, tuple):
            tokenizer_1, tokenizer_2 = self.tokenizer
            input_prompt_heatmap_tokens_1 = tokenizer_1([input_prompt_heatmap])[0]
            input_prompt_heatmap_tokens_2 = tokenizer_2([input_prompt_heatmap])[0]
            input_prompt_heatmap_tokens = (input_prompt_heatmap_tokens_1, input_prompt_heatmap_tokens_2)
        else:
            input_prompt_heatmap_tokens = self.tokenizer([input_prompt_heatmap])
        
        target_concept = [self.concept_dict[c] for c in target_concept]
        target_concept = int_to_onehot(target_concept, self.max_concept_length)

        # Load image
        image_path = self.image_paths[index]
        x = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            x = self.transform(x)

        # Load heatmap if available
        heatmap_path = self.heatmap_paths[index]
        if heatmap_path is not None and os.path.exists(heatmap_path):
            heatmap = Image.open(heatmap_path).convert("RGB")
            if self.transform is not None:
                heatmap = self.transform(heatmap)
        else:
            # If heatmap is not available, create a zero tensor with same shape as image
            heatmap = torch.zeros_like(x)

        return x, input_prompt_tokens, target_concept, heatmap, input_prompt_heatmap_tokens

    def __len__(self):
        return len(self.image_paths)


def get_dataloader(image_folder, batch_size, transform, tokenizer, collate_fn=None, num_workers=4, shuffle=False,
                   max_concept_length=100, select="random"):
    dataset = TrainingDataset(image_folder, transform=transform, tokenizer=tokenizer, select=select,
                              max_concept_length=max_concept_length)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                                             collate_fn=collate_fn)
    return dataloader


def parse_concept(input_concept):
    """
    parse the input concept into a list of concepts for evaluation, supported formats:
    concept: str:
        'man' -> ['man'] (generate an image of a man)
    concept: str:
        'man,young' -> ['man', 'young'] (generate an image of a young man)
    concept: list[str]:
        ['man', 'woman'] -> ['man'], ['woman'] (generate two images: a man, a woman)
    concept: list[str]:
        ['man,young', 'woman,young'] -> [['man','young'],['woman','young']] (generate two images: a young man, an old woman)

    The output of this function is directly fed to int_to_onehot and return a multi-hot vector which can be directly used by the model
    """

    def parse_concept_string(concept):
        assert isinstance(concept, str)
        concept = concept.split(',')
        concept = [x.strip() for x in concept]
        return concept

    if isinstance(input_concept, str):
        input_concept = parse_concept_string(input_concept)
        input_concept = [input_concept]

    elif isinstance(input_concept, list):
        input_concept = [parse_concept_string(x) for x in input_concept]

    else:
        raise ValueError(input_concept)

    return input_concept


def get_test_data(data_dir, given_prompt=None, given_concept=None, with_baseline=True, device='cuda',
                  max_concept_length=100):
    """
    data_dir: path to data file
    prompt: str
    concept: str or list[str]
    """
    concept_dict = json.load(open(os.path.join(data_dir, 'concept_dict.json'), 'r'))
    if not given_prompt or not given_concept:
        prompt, concept = json.load(open(os.path.join(data_dir, 'test.json'), 'r'))
    if given_prompt:
        prompt = given_prompt
    if given_concept:
        concept = given_concept

    concept = parse_concept(concept)
    print(f'eval with concept: {concept}')

    concept = [int_to_onehot([concept_dict[c_i] for c_i in c], max_concept_length).to(device).unsqueeze(0) for c in
               concept]
    if with_baseline:
        concept.insert(0, None)
    prompt = [prompt] * len(concept)
    return prompt, concept


def get_i2p_data(data_dir=None, given_prompt=None, given_concept=None, with_baseline=True, device='cuda',
                 max_concept_length=100):
    i2p = pd.read_csv("./i2p_benchmark.csv")
    if given_prompt:
        prompts = i2p[i2p.categories.apply(lambda x: given_prompt in x)].prompt.values.tolist()
    else:
        prompts = i2p.prompt.values.tolist()

    concept_label = [given_concept] if isinstance(given_concept, str) else given_concept
    concept_dict = json.load(open(os.path.join(data_dir, 'concept_dict.json'), 'r'))
    concept = [int_to_onehot(concept_dict[x], max_concept_length).to(device).unsqueeze(0) for x in concept_label]
    if with_baseline:
        concept.insert(0, None)
        concept_label.insert(0, 'none')

    inputs = []
    for prompt in prompts:
        for c_i, c in zip(concept, concept_label):
            inputs.append([prompt, c_i, c])
    return inputs