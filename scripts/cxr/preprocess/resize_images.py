import argparse
import json
import os
from p_tqdm import p_map

from torchvision.datasets.folder import default_loader
from torchvision.transforms import Resize
from torchvision.transforms.functional import InterpolationMode

IMAGE_COLUMN = "image_path"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resize images")
    parser.add_argument("--jsonl_dir", help="Directory containing the jsonl files")
    parser.add_argument("--root_dir", help="Input directory")
    parser.add_argument("--resized_dir", help="Output directory")
    parser.add_argument("--size", help="Size of the output images", type=int)
    args = parser.parse_args()

    r = Resize([args.size], interpolation=InterpolationMode.BICUBIC)

    def process_example(example):
        image = default_loader(os.path.join(args.root_dir, example[IMAGE_COLUMN]))
        image = r(image)
        path = os.path.join(args.resized_dir, example[IMAGE_COLUMN])
        os.makedirs(os.path.dirname(path), exist_ok=True)
        image.save(path)

    # Load the dataset
    for split in ['train', 'dev', 'validate', 'test']:
        with open(os.path.join(args.jsonl_dir, f"{split}.jsonl"), "r") as f:
            examples = [json.loads(line) for line in f]
            p_map(process_example, examples)
