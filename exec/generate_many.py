import subprocess
import multiprocessing
import sys

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--starting_seed', type=int, default=42)
parser.add_argument('--num_seeds', type=int, default=50)
parser.add_argument('--output_name_dir', type=str, default='train_scorer_0_2000')
parser.add_argument('--starting_x', type=int, default=0)
parser.add_argument('--max_predict_samples', type=int, default=2000)
parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0, 1, 2, 3, 4, 5, 6, 7])

args = parser.parse_args()

gpu_ids = args.gpu_ids

commands = []
for seed in range(args.starting_seed, args.starting_seed + args.num_seeds):
    cmd = f"python scripts/cxr/postprocess/generate.py  --data_dir /Mounts/rbg-storage1/datasets/MIMIC/splits/ap_pa_per_dicom_id  --mimic_root /storage/quach/MIMIC/physionet.org/files/mimic-cxr-resized-224 --checkpoint /Mounts/rbg-storage1/snapshots/repg2/ap_and_pa/checkpoint-15000 --strategy sample --predict_split dev --seed {seed} --output_name {args.output_name_dir}/dev_sample_seed_{seed}.jsonl --num_workers 1 --max_predict_samples {args.max_predict_samples} --starting_x {args.starting_x}"
    commands.append(cmd)


def run_command_on_gpu(command, gpu_id):
    print(f"[{gpu_id}]\n{command}")
    subprocess.call(f"CUDA_VISIBLE_DEVICES={gpu_id} " + command, shell=True)

if __name__ == '__main__':
    pool = multiprocessing.Pool(processes=len(gpu_ids))
    for i, command in enumerate(commands):
        gpu_id = gpu_ids[i % len(gpu_ids)]
        pool.apply_async(run_command_on_gpu, args=(command, gpu_id))
    pool.close()
    pool.join()

