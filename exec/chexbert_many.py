import subprocess
import multiprocessing
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--starting_seed', type=int, default=42)
parser.add_argument('--num_seeds', type=int, default=50)
parser.add_argument('--pred_dir', type=str, default='train_scorer_0_2000')
parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0, 1, 2, 3, 4, 5, 6, 7])

args = parser.parse_args()

gpu_ids = args.gpu_ids

commands = []

for seed in range(args.starting_seed, args.starting_seed + args.num_seeds):
    cmd = f"python scripts/cxr/postprocess/chexbert.py  --jsonl /Mounts/rbg-storage1/snapshots/repg2/ap_and_pa/checkpoint-15000/preds/{args.pred_dir}/dev_sample_seed_{seed}.jsonl"
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
