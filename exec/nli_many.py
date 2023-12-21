import subprocess
import multiprocessing
import argparse
import math

parser = argparse.ArgumentParser()
parser.add_argument('--pred_dir', type=str, default='train_scorer_0_2000')
parser.add_argument('--num_x', type=int, required=True)
parser.add_argument('--starting_seed', type=int, default=42)
parser.add_argument('--num_seeds', type=int, default=50)
parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0, 1, 2, 3, 4, 5, 6, 7])

args = parser.parse_args()

gpu_ids = args.gpu_ids

commands = []

num_x = args.num_x
num_gpus = len(gpu_ids)

num_x_per_gpu = math.ceil(num_x / num_gpus)


start_seed = args.starting_seed
end_seed = start_seed + args.num_seeds

for start_x in range(0, num_x, num_x_per_gpu):
    end_x = min(start_x + num_x_per_gpu, num_x)
    cmd = f"python scripts/cxr/postprocess/nli.py  --base /Mounts/rbg-storage1/snapshots/repg2/ap_and_pa/checkpoint-15000/preds/{args.pred_dir} --start_x {start_x} --end_x {end_x} --start_seed {start_seed} --end_seed {end_seed}"
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
