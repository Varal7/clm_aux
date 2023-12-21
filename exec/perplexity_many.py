import subprocess
import multiprocessing
import os

gpu_ids = [0, 1, 2, 3, 4, 5, 6, 7]


FOLDERS = [
        'train_scorer_0_2000',
        'calibration_2000_5000',
        'valid_5000_8000',
        'test_8000_13000',
        'valid_scorer_13000_14000',
]

commands = []

for folder in FOLDERS:
    for seed in range(42, 62):
        cmd = f"""python scripts/cxr/postprocess/perplexity.py \
    --data_dir /Mounts/rbg-storage1/datasets/MIMIC/splits/ap_pa_per_dicom_id \
    --mimic_root /storage/quach/MIMIC/physionet.org/files/mimic-cxr-resized-224 \
    --checkpoint /Mounts/rbg-storage1/snapshots/repg2/ap_and_pa/checkpoint-15000 \
    --batch_size 16 \
    --num_workers 1 \
    --jsonl /Mounts/rbg-storage1/snapshots/repg2/ap_and_pa/checkpoint-15000/preds/{folder}/dev_sample_seed_{seed}.jsonl
        """
        commands.append(cmd)


def run_command_on_gpu(command, gpu_id):
    filename = command.split()[-1]
    output_name = filename.replace(".jsonl", "_normalized_likelihoods.json")
    if os.path.exists(output_name):
        print(f"Skipping {output_name}")
        return

    print(f"[{gpu_id}]\n{command}")
    subprocess.call(f"CUDA_VISIBLE_DEVICES={gpu_id} " + command, shell=True)

if __name__ == '__main__':
    pool = multiprocessing.Pool(processes=len(gpu_ids))
    for i, command in enumerate(commands):
        gpu_id = gpu_ids[i % len(gpu_ids)]
        pool.apply_async(run_command_on_gpu, args=(command, gpu_id))
    pool.close()
    pool.join()
