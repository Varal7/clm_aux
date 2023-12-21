# notify CUDA_VISIBLE_DEVICES=6 python scripts/generate.py \
  # --data_dir /Mounts/rbg-storage1/datasets/MIMIC/splits/ap_pa_per_dicom_id \
  # --mimic_root /storage/quach/MIMIC/physionet.org/files/mimic-cxr-resized-224 \
  # --checkpoint /Mounts/rbg-storage1/snapshots/repg2/ap_and_pa/checkpoint-15000 \
  # --strategy sample \
  # --predict_split dev \
  # --output_name ann/dev_sample_seed_41.jsonl \
  # --seed 41 \
  # --max_predict_samples 1000 \
  # --num_workers 1

# notify CUDA_VISIBLE_DEVICES=6 python scripts/chexbert.py \
  # --jsonl /Mounts/rbg-storage1/snapshots/repg2/ap_and_pa/checkpoint-15000/preds/validate_sample.jsonl
  
CUDA_VISIBLE_DEVICES=0 python scripts/generate.py  --data_dir /Mounts/rbg-storage1/datasets/MIMIC/splits/ap_pa_per_dicom_id  --mimic_root /storage/quach/MIMIC/physionet.org/files/mimic-cxr-resized-224 --checkpoint /Mounts/rbg-storage1/snapshots/repg2/ap_and_pa/checkpoint-15000 --strategy sample --predict_split dev --seed 42 --output_name many_part2/dev_sample_seed_42.jsonl --num_workers 1 --max_predict_samples 1000 --starting_x 1000

