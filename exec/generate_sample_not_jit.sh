notify CUDA_VISIBLE_DEVICES=4 python scripts/generate.py \
  --data_dir /Mounts/rbg-storage1/datasets/MIMIC/splits/ap_pa_per_dicom_id \
  --mimic_root /storage/quach/MIMIC/physionet.org/files/mimic-cxr-resized-224 \
  --checkpoint /Mounts/rbg-storage1/snapshots/repg2/ap_and_pa/checkpoint-15000 \
  --strategy sample \
  --predict_split validate \
  --image_processor_jit False \
  --output_name validate_sample_not_jit.jsonl \
  --num_workers 1

notify CUDA_VISIBLE_DEVICES=4 python scripts/chexbert.py \
  --jsonl /Mounts/rbg-storage1/snapshots/repg2/ap_and_pa/checkpoint-15000/preds/validate_sample_not_jit.jsonl


