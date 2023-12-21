notify CUDA_VISIBLE_DEVICES=6 python scripts/cxr/postprocess/perplexity.py \
  --data_dir /Mounts/rbg-storage1/datasets/MIMIC/splits/ap_pa_per_dicom_id \
  --mimic_root /storage/quach/MIMIC/physionet.org/files/mimic-cxr-resized-224 \
  --checkpoint /Mounts/rbg-storage1/snapshots/repg2/ap_and_pa/checkpoint-15000 \
  --jsonl /Mounts/rbg-storage1/snapshots/repg2/ap_and_pa/checkpoint-15000/preds/valid_scorer_13000_14000/dev_sample_seed_42.jsonl \
  --batch_size 32 \
  --num_workers 1

