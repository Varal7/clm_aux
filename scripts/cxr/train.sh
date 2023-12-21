notify COMET_PROJECT_NAME="rep-3" CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python scripts/cxr/train.py \
  --data_dir /Mounts/rbg-storage1/datasets/MIMIC/splits/ap_pa_per_dicom_id \
  --mimic_root /storage/quach/MIMIC/physionet.org/files/mimic-cxr-resized-224 \
  --output_dir /Mounts/rbg-storage1/snapshots/repg2/ap_and_pa \
  --num_workers 1 \
  --max_eval_samples 2000 \
  --num_train_epochs 50 \
  --remove_unused_columns False \
  --do_train \
  --do_eval \
  --evaluation_strategy steps \
  --eval_steps 500 \
  --save_strategy steps \
  --save_steps 500 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 16
  # --max_train_samples 1000 \
  # --overwrite_output_dir \
