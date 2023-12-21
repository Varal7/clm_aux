cd /Mounts/rbg-storage1/users/quach/cxr-project/rep


notify COMET_PROJECT_NAME="rep-2" CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_gpt.py --num_train_epochs 15 --metadata_base /Mounts/rbg-storage1/datasets/MIMIC/metadata_files/mimic_findings_all/ --save_name  report-to-text-mimic-all 

notify COMET_PROJECT_NAME="rep-2" CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_gpt.py --num_train_epochs 50 --metadata_base /Mounts/rbg-storage1/datasets/MIMIC/metadata_files/mimic_findings_100/ --save_name  report-to-text-mimic-100 

notify COMET_PROJECT_NAME="rep-2" CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_gpt.py --num_train_epochs 20 --metadata_base /Mounts/rbg-storage1/datasets/MIMIC/metadata_files/mimic_findings_1000/ --save_name  report-to-text-mimic-1000 

notify COMET_PROJECT_NAME="rep-2" CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_gpt.py --num_train_epochs 20 --metadata_base /Mounts/rbg-storage1/datasets/MIMIC/metadata_files/mimic_findings_10000/ --save_name  report-to-text-mimic-10000 

notify COMET_PROJECT_NAME="rep-2" CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_gpt.py --num_train_epochs 20 --metadata_base /Mounts/rbg-storage1/datasets/MIMIC/metadata_files/mimic_findings_100000/ --save_name  report-to-text-mimic-100000 

notify COMET_PROJECT_NAME="rep-2" CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_gpt.py --num_train_epochs 1 --metadata_base /Mounts/rbg-storage1/datasets/MIMIC/metadata_files/mimic_findings_all/ --save_name  report-to-text-mimic-all-one

notify COMET_PROJECT_NAME="rep-1" CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_gpt.py --num_train_epochs 100 --metadata_base /Mounts/rbg-storage1/users/quach/rep/data/mimic-10000/ --save_name  report-to-text-mimic-10000

notify COMET_PROJECT_NAME="rep-1"  python train_gpt.py --num_train_epochs 100 --metadata_base /Mounts/rbg-storage1/users/quach/rep/data/mimic-all/ --save_name  report-to-text-mimic-all --num_proc 32 

notify COMET_PROJECT_NAME="rep-1"  python train_gpt.py --num_train_epochs 100 --metadata_base /Mounts/rbg-storage1/users/quach/rep/data/mimic-all/ --save_name  report-to-text-mimic-all --num_proc 1

sudo docker run -v /Mounts/rbg-storage1/users/quach/rep/data:/data chexpert-labeler:latest  python label.py --reports_path /data/mimic-100/dev.csv --output_path /data/mimic-100/dev-gold.csv

sudo docker run -v /Mounts/rbg-storage1/snapshots/repg:/repg chexpert-labeler:latest  python label.py --reports_path /repg/report-to-text-mimic-10000/checkpoint-2000/predictions/dev.csv --output_path /repg/report-to-text-mimic-10000/checkpoint-2000/predictions/dev-annotated.csv

sudo docker run -v /Mounts/rbg-storage1/snapshots/repg:/repg chexpert-labeler:latest  python label.py --reports_path /repg/report-to-text-mimic-all/checkpoint-9500/predictions/dev.csv --output_path /repg/report-to-text-mimic-all/checkpoint-9500/predictions/dev-annotated.csv

notify COMET_PROJECT_NAME="rep-bio1" CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py --num_train_epochs 15 --metadata_base /Mounts/rbg-storage1/datasets/MIMIC/metadata_files/mimic_findings_all/ --save_name  report-to-text-mimic-all 

rsync -avh /Users/varal7/Projects/NeuraCrypt/ quach@rbgquanta1.csail.mit.edu:/Mounts/rbg-storage1/users/quach/Syfer --delete


# March 8
python scripts/preprocess/make_jsonl.py --split_dir /Mounts/rbg-storage1/datasets/MIMIC/splits 

python scripts/preprocess/resize_images.py --jsonl_dir /Mounts/rbg-storage1/datasets/MIMIC/splits/ap_pa_per_dicom_id --root_dir /Mounts/rbg-storage1/datasets/MIMIC/physionet.org/files/mimic-cxr-jpg --resized_dir /storage/quach/MIMIC/physionet.org/files/mimic-cxr-resized-224 --size 224

python scripts/preprocess/resize_images.py --jsonl_dir /Mounts/rbg-storage1/datasets/MIMIC/splits/ap_pa_per_dicom_id --root_dir /storage/quach/MIMIC/physionet.org/files/mimic-cxr-jpg --resized_dir /storage/quach/MIMIC/physionet.org/files/mimic-cxr-resized-224 --size 224


notify COMET_PROJECT_NAME="rep-3" CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python scripts/main.py \
  --data_dir /Mounts/rbg-storage1/datasets/MIMIC/splits/ap_pa_per_dicom_id \
  --mimic_root /storage/quach/MIMIC/physionet.org/files/mimic-cxr-resized-224 \
  --output_dir /Mounts/rbg-storage1/snapshots/repg/ap_and_pa \
  --num_workers 1 \
  --max_eval_samples 2000 \
  --num_train_epochs 10 \
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



CUDA_VISIBLE_DEVICES=4 python scripts/chexbert.py --jsonl /Mounts/rbg-storage1/snapshots/repg/ap_and_pa/checkpoint-9000/preds/validate_beam_5.jsonl

mkdir -p /storage/quach/MIMIC/physionet.org/files
rsync -avh rbgquanta1:/storage/quach/MIMIC/physionet.org/files/mimic-cxr-resized-224 /storage/quach/MIMIC/physionet.org/files


CUDA_VISIBLE_DEVICES=0 python scripts/nli.py  --start_x 400 --end_x 500 --base /Mounts/rbg-storage1/snapshots/repg2/ap_and_pa/checkpoint-15000/preds/many_part2
CUDA_VISIBLE_DEVICES=1 python scripts/nli.py  --start_x 500 --end_x 600 --base /Mounts/rbg-storage1/snapshots/repg2/ap_and_pa/checkpoint-15000/preds/many_part2

CUDA_VISIBLE_DEVICES=2 python scripts/nli.py  --start_x 600 --end_x 700 --base /Mounts/rbg-storage1/snapshots/repg2/ap_and_pa/checkpoint-15000/preds/many_part2

CUDA_VISIBLE_DEVICES=3 python scripts/nli.py  --start_x 700 --end_x 800 --base /Mounts/rbg-storage1/snapshots/repg2/ap_and_pa/checkpoint-15000/preds/many_part2

CUDA_VISIBLE_DEVICES=4 python scripts/nli.py  --start_x 800 --end_x 900 --base /Mounts/rbg-storage1/snapshots/repg2/ap_and_pa/checkpoint-15000/preds/many_part2

CUDA_VISIBLE_DEVICES=5 python scripts/nli.py  --start_x 900 --end_x 1000 --base /Mounts/rbg-storage1/snapshots/repg2/ap_and_pa/checkpoint-15000/preds/many_part2

CUDA_VISIBLE_DEVICES=6 python scripts/nli.py  --start_x 0 --end_x 100 --base /Mounts/rbg-storage1/snapshots/repg2/ap_and_pa/checkpoint-15000/preds/many_part2

CUDA_VISIBLE_DEVICES=7 python scripts/nli.py  --start_x 100 --end_x 200 --base /Mounts/rbg-storage1/snapshots/repg2/ap_and_pa/checkpoint-15000/preds/many_part2

CUDA_VISIBLE_DEVICES=0 COMET_PROJECT_NAME="rep-gnn-1" python scripts/gnn.py --save_dir run1

python scripts/dispatcher.py configs/gnn.json5 --script scripts/gnn.py  --dry_run

notify WANDB_DISABLED=true CUDA_VISIBLE_DEVICES=0 /data/rsg/nlp/quach/miniconda3/envs/pt113/bin/python -m pdb /Mounts/rbg-storage1/users/quach/cxr-project/rep/scripts/train.py --data_dir /Mounts/rbg-storage1/datasets/MIMIC/splits/ap_pa_per_dicom_id --mimic_root /storage/quach/MIMIC/physionet.org/files/mimic-cxr-resized-224 --output_dir /storage/quach/snapshots/repg2/debug --num_workers 1 --max_eval_samples 2000 --num_train_epochs 50 --remove_unused_columns False --do_train --do_eval --evaluation_strategy steps --eval_steps 500 --save_strategy steps --save_steps 500  --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --text_decoder_model gpt2 --use_fp16 True --fp16 True --load_best_model_at_end True --save_total_limit 2 --overwrite_output_dir

notify WANDB_DISABLED=true COMET_PROJECT_NAME="rep-3 "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 /data/rsg/nlp/quach/miniconda3/envs/pt113/bin/python /Mounts/rbg-storage1/users/quach/cxr-project/rep/scripts/train.py --data_dir /Mounts/rbg-storage1/datasets/MIMIC/splits/ap_pa_per_dicom_id --mimic_root /storage/quach/MIMIC/physionet.org/files/mimic-cxr-resized-224 --output_dir /storage/quach/snapshots/repg2/llama --num_workers 1 --max_eval_samples 2000 --num_train_epochs 50 --remove_unused_columns False --do_train --do_eval --evaluation_strategy steps --eval_steps 500 --save_strategy steps --save_steps 500  --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --text_decoder_model /storage/quach/weights/llama-hf-7b --use_fp16 True --fp16 True --load_best_model_at_end True --save_total_limit 2


rsync -avh rbgquanta1:/storage/quach/MIMIC/physionet.org/files/mimic-cxr-resized-224 .


CUDA_VISIBLE_DEVICES=0 COMET_PROJECT_NAME="rep-text-1" python scripts/ablation/text_only.py --save_dir run3

python scripts/dispatcher.py configs/text-only.json5 --script scripts/ablation/text_only.py  --dry_run

1,CUDA_VISIBLE_DEVICES=0 python scripts/qa/run_triviaqa.py --output_name "/Mounts/rbg-storage1/users/quach/outputs/llama/triviaqa/fewshot/0/dev_greedy.jsonl" --strategy greedy --num_generations 1 --few_shot 0
1,CUDA_VISIBLE_DEVICES=1 python scripts/qa/run_triviaqa.py --output_name "/Mounts/rbg-storage1/users/quach/outputs/llama/triviaqa/fewshot/1/dev_greedy.jsonl" --strategy greedy --num_generations 1 --few_shot 1
2,CUDA_VISIBLE_DEVICES=2 python scripts/qa/run_triviaqa.py --output_name "/Mounts/rbg-storage1/users/quach/outputs/llama/triviaqa/fewshot/2/dev_greedy.jsonl" --strategy greedy --num_generations 1 --few_shot 3
3,CUDA_VISIBLE_DEVICES=3 python scripts/qa/run_triviaqa.py --output_name "/Mounts/rbg-storage1/users/quach/outputs/llama/triviaqa/fewshot/4/dev_greedy.jsonl" --strategy greedy --num_generations 1 --few_shot 4
4,CUDA_VISIBLE_DEVICES=4 python scripts/qa/run_triviaqa.py --output_name "/Mounts/rbg-storage1/users/quach/outputs/llama/triviaqa/fewshot/8/dev_greedy.jsonl" --strategy greedy --num_generations 1 --few_shot 8
5,CUDA_VISIBLE_DEVICES=5 python scripts/qa/run_triviaqa.py --output_name "/Mounts/rbg-storage1/users/quach/outputs/llama/triviaqa/fewshot/16/dev_greedy.jsonl" --strategy greedy --num_generations 1 --few_shot 16
6,CUDA_VISIBLE_DEVICES=6 python scripts/qa/run_triviaqa.py --output_name "/Mounts/rbg-storage1/users/quach/outputs/llama/triviaqa/fewshot/32/dev_greedy.jsonl" --strategy greedy --num_generations 1 --few_shot 32
7,CUDA_VISIBLE_DEVICES=7 python scripts/qa/run_triviaqa.py --output_name "/Mounts/rbg-storage1/users/quach/outputs/llama/triviaqa/fewshot/32/dev_sample.jsonl" --strategy sample --num_generations 20 --few_shot 32 --batch_size 4


notify WANDB_DISABLED=true COMET_PROJECT_NAME="flan-summarization"  python scripts/summarization/train-t5flan.py \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --predict_with_generate True \
  --learning_rate 5e-5 \
  --num_train_epochs 5 \
  --do_train \
  --do_eval \
  --evaluation_strategy epoch \
  --save_strategy epoch \
  --save_total_limit 2 \
  --load_best_model_at_end True \
  --output_dir /storage/quach/snapshots/flan-summarization/apr16-base

notify WANDB_DISABLED=true COMET_PROJECT_NAME="t5-summarization"  python scripts/summarization/train-t5.py \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --predict_with_generate True \
  --learning_rate 5e-5 \
  --num_train_epochs 5 \
  --do_train \
  --do_eval \
  --evaluation_strategy epoch \
  --save_strategy epoch \
  --save_total_limit 2 \
  --load_best_model_at_end True \
  --output_dir /storage/quach/snapshots/t5-summarization/apr18-base

notify python exec/generate_many.py --output_name_dir train_scorer_0_2000 --starting_x 0 --max_predict_samples 2000  # ok
notify python exec/generate_many.py --output_name_dir calibration_2000_5000 --starting_x 2000 --max_predict_samples 3000  # ok
notify python exec/generate_many.py --output_name_dir valid_5000_8000 --starting_x 5000 --max_predict_samples 3000  # ok
notify python exec/generate_many.py --output_name_dir test_8000_13000 --starting_x 8000 --max_predict_samples 5000  # ok
notify python exec/generate_many.py --output_name_dir valid_scorer_13000_14000 --starting_x 13000 --max_predict_samples 1000 --gpu_ids 0 1 2 4 5 6 7  # ok

notify python exec/chexbert_many.py --pred_dir train_scorer_0_2000   # ok
notify python exec/chexbert_many.py --pred_dir calibration_2000_5000 --gpu_ids 1 2 4 5 7 # ok
notify python exec/chexbert_many.py --pred_dir valid_5000_8000  #  ok
notify python exec/chexbert_many.py --pred_dir test_8000_13000  # ok
notify python exec/chexbert_many.py --pred_dir valid_scorer_13000_14000 --gpu_ids 1 2 3 4 5 6 7 # ok


notify python exec/nli_many.py --starting_seed 42 --num_seeds 20 --pred_dir train_scorer_0_2000 --num_x 2000 --gpu_ids 1 2 3 4 5 6 7  # ok
notify python exec/nli_many.py --starting_seed 42 --num_seeds 20 --pred_dir calibration_2000_5000 --num_x 3000 --gpu_ids 1 2 4 5 7 # ok
notify python exec/nli_many.py --starting_seed 42 --num_seeds 20 --pred_dir valid_5000_8000 --num_x 3000 --gpu_ids 1 2 4 5 7 #  ok
notify python exec/nli_many.py --starting_seed 42 --num_seeds 20 --pred_dir test_8000_13000 --num_x 5000  --gpu_ids 1 2 4 5 7 # ok
notify python exec/nli_many.py --starting_seed 42 --num_seeds 20 --pred_dir valid_scorer_13000_14000 --num_x 1000  # ok

notify python exec/nli_many.py --pred_dir train_scorer_0_2000 --num_x 2000 --gpu_ids 1 2 3 4 5 6 7  # ok
notify python exec/nli_many.py --pred_dir calibration_2000_5000 --num_x 3000 --gpu_ids 1 2 4 5 7 
notify python exec/nli_many.py --pred_dir valid_5000_8000 --num_x 3000
notify python exec/nli_many.py --pred_dir test_8000_13000 --num_x 5000  
notify python exec/nli_many.py --pred_dir valid_scorer_13000_14000 --num_x 1000  # ok

notify python scripts/dispatcher.py configs/gnn.json5 --script scripts/cxr/postprocess/train_gnn.py  # ok

python scripts/dispatcher.py configs/text-only.json5 --script scripts/cxr/postprocess/train_text_only.py  #  ok

notify python scripts/cxr/postprocess/predict_gnn.py --base /Mounts/rbg-storage1/snapshots/repg2/ap_and_pa/checkpoint-15000/preds/train_scorer_0_2000 --num_x 2000 --name apr26-20seeds-f0f49f2bd9e9bb124e97d1aa9713f417
notify python scripts/cxr/postprocess/predict_gnn.py --base /Mounts/rbg-storage1/snapshots/repg2/ap_and_pa/checkpoint-15000/preds/calibration_2000_5000 --num_x 3000 --name apr26-20seeds-f0f49f2bd9e9bb124e97d1aa9713f417
notify python scripts/cxr/postprocess/predict_gnn.py --base /Mounts/rbg-storage1/snapshots/repg2/ap_and_pa/checkpoint-15000/preds/valid_5000_8000 --num_x 3000 --name apr26-20seeds-f0f49f2bd9e9bb124e97d1aa9713f417
notify python scripts/cxr/postprocess/predict_gnn.py --base /Mounts/rbg-storage1/snapshots/repg2/ap_and_pa/checkpoint-15000/preds/test_8000_13000 --num_x 5000 --name apr26-20seeds-f0f49f2bd9e9bb124e97d1aa9713f417
notify python scripts/cxr/postprocess/predict_gnn.py --base /Mounts/rbg-storage1/snapshots/repg2/ap_and_pa/checkpoint-15000/preds/valid_scorer_13000_14000 --num_x 1000 --name apr26-20seeds-f0f49f2bd9e9bb124e97d1aa9713f417


notify python scripts/dispatcher.py configs/sentence-text-only.json5 --script scripts/cxr/postprocess/train_sentence_text_only.py 

notify python scripts/dispatcher.py configs/image-sentence.json5 --script scripts/cxr/postprocess/train_image_sentence.py 


python scripts/cxr/postprocess/train_image_sentence.py --cache_only

notify python scripts/dispatcher.py configs/overfit-image-sentence.json5 --script scripts/cxr/postprocess/train_image_sentence.py 


python scripts/cxr/postprocess/train_image_sentence.py --base /Mounts/rbg-storage1/snapshots/repg2/ap_and_pa/checkpoint-15000/preds/train_scorer_0_2000 --test_base /Mounts/rbg-storage1/snapshots/repg2/ap_and_pa/checkpoint-15000/preds/valid_scorer_13000_14000 --start_seed 42 --end_seed 62 --prefix dev_sample_seed --dropout 0 --hidden_dim 512 --learning_rate 1e-05 --weight_decay 0.001 --pretrained_bert emilyalsentzer/Bio_ClinicalBERT --num_epochs 5 --batch_size 32 --accumulate_grad_batches 1 --save_dir /storage/quach/snapshots/repg2/image-sentence --workspace varal7 --project_name rep-image-sentence --comet_tags may1-single- --results_path /Mounts/rbg-storage1/logs/repg2/may1-20seeds-single.results

notify python scripts/cxr/postprocess/predict_text_only.py --base /Mounts/rbg-storage1/snapshots/repg2/ap_and_pa/checkpoint-15000/preds/train_scorer_0_2000 --num_x 2000 --name apr26-20seeds-ab1d03c3d71487003e792efb62916061
notify python scripts/cxr/postprocess/predict_text_only.py --base /Mounts/rbg-storage1/snapshots/repg2/ap_and_pa/checkpoint-15000/preds/calibration_2000_5000 --num_x 3000 --name apr26-20seeds-ab1d03c3d71487003e792efb62916061
notify python scripts/cxr/postprocess/predict_text_only.py --base /Mounts/rbg-storage1/snapshots/repg2/ap_and_pa/checkpoint-15000/preds/valid_5000_8000 --num_x 3000 --name apr26-20seeds-ab1d03c3d71487003e792efb62916061
notify python scripts/cxr/postprocess/predict_text_only.py --base /Mounts/rbg-storage1/snapshots/repg2/ap_and_pa/checkpoint-15000/preds/test_8000_13000 --num_x 5000 --name apr26-20seeds-ab1d03c3d71487003e792efb62916061
notify python scripts/cxr/postprocess/predict_text_only.py --base /Mounts/rbg-storage1/snapshots/repg2/ap_and_pa/checkpoint-15000/preds/valid_scorer_13000_14000 --num_x 1000 --name apr26-20seeds-ab1d03c3d71487003e792efb62916061

python scripts/cxr/postprocess/gather_scores.py --scores text-only_42_62/text-only-apr26-20seeds-ab1d03c3d71487003e792efb62916061.pt --score_name text

python scripts/cxr/postprocess/train_image_report.py

notify python scripts/dispatcher.py configs/image-report.json5 --script scripts/cxr/postprocess/train_image_report.py 



notify python scripts/cxr/postprocess/predict_image_report.py --base /Mounts/rbg-storage1/snapshots/repg2/ap_and_pa/checkpoint-15000/preds/train_scorer_0_2000 --num_x 2000 --name may2-20seeds-67ae9a0d66dfbec7893eb2a47ac09972
notify python scripts/cxr/postprocess/predict_image_report.py --base /Mounts/rbg-storage1/snapshots/repg2/ap_and_pa/checkpoint-15000/preds/calibration_2000_5000 --num_x 3000 --name may2-20seeds-67ae9a0d66dfbec7893eb2a47ac09972
notify python scripts/cxr/postprocess/predict_image_report.py --base /Mounts/rbg-storage1/snapshots/repg2/ap_and_pa/checkpoint-15000/preds/valid_5000_8000 --num_x 3000 --name may2-20seeds-67ae9a0d66dfbec7893eb2a47ac09972
notify python scripts/cxr/postprocess/predict_image_report.py --base /Mounts/rbg-storage1/snapshots/repg2/ap_and_pa/checkpoint-15000/preds/test_8000_13000 --num_x 5000 --name may2-20seeds-67ae9a0d66dfbec7893eb2a47ac09972
notify python scripts/cxr/postprocess/predict_image_report.py --base /Mounts/rbg-storage1/snapshots/repg2/ap_and_pa/checkpoint-15000/preds/valid_scorer_13000_14000 --num_x 1000 --name may2-20seeds-67ae9a0d66dfbec7893eb2a47ac09972

python scripts/cxr/postprocess/gather_scores.py --scores image-report_42_62/image-report-may2-20seeds-67ae9a0d66dfbec7893eb2a47ac09972.pt --score_name image-report


CUDA_VISIBLE_DEVICES=0 notify python scripts/cxr/postprocess/predict_image_sentence.py --base /Mounts/rbg-storage1/snapshots/repg2/ap_and_pa/checkpoint-15000/preds/train_scorer_0_2000 --num_x 2000
CUDA_VISIBLE_DEVICES=1 notify python scripts/cxr/postprocess/predict_image_sentence.py --base /Mounts/rbg-storage1/snapshots/repg2/ap_and_pa/checkpoint-15000/preds/calibration_2000_5000 --num_x 3000
CUDA_VISIBLE_DEVICES=2 notify python scripts/cxr/postprocess/predict_image_sentence.py --base /Mounts/rbg-storage1/snapshots/repg2/ap_and_pa/checkpoint-15000/preds/valid_5000_8000 --num_x 3000
CUDA_VISIBLE_DEVICES=3 notify python scripts/cxr/postprocess/predict_image_sentence.py --base /Mounts/rbg-storage1/snapshots/repg2/ap_and_pa/checkpoint-15000/preds/test_8000_13000 --num_x 5000
CUDA_VISIBLE_DEVICES=4 notify python scripts/cxr/postprocess/predict_image_sentence.py --base /Mounts/rbg-storage1/snapshots/repg2/ap_and_pa/checkpoint-15000/preds/valid_scorer_13000_14000 --num_x 1000



notify python scripts/dispatcher.py configs/image-sentence.json5 --script scripts/cxr/postprocess/train_image_sentence.py 

python scripts/qa/run_triviaqa_self_eval.py  --filename /Mounts/rbg-storage1/users/quach/outputs/llama/triviaqa/fewshot/32/small_2000_3000/dev_sample.jsonl
