import os
from tqdm.auto import tqdm
import json
import argparse
from repcal.utils.data import ANNOTATION_FILENAME, FINAL_REPORT_START, METADATA_FILENAME, get_merged_df, get_image_path_from_row

DEFAULT_QUERY = 'ViewPosition == "PA" or ViewPosition == "AP"'

def make_jsonl_with_query(query, split_dir, out_dir, annotation_filename, metadata_filename):
    split_filename = os.path.join(split_dir, "splits_with_reports.csv")
    df = get_merged_df(split_filename, annotation_filename=annotation_filename, metadata_filename=metadata_filename)

    df = df.query(query).groupby("study_id").first().reset_index()
    starts_with_final_report = df['report'].str.startswith(FINAL_REPORT_START)
    df = df[starts_with_final_report]

    for split in ['train', 'dev', 'validate', 'test']:
        split_df = df.query(f"split == '{split}'")

        with open(os.path.join(out_dir, f"{split}.jsonl"), "w") as f:
            for row in tqdm(split_df.to_dict("records")):
                image_path = get_image_path_from_row(row)
                row['image_path'] = image_path
                f.write(json.dumps(row) + "\n")


def make_jsonl_per_dicom_id(split_dir, out_dir, annotation_filename, metadata_filename):
    split_filename = os.path.join(split_dir, "splits_with_reports.csv")
    df = get_merged_df(split_filename, annotation_filename=annotation_filename, metadata_filename=metadata_filename)

    for split in ['train', 'dev', 'validate', 'test']:
        split_df = df.query(f"split == '{split}'")

        with open(os.path.join(out_dir, f"{split}.jsonl"), "w") as f:
            for row in tqdm(split_df.to_dict("records")):
                image_path = get_image_path_from_row(row)
                row['image_path'] = image_path
                f.write(json.dumps(row) + "\n")

def make_jsonl_per_study_id(split_dir, out_dir, annotation_filename, metadata_filename):
    split_filename = os.path.join(split_dir, "splits_with_reports.csv")
    df = get_merged_df(split_filename, annotation_filename=annotation_filename, metadata_filename=metadata_filename)

    for split in ['train', 'dev', 'validate', 'test']:
        sdf = df.query(f"split == '{split}'")

        split_df = sdf.groupby("study_id").first().reset_index()
        dicom_ids = sdf.groupby("study_id")["dicom_id"].apply(list).to_dict()

        with open(os.path.join(out_dir, f"{split}.jsonl"), "w") as f:
            for row in tqdm(split_df.to_dict("records")):
                row['image_path'] = []
                row['dicom_ids'] = dicom_ids[row['study_id']]
                for dicom_id in dicom_ids[row['study_id']]:
                    image_path = get_image_path_from_row(row, dicom_id=dicom_id)
                    row['image_path'].append(image_path)
                f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--split_dir', type=str, required=True)
    parser.add_argument('--annotation_filename', type=str, default=ANNOTATION_FILENAME)
    parser.add_argument('--metadata_filename', type=str, default=METADATA_FILENAME)
    parser.add_argument('--query', type=str, default=DEFAULT_QUERY)

    args = parser.parse_args()

    per_dicom_id_dir = os.path.join(args.split_dir, "ap_pa_per_dicom_id")
    os.makedirs(per_dicom_id_dir, exist_ok=True)
    make_jsonl_with_query(args.query, args.split_dir, per_dicom_id_dir, args.annotation_filename, args.metadata_filename)

    #  per_dicom_id_dir = os.path.join(args.split_dir, "per_dicom_id")
    #  os.makedirs(per_dicom_id_dir, exist_ok=True)
    #  make_jsonl_per_dicom_id(args.split_dir, per_dicom_id_dir, args.annotation_filename, args.metadata_filename)

    #  per_study_id_dir = os.path.join(args.split_dir, "per_study_id")
    #  os.makedirs(per_study_id_dir, exist_ok=True)
    #  make_jsonl_per_study_id(args.split_dir, per_study_id_dir, args.annotation_filename, args.metadata_filename)
