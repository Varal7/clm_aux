import os
import json
import nltk
import pandas as pd

# Legacy postprocessing code

DEFAULT_CHECKPOINT = "/Mounts/rbg-storage1/snapshots/repg2/report-to-text-mimic-all/best"
DEFAULT_FILENAME = os.path.join(DEFAULT_CHECKPOINT, "preds", "dev_beam_5.jsonl")
ANNOTATION_FILENAME = "/Mounts/rbg-storage1/datasets/MIMIC/physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-chexpert.csv"
METADATA_FILENAME = "/Mounts/rbg-storage1/datasets/MIMIC/physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-metadata.csv"

sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


def get_predictions(filename=None):
    if filename is None:
        filename = DEFAULT_FILENAME

    with open(filename) as f:
        data = list(map(json.loads, f.readlines()))

    data = [remove_prompt(x) for x in data]

    for sample in data:
        sample['ori_sentences'] =  []
        for sen in to_sentences(sample['ori']):
            try:
                float(sen)
            except ValueError:
                sample['ori_sentences'].append(sen)

        sample['gen_sentences'] =  []
        for sen in to_sentences(sample['gen']):
            try:
                float(sen)
            except ValueError:
                sample['gen_sentences'].append(sen)

    return data

def remove_prompt(sample):
    n = len(sample['prompt'])
    sample['ori'] = sample['report'][n:]
    sample['gen'] = sample['beam_5'][n:]
    return sample

def to_sentences(report):
    return sent_tokenizer.tokenize(report.replace('\n', '').strip())

def get_predictions_df(filename=None):
    return pd.DataFrame.from_records(get_predictions(filename))

def merge_df_with_annotations(df, annotation_filename=None):
    if annotation_filename is None:
        annotation_filename = ANNOTATION_FILENAME

    annotations = pd.read_csv(annotation_filename)
    return df.merge(right=annotations, how="left", left_on=['study_id', 'subject_id'], right_on=['study_id', 'subject_id'])

def merge_df_with_metadata(df, metadata_filename=None):
    if metadata_filename is None:
        metadata_filename = METADATA_FILENAME

    meta = pd.read_csv(metadata_filename)
    df = df.copy()
    df['dicom_id'] = df['path'].apply(lambda x: os.path.basename(x)[:-4])
    return df.merge(right=meta, how="left", left_on='dicom_id', right_on='dicom_id')

def get_full_df(filename=None, annotation_filename=None, metadata_filename=None):
    df = get_predictions_df(filename)
    df = merge_df_with_annotations(df, annotation_filename)
    df = merge_df_with_metadata(df, metadata_filename)
    return df

