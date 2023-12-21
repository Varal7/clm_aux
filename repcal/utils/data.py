import os
import pandas as pd
from tqdm.auto import tqdm
from nltk.tokenize import sent_tokenize
import json

#  from repcal.utils.legacy_data import (\
#          get_predictions, \
#          remove_prompt, \
#          to_sentences, \
#          get_full_df, \
#      )

ANNOTATION_FILENAME = "/Mounts/rbg-storage1/datasets/MIMIC/physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-chexpert.csv"
METADATA_FILENAME = "/Mounts/rbg-storage1/datasets/MIMIC/physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-metadata.csv"
SPLITS_FILENAME = "/Mounts/rbg-storage1/datasets/MIMIC/physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-split.csv.gz"

JPG_ROOT = "/Mounts/rbg-storage1/datasets/MIMIC/physionet.org/files/mimic-cxr-jpg/"

TEXT_ROOT = "/Mounts/rbg-storage1/datasets/MIMIC/physionet.org/files/mimic-cxr/"


# Data preprocessing

def get_image_path_from_row(row, dicom_id=None, prefix=None, extension="jpg"):
    subject_id = str(row['subject_id'])
    if dicom_id is None:
        dicom_id = row['dicom_id']
    components = ["2.0.0/files/", f"p{subject_id[:2]}", f"p{subject_id}", f"s{row['study_id']}", f"{dicom_id}.{extension}"]
    if prefix is not None:
        components = [prefix] + components
    return os.path.join(*components)


def get_text_path_from_row(row, prefix=TEXT_ROOT):
    subject_id = str(row['subject_id'])
    components = ["2.0.0/files/", f"p{subject_id[:2]}", f"p{subject_id}", f"s{row['study_id']}.txt"]
    if prefix is not None:
        components = [prefix] + components
    return os.path.join(*components)

def get_merged_df(splits_filename=SPLITS_FILENAME, annotation_filename=ANNOTATION_FILENAME, metadata_filename=METADATA_FILENAME):
    splits = pd.read_csv(splits_filename)
    metadata = pd.read_csv(metadata_filename)
    chexpert = pd.read_csv(annotation_filename)

    r = splits.merge(right=metadata, how="left", left_on=['dicom_id'], right_on=['dicom_id'], suffixes = ("", "_"))\
              .merge(right=chexpert, how="left", left_on=['study_id'], right_on=['study_id'], suffixes = ("", "_"))

    for col in set(r.columns):
        if col.endswith("_"):
            del r[col]

    return r


# Report preprocessing

first_split = "FINDINGS AND IMPRESSION:"
second_split = "FINDINGS:"
third_split = "IMPRESSION:"
fourth_split = "FINDINGS"

FINDINGS_STARTS = [first_split, second_split, third_split, fourth_split]
FINAL_REPORT_START = " " * 33 + "FINAL REPORT"

def has_findings(report):
    return any([keyword in report for keyword in FINDINGS_STARTS])

def get_before_findings(t):
    """We want to get the earliest beginning of a findings paragraph"""
    if first_split in t:
        return t.split(first_split)[0] + first_split
    if second_split in t:
        return t.split(second_split)[0] + second_split
    if third_split in t:
        return t.split(third_split)[0] + third_split
    return t.split(fourth_split)[0] + fourth_split

def get_after_findings(t):
    if first_split in t:
        return first_split + t.split(first_split)[1]
    if second_split in t:
        return second_split + t.split(second_split)[1]
    if third_split in t:
        return third_split + t.split(third_split)[1]
    return t.split(fourth_split)[0] + fourth_split

def minibatch(data, bs):
    """Takes a list of dicts ans returns a dict of lists by minibatch"""
    for i in tqdm(range(0, len(data), bs), ncols=80):
        batch = data[i : min(len(data), i + bs)]
        yield {k: [dic[k] for dic in batch] for k in batch[0]}

def to_sentences(text):
    return sent_tokenize(text.replace('\n', '').strip())

def get_sentences(start_seed, end_seed, base, prefix="dev_sample_seed"):
    """Returns a list of list of list of sentences.
        The first level is for each report
        The second level is for each seed
        The third level is for each sentence.
    """
    def read_studies(seed, base):
        with open(os.path.join(base, f"{prefix}_{seed}.jsonl")) as f:
            return [json.loads(l) for l in f]

    def extract_sentences(studies):
        return [to_sentences(r['generated'][len(r['prompt']):]) for r in (studies)]

    num_seeds = end_seed - start_seed
    sentences_by_seed = {j: extract_sentences(read_studies(start_seed + j, base)) for j in tqdm(range(num_seeds))}

    num_x = len(sentences_by_seed[0])

    sentences_by_x = [
        [sentences_by_seed[j][i] for j in range(num_seeds)]
        for i in range(num_x)
    ]

    return sentences_by_x
