import gzip
import json
import os
from tqdm.auto import tqdm

from repcal.utils.nq import simplify_nq_example

def preprocess_nq(input_file, output_file):
    with gzip.open(input_file, 'rt') as fin, open(output_file, "a") as fout:
        for line in tqdm(fin):
            example = json.loads(line)
            example = simplify_nq_example(example)
            output = {
                'id': example['example_id'],
                'question': example['question_text'],
                'answer': []
            }
            for annotation in example['annotations']:
                if annotation['yes_no_answer'] != 'NONE':
                    output['answer'].append(annotation['yes_no_answer'])

                answers = annotation['short_answers']

                if len(answers) == 0:
                    continue

                components = []
                for answer in answers:
                    answer = " ".join(example["document_text"].split(" ")[answer['start_token']:answer['end_token']])
                    components.append(answer)
                output['answer'].append(" ".join(components))


            fout.write(json.dumps(output) + "\n")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('input_directory', type=str)
    parser.add_argument('output_name', type=str)
    args = parser.parse_args()


    input_files = [os.path.join(args.input_directory, f) for f in os.listdir(args.input_directory) if f.endswith('.jsonl.gz')]
    input_files.sort()

    output_file = os.path.join(args.input_directory, args.output_name)
    if os.path.exists(output_file):
        os.remove(output_file)

    for input_file in input_files:
        preprocess_nq(input_file, output_file)

