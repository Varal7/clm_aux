import torch
import math
from tqdm.auto import tqdm
import os
import sys
import json
from dataclasses import dataclass, field
import logging


from transformers import LlamaForCausalLM, LlamaTokenizer

from transformers import (
    HfArgumentParser,
)

LLAMA_PATH = "/storage/quach/weights/llama-hf-13b"
FILENAME = "/Mounts/rbg-storage1/users/quach/outputs/llama/triviaqa/fewshot/32/small_2000_3000/dev_sample.jsonl"

logger = logging.getLogger(__name__)

prompt = """Answer these questions

Q: Who appears on the reverse of the current Bank of England £10 note?
A: Adam Smith
Incorrect

Q: The Archibald Fountain is in which Australian city?
A: Sydney
Correct

Q: The Malagasy people form the main ethnic group of which country?
A: Madagascar
Correct

Q: On July 24, 1911, future US. Senator (R, Connecticut) Hiram Bingham III “discovered” what Peruvian city?
A: Machu Picchu
Correct

Q: Who won best British group and best British album at the 2000 Brit Awards
A: Robbie Williams
Incorrect

Q: Which body of water was previously called the Euxine, after early colonisation of its southern coastline by Greeks, derived from their word for 'hospitable'?
A: Black Sea
Correct

Q: The name of which chemical element takes its name from the Greek for light-bringing?
A: Mercury
Incorrect

Q: What happens at the 'Tower of London' at 9.40 every evening?
A: The Beefeater raises the national flag
Incorrect

Q: Who had a U.K. No 1 in the 80's with Goody Two Shoes
A: Adam Ant
Correct

Q: The singer 'Sting' featured on a track of which 1985 Dire Straits album?
A: Brothers In Arms
Correct

Q: "In 1976 or 1977, John Carpenter and Debra Hill began drafting a story titled ""The Babysitter Murders"", which eventually became what film?"
A: HALLOWEEN
Correct

Q: Which Greek philosopher taught at the Lyceum?
A: Aristotle
Correct

Q: Which is the only country on mainland Europe to be in the same time zone as the UK?
A: Portugal
Correct

Q: What element is often called quicksilver?
A: Mercury
Correct

Q: The first ten series of which long running TV drama featured the character Claude Jeremiah Greengrass?
A: Yes, Minister
Incorrect

Q: Which element has the atomic number 1?
A: Hydrogen
Correct

Q: Who plays the title character in the recent BBC TV series 'Sherlock'?
A: Benedict Cumberbatch
Correct

Q: What is the name of Tony and Cherie Blair's daughter?
A: Nancy
Incorrect

Q: Which is the smallest marine mammal?
A: Walrus
Incorrect

Q: Which group recorded the 90s albums 'Nevermind' and 'In Utero'?
A: Nirvana
Correct

Q: By population, which is the second biggest city in France?
A: Lyon
Correct

Q: According to the Bible, who was the high priest of Judea at the time of Christ's arrest and execution?
A: Caiaphas
Correct

Q: In 1938 Orson Welles, who was 22 at the time, wrote, produced, and narrated a radio play adaptation of what work, the US broadcast of which sparked widespread upheaval and panic?
A: War of the Worlds by H G Wells
Correct

Q: Which country singer released the 1975 concept album Red Headed Stranger?
A: Willie Nelson
Correct

Q: What is the usual colour of the drink Grenadine?
A: Red
Correct

Q: Which teacher taught Helen Keller to communicate?
A: Annie Sullivan Macy
Correct

Q: Which acid is found in stinging nettles?
A: Acetic acid
Incorrect

Q: Which Dickens novel takes place during the French revolution
A: A Tale of Two Cities
Correct

Q: The name of the city of Firenze in Italy is usually anglicised to what?
A: Florence
Correct

Q: What device uses radio waves bounced off of objects to identify their range, altitude, direction, and speed?
A: Radar
Correct

Q: What is the subject of the Surgeon's Photograph of 1934
A: Purported monster of Loch Ness
Incorrect

Q: Near which village in North Wales were the rowing events held at the 1958 British Empire and Commonwealth Games?
A: Llanwrst
Incorrect

Q: {question}
A: {answer}
"""

CORRECT = 12521
INCORRECT = 797

@dataclass
class PredictArguments:
    """
    Arguments pertaining to how to run the prediction.
    """
    filename: str = field(default=FILENAME)
    checkpoint: str = field(default=LLAMA_PATH)
    batch_size: int = field(default=4)


def main():
    parser = HfArgumentParser(PredictArguments)
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        args, = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        args, = parser.parse_args_into_dataclasses()

    args: PredictArguments

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO)

    logger.info(f"Self eval with {args}")

    examples = []

    with open(os.path.join(args.filename)) as f:
        for line in tqdm(f):
            example = json.loads(line)
            examples.append(example)

    logger.info("Using prompt: %s", prompt)

    logger.info("Loading model")

    tokenizer = LlamaTokenizer.from_pretrained(args.checkpoint)
    model = LlamaForCausalLM.from_pretrained(args.checkpoint, torch_dtype=torch.float16)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Moving model to device")

    model.to(device)

    tokenizer.eos_token = tokenizer.decode(tokenizer.eos_token_id)
    tokenizer.bos_token = tokenizer.decode(tokenizer.bos_token_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    def run(sample, batch_size):
        question = sample['question']

        answers = []
        answerw2idx = {}

        for generation in sample['generations']:
            answer = generation['decoded']
            if answer not in answerw2idx:
                answerw2idx[answer] = len(answers)
                answers.append(answer)

        input_texts = []
        for answer in answers:
            input_text = prompt.format(question=question, answer=answer)
            input_texts.append(input_text)

        num_batches = math.ceil(len(input_texts) / batch_size)

        scores = []

        for i in tqdm(range(num_batches)):
            batch = input_texts[i * batch_size: (i + 1) * batch_size]
            batch = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
            batch = batch.to(device)
            with torch.no_grad():
                outputs = model(**batch, return_dict=True)
            logits = outputs.logits.cpu().float()
            logits = logits[:, -1, :]
            logits = logits[:, [CORRECT, INCORRECT]]
            logits = torch.softmax(logits, dim=-1)
            scores.extend(logits[:, 0].tolist())

        for generation in sample['generations']:
            answer = generation['decoded']
            generation['self_eval'] = scores[answerw2idx[answer]]


        return sample

    output_name = args.filename.replace(".jsonl", "_self_eval.jsonl")

    output_dir = os.path.dirname(output_name)
    os.makedirs(output_dir, exist_ok=True)

    with open(output_name, "w") as w:
        for sample in tqdm(examples):
            sample = run(sample, args.batch_size)
            w.write(json.dumps(sample) + "\n")


if __name__ == "__main__":
    main()
