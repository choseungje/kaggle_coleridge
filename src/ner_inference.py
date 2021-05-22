from pytorch_pretrained_bert import BertTokenizer
import torch
import json
from tqdm import tqdm
import pandas as pd
import re
import numpy as np

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

PATH = '../output/ner_model.pt'
model = torch.load(PATH)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

submission = pd.read_csv('../input/sample_submission.csv')
print(submission)
print(submission.keys())

paper_test_folder = '../input/test'

papers = {}
for paper_id in tqdm(submission['Id'].unique()):
    with open(f'{paper_test_folder}/{paper_id}.json', 'r') as f:
        paper = json.load(f)
        papers[paper_id] = paper
# print(papers)

MAX_LENGTH = 64 # max no. words for each sentence.
OVERLAP = 20 # if a sentence exceeds MAX_LENGTH, we split it to multiple sentences with overlapping


def clean_training_text(txt):
    """
    similar to the default clean_text function but without lowercasing.
    """
    return re.sub('[^A-Za-z0-9]+', ' ', str(txt)).strip()


def shorten_test_sentences(sentences):
    short_sentences = []
    for sentence in sentences:
        words = sentence.split()
        if len(words) > MAX_LENGTH:
            short_sentences.append(tokenizer.tokenize(' '.join(words[:MAX_LENGTH])))
        else:
            short_sentences.append(tokenizer.tokenize(sentence))
    return short_sentences


ner_test_data = []
sentence_per_paper = []
c = 0

pbar = tqdm(total=len(submission))
for i, p_id in submission[['Id']].itertuples():
    # paper
    print(p_id)
    paper = papers[p_id]

    # sentences
    sentences = set([clean_training_text(sentence) for section in paper
                     for sentence in section['text'].split('. ')
                     ])
    #     if i == 1:
    #         print(sentences)
    sentences = shorten_test_sentences(sentences)  # make sentences short
    sentences = [sentence for sentence in sentences if
                 len(sentence) > 10]  # only accept sentences with length > 10 chars
    #     if i == 1:
    #         print(sentences)
    for sentence in sentences:
        ner_test_data.append(sentence)
        c += 1
    sentence_per_paper.append(c)

    pbar.update(1)

print(ner_test_data[0])


def padding(seq, value='[PAD]', pad_len=64):
    if len(seq) >= pad_len:
        return seq[:pad_len]
    pad = [value for i in range(pad_len - len(seq))]
    seq.extend(pad)

    return seq


for e, inp in enumerate(ner_test_data):
    inp = padding(inp)
    ner_test_data[e] = tokenizer.convert_tokens_to_ids(inp)

print(ner_test_data[0])


class TestBERTDataset(torch.utils.data.Dataset):
    def __init__(self, x):
        self.input_ids = [np.array(i) for i in x]
        self.attention_masks = x

    def __getitem__(self, idx):
        input_ids = torch.LongTensor(self.input_ids[idx])
        attention_masks = torch.LongTensor([float(i > 0) for i in self.attention_masks[idx]])
        return input_ids, attention_masks

    def __len__(self):
        return len(self.input_ids)


bs = 120
data_test = TestBERTDataset(ner_test_data)
test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=bs)

convert_to_token = []

model.eval()
for batch, (input_ids, attention_masks) in enumerate(test_dataloader):

    input_ids = input_ids.to(device)
    attention_masks = attention_masks.to(device)

    with torch.no_grad():
        logits = model(input_ids, None, attention_masks, None)

        predicted_index = torch.argmax(logits, dim=-1)

        for inp, seq in zip(input_ids.tolist(), predicted_index.tolist()):
            check_bio = ""
            for e, (inp_token, seq_token) in enumerate(zip(inp, seq)):
                if seq_token != 2:
                    convert = tokenizer.convert_ids_to_tokens([inp_token])[0]
                    if convert != '[PAD]':
                        check_bio += convert + " "
            convert_to_token.append(check_bio)

    if batch % 100 == 1:
        print(predicted_index.shape)
        #         print(logits)
        print("| Number of predict data {:3d} |".format(input_ids.shape[0] * (batch)))
print(len(convert_to_token))


prediction = []
for e, i in enumerate(sentence_per_paper):
    #     print("i", i)
    if e == 0:
        pred = convert_to_token[:i]
    else:
        pred = convert_to_token[sentence_per_paper[e - 1]:i]
    print(set(pred))
    prediction.append('|'.join([i.strip() for i in set(pred) if i != '']))

for e_pred, p in enumerate(prediction):
    temp = ''
    for e, i in enumerate(range(len(p))):
        if p[i] == '#':
            if p[i + 1] != '#':
                temp = temp[:-1]
        else:
            temp += p[i]
    prediction[e_pred] = temp.lower()

submission['PredictionString'] = prediction
print(submission)

for i in prediction:
    print(i)
submission.to_csv('../output/submission.csv', index=False)
