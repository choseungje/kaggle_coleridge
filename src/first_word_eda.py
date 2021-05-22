import pandas as pd
import re
from pytorch_pretrained_bert import BertTokenizer
from tqdm import tqdm
import json

df = pd.read_csv('../input/train.csv')
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

paper_train_folder = '../input/train'
papers = {}
for paper_id in tqdm(df['Id'].unique()):
    with open(f'{paper_train_folder}/{paper_id}.json', 'r') as f:
        paper = json.load(f)
        papers[paper_id] = paper

MAX_LENGTH = 64  # max no. words for each sentence.
OVERLAP = 20  # if a sentence exceeds MAX_LENGTH, we split it to multiple sentences with overlapping

MAX_SAMPLE = None  # set a small number for experimentation, set None for production.


def clean_training_text(txt):
    """
    similar to the default clean_text function but without lowercasing.
    """
    return re.sub('[^A-Za-z0-9]+', ' ', str(txt)).strip()


def shorten_sentences(sentences):
    short_sentences = []
    for sentence in sentences:
        words = sentence.split()
        # 0~9로 시작하는 문장 전처리
        try:
            for n in range(len(words)):
                if int(words[n]):
                    del words[n]
                else:
                    break
        except:
            pass
        # print(len(words), words)
        if len(words) > MAX_LENGTH:
            short_sentences.append(' '.join(words[:MAX_LENGTH]))
        else:
            short_sentences.append(' '.join(words))
    return short_sentences


cnt_pos, cnt_neg = 0, 0  # number of sentences that contain/not contain labels
ner_data = []
dic = {}
pbar = tqdm(total=len(df))
for i, id, dataset_label in df[['Id', 'dataset_label']].itertuples():
    # paper
    paper = papers[id]

    # labels
    labels = dataset_label.split('|')
    labels = [clean_training_text(label) for label in labels]

    # sentences
    sentences = set([clean_training_text(sentence) for section in paper
                     for sentence in section['text'].split('. ')
                     ])
    sentences = shorten_sentences(sentences)  # make sentences short
    sentences = [sentence for sentence in sentences if
                 len(sentence) > 10]  # only accept sentences with length > 10 chars

    for i in sentences:
        # print(i)
        if i.split(' ')[0] not in dic:
            dic[i.split(' ')[0]] = 1
        else:
            dic[i.split(' ')[0]] += 1
            # if i.split(' ')[0] == 'NCES':
            #     print(i)
    pbar.update(1)

print(len(dic))
c = []
for i in dic:
    c.append((i, dic[i]))

c = sorted(c, reverse=True, key=lambda x: x[1])
n = 0
for i in range(100):
    print(i, c[i])
    n += c[i][1]
print(n)
