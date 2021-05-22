import numpy as np
import pandas as pd
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForTokenClassification
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import os
import time

# Load pre-trained model (weights)
model = BertForTokenClassification.from_pretrained('bert-base-cased', num_labels=3)
# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
# print(model)
print(tokenizer)

print("vocab_size : {}\n" .format(len(tokenizer.vocab)))
print(tokenizer.convert_tokens_to_ids(['to', '[PAD]']))


df = pd.read_csv('../input/train.csv')
# print(df)
print("df key :", df.keys())

char_label_maxlen = 0
word_label_maxlen = 0
# check label length at char level
char_length_dic = {}
# word level length check
word_length_dic = {}

for label in df['dataset_label']:
    # check char level
    if len(label) not in char_length_dic:
        char_length_dic[len(label)] = 1
    else:
        char_length_dic[len(label)] += 1

    # char max len
    if len(label) > char_label_maxlen:
        char_label_maxlen = len(label)

    # check word level
    if len(label.split(' ')) not in word_length_dic:
        word_length_dic[len(label.split(' '))] = 1
        if len(label.split(' ')) == 1:
            print(f"A word of length 1 is '{label}'")
    else:
        word_length_dic[len(label.split(' '))] += 1

    # word max len
    if len(label.split(' ')) > word_label_maxlen:
        word_label_maxlen = len(label.split(' '))

print("char level max length :", char_label_maxlen)
print("word level max length :", word_label_maxlen)

length_data = []
for i in char_length_dic:
    length_data.append((i, char_length_dic[i]))

x, y = zip(*length_data)
plt.title('Label length at char level')
plt.bar(x, y)
plt.show()


sentence_length_data = []
for i in word_length_dic:
    sentence_length_data.append((i, word_length_dic[i]))

x, y = zip(*sentence_length_data)
print("word_length_dic :", word_length_dic)
plt.title('Label length at word level')
plt.bar(x, y)
plt.show()

df = df.groupby('Id').agg({
    'pub_title': 'first',
    'dataset_title': '|'.join,
    'dataset_label': '|'.join,
    'cleaned_label': '|'.join
}).reset_index()
# print(df)

paper_train_folder = '../input/train'

papers = {}
for paper_id in tqdm(df['Id'].unique()):
    with open(f'{paper_train_folder}/{paper_id}.json', 'r') as f:
        paper = json.load(f)
        papers[paper_id] = paper

MAX_LENGTH = 64 # max no. words for each sentence.
OVERLAP = 20 # if a sentence exceeds MAX_LENGTH, we split it to multiple sentences with overlapping

MAX_SAMPLE = None # set a small number for experimentation, set None for production.

import re


def clean_training_text(txt):
    """
    similar to the default clean_text function but without lowercasing.
    """
    return re.sub('[^A-Za-z0-9]+', ' ', str(txt)).strip()


def shorten_sentences(sentences):
    short_sentences = []
    for sentence in sentences:
        words = sentence.split()
        if len(words) > MAX_LENGTH:
            for p in range(0, len(words), MAX_LENGTH - OVERLAP):
                short_sentences.append(' '.join(words[p:p + MAX_LENGTH]))
        else:
            short_sentences.append(sentence)
    return short_sentences


def find_sublist(big_list, small_list):
    all_positions = []
    for i in range(len(big_list) - len(small_list) + 1):
        if small_list == big_list[i:i + len(small_list)]:
            all_positions.append(i)

    return all_positions


def tag_sentence(sentence, labels):  # requirement: both sentence and labels are already cleaned
    #     sentence_words = sentence.split()
    #     print(sentence)
    sentence_words = tokenizer.tokenize(sentence)

    if labels is not None and any(re.findall(f'\\b{label}\\b', sentence)
                                  for label in labels):  # positive sample
        nes = ['O'] * len(sentence_words)
        for label in labels:
            #             label_words = label.split()
            label_words = tokenizer.tokenize(label)

            all_pos = find_sublist(sentence_words, label_words)
            for pos in all_pos:
                nes[pos] = 'B'
                for i in range(pos + 1, pos + len(label_words)):
                    nes[i] = 'I'

        return True, list(zip(sentence_words, nes))

    else:  # negative sample
        nes = ['O'] * len(sentence_words)
        return False, list(zip(sentence_words, nes))


# cnt_pos, cnt_neg = 0, 0 # number of sentences that contain/not contain labels
# ner_data = []
#
# pbar = tqdm(total=len(df))
# for i, id, dataset_label in df[['Id', 'dataset_label']].itertuples():
#     # paper
#     paper = papers[id]
#
#     # labels
#     labels = dataset_label.split('|')
#     labels = [clean_training_text(label) for label in labels]
#
#     # sentences
#     sentences = set([clean_training_text(sentence) for section in paper
#                  for sentence in section['text'].split('. ')
#                 ])
#     sentences = shorten_sentences(sentences) # make sentences short
#     sentences = [sentence for sentence in sentences if len(sentence) > 10] # only accept sentences with length > 10 chars
#
#     for e, sentence in enumerate(sentences):
#         is_positive, tags = tag_sentence(sentence, labels)
# #         print("tags", tags)
#         if is_positive:
#             cnt_pos += 1
#             ner_data.append(tags)
#         elif any(word in sentence.lower() for word in ['data', 'study', 'dataset', 'of', 'in', 'and']):
#             if cnt_neg < 500000:
#                 ner_data.append(tags)
#                 cnt_neg += 1
#
#     pbar.update(1)
#     pbar.set_description(f"Training data size: {cnt_pos} positives + {cnt_neg} negatives")
#
#
# print(len(ner_data))
#
# train2 = pd.DataFrame(columns=['data'])
# train2['data'] = ner_data
# train2.to_csv('../input/train_dataset_test2.csv', index=False)
# print("Save")

# ==========================================================================================
test=pd.read_csv('../input/train_dataset_test.csv')

print(test)
ner_data = test['data'].values
print(ner_data[0])
print(len(ner_data[0]))

for e, data in enumerate(tqdm(ner_data)):
    temp = []
    for i in data[3:-3].split('\'), (\''):
        temp.append(i.split('\', \''))
    ner_data[e] = temp

print(len(ner_data))

# ==========================================================================================
inputs = []
target = []

for row in ner_data:
    inp, tar = list(zip(*row))
    inputs.append(list(inp))
    target.append(list(tar))
print(f"len inputs : {len(inputs)}")
print(f"len target : {len(target)}")


def padding(seq, value='[PAD]', pad_len=64):
    if len(seq) >= pad_len:
        return seq[:pad_len]
    pad = [value for i in range(pad_len - len(seq))]
    seq.extend(pad)

    return seq


tag_dic = {'B': 0,
           'I': 1,
           'O': 2}
print("target vocab_size : {}".format(len(tag_dic)))
print(tag_dic)

for e, (inp, tar) in enumerate(zip(inputs, target)):
    inp = padding(inp)
    inputs[e] = tokenizer.convert_tokens_to_ids(inp)

    tar = padding(tar, value='O')
    target[e] = [tag_dic[i] for i in tar]

print(inputs[0])
print(target[0])


train_x, valid_x, train_y, valid_y = train_test_split(
    inputs, target, test_size=0.01, shuffle=True, random_state=45)

print(f"len train_x : {len(train_x)}")
print(f"len valid_x : {len(valid_x)}")
print(f"len train_y : {len(train_y)}")
print(f"len valid_y : {len(valid_y)}")


class BERTDataset(Dataset):
    def __init__(self, x, y):
        self.input_ids = [np.array(i) for i in x]
        #         self.attention_masks = [[float(i>0) for i in ii] for ii in x]
        self.attention_masks = x

        self.labels = [np.array(i) for i in y]

    def __getitem__(self, idx):
        input_ids = torch.LongTensor(self.input_ids[idx])
        attention_masks = torch.LongTensor([float(i > 0) for i in self.attention_masks[idx]])
        #         attention_masks = torch.LongTensor(self.attention_masks[idx])
        labels = torch.LongTensor(self.labels[idx])
        return input_ids, attention_masks, labels

    def __len__(self):
        return len(self.labels)


data_train = BERTDataset(train_x, train_y)
data_val = BERTDataset(valid_x, valid_y)


bs = 230
train_dataloader = torch.utils.data.DataLoader(data_train, batch_size=bs)
valid_dataloader = torch.utils.data.DataLoader(data_val, batch_size=bs)
for e, (i, j, k) in enumerate(train_dataloader):
    if e == 2:
        print(i)
        print(i, j, k)
        break

device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=5 * 1e-5, betas=(0.9, 0.98), eps=1e-9)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2100 * 3, eta_min=1e-6)


def save_torch(model, optimizer):
    if not os.path.exists('../output/'):
        os.makedirs('../output/')
    torch.save(model, '../output/model.pt')
    state = {
        'State_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(state, '../output/model_state.pt')


epochs = 1
max_grad_norm = 1

for epoch in range(epochs):
    model.train()
    t_loss = 0
    v_loss = 0
    start = time.time()

    for batch, (input_ids, attention_masks, labels) in enumerate(train_dataloader):
        optimizer.zero_grad()

        input_ids = input_ids.to(device)
        attention_masks = attention_masks.to(device)
        labels = labels.to(device)

        loss = model(input_ids, None, attention_masks, labels)
        t_loss += loss.item()

        loss.backward()

        # gradient clipping
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
        # update parameters
        optimizer.step()
        scheduler.step()

        if batch % 100 == 1:
            print("| epoch {:2d} | batch {:3d} | loss {:6f} | time {} sec | LR {}".format(epoch + 1, batch,
                                                                                          t_loss / batch,
                                                                                          time.time() - start,
                                                                                          scheduler.get_lr()))
    print("| Epoch {:2d} | Loss {:6f}".format(epoch + 1, t_loss / (batch + 1)))

    save_torch(model, optimizer)

    model.eval()
    for batch, (input_ids, attention_masks, labels) in enumerate(valid_dataloader):

        input_ids = input_ids.to(device)
        attention_masks = attention_masks.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            loss = model(input_ids, None, attention_masks, labels)
            v_loss += loss.item()

        if batch % 100 == 1:
            print("| epoch {:2d} | batch {:3d} | val loss {:6f} |".format(epoch + 1, batch, v_loss / batch))
    print("| Epoch {:2d} | Val Loss {:6f}".format(epoch + 1, v_loss / (batch + 1)))
    print()


# =======================================================================================
# inference
PATH = '../output/model.pt'
model = torch.load(PATH)
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
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
