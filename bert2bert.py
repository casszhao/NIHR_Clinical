import torch
import pandas as pd
import numpy as np

'''
sorted_cat = pd.read_csv('./data/sorted_cats.csv', header=None)[0].to_list()
from mediawiki import MediaWiki
wikipedia = MediaWiki()

article_list = []
for cat in sorted_cat:
  try:
    p = wikipedia.page(cat).summary
    article_list.append(p)
  except:
    print('issue with ', cat)

df = pd.DataFrame(list(zip(sorted_cat, article_list)),
               columns =['label', 'article']).dropna()

df.to_csv('./data/cat_article.csv')

'''


df = pd.read_csv('./data/cat_article.csv')
model_name = "emilyalsentzer/Bio_ClinicalBERT"

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


doc = df['article'].to_list()
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(doc)
indices = np.argsort(vectorizer.idf_)[::-1]
features = vectorizer.get_feature_names()
top_n = 30
top_features = [features[i] for i in indices[:top_n]]
print(top_features)



tfidf_vectorizer=TfidfVectorizer(use_idf=True, lowercase=True, stop_words='english')
tfidf_vectorizer_vectors=tfidf_vectorizer.fit_transform(df['article'].to_list())

tfidf = tfidf_vectorizer_vectors.todense()
# TFIDF of words not in the doc will be 0, so replace them with nan
tfidf[tfidf == 0] = np.nan
# Use nanmean of numpy which will ignore nan while calculating the mean
means = np.nanmean(tfidf, axis=0)
# convert it into a dictionary for later lookup
means = dict(zip(tfidf_vectorizer.get_feature_names(), means.tolist()[0]))



sorted_cat_lowervob = list((map(lambda x: x.lower(), df['label'].to_list())))
sorted_cat_lowervob


df['label']=sorted_cat_lowervob



top_30tfidf_list = []
for row in df['article'].to_list():
  tokenized_list = row.split()
  each_tfidf = []
  for word in tokenized_list:
    tfidf = means.get(word)
    each_tfidf.append(tfidf)
  word_tfidf = pd.DataFrame(list(zip(tokenized_list, each_tfidf)),
                            columns=['word', 'tfidf']).sort_values(by='tfidf', ascending=False).drop_duplicates().head(30)
  top_30tfidf = word_tfidf['word'].to_list()
  top_30tfidf_list.append(top_30tfidf)

df['top30_tfidf'] = top_30tfidf_list


df.to_csv('sorted_cat_articles_top30tfidf.csv')



################  bert2bert


from transformers import BertGenerationEncoder,BertGenerationDecoder,EncoderDecoderModel
from transformers import BertTokenizer
from datasets import load_dataset, Dataset

# leverage checkpoints for Bert2Bert model...
# use BERT's cls token as BOS token and sep token as EOS token
encoder = BertGenerationEncoder.from_pretrained("bert-base-uncased", bos_token_id=101, eos_token_id=102)
# add cross attention layers and use BERT's cls token as BOS token and sep token as EOS token
decoder = BertGenerationDecoder.from_pretrained("bert-base-uncased", add_cross_attention=True, is_decoder=True, bos_token_id=101, eos_token_id=102)
bert2bert = EncoderDecoderModel(encoder=encoder, decoder=decoder) #model

# create tokenizer...
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", padding=True)

sentence_format = []
for top_30 in top_30tfidf_list:
  sentence = ' '.join(top_30)
  sentence_format.append(sentence)


df['sent'] = sentence_format

input_ids = tokenizer(sentence_format[2], add_special_tokens=False, return_tensors="pt").input_ids
labels = tokenizer(sorted_cat_lowervob[2], return_tensors="pt").input_ids

# train...
output = bert2bert(input_ids=input_ids, decoder_input_ids=labels, labels=labels)

print('output:')

print(output)

#########################    TRAIN #####################
raw_datasets = Dataset.from_pandas(df)

device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
from sklearn.model_selection import train_test_split
train_texts, val_texts, train_labels, val_labels = train_test_split(df['sent'], df['label'], test_size=.2)
test_texts, val_texts, test_labels, val_labels = train_test_split(val_texts, val_labels, test_size=.5)


train_encodings = tokenizer(train_texts.to_list(), add_special_tokens=False, return_tensors="pt", padding=True).input_ids
val_encodings = tokenizer(val_texts.to_list(), add_special_tokens=False, return_tensors="pt", padding=True).input_ids
test_encodings = tokenizer(test_texts.to_list(), add_special_tokens=False, return_tensors="pt", padding=True).input_ids

train_labels = tokenizer(train_labels.to_list(), return_tensors="pt", padding=True).input_ids
val_labels = tokenizer(val_labels.to_list(), return_tensors="pt", padding=True).input_ids
test_labels = tokenizer(test_labels.to_list(), return_tensors="pt", padding=True).input_ids

batch_size = 16

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

train_inputs = torch.tensor(train_encodings)
validation_inputs = torch.tensor(val_encodings)
train_labels = torch.tensor(train_labels)
validation_labels = torch.tensor(val_labels)




# Create an iterator of our data with torch DataLoader. This helps save on memory during training because, unlike a for loop,
# with an iterator the entire dataset does not need to be loaded into memory

train_data = TensorDataset(train_inputs, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

validation_data = TensorDataset(validation_inputs, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)


from transformers import get_linear_schedule_with_warmup
from tqdm import  tqdm_notebook

lr = 2e-5
max_grad_norm = 1.0
num_total_steps = 1000
num_warmup_steps = 100
warmup_proportion = float(num_warmup_steps) / float(num_total_steps)  # 0.1


### In PyTorch-Transformers, optimizer and schedules are splitted and instantiated like this:
optimizer = AdamW(bert2bert.parameters(), lr=lr, correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False
#scheduler = WarmupLinearSchedule(optimizer, warmup_steps=num_warmup_steps, t_total=num_total_steps)  # PyTorch scheduler
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps = -1) #num_total_steps

epochs = 2

# trange is a tqdm wrapper around the normal python range
for epoch in tqdm_notebook(range(epochs)):

  # Training
  # Set our model to training mode (as opposed to evaluation mode)
  bert2bert.train()

  # # Tracking variables
  # tr_loss = 0
  # nb_tr_examples, nb_tr_steps = 0, 0

  # # Train the data for one epoch
  for i, batch in enumerate(train_dataloader):
    # Add batch to GPU
    batch = tuple(t.to(device) for t in batch)
    # Unpack the inputs from our dataloader
    b_input_ids, b_labels = batch
    # Forward pass
    # outputs = bert2bert(input_ids = b_input_ids, decoder_input_ids=b_labels)
    # loss = outputs[0]
    loss = bert2bert(input_ids=b_input_ids, decoder_input_ids=b_labels, labels=b_labels).loss  # , labels=labels
    print(loss)
    # train_loss_set.append(loss.item())
    # Backward pass
    loss.backward()
    # Update parameters and take a step using the computed gradient
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
    if (i) % 50 == 0:
      print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
            .format(epoch + 1, epochs, i + 1, total_step, loss.item()))



bert2bert.save_pretrained('./model/bert2bert/')

