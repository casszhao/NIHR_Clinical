import torch
from transformers import BertTokenizer, BertForMultipleChoice
import pandas as pd
import nltk


choice_list = pd.read_csv('./data/sorted_cats.csv', header=None)[0].to_list()
print(choice_list)
print(len(choice_list))
print(len(str(choice_list)))
testing_data = pd.read_csv('./data/new_cat_list_3match_results.csv', usecols=[2,3,6])
print(testing_data)
print(len(testing_data))
text_list = testing_data['Abstract'].to_list()

model_name = 'emilyalsentzer/Bio_ClinicalBERT'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForMultipleChoice.from_pretrained(model_name)

def get_prompt_list(prompt):
    prompt_list = []
    for i in range(len(choice_list)):
        prompt_list.append(prompt)
    print(len(prompt_list))
    return prompt_list



prediction_list = []
for prompt in text_list:
    #print(prompt)
    labels = torch.tensor(0).unsqueeze(0)
    prompt_list = get_prompt_list(str(prompt))
    #print(len(prompt_list))
    assert len(prompt_list)==len(choice_list)
    encoding = tokenizer(prompt_list, choice_list, return_tensors='pt', padding=True)
    outputs = model(**{k: v.unsqueeze(0) for k, v in encoding.items()}, labels=labels)  # batch size is 1

    # the linear classifier still needs to be trained
    loss = outputs.loss
    logits = outputs.logits

    prediction_list.append(logits)

