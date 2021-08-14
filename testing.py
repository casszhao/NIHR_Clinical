import torch
import tensorflow as tf
import pandas as pd

choice_list = pd.read_csv('./data/sorted_cats.csv', header=None)[0].to_list()
print(choice_list)
cate_dic = {k: v for k, v in enumerate(choice_list)}
print(cate_dic)


x = torch.rand(1, 3)
print('the tensor is')
print(x)

values, indices = tf.nn.top_k(x,2)
print('------value ')
print(values)
print('')
print('------indices ')
print(indices)
indices = indices.numpy().flatten()
print(indices)
# print(indices.type())

def num2cat_dic(indices_list):
    map_cat = []
    for ind in indices_list:
        cat_value = cate_dic[ind]
        map_cat.append(cat_value)

    return map_cat
