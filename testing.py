import torch
import tensorflow as tf

x = torch.rand(1, 3)
print('the tensor is')
print(x)

values, indices = tf.nn.top_k(x,2)
print('------value ')
print(values)
print('------indices ')
print(indices)
