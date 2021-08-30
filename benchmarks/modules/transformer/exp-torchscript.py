#!/usr/bin/env python
# coding: utf-8

# In[1]:


from allennlp.modules.transformer import util, util_script


# In[2]:


import numpy as np
from time import perf_counter


def timer(f, *args):

    start = perf_counter()
    f(*args)
    return 1000 * (perf_counter() - start)


# In[4]:


import torch

bs = 5
num_attention_heads = 8
source_seq_len = 32
target_seq_len = 4

values = torch.randn(bs, num_attention_heads, source_seq_len, target_seq_len)
mask = torch.randn(bs, target_seq_len)


# In[9]:


print(np.mean([timer(util.apply_mask, values, mask) for _ in range(100)]))


# In[12]:


scripted = torch.jit.script(util_script.apply_mask)
print(np.mean([timer(scripted, values, mask) for _ in range(100)]))


# In[ ]:
