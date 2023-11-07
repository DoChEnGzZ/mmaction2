'''
Author: dochengzz
Date: 2023-11-07 18:30:41
LastEditTime: 2023-11-07 18:30:56
LastEditors: dochengzz
Description: 
FilePath: /mmaction2/tools/data/skeleton/merge_pkl.py
'''
import os.path as osp
import os
import pickle
result = []
path = './'
for d in os.listdir(path):
    if d.endswith('.pkl'):
        with open(osp.join(path, d), 'rb') as f:
            content = pickle.load(f)
        result.append(content)
with open('train.pkl', 'wb') as out:
    pickle.dump(result, out, protocol=pickle.HIGHEST_PROTOCOL)
