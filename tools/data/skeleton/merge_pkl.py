'''
Author: dochengzz
Date: 2023-11-07 18:30:41
LastEditTime: 2023-11-10 15:37:08
LastEditors: dochengzz
Description: 
FilePath: /mmaction2/tools/data/skeleton/merge_pkl.py
'''
import os.path as osp
import os
import pickle
result = dict()
result['split'] = dict()
result['split']['train'] = []
result['split']['val'] = []
annotations = []
path = './'
for d in os.listdir(path):
    if d.endswith('.pkl'):
        print(d)
        with open(osp.join(path, d), 'rb') as f:
            content = pickle.load(f)
            # val前缀为S003
            if d[:4] == 'S003' or d[:4] == 'S009':
                result['split']['val'].append(content['frame_dir'])  
            else:    
                result['split']['train'].append(content['frame_dir'])
            annotations.append(content)
result['annotations'] = annotations       
with open('my_dataset.pkl', 'wb') as out:
    pickle.dump(result, out)
