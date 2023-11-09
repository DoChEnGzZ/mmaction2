'''
Author: dochengzz
Date: 2023-11-07 18:30:41
LastEditTime: 2023-11-09 16:33:05
LastEditors: dochengzz
Description: 
FilePath: /mmaction2/tools/data/skeleton/merge_pkl.py
'''
import os.path as osp
import os
import mmengine
result = dict()
result['split'] = dict()
result['split']['train'] = []
result['split']['val'] = []
annotations = []
path = './'
for d in os.listdir(path):
    if d.endswith('.pkl'):
        with open(osp.join(path, d), 'rb') as f:
            content = mmengine.load(f)
            result['split']['train'].append(content['frame_dir'])  
            result['split']['val'].append(content['frame_dir'])
            annotations.append(content)
result['annotations'] = annotations       
with open('my_dataset.pkl', 'wb') as out:
    mmengine.dump(result, out)
