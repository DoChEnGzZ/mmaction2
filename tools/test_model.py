'''
Author: dochengzz
Date: 2023-11-10 12:28:04
LastEditTime: 2023-11-10 14:55:13
LastEditors: dochengzz
Description: 
FilePath: /mmaction2/tools/test_model.py
'''
from mmaction.apis import inference_skeleton, init_recognizer  
import mmengine
import numpy as np

config_path = 'configs/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_gym-keypoint.py'
checkpoint_path = 'myCheckpoint.pth' # 可以是本地路径
img_path = 'data/S003C001P001R001A001.pkl'   # 您可以指定自己的图片路径
data = mmengine.load(img_path)
# print(data['keypoint'])

# 获取数目
total_frame = data['total_frames']
person_num = len(data['keypoint'])
key_point_num = len(data['keypoint'][0][0])
img_shape = data['img_shape']

pose_results = []
for i in range (0,total_frame):
  pose = dict()
  # 单人
  keypoints = data['keypoint'][0][i]
  keypoint_scores = data['keypoint_score'][0][i]
  pose['keypoints'] = np.array([keypoints])
  pose['keypoint_scores'] = np.array([keypoint_scores])
  pose_results.append(pose)

# 从配置文件和权重文件中构建模型
model = init_recognizer(config_path, checkpoint_path, device="cuda:0")  # device 可以是 'cuda:0'
# 对单个视频进行测试
result = inference_skeleton(model, pose_results,img_shape)
print(results.pred_label)