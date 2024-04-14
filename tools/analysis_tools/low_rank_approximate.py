
import torch
import collections

import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

import json

def low_rank_approximate(mat_org: torch.tensor, rank=16):
    """ Learning a low-rank decomposition for the given matrix.

    Args:
        mat_org (torch.tensor): the given matrix.
        rank (int, optional): defined rank value. Defaults to 16.
    """
    device = mat_org.device

    if not device == 'cpu':
        mat_org = mat_org.cpu()
    u, s, vh = np.linalg.svd(mat_org.detach().numpy(), full_matrices=True)

    s_val = np.sqrt(np.diag(s[:rank])) # half singular value
    mat_q = torch.tensor(u[:, :rank] @ s_val)
    mat_r = torch.tensor(s_val @ vh[:rank, :])
    error = nn.functional.mse_loss(mat_q @ mat_r, mat_org)

    mat_q = mat_q.to(device)
    mat_r = mat_r.to(device)

    output = {'mat_q': mat_q,
              'mat_r': mat_r.t(),
              'error': error}
    return output


#new_pth = '/remote-home/pxy/mmrotate/work_dirs/oriented_rcnn_swin_tiny_fpn_1x_dota_le90_train_test_50v2/epoch_12.pth'
new_pth = '/remote-home/pxy/mmrotate/work_dirs/oriented_rcnn_swin_tiny_fpn_1x_dota_le90/epoch_12.pth'

new_state_dict_load = torch.load(new_pth, map_location=lambda storage, loc: storage.cuda(1))
# print(type(new_state_dict_load))
# print(new_state_dict_load.keys())
# print(list(new_state_dict_load['state_dict'].keys()))
# print(type(new_state_dict_load['state_dict']))

state_dict = new_state_dict_load['state_dict']

#grad_state_dict = collections.OrderedDict()
rank_list = [8,16,32,48,64,96,192,384,768]
low_rank_approximate_result = {}

for layer in state_dict:
    

    if 'weight' in layer and 'conv' not in layer and 'norm' not in layer and 'cls' not in layer and 'reg' not in layer:# and 'qkv' in layer:

        if 'backbone.stages.0' in layer:
            weight = state_dict[layer]
            layer_rank_error = np.zeros(6)
            if len(weight.shape) ==2:
                print(layer)
                print(state_dict[layer].shape) 
                for i,rank in enumerate(rank_list[:6]):
                    low_rank_mat_dict = low_rank_approximate(weight,rank= int(rank))
                    #print(low_rank_mat_dict['error'])
                # layer_rank_error.append(low_rank_mat_dict['error'].numpy())
                    layer_rank_error[i] = low_rank_mat_dict['error'].numpy()
                print(layer_rank_error)
                low_rank_approximate_result[layer] = layer_rank_error.tolist()
        
        if 'backbone.stages.1' in layer:
            weight = state_dict[layer]
            layer_rank_error = np.zeros(7)
            if len(weight.shape) ==2:
                print(layer)
                print(state_dict[layer].shape) 
                for i,rank in enumerate(rank_list[:7]):
                    low_rank_mat_dict = low_rank_approximate(weight,rank= int(rank))
                    #print(low_rank_mat_dict['error'])
                # layer_rank_error.append(low_rank_mat_dict['error'].numpy())
                    layer_rank_error[i] = low_rank_mat_dict['error'].numpy()
                print(layer_rank_error)
                low_rank_approximate_result[layer] = layer_rank_error.tolist()
        
        if 'backbone.stages.2' in layer:
            weight = state_dict[layer]
            layer_rank_error = np.zeros(8)
            if len(weight.shape) ==2:
                print(layer)
                print(state_dict[layer].shape) 
                for i,rank in enumerate(rank_list[:8]):
                    low_rank_mat_dict = low_rank_approximate(weight,rank= int(rank))
                    #print(low_rank_mat_dict['error'])
                # layer_rank_error.append(low_rank_mat_dict['error'].numpy())
                    layer_rank_error[i] = low_rank_mat_dict['error'].numpy()
                print(layer_rank_error)
                low_rank_approximate_result[layer] = layer_rank_error.tolist()
        
        if 'backbone.stages.3' in layer:
            weight = state_dict[layer]
            layer_rank_error = np.zeros(9)
            if len(weight.shape) ==2:
                print(layer)
                print(state_dict[layer].shape) 
                for i,rank in enumerate(rank_list):
                    low_rank_mat_dict = low_rank_approximate(weight,rank= int(rank))
                    #print(low_rank_mat_dict['error'])
                # layer_rank_error.append(low_rank_mat_dict['error'].numpy())
                    layer_rank_error[i] = low_rank_mat_dict['error'].numpy()
                print(layer_rank_error)
                low_rank_approximate_result[layer] = layer_rank_error.tolist()

        if 'roi_head' in layer:
            weight = state_dict[layer]
            layer_rank_error = np.zeros(9)
            if len(weight.shape) ==2:
                print(layer)
                print(state_dict[layer].shape) 
                for i,rank in enumerate(rank_list):
                    low_rank_mat_dict = low_rank_approximate(weight,rank= int(rank))
                    #print(low_rank_mat_dict['error'])
                # layer_rank_error.append(low_rank_mat_dict['error'].numpy())
                    layer_rank_error[i] = low_rank_mat_dict['error'].numpy()
                print(layer_rank_error)
                low_rank_approximate_result[layer] = layer_rank_error.tolist()


print(len(low_rank_approximate_result))


# 指定保存的JSON文件路径
json_file_path = 'low_rank_approximate_result.json'

# 将字典保存为JSON文件
with open(json_file_path, 'w') as json_file:
    json.dump(low_rank_approximate_result, json_file)

#plt.figure(figsize=(10, 6))
line_styles = ['-', '--', '-.', ':']  # 实线、虚线、点线、点划线
markers = ['o', 's', '^', 'x', '*']  # 圆圈、正方形、三角形、叉叉、星号
# 遍历字典中的每一项，绘制曲线
for idx, (description, values) in enumerate(low_rank_approximate_result.items()):
    # 使用 range(len(values)) 创建 x 坐标轴
    line_style = line_styles[idx % len(line_styles)]
    marker = markers[idx % len(markers)]
    plt.plot(range(len(values)), values, label=description, linestyle=line_style, marker=marker, linewidth=1.5)

# 添加图例
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
# 设置横轴刻度和标签
#custom_labels = ['a', 'b', 'c', 'd', 'e']  # 自定义标签
plt.xticks(range(9), rank_list)
# 添加标签和标题
plt.xlabel('rank')
plt.ylabel('layer_rank_approximate_error')
plt.title('low_rank_approximate_result')
plt.savefig('low_rank_anaylsis.png')






# state_dict_save_lora = {'meta':new_state_dict_load['meta'],'state_dict': grad_state_dict}  #,'layers_info':list(new_state_dict_load['state_dict'].keys())} 
# torch.save(state_dict_save_lora , save_lora_pth)

#print(new_state_dict_load['meta'])
#print(grad_state_dict.keys())

# old_pth = '/remote-home/pxy/mmrotate/work_dirs/oriented_rcnn_swin_tiny_fpn_1x_dota_le90_train_test/epoch_12.pth'
# old_pth_load = torch.load(old_pth)
# print(type(new_state_dict_load))
# print(new_state_dict_load.keys())
# print(list(new_state_dict_load['state_dict'].keys()))
#print(type(new_state_dict_load['state_dict']))

