import torch
import sys
import toml
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter   
from collections import OrderedDict
from Network.VONet import VONet




flowNetPath = "/home/lzl/code/python/tartanvo_train_implementation/checkpoints/pwc_net_chairs.pth.tar"
poseNetPath = "/home/lzl/code/python/tartanvo_train_implementation/checkpoints/poseonly_no_rcr/flowpose_model_iteration_100000.pth"
savepath = "checkpoints/whole_no_rcr/init.pth"
checkpoint_pwc = torch.load(flowNetPath)
checkpoint_pose = torch.load(poseNetPath)

def remove_module_prefix(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_key = k[7:]  # Remove 'module.' prefix
        else:
            new_key = k
        new_state_dict[new_key] = v
    return new_state_dict
def add_module_prefix(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if not k.startswith('module.'):
            new_key = 'module.' + k
        else:
            new_key = k
        new_state_dict[new_key] = v
    return new_state_dict
checkpoint_pose_state_dict = remove_module_prefix(checkpoint_pose['model_state_dict']) 
VONET = VONet()
print(VONET.flowNet.load_state_dict(checkpoint_pwc))
print(VONET.flowPoseNet.load_state_dict(checkpoint_pose_state_dict))
state_dict = VONET.state_dict()
new_state_dict = add_module_prefix(state_dict)
print(new_state_dict.keys())

torch.save({"model_state_dict":new_state_dict,"iteration":0},savepath)