import torch
import sys
import toml
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter   
from collections import OrderedDict
from Network.VONet import VONet

from Network.PWC import PWCDCNet as FlowNet


flowNetPath = "/home/lzl/code/python/tartanvo_train_implementation/checkpoints/pwc_net_chairs.pth.tar"
savepath = "checkpoints/flow_test/init.pth"
checkpoint_pwc = torch.load(flowNetPath)

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


state_dict = checkpoint_pwc
new_state_dict = add_module_prefix(state_dict)
print(new_state_dict.keys())

torch.save({"model_state_dict":new_state_dict,"iteration":0},savepath)