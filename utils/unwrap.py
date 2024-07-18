import torch
import sys
import toml
from torch.optim.lr_scheduler import LambdaLR

from Network.VOFlowNet import VOFlowRes as FlowPoseNet

from utils.train_pose_utils import load_from_file,calculate_loss
import argparse
from accelerate import DistributedDataParallelKwargs,Accelerator
from accelerate.utils import ProjectConfiguration,broadcast_object_list
from accelerate.utils import set_seed

if __name__ == '__main__':
    set_seed(42)

    
    checkpoint = "/home/lzl/code/python/tartanvo_train_implementation/checkpoints/poseonly_no_rcrr/458"
    learning_rate = 1e-4
    total_iterations = 1000000
    def lr_lambda(iteration):
        if iteration/4 < 0.5 * total_iterations:
            return 1.0
        elif iteration/4 < 0.875 * total_iterations:
            return 0.2
        else:
            return 0.04
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)

    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

    device = accelerator.device

    model = FlowPoseNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = LambdaLR(optimizer, lr_lambda)
    accelerator.register_for_checkpointing(scheduler)

    model, optimizer,  scheduler = accelerator.prepare( model, optimizer,  scheduler)
    if checkpoint != "":
        accelerator.load_state(checkpoint)

    torch.save(accelerator.unwrap_model(model).state_dict(), "model.pth")

        

        



