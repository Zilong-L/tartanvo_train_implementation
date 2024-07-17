import torch
import sys
import toml
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter   

from Network.VONet import VONet

from utils.train_whole_utils import calculate_loss, load_from_file,load_model, save_model,  validate
import argparse
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from accelerate.logging import get_logger

import vo_trajectory_from_folder


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--config',  default='configs/train_whole_no_rcr.toml')
    config_file = args.parse_args().config
    
    with open(config_file, 'r') as file:
        config = toml.load(file)
        

    datastr = config['datastr']
    dataset_path = config['dataset_path']
    batch_size = int(config['batch_size'])
    worker_num = int(config['worker_num'])

    model_name = config['model_name']
    summary_path = config['summary_path']
    model_path  = config['model_path']
    image_width = int(config['image_width'])
    image_height = int(config['image_height'])
    flow_only = config['flow_only']
    rcr_type = config['rcr_type']
    
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    device = accelerator.device
    def lr_lambda(iteration):
        if iteration < 0.5 * total_iterations:
            return 1.0
        elif iteration < 0.875 * total_iterations:
            return 0.2
        else:
            return 0.04
    
    iteration = 0 
    num_epochs = 9999
    total_iterations = int(config['total_iterations'])
    learning_rate = float(config['learning_rate'] )
    
    model = VONet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = LambdaLR(optimizer, lr_lambda)
    summaryWriter = SummaryWriter(summary_path)
    start_epoch,iteration = load_model(model, optimizer, scheduler, model_name)


    train_scene_dataloaders = {}
    with open(dataset_path, 'r') as f:
        posefiles = f.readlines()
    posefiles = [posefile.strip() for posefile in posefiles]
    for posefile in posefiles:
        train_scene_dataloaders[posefile] = load_from_file(posefile, datastr, image_height, image_width, batch_size, worker_num, flow_only=flow_only, rcr_type=rcr_type)
    
    model, optimizer,  scheduler = accelerator.prepare( model, optimizer,  scheduler)
    train_dataloaders = {}
    for k,v in train_scene_dataloaders.items():
        train_dataloaders[k] = accelerator.prepare(v)
    
    model.train()
    logger = get_logger(__name__)
    for epoch in range(start_epoch,num_epochs):
        for posefile in posefiles:
            for sample in train_dataloaders[posefile]:
                img1 = sample['img1']
                img2 = sample['img2']
                flow_gt = sample['flow']
                intrinsic_gt = sample['intrinsic']
                motions_gt = sample['motion']
                optimizer.zero_grad()  # Zero the parameter gradients
                
                # forward------------------------------------------------------------------
                flow, pose = model([img1,img2,intrinsic_gt])

                # loss calculation---------------------------------------------------------
                total_loss,flow_loss,pose_loss,trans_loss,rot_loss = calculate_loss(pose, motions_gt,flow,flow_gt)
                
                # taking steps
                accelerator.backward(total_loss)
                optimizer.step()
                scheduler.step()
                
                # loggings
                if accelerator.is_main_process:
                    print(f"train epoch {epoch}, iteration {iteration}")
                    iteration += 1
                    if iteration % 10 == 0:
                        summaryWriter.add_scalar('Loss/train_total', total_loss, iteration)
                        summaryWriter.add_scalar('Loss/train_flow', flow_loss, iteration)
                        summaryWriter.add_scalar('Loss/train_pose', pose_loss, iteration)
                        summaryWriter.add_scalar('Loss/train_trans', trans_loss, iteration)
                        summaryWriter.add_scalar('Loss/train_rot', rot_loss, iteration)
                        print(f"Epoch {epoch }, Step {iteration}, Loss: {total_loss}")
                        print(f"flow loss: {flow_loss}, pose loss: {pose_loss}")
                        print(f"translation loss: {trans_loss}, rotation loss: {rot_loss}")
        if accelerator.is_main_process:
            model_save_path = f'{model_path}/{iteration}.pth'
            save_model(model, optimizer, scheduler, epoch, iteration, model_save_path)



        



