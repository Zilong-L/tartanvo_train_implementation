from cv2 import IMWRITE_JPEG2000_COMPRESSION_X1000
import cv2
import flow_vis
import torch
import toml
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter   
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from Network.VONet import VONet
from Datasets.utils import visflow
from utils.train_utils import  load_checkpoint, get_loader
import argparse
import numpy as np
from Network.PWC import PWCDCNet as FlowNet

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--config',  default='configs/train_whole_rcr_lambda_1.toml')
    config_file = args.parse_args().config
    
    with open(config_file, 'r') as file:
        config = toml.load(file)
        
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device_id = rank % torch.cuda.device_count()

    datastr = config['datastr']
    train_path = config['train_path']
    val_path = config['val_path']
    batch_size = int(config['batch_size'])
    image_width = int(config['image_width'])
    image_height = int(config['image_height'])
    flow_only = config['flow_only']
    rcr_type = config['rcr_type']
    shuffle = config['shuffle']
    pretrained_model_path = config['pretrained_model_path']
    save_path  = config['save_path']
    summary_path = config['summary_path']
    
    total_iterations = int(config['total_iterations'])
    learning_rate = float(config['learning_rate'] )
    lambda_flow = float(config['lambda_flow'])
    
    def lr_lambda(iteration):
        if iteration < 0.5 * total_iterations:
            return 1.0
        elif iteration < 0.875 * total_iterations:
            return 0.2
        else:
            return 0.04
        
    iteration = 0 
    torch.cuda.set_device(rank)
    torch.cuda.empty_cache()
    model = FlowNet().to(device_id)
    ddp_model = DDP(model, device_ids=[device_id],find_unused_parameters=True)
    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=learning_rate)
    scheduler = LambdaLR(optimizer, lr_lambda)
    summaryWriter = SummaryWriter(summary_path)
    iteration = load_checkpoint(ddp_model, optimizer, scheduler, pretrained_model_path,map_location=map_location)


    train_dataloader = get_loader(train_path, datastr,image_height,image_width, batch_size, flow_only=flow_only, rcr_type=rcr_type,shuffle=shuffle,rank=rank,world_size=dist.get_world_size())
    val_dataloader = get_loader(val_path, datastr,image_height,image_width, batch_size, flow_only=flow_only, rcr_type=rcr_type,shuffle=shuffle,rank=rank,world_size=dist.get_world_size())

if rank == 0:
    with torch.no_grad():
        for sample in train_dataloader:
            flow_gt = sample['flow']
            img1 = sample['img1']
            img2 = sample['img2']
            img1 = img1.to(device_id)
            img2 = img2.to(device_id)
            flow = ddp_model([img1, img2])
            for img1, img2, flow_uv,flow_gt_uv in zip(img1, img2, flow,flow_gt):
                
                
                # Convert image tensor to numpy array and ensure it's in the correct format
                img1_np = img1.permute(1, 2, 0).cpu().numpy()
                img1_np = (img1_np * 255).astype(np.uint8)  # Assuming the image tensor is normalized [0, 1]
                img2_np = img2.permute(1, 2, 0).cpu().numpy()
                img2_np = (img2_np * 255).astype(np.uint8)  # Assuming the image tensor is normalized [0, 1]
                
                flow_np = flow_uv.permute(1, 2, 0).cpu().detach().numpy()
                
                # Convert flow to color
                flow_color = flow_vis.flow_to_color(flow_np, convert_to_bgr=False)
                flow_gt_uv_np = flow_gt_uv.permute(1, 2, 0).cpu().detach().numpy()
                flow_gt_color = flow_vis.flow_to_color(flow_gt_uv_np, convert_to_bgr=False)
                
                # Display the images using OpenCV
                cv2.imshow('img1', img1_np)
                cv2.imshow('img2', img2_np)
                cv2.imshow('flow', flow_color)
                cv2.imshow('flow_gt', flow_gt_color)
                cv2.waitKey(0)
            
    