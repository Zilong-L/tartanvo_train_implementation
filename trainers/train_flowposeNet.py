import torch
import toml
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter   
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from Network.VOFlowNet import VOFlowRes as FlowPoseNet

from utils.train_pose_utils import get_loaders, load_checkpoint, save_checkpoint, calculate_loss,get_loader
import argparse


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--config',  default='configs/flowpose_overfit.toml')
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
    model = FlowPoseNet().to(device_id)
    ddp_model = DDP(model, device_ids=[device_id])
    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=learning_rate)
    scheduler = LambdaLR(optimizer, lr_lambda)
    summaryWriter = SummaryWriter(summary_path)
    iteration = load_checkpoint(ddp_model, optimizer, scheduler, pretrained_model_path,map_location=map_location)


    train_dataloader = get_loader(train_path, datastr,image_height,image_width, batch_size, flow_only=flow_only, rcr_type=rcr_type,shuffle=shuffle,rank=rank,world_size=dist.get_world_size())
    val_dataloader = get_loader(val_path, datastr,image_height,image_width, batch_size, flow_only=flow_only, rcr_type=rcr_type,shuffle=shuffle,rank=rank,world_size=dist.get_world_size())

    while iteration < total_iterations:
        for sample in train_dataloader:
            ddp_model.train()
            if iteration >= total_iterations:
                print(f"Successfully completed training for {iteration} iterations")
                break
            sample = {k: v.to(device_id) for k, v in sample.items()} 
            # inputs-------------------------------------------------------------------
            flow_gt = sample['flow']
            intrinsic_layer = sample['intrinsic']
            flow_input = torch.cat( ( flow_gt, intrinsic_layer ), dim=1 ) 

            # forward------------------------------------------------------------------
            optimizer.zero_grad()  # Zero the parameter gradients
            relative_motion = ddp_model(flow_input)

            # loss calculation---------------------------------------------------------
            motions_gt = sample['motion']
            total_loss,trans_loss,rot_loss = calculate_loss(relative_motion, motions_gt,device_id)

            # backpropagation----------------------------------------------------------
            total_loss.backward()
            optimizer.step()
            scheduler.step()

            iteration += 1
            if rank == 0:
                if iteration % 10 == 0:
                    summaryWriter.add_scalar('Loss/train_pose', total_loss, iteration)
                    summaryWriter.add_scalar('Loss/train_trans', trans_loss, iteration)
                    summaryWriter.add_scalar('Loss/train_rot', rot_loss, iteration)
                    print(f"Step {iteration}, Loss: {total_loss}, translation loss: {trans_loss}, rotation loss: {rot_loss}")
                if iteration % 500 == 0:
                    ddp_model.eval()
                    val_pose, val_trans, val_rot, samplecount = 0, 0, 0, 0
                    with torch.no_grad():
                        for sample in val_dataloader:
                            sample = {k: v.to(device_id) for k, v in sample.items()} 
                            # inputs-------------------------------------------------------------------
                            flow_gt = sample['flow']
                            intrinsic_layer = sample['intrinsic']
                            flow_input = torch.cat( ( flow_gt, intrinsic_layer ), dim=1 ) 

                            # forward------------------------------------------------------------------
                            relative_motion = ddp_model(flow_input)

                            # loss calculation---------------------------------------------------------
                            motions_gt = sample['motion']
                            total_loss,trans_loss,rot_loss = calculate_loss(relative_motion, motions_gt,device_id)
                            val_pose += total_loss.item()
                            val_trans += trans_loss.item()
                            val_rot += rot_loss.item()
                            samplecount += 1
                        val_pose = val_pose/samplecount
                        val_trans = val_trans/samplecount
                        val_rot = val_rot/samplecount
                        
                        if rank == 0 :
                            summaryWriter.add_scalar('Loss/val_pose', val_pose, iteration)
                            summaryWriter.add_scalar('Loss/val_trans', val_trans, iteration)
                            summaryWriter.add_scalar('Loss/val_rot', val_rot, iteration)
                            print(f"Step {iteration}, validation Loss: {val_pose}, translation loss: {val_trans}, rotation loss: {val_rot}")
                if iteration % 2500 == 0:
                    model_save_path = f'{save_path}/flowpose_model_iteration_{iteration}.pth'
                    save_checkpoint( ddp_model, optimizer, scheduler,  iteration, model_save_path)
        if rank == 0:
            model_save_path = f'{save_path}/flowpose_model_iteration_{iteration}.pth'
            save_checkpoint( ddp_model, optimizer, scheduler,  iteration, model_save_path)
        
    dist.destroy_process_group()

            



