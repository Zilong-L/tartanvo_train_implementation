import torch
import sys
import toml
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter   

from Network.VOFlowNet import VOFlowRes as FlowPoseNet

from utils.train_pose_utils import load_from_file,calculate_loss
import argparse
from accelerate import DistributedDataParallelKwargs,Accelerator
from accelerate.utils import ProjectConfiguration,broadcast_object_list
from accelerate.utils import set_seed

if __name__ == '__main__':
    set_seed(42)
    args = argparse.ArgumentParser()
    args.add_argument('--config',  default='configs/train_pose_no_rcr.toml')
    config_file = args.parse_args().config
    
    with open(config_file, 'r') as file:
        config = toml.load(file)
        

    datastr = config['datastr']
    dataset_path = config['dataset_path']
    val_path = config['val_path']
    batch_size = int(config['batch_size'])
    worker_num = int(config['worker_num'])

    summary_path = config['summary_path']
    image_width = int(config['image_width'])
    image_height = int(config['image_height'])
    flow_only = config['flow_only']
    rcr_type = config['rcr_type']
    
    checkpoint = config['checkpoint']
    checkpoint_path = config['checkpoint_path']
    
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    # project_config = ProjectConfiguration(project_dir=".", logging_dir=summary_path)
    # accelerator = Accelerator(kwargs_handlers=[ddp_kwargs],log_with="tensorboard", project_config=project_config)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    # hps = {"num_iterations": 5, "learning_rate": 1e-2}
    # accelerator.init_trackers("tartanvo", config=hps)
    device = accelerator.device
    
    def lr_lambda(iteration):
        if iteration/4 < 0.5 * total_iterations:
            return 1.0
        elif iteration/4 < 0.875 * total_iterations:
            return 0.2
        else:
            return 0.04
    
    iteration = 0 
    total_iterations = int(config['total_iterations']) 
    learning_rate = float(config['learning_rate'] )
    learning_rate *= accelerator.num_processes
    
    model = FlowPoseNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = LambdaLR(optimizer, lr_lambda)
    summaryWriter = SummaryWriter(summary_path)
    accelerator.register_for_checkpointing(scheduler)

    train_scene_dataloaders = []
    with open(dataset_path, 'r') as f:
        posefiles = f.readlines()
    posefiles = [posefile.strip() for posefile in posefiles]
    for posefile in posefiles:
        train_scene_dataloaders.append( load_from_file(posefile, datastr, image_height, image_width, batch_size, worker_num, flow_only=flow_only, rcr_type=rcr_type))
        
    val_scene_dataloaders = []
    with open(val_path, 'r') as f:
        val_posefiles = f.readlines()
    val_posefiles = [posefile.strip() for posefile in val_posefiles]
    for posefile in val_posefiles:
        val_scene_dataloaders.append( load_from_file(posefile, datastr, image_height, image_width, batch_size, worker_num, flow_only=flow_only, rcr_type=rcr_type))

    model, optimizer,  scheduler = accelerator.prepare( model, optimizer,  scheduler)
    if checkpoint != "":
        accelerator.load_state(checkpoint)
    
    train_dataloaders = []
    for loader in train_scene_dataloaders:
        train_dataloaders.append( accelerator.prepare(loader))
        
    val_dataloaders = []
    for loader in val_scene_dataloaders:
        val_dataloaders.append( accelerator.prepare(loader))
        
    iteration = int((scheduler.scheduler._step_count - 1) /4)
    training = [True]
    while training[0]:
        model.train()
        for dataloader in train_dataloaders:
            if training[0] is not True:
                break
            for sample in dataloader:
                if training[0] is not True:
                    break
                flow_gt = sample['flow']
                intrinsic_gt = sample['intrinsic']
                
                # forward------------------------------------------------------------------
                optimizer.zero_grad()  # Zero the parameter gradients
                flow_input = torch.cat( ( flow_gt, intrinsic_gt ), dim=1 ) 
                relative_motion = model(flow_input)

                # loss calculation---------------------------------------------------------
                motions_gt = sample['motion']
                total_loss,trans_loss,rot_loss = calculate_loss(relative_motion, motions_gt)

                # loss calculation---------------------------------------------------------
                # taking steps
                accelerator.backward(total_loss)
                optimizer.step()
                scheduler.step()
                
                # loggings
                iteration += 1
                if accelerator.is_main_process:
                    print(f"iteration {iteration}")
                    if iteration % 20 == 0:
                        summaryWriter.add_scalar('Loss/train_total', total_loss, iteration)
                        summaryWriter.add_scalar('Loss/train_trans', trans_loss, iteration)
                        summaryWriter.add_scalar('Loss/train_rot', rot_loss, iteration)
                        print(f"Step {iteration}, Loss: {total_loss}")
                        print(f"translation loss: {trans_loss}, rotation loss: {rot_loss}")
                    if iteration >= total_iterations:
                        print(f"Successfully completed training for {iteration} iterations")
                        training = [False]
                accelerator.wait_for_everyone()
                broadcast_object_list(training)
        accelerator.save_state(f'{checkpoint_path}/{iteration}')
        model.eval()
        with torch.no_grad():
            val_total,val_trans,val_rot = 0,0,0
            total_samples = 0
            for dataloader in val_dataloaders:
                for sample in dataloader:
                    total_samples += 1
                    flow_gt = sample['flow']
                    intrinsic_gt = sample['intrinsic']

                    # Prepare validation input
                    flow_input = torch.cat((flow_gt, intrinsic_gt), dim=1)
                    relative_motion_val = model(flow_input)

                    # Calculate validation loss
                    motions_gt = sample['motion']
                    val_total_loss, val_trans_loss, val_rot_loss = calculate_loss(relative_motion_val, motions_gt)

                    # Accumulate validation loss
                    val_total += val_total_loss.item()
                    val_trans += val_trans_loss.item()
                    val_rot += val_rot_loss.item()

                    # Optional: log validation losses
                    if accelerator.is_main_process:
                        print(f"Validation - Total Loss: {val_total_loss}, Translation Loss: {val_trans_loss}, Rotation Loss: {val_rot_loss}")

            # Average validation loss
            if total_samples > 0:
                val_total /= total_samples
                val_trans /= total_samples
                val_rot /= total_samples
            else:
                val_total,val_trans,val_rot = 0,0,0

            # Log the averaged validation loss
            if accelerator.is_main_process:
                summaryWriter.add_scalar('Loss/val_total', val_total, iteration)
                summaryWriter.add_scalar('Loss/val_trans', val_trans, iteration)
                summaryWriter.add_scalar('Loss/val_rot', val_rot, iteration)
                print(f"Averaged Validation Loss: {val_total}, Translation Loss: {val_trans}, Rotation Loss: {val_rot}")
            
                
    accelerator.end_training()
    




        

        



