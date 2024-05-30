import torch
import numpy as np
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter   
from torch.utils.data import DataLoader
from Datasets.utils import ToTensor, Compose, CropCenter, dataset_intrinsics, DownscaleFlow, plot_traj, visflow
from Datasets.tartanDataset import TartanDataset
from Datasets.transformation import ses2poses_quat
from evaluator.tartanair_evaluator import TartanAirEvaluator
from Network.VOFlowNet import VOFlowRes as FlowPoseNet

from utils.train_pose_utils import load_model, save_model, train_pose_batch
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='HRL')

    parser.add_argument('--batch-size', type=int, default=1,
                        help='batch size (default: 1)')
    parser.add_argument('--worker-num', type=int, default=16,
                        help='data loader worker number (default: 1)')
    parser.add_argument('--image-width', type=int, default=640,
                        help='image width (default: 640)')
    parser.add_argument('--image-height', type=int, default=448,
                        help='image height (default: 448)')
    parser.add_argument('--model-name', default='',
                        help='name of pretrained model (default: "")')
    parser.add_argument('--euroc', action='store_true', default=False,
                        help='euroc test (default: False)')
    parser.add_argument('--kitti', action='store_true', default=False,
                        help='kitti test (default: False)')
    parser.add_argument('--kitti-intrinsics-file',  default='',
                        help='kitti intrinsics file calib.txt (default: )')
    parser.add_argument('--test-dir', default='',
                        help='test trajectory folder where the RGB images are (default: "")')
    parser.add_argument('--pose-file', default='',
                        help='test trajectory gt pose file, used for scale calculation, and visualization (default: "")')
    parser.add_argument('--save-flow', action='store_true', default=False,
                        help='save optical flow (default: False)')

    args = parser.parse_args()

    return args

def lr_lambda(iteration):
        if iteration < 0.5 * total_iterations:
            return 1.0
        elif iteration < 0.875 * total_iterations:
            return 0.2
        else:
            return 0.04
    
if __name__ == '__main__':

    args = get_args()

    # load trajectory data from a folder
    datastr = 'tartanair'
    focalx, focaly, centerx, centery = dataset_intrinsics('tartanair') 
    with open('/root/volume/code/python/tartanvo/data/pose_left_paths.txt', 'r') as f:
        posefiles = f.readlines()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    iteration = 0 
    num_epochs = 9999
    learning_rate = 1e-4
    total_iterations = 100000
    
    model = FlowPoseNet()
    model = torch.nn.DataParallel(model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = LambdaLR(optimizer, lr_lambda)
    summaryWriter = SummaryWriter('runs/NORCR_POSEONLY')
    start_epoch,iteration = load_model(model, optimizer, scheduler, args.model_name)

    done = False
    for epoch in range(start_epoch,num_epochs):
        for posefile in posefiles:
            posefile = posefile.strip()
            print(posefile)
            transform = Compose([CropCenter((args.image_height, args.image_width)), DownscaleFlow(), ToTensor()])
            trainDataset = TartanDataset( posefile = posefile, transform=transform, 
                                                focalx=focalx, focaly=focaly, centerx=centerx, centery=centery,flowonly=True)
            
            trainDataloader = DataLoader(trainDataset, batch_size=args.batch_size,
                                                shuffle=False, num_workers=args.worker_num)
            trainDataiter = iter(trainDataloader)

            for batch_idx, sample in enumerate(trainDataiter):
                sample = {k: v.to(device) for k, v in sample.items()} 
                total_loss,trans_loss,rot_loss = train_pose_batch(model, optimizer, sample )
                iteration += 1
                scheduler.step()
                if iteration % 10 == 0:
                    summaryWriter.add_scalar('Loss/train_pose', total_loss, iteration)
                    summaryWriter.add_scalar('Loss/train_trans', trans_loss, iteration)
                    summaryWriter.add_scalar('Loss/train_rot', rot_loss, iteration)
                    print(f"Epoch {epoch + 1}, Step {iteration}, Loss: {total_loss}")
                    print(f"translation loss: {trans_loss}, rotation loss: {rot_loss}")
                if iteration >= total_iterations:
                    done = True
                    break
        model_save_path = f'models/NORCR_POSEONLY/flowpose_model_iteration_{iteration}.pth'
        save_model(model, optimizer, scheduler, epoch, iteration, model_save_path)
        if done:
            break


        



