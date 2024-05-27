import torch
import numpy as np
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter   
from torch.utils.data import DataLoader
from Datasets.utils import ToTensor, Compose, CropCenter, dataset_intrinsics, DownscaleFlow, plot_traj, visflow
from Datasets.tartanDataset import TartanDataset
from Datasets.transformation import ses2poses_quat
from evaluator.tartanair_evaluator import TartanAirEvaluator
from TartanVO import TartanVO
from utils.train_utils import train_pose_batch
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='HRL')

    parser.add_argument('--batch-size', type=int, default=1,
                        help='batch size (default: 1)')
    parser.add_argument('--worker-num', type=int, default=1,
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


    
if __name__ == '__main__':

    args = get_args()

    # load trajectory data from a folder
    datastr = 'tartanair'
    focalx, focaly, centerx, centery = dataset_intrinsics('tartanair') 
    with open('/root/volume/code/python/tartanvo/data/pose_left_paths.txt', 'r') as f:
        posefiles = f.readlines()

    learning_rate = 1e-4
    num_epochs = 10
    model = TartanVO().vonet
    model.train()

    total_iterations = 100000
    def lr_lambda(current_step):
        if current_step < 0.5 * total_iterations:
            return 1.0
        elif current_step < 0.875 * total_iterations:
            return 0.2
        else:
            return 0.04
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = LambdaLR(optimizer, lr_lambda)

    summaryWriter = SummaryWriter()
    loss_values = []
    current_step = 0 

    for epoch in range(num_epochs):
        for posefile in posefiles:
            posefile = posefile.strip()
            print(posefile)
            transform = Compose([CropCenter((args.image_height, args.image_width)), DownscaleFlow(), ToTensor()])
            trainDataset = TartanDataset( posefile = posefile, transform=transform, 
                                                focalx=focalx, focaly=focaly, centerx=centerx, centery=centery)
            
            trainDataloader = DataLoader(trainDataset, batch_size=args.batch_size,
                                                shuffle=False, num_workers=args.worker_num)
            trainDataiter = iter(trainDataloader)

            for batch_idx, sample in enumerate(trainDataiter):
                total_loss,flow_loss,pose_loss,trans_loss,rot_loss = train_pose_batch(model, optimizer, sample)
                current_step += 1
                scheduler.step()
                if current_step % 10 == 0:
                    summaryWriter.add_scalar('Loss/train_total', total_loss, current_step)
                    summaryWriter.add_scalar('Loss/train_flow', flow_loss, current_step)
                    summaryWriter.add_scalar('Loss/train_pose', pose_loss, current_step)
                    summaryWriter.add_scalar('Loss/train_trans', trans_loss, current_step)
                    summaryWriter.add_scalar('Loss/train_rot', rot_loss, current_step)
                    print(f"Epoch {epoch + 1}, Step {current_step}, Loss: {total_loss}")
                    print(f"Flow Loss: {flow_loss}")
                    print(f"Pose Loss: {pose_loss}")

        model_save_path = f'mymodels/model_epoch_{epoch + 1}.pkl'
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")


        



