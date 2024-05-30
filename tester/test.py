import torch
import matplotlib.pyplot as plt
from Datasets.tartanDataset import TartanDataset
from torch.utils.data import DataLoader
from Datasets.utils import ToTensor, Compose, CropCenter, dataset_intrinsics, DownscaleFlow, plot_traj, visflow
from Datasets.tartanTrajFlowDataset import TrajFolderDataset
from Datasets.transformation import ses2poses_quat
from evaluator.tartanair_evaluator import TartanAirEvaluator

from utils.train_pose_utils import test_pose_batch,load_model
from Network.VOFlowNet import VOFlowRes as FlowPoseNet

import argparse
import numpy as np


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
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = FlowPoseNet()
    model = torch.nn.DataParallel(model).to(device)
    start_epoch,iteration = load_model(model,  filepath=args.model_name)
    print(f"Model loaded from {args.model_name}, start from epoch {start_epoch}, iteration {iteration}")
    pose_std_tensor = torch.tensor(np.array([ 0.13,  0.13,  0.13,  0.013 ,  0.013,  0.013], dtype=np.float32) ).cuda()

    datastr = 'tartanair'
    focalx, focaly, centerx, centery = dataset_intrinsics(datastr) 

    # with open('/root/volume/code/python/tartanvo/data/pose_left_paths.txt', 'r') as f:
    #     posefiles = f.readlines()
    

    posefile = '/root/volume/code/python/tartanvo/data/test/carwelding/Easy/P002/pose_left.txt'
    transform = Compose([CropCenter((args.image_height, args.image_width)), DownscaleFlow(), ToTensor()])
    testDataset = TartanDataset( posefile = posefile, transform=transform, 
                                        focalx=focalx, focaly=focaly, centerx=centerx, centery=centery)
    
    testDataloader = DataLoader(testDataset, batch_size=args.batch_size,
                                        shuffle=False, num_workers=args.worker_num)
    TestDataiter =  iter(testDataloader)
    motionlist = []
    motion_gts = []
    for sample in TestDataiter:
        sample = {k: v.to(device) for k, v in sample.items()} 
        relative_motion,total_loss,trans_loss,rot_loss = test_pose_batch(model, sample,pose_std_tensor)
        motionlist.extend(relative_motion)

    print(f"Loss: {total_loss.item()}, translation loss: {trans_loss.item()}")
    poselist = ses2poses_quat(np.array(motionlist))
    positions = np.array([pose[:3] for pose in poselist])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], label='Model Output')
    ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], c='red', marker='o', label='Start')
    ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], c='green', marker='o', label='End')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    plt.title('Trajectory Without Alignment')
    plt.show()


    evaluator = TartanAirEvaluator()
    results = evaluator.evaluate_one_trajectory(posefile, poselist, scale=True, kittitype=(datastr=='kitti'))
    if datastr=='euroc':
        print("==> ATE: %.4f" %(results['ate_score']))
    else:
        print("==> ATE: %.4f,\t KITTI-R/t: %.4f, %.4f" %(results['ate_score'], results['kitti_score'][0], results['kitti_score'][1]))

    # save results and visualization
    plot_traj(results['gt_aligned'], results['est_aligned'], vis=False, savefigname='results/test.png', title='ATE %.4f' %(results['ate_score']))