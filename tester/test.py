import torch
import matplotlib.pyplot as plt
from Datasets.tartanDataset import TartanDataset
from torch.utils.data import DataLoader
from Datasets.utils import ToTensor, Compose, CropCenter, dataset_intrinsics, DownscaleFlow, plot_traj, visflow
from Datasets.tartanTrajFlowDataset import TrajFolderDataset
from Datasets.transformation import ses2poses_quat,se2SE
from evaluator.tartanair_evaluator import TartanAirEvaluator

from utils.train_utils import test_pose_batch,load_checkpoint
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


# FIXME: change model_name with --and testing pose file arguments 
if __name__ == '__main__':
    args = get_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = FlowPoseNet().to(device)
    state_dict = torch.load(args.model_name)['model_state_dict']
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    print(f"Model loaded from {args.model_name}, start from ")
    pose_std_tensor = torch.tensor(np.array([ 0.13,  0.13,  0.13,  0.013 ,  0.013,  0.013], dtype=np.float32) ).cuda()

    datastr = 'tartanair'
    focalx, focaly, centerx, centery = dataset_intrinsics(datastr) 

    # with open('/root/volume/code/python/tartanvo/data/pose_left_paths.txt', 'r') as f:
    #     posefiles = f.readlines()
    
    #FIXME: change pose file here
    posefile = args.pose_file
    transform = Compose([CropCenter((args.image_height, args.image_width)), DownscaleFlow(), ToTensor()])
    testDataset = TartanDataset( posefile = posefile, transform=transform, 
                                        focalx=focalx, focaly=focaly, centerx=centerx, centery=centery)
    
    testDataloader = DataLoader(testDataset, batch_size=args.batch_size,
                                        shuffle=False, num_workers=args.worker_num)
    TestDataiter =  iter(testDataloader)
    motionlist = []
    motion_gts = []
    model.eval()
    with torch.no_grad():
        for sample in TestDataiter:
            sample = {k: v.to(device) for k, v in sample.items()} 
            relative_motion,total_loss,trans_loss,rot_loss = test_pose_batch(model, sample)
            motion_gts.extend(sample['motion'].cpu().numpy())
            print(total_loss)
            motions_gt = sample['motion'].cpu().numpy()
            
            posenp = np.array(relative_motion)
            scale = np.linalg.norm(motions_gt[:,:3], axis=1)
            trans_est = posenp[:,:3]
            trans_est = trans_est/np.linalg.norm(trans_est,axis=1).reshape(-1,1)*scale.reshape(-1,1)
            posenp[:,:3] = trans_est 
            motionlist.extend(posenp)
            
    
    motionlist *= pose_std_tensor.cpu().numpy()
    initial_pose = np.eye(4)
    mostions_gt_SE = [se2SE(x) for x in motionlist]
    poses = []
    for motion in mostions_gt_SE:
        initial_pose = np.matmul(initial_pose, motion)
        poses.append(initial_pose[:3, 3])
        
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot each point in the list
    for x, y, z in poses:
        ax.scatter(x, y, z, marker='o')  # Plot each point

    # Set labels for each axis
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')

    # Show the plot
    plt.show()
    poselist = ses2poses_quat(np.array(motionlist))

    evaluator = TartanAirEvaluator()
    results = evaluator.evaluate_one_trajectory(posefile, poselist, scale=True, kittitype=(datastr=='kitti'))
    if datastr=='euroc':
        print("==> ATE: %.4f" %(results['ate_score']))
    else:
        print("==> ATE: %.4f,\t KITTI-R/t: %.4f, %.4f" %(results['ate_score'], results['kitti_score'][0], results['kitti_score'][1]))

    # save results and visualization
    plot_traj(results['gt_aligned'], results['est_aligned'], vis=False, savefigname='results/test.png', title='ATE %.4f' %(results['ate_score']))