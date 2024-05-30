import numpy as np
import os
import cv2
from torch import float32
from torch.utils.data import Dataset, DataLoader
from os import listdir
from .transformation import pos_quats2SEs, pose2motion, SEs2ses,pose2motion_bug
from .utils import make_intrinsics_layer

def get_data_dir(scene_path):
    base_dir = os.path.dirname(scene_path)
    rgb_dir = os.path.join(base_dir, 'image_left')
    flow_dir = os.path.join(base_dir, 'flow')
    return rgb_dir, flow_dir

class TartanDataset(Dataset):
    """scene flow synthetic dataset. """

    def __init__(self,  posefile = None, transform = None, 
                    focalx = 320.0, focaly = 320.0, centerx = 320.0, centery = 240.0,flowonly=False):
        self.focalx = focalx
        self.focaly = focaly
        self.centerx = centerx
        self.centery = centery
        self.flowonly = flowonly
        
        self.transform = transform
        self.pose_std = np.array([ 0.13,  0.13,  0.13,  0.013 ,  0.013,  0.013], dtype=np.float32)

        rgb_dir,flow_dir = get_data_dir(posefile)
        if not flowonly: 
            rgb_files = listdir(rgb_dir)
            self.rgbfiles = [(rgb_dir +'/'+ ff) for ff in rgb_files if (ff.endswith('.png') or ff.endswith('.jpg'))]
            self.rgbfiles.sort()

        flow_files = listdir(flow_dir)
        self.flowfiles = [(flow_dir +'/'+ ff) for ff in flow_files if ff.endswith('.npy')]
        self.flowfiles.sort()

        if not flowonly: 
            print('Find {} image files in {}'.format(len(self.rgbfiles), rgb_dir))
        print('Find {} flow files in {}'.format(len(self.flowfiles), flow_dir))

        if posefile is not None and posefile!="":
            poselist = np.loadtxt(posefile).astype(np.float32)
            assert(poselist.shape[1]==7) # position + quaternion
            poses = pos_quats2SEs(poselist) # quats to 4x3 matrix
            self.matrix = pose2motion(poses)
            self.motions = SEs2ses(self.matrix).astype(np.float32)
            #  normalization for training 
            self.motions = self.motions / self.pose_std
            #  normalization for training  
            assert(len(self.motions) == len(self.flowfiles))
        else:
            self.motions = None

        self.N = len(self.flowfiles)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        res = {}
        if not self.flowonly:
            imgfile1 = self.rgbfiles[idx].strip()
            imgfile2 = self.rgbfiles[idx+1].strip()

            img1 = cv2.imread(imgfile1)
            img2 = cv2.imread(imgfile2)

            res['img1'] = img1
            res['img2'] = img2

        flowfile = self.flowfiles[idx].strip()
        flow = np.load(flowfile)
        res['flow'] = flow

        h, w, _ = flow.shape
        intrinsicLayer = make_intrinsics_layer(w, h, self.focalx, self.focaly, self.centerx, self.centery)
        res['intrinsic'] = intrinsicLayer
        if self.transform:
            res = self.transform(res)

        res['motion'] = self.motions[idx]
        return res


