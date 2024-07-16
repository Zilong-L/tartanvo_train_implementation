import numpy as np
import os
import cv2
from torch import float32
from torch.utils.data import Dataset
from os import listdir
from .transformation import pos_quats2SEs, pose2motion, SEs2ses, pose2motion_bug
from .utils import make_intrinsics_layer

def get_data_dir(scene_path):
    base_dir = os.path.dirname(scene_path)
    rgb_dir = os.path.join(base_dir, 'image_left')
    flow_dir = os.path.join(base_dir, 'flow')
    return rgb_dir, flow_dir

class TartanDataset(Dataset):
    """scene flow synthetic dataset. """

    def __init__(self, posefile=None, transform=None, 
                 focalx=320.0, focaly=320.0, centerx=320.0, centery=240.0, flow_only=False):
        self.focalx = focalx
        self.focaly = focaly
        self.centerx = centerx
        self.centery = centery
        self.flow_only = flow_only
        self.transform = transform
        self.pose_std = np.array([0.13, 0.13, 0.13, 0.013, 0.013, 0.013], dtype=np.float32)

        rgb_dir, flow_dir = get_data_dir(posefile)
        
        self.data = {'images': {}, 'flows': {}}
        
        if not flow_only: 
            rgb_files = [os.path.join(rgb_dir, ff) for ff in listdir(rgb_dir) if (ff.endswith('.png') or ff.endswith('.jpg'))]
            rgb_files.sort()
            self.data['images'] = {i: cv2.imread(file) for i, file in enumerate(rgb_files)}

        flow_files = [os.path.join(flow_dir, ff) for ff in listdir(flow_dir) if ff.endswith('.npy')]
        flow_files.sort()
        self.data['flows'] = {i: np.load(file) for i, file in enumerate(flow_files)}
        
        print(f'Find {len(self.data["flows"])} flow files in {flow_dir}')

        if posefile is not None and posefile != "":
            poselist = np.loadtxt(posefile).astype(np.float32)
            assert poselist.shape[1] == 7  # position + quaternion
            poses = pos_quats2SEs(poselist)  # quats to 4x3 matrix
            self.matrix = pose2motion(poses)
            self.motions = SEs2ses(self.matrix).astype(np.float32)
            # normalization for training 
            self.motions = self.motions / self.pose_std
            assert len(self.motions) == len(self.data['flows'])
        else:
            self.motions = None

        self.N = len(self.data['flows'])

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        res = {}
        if not self.flow_only:
            img1 = self.data['images'][idx]
            img2 = self.data['images'][idx + 1]

            res['img1'] = img1
            res['img2'] = img2

        flow = self.data['flows'][idx]
        res['flow'] = flow

        h, w, _ = flow.shape
        intrinsic_layer = make_intrinsics_layer(w, h, self.focalx, self.focaly, self.centerx, self.centery)
        res['intrinsic'] = intrinsic_layer
        
        if self.transform:
            res = self.transform(res)

        res['motion'] = self.motions[idx]
        return res
