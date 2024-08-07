from __future__ import division
import torch
import math
import random
import numpy as np
import numbers
import cv2
import matplotlib.pyplot as plt
import torchvision
from torchvision.transforms import RandomResizedCrop
from torchvision.transforms.functional import resized_crop
import os
if ( not ( "DISPLAY" in os.environ ) ):
    plt.switch_backend('agg')
    print("Environment variable DISPLAY is not present in the system.")
    print("Switch the backend of matplotlib to agg.")
import time
# ===== general functions =====

class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (List[Transform]): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


class DownscaleFlow(object):
    """
    Scale the flow and mask to a fixed size

    """
    def __init__(self, scale=4):
        '''
        size: output frame size, this should be NO LARGER than the input frame size! 
        '''
        self.downscale = 1.0/scale

    def __call__(self, sample): 
        if self.downscale!=1 and 'flow' in sample :
            sample['flow'] = cv2.resize(sample['flow'], 
                (0, 0), fx=self.downscale, fy=self.downscale, interpolation=cv2.INTER_LINEAR)

        if self.downscale!=1 and 'intrinsic' in sample :
            sample['intrinsic'] = cv2.resize(sample['intrinsic'], 
                (0, 0), fx=self.downscale, fy=self.downscale, interpolation=cv2.INTER_LINEAR)

        if self.downscale!=1 and 'fmask' in sample :
            sample['fmask'] = cv2.resize(sample['fmask'],
                (0, 0), fx=self.downscale, fy=self.downscale, interpolation=cv2.INTER_LINEAR)
        return sample
    
    
class CropCenter(object):
    """Crops the a sample of data (tuple) at center
    if the image size is not large enough, it will be first resized with fixed ratio
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, sample):
        kks = list(sample.keys())
        th, tw = self.size
        h, w = sample[kks[0]].shape[0], sample[kks[0]].shape[1]
        if w == tw and h == th:
            return sample

        # resize the image if the image size is smaller than the target size
        scale_h, scale_w, scale = 1., 1., 1.
        if th > h:
            scale_h = float(th)/h
        if tw > w:
            scale_w = float(tw)/w
        if scale_h>1 or scale_w>1:
            scale = max(scale_h, scale_w)
            w = int(round(w * scale)) # w after resize
            h = int(round(h * scale)) # h after resize

        x1 = int((w-tw)/2)
        y1 = int((h-th)/2)

        for kk in kks:
            if sample[kk] is None:
                continue
            img = sample[kk]
            if len(img.shape)==3:
                if scale>1:
                    img = cv2.resize(img, (w,h), interpolation=cv2.INTER_LINEAR)
                sample[kk] = img[y1:y1+th,x1:x1+tw,:]
            elif len(img.shape)==2:
                if scale>1:
                    img = cv2.resize(img, (w,h), interpolation=cv2.INTER_LINEAR)
                sample[kk] = img[y1:y1+th,x1:x1+tw]

        return sample

class ToTensor(object):
    def __call__(self, sample):
        sss = time.time()

        kks = list(sample)

        for kk in kks:
            data = sample[kk]
            data = data.astype(np.float32) 
            if len(data.shape) == 3: # transpose image-like data
                data = data.transpose(2,0,1)
            elif len(data.shape) == 2:
                data = data.reshape((1,)+data.shape)

            if len(data.shape) == 3 and data.shape[0]==3: # normalization of rgb images
                data = data/255.0
            
            sample[kk] = torch.from_numpy(data.copy()) # copy to make memory continuous

        return sample


def tensor2img(tensImg,mean,std):
    """
    convert a tensor a numpy array, for visualization
    """
    # undo normalize
    for t, m, s in zip(tensImg, mean, std):
        t.mul_(s).add_(m) 
    tensImg = tensImg * float(255)
    # undo transpose
    tensImg = (tensImg.numpy().transpose(1,2,0)).astype(np.uint8)
    return tensImg

def bilinear_interpolate(img, h, w):
    # assert round(h)>=0 and round(h)<img.shape[0]
    # assert round(w)>=0 and round(w)<img.shape[1]

    h0 = int(math.floor(h))
    h1 = h0 + 1
    w0 = int(math.floor(w))
    w1 = w0 + 1

    a = h - h0 
    b = w - w0

    h0 = max(h0, 0)
    w0 = max(w0, 0)
    h1 = min(h1, img.shape[0]-1)
    w1 = min(w1, img.shape[1]-1)

    A = img[h0,w0,:]
    B = img[h1,w0,:]
    C = img[h0,w1,:]
    D = img[h1,w1,:]

    res = (1-a)*(1-b)*A + a*(1-b)*B + (1-a)*b*C + a*b*D

    return res 

def calculate_angle_distance_from_du_dv(du, dv, flagDegree=False):
    a = np.arctan2( dv, du )

    angleShift = np.pi

    if ( True == flagDegree ):
        a = a / np.pi * 180
        angleShift = 180
        # print("Convert angle from radian to degree as demanded by the input file.")

    d = np.sqrt( du * du + dv * dv )

    return a, d, angleShift

def visflow(flownp, maxF=500.0, n=8, mask=None, hueMax=179, angShift=0.0): 
    """
    Show a optical flow field as the KITTI dataset does.
    Some parts of this function is the transform of the original MATLAB code flow_to_color.m.
    """

    ang, mag, _ = calculate_angle_distance_from_du_dv( flownp[:, :, 0], flownp[:, :, 1], flagDegree=False )

    # Use Hue, Saturation, Value colour model 
    hsv = np.zeros( ( ang.shape[0], ang.shape[1], 3 ) , dtype=np.float32)

    am = ang < 0
    ang[am] = ang[am] + np.pi * 2

    hsv[ :, :, 0 ] = np.remainder( ( ang + angShift ) / (2*np.pi), 1 )
    hsv[ :, :, 1 ] = mag / maxF * n
    hsv[ :, :, 2 ] = (n - hsv[:, :, 1])/n

    hsv[:, :, 0] = np.clip( hsv[:, :, 0], 0, 1 ) * hueMax
    hsv[:, :, 1:3] = np.clip( hsv[:, :, 1:3], 0, 1 ) * 255
    hsv = hsv.astype(np.uint8)

    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    if ( mask is not None ):
        mask = mask != 255
        bgr[mask] = np.array([0, 0 ,0], dtype=np.uint8)

    return bgr


def dataset_intrinsics(dataset='tartanair'):
    if dataset == 'kitti':
        focalx, focaly, centerx, centery = 707.0912, 707.0912, 601.8873, 183.1104
    elif dataset == 'euroc':
        focalx, focaly, centerx, centery = 458.6539916992, 457.2959899902, 367.2149963379, 248.3750000000
    elif dataset == 'tartanair':
        focalx, focaly, centerx, centery = 320.0, 320.0, 320.0, 240.0
    else:
        return None
    return focalx, focaly, centerx, centery



def plot_traj(gtposes, estposes, vis=False, savefigname=None, title=''):
    fig = plt.figure(figsize=(4,4))
    cm = plt.cm.get_cmap('Spectral')

    plt.subplot(111)
    plt.plot(gtposes[:,0],gtposes[:,1], linestyle='dashed',c='k')
    plt.plot(estposes[:, 0], estposes[:, 1],c='#ff7f0e')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.legend(['Ground Truth','TartanVO'])
    plt.title(title)
    if savefigname is not None:
        plt.savefig(savefigname)
    if vis:
        plt.show()
    plt.close(fig)

def make_intrinsics_layer(w, h, fx, fy, ox, oy):
    ww, hh = np.meshgrid(range(w), range(h))
    ww = (ww.astype(np.float32) - ox + 0.5 )/fx
    hh = (hh.astype(np.float32) - oy + 0.5 )/fy
    intrinsicLayer = np.stack((ww,hh)).transpose(1,2,0)

    return intrinsicLayer

def load_kiiti_intrinsics(filename):
    '''
    load intrinsics from kitti intrinsics file
    '''
    with open(filename, 'r') as f:
        lines = f.readlines()
    cam_intrinsics = lines[2].strip().split(' ')[1:]
    focalx, focaly, centerx, centery = float(cam_intrinsics[0]), float(cam_intrinsics[5]), float(cam_intrinsics[2]), float(cam_intrinsics[6])

    return focalx, focaly, centerx, centery

import torch
import random
import torchvision.transforms.functional as F

class RandomResizeCrop(object):
    """
    Random scale to cover continuous focal length
    Due to the tartanair focal is already small, we only up scale the image

    """

    def __init__(self, size, max_scale=2.5, keep_center=False, fix_ratio=False, scale_disp=False):
        '''
        size: output frame size, this should be NO LARGER than the input frame size! 
        scale_disp: when training the stereovo, disparity represents depth, which is not scaled with resize 
        '''
        if isinstance(size, numbers.Number):
            self.target_h = int(size)
            self.target_w = int(size)
        else:
            self.target_h = size[0]
            self.target_w = size[1]

        # self.max_focal = max_focal
        self.keep_center = keep_center
        self.fix_ratio = fix_ratio
        self.scale_disp = scale_disp
        # self.tartan_focal = 320.

        # assert self.max_focal >= self.tartan_focal
        self.scale_base = max_scale #self.max_focal /self.tartan_focal

    def __call__(self, sample): 
        for kk in sample:
            if len(sample[kk].shape)>=2:
                h, w = sample[kk].shape[0], sample[kk].shape[1]
                break
        self.target_h = min(self.target_h, h)
        self.target_w = min(self.target_w, w)

        scale_w, scale_h, x1, y1, crop_w, crop_h = generate_random_scale_crop(h, w, self.target_h, self.target_w, 
                                                    self.scale_base, self.keep_center, self.fix_ratio)

        for kk in sample:
            # if kk in ['flow', 'flow2', 'img0', 'img0n', 'img1', 'img1n', 'intrinsic', 'fmask', 'disp0', 'disp1', 'disp0n', 'disp1n']:
            if len(sample[kk].shape)>=2 or kk in ['fmask', 'fmask2']:
                sample[kk] = sample[kk][y1:y1+crop_h, x1:x1+crop_w]
                sample[kk] = cv2.resize(sample[kk], (0,0), fx=scale_w, fy=scale_h, interpolation=cv2.INTER_LINEAR)
                # Note opencv reduces the last dimention if it is one
                sample[kk] = sample[kk][:self.target_h,:self.target_w]

        # scale the flow
        if 'flow' in sample:
            sample['flow'][:,:,0] = sample['flow'][:,:,0] * scale_w
            sample['flow'][:,:,1] = sample['flow'][:,:,1] * scale_h
        # scale the flow
        if 'flow2' in sample:
            sample['flow2'][:,:,0] = sample['flow2'][:,:,0] * scale_w
            sample['flow2'][:,:,1] = sample['flow2'][:,:,1] * scale_h

        if self.scale_disp: # scale the depth
            if 'disp0' in sample:
                sample['disp0'][:,:] = sample['disp0'][:,:] * scale_w
            if 'disp1' in sample:
                sample['disp1'][:,:] = sample['disp1'][:,:] * scale_w
            if 'disp0n' in sample:
                sample['disp0n'][:,:] = sample['disp0n'][:,:] * scale_w
            if 'disp1n' in sample:
                sample['disp1n'][:,:] = sample['disp1n'][:,:] * scale_w
        else:
            sample['scale_w'] = np.array([scale_w ])# used in e2e-stereo-vo

        return sample
def generate_random_scale_crop(h, w, target_h, target_w, scale_base, keep_center, fix_ratio):
    '''
    Randomly generate scale and crop params
    H: input image h
    w: input image w
    target_h: output image h
    target_w: output image w
    scale_base: max scale up rate
    keep_center: crop at center
    fix_ratio: scale_h == scale_w
    '''
    scale_w = random.random() * (scale_base - 1) + 1
    if fix_ratio:
        scale_h = scale_w
    else:
        scale_h = random.random() * (scale_base - 1) + 1

    crop_w = int(math.ceil(target_w/scale_w)) # ceil for redundancy
    crop_h = int(math.ceil(target_h/scale_h)) # crop_w * scale_w > w

    if keep_center:
        x1 = int((w-crop_w)/2)
        y1 = int((h-crop_h)/2)
    else:
        x1 = random.randint(0, w - crop_w)
        y1 = random.randint(0, h - crop_h)

    return scale_w, scale_h, x1, y1, crop_w, crop_h
class RandomCropAndResized(object):
    # TODO: Implement handling for RGB images in phase two of development.
    # DONE: RGB images are currently not included in the RandomResizedCrop (RCR) process.
    # TODO: Consider implementing a "Consistent RandomResizedCrop" mechanism.
    # I am not sure whether is function meets the paper's requirements. 
    # Current implementation results in different crops even within the same scene, which may not be ideal.
    # Consideration: Ensure the cropped region remains consistent across a single scene.
    """
    Crop the input data at a random location and resize it to the target size.
    """
    def __init__(self):
        self.transform = RandomResizedCrop(size=(448, 640), scale=(0.08, 1.0), ratio=(3./4., 4./3.))
        
        
    def __call__(self, sample): 
        flow = torch.tensor(sample['flow']).permute(2, 0, 1)  # (H, W, 2) -> (2, H, W)
        intrinsic = torch.tensor(sample['intrinsic']).permute(2, 0, 1)  # (H, W, 2) -> (2, H, W)
        img1 = torch.tensor(sample['img1']).permute(2, 0, 1)  # (H, W, 3) -> (3, H, W)
        img2 = torch.tensor(sample['img2']).permute(2, 0, 1)  # (H, W, 3) -> (3, H, W)
        # Resulting shape [10, H, W] - 2 for flow, 2 for intrinsic, 3 for img1, and 3 for img2
        combined = torch.cat([flow,intrinsic,img1,img2], dim=0)

        # Apply the same transform to the combined tensor
        transformed = self.transform(combined)

        # Split the transformed tensor back into 'flow', 'intrinsic', 'img1', and 'img2'
        sample['flow'] = transformed[:2].permute(1, 2, 0).numpy()  # First two channels
        sample['intrinsic'] = transformed[2:4].permute(1, 2, 0).numpy()  # Next two channels
        sample['img1'] = transformed[4:7].permute(1, 2, 0).numpy()   # Next three channels
        sample['img2'] = transformed[7:10].permute(1, 2, 0).numpy()   # Last three channels
        return sample
class RandomCropAndResizedFlow(object):
    # TODO: Implement handling for RGB images in phase two of development.
    # DONE: RGB images are currently not included in the RandomResizedCrop (RCR) process.
    # TODO: Consider implementing a "Consistent RandomResizedCrop" mechanism.
    # I am not sure whether is function meets the paper's requirements. 
    # Current implementation results in different crops even within the same scene, which may not be ideal.
    # Consideration: Ensure the cropped region remains consistent across a single scene.
    """
    Crop the input data at a random location and resize it to the target size.
    """
    def __init__(self):
        self.transform = RandomResizedCrop(size=(448, 640), scale=(0.08, 1.0), ratio=(3./4., 4./3.))
        
        
    def __call__(self, sample): 
        flow = torch.tensor(sample['flow']).permute(2, 0, 1)  # (H, W, 2) -> (2, H, W)
        intrinsic = torch.tensor(sample['intrinsic']).permute(2, 0, 1)  # (H, W, 2) -> (2, H, W)
        combined = torch.cat([flow,intrinsic], dim=0)

        # Apply the same transform to the combined tensor
        transformed = self.transform(combined)

        # Split the transformed tensor back into 'flow', 'intrinsic', 'img1', and 'img2'
        sample['flow'] = transformed[:2].permute(1, 2, 0).numpy()  # First two channels
        sample['intrinsic'] = transformed[2:4].permute(1, 2, 0).numpy()  # Next two channels
        return sample
class ConsistentRandomResizedCrop:
    def __init__(self):
        self.input_size = (112,160)
        self.min_scale = 0.4
        self.max_scale = 1.0
        # Initialize cropping parameters
        self.initialize_cropping_params()
        print("Random crop parameters: ", self.top, self.left, self.crop_height, self.crop_width)

    def initialize_cropping_params(self):
        image_size = [112, 160]
        # Randomly determine the scale of the crop
        scale = random.uniform(self.min_scale, self.max_scale)
        self.crop_height = int(image_size[0] * scale)
        self.crop_width = int(image_size[1] * scale)

        # Randomly choose the top left corner of the crop area
        self.top = random.randint(0, image_size[0] - self.crop_height)
        self.left = random.randint(0, image_size[1] - self.crop_width)


    def __call__(self, sample):
        # Resulting shape [4, H, W]
        combined = torch.cat([sample['flow'], sample['intrinsic']], dim=0)  
        
        # Perform the crop and resize
        transformed = resized_crop(combined, self.top, self.left, self.crop_height, self.crop_width, self.input_size,
                                     interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
        # Split the transformed tensor back into 'flow' and 'intrinsic'
        sample['flow'] = transformed[:2]  # First two channels
        sample['intrinsic'] = transformed[2:]  # Next two channels
        return sample

