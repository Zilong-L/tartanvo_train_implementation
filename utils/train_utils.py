import torch
from Datasets.utils import ToTensor,  CropCenter, dataset_intrinsics, DownscaleFlow, Compose,RandomCropAndResized,RandomCropAndResizedFlow,RandomResizeCrop
from Datasets.tartanDataset import TartanDataset

from torch.utils.data import DataLoader,ConcatDataset,DistributedSampler

def load_dataset(posefile,datastr,height,width,flow_only=True,rcr_type="NO_RCR"):
    focalx, focaly, centerx, centery = dataset_intrinsics(datastr) 
    transform = Compose([CropCenter((height,width)), DownscaleFlow(), ToTensor()])
    if rcr_type == "RCR" :
        # if flow_only:
        #     transform = Compose([CropCenter((height,width)), RandomCropAndResizedFlow(),DownscaleFlow(),ToTensor()])
        # else:
        transform = Compose([CropCenter((height,width)), RandomResizeCrop(size=(448, 640)),DownscaleFlow(),ToTensor()])

    dataset = TartanDataset( posefile = posefile, transform=transform, 
                                                focalx=focalx, focaly=focaly, centerx=centerx, centery=centery,flow_only=flow_only)
    return dataset
def get_loader(posefile_path, datastr, height, width, batch_size,  flow_only=True, rcr_type="NO_RCR",shuffle=False,rank=None,world_size=None):
    scene_datasets = []
    print(posefile_path)
    with open(posefile_path, 'r') as f:
        posefiles = f.readlines()
    posefiles = [posefile.strip() for posefile in posefiles]
    for posefile in posefiles:
        scene_datasets.append( load_dataset(posefile, datastr, height, width, flow_only=flow_only, rcr_type=rcr_type))
    dataset = ConcatDataset(scene_datasets)
    dataloader = DataLoader(dataset,sampler=DistributedSampler(dataset=dataset, num_replicas=world_size, rank=rank, shuffle=shuffle),batch_size=batch_size)    
    return dataloader

def save_checkpoint(model, optimizer, scheduler,  iteration, filepath):
    """
    保存模型、优化器和调度器的状态

    参数:
    model (torch.nn.Module): 要保存的模型
    optimizer (torch.optim.Optimizer): 要保存的优化器
    scheduler (torch.optim.lr_scheduler._LRScheduler): 要保存的调度器
    epoch (int): 当前的epoch
    iteration (int): 当前的迭代步数
    filepath (str): 保存文件的路径
    """
    torch.save({
        'iteration': iteration,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, filepath)
def load_checkpoint(model, optimizer=None, scheduler=None, filepath="",map_location='cuda:0'):
    """
    加载模型、优化器和调度器的状态

    参数:
    model (torch.nn.Module): 要加载状态的模型
    optimizer (torch.optim.Optimizer): 要加载状态的优化器
    scheduler (torch.optim.lr_scheduler._LRScheduler): 要加载状态的调度器
    filepath (str): 要加载文件的路径

    返回:
    int: 加载的epoch
    int: 加载的迭代步数
    """
    if filepath=="":
        return 0
    checkpoint = torch.load(filepath, map_location=map_location)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None and 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict'] is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    iteration = checkpoint['iteration']
    print(f"successfully load model from {filepath}")
    return iteration




def process_whole_sample(ddp_model,sample,lambda_flow,device_id):
    sample = {k: v.to(device_id) for k, v in sample.items()} 
    # inputs-------------------------------------------------------------------
    img1 = sample['img1']
    img2 = sample['img2']
    intrinsic_layer = sample['intrinsic']
        
    # forward------------------------------------------------------------------
    flow, relative_motion = ddp_model([img1,img2,intrinsic_layer])


    # loss calculation---------------------------------------------------------
    flow_gt = sample['flow']
    motions_gt = sample['motion']
    flow_loss = ddp_model.module.flowNet.get_loss(flow,flow_gt,small_scale=True)
    pose_loss,trans_loss,rot_loss = ddp_model.module.flowPoseNet.linear_norm_trans_loss(relative_motion, motions_gt)
    total_loss = flow_loss*lambda_flow + pose_loss
    
    return total_loss,flow_loss,pose_loss,trans_loss,rot_loss
def process_flow_sample(ddp_model,sample,lambda_flow,device_id):
    sample = {k: v.to(device_id) for k, v in sample.items()} 
    # inputs-------------------------------------------------------------------
    img1 = sample['img1']
    img2 = sample['img2']
        
    # forward------------------------------------------------------------------
    flow = ddp_model([img1,img2])
    # loss calculation---------------------------------------------------------
    flow_gt = sample['flow']
    flow_loss =  ddp_model.module.get_loss(flow,flow_gt,small_scale=True)
    return flow_loss

def process_flowpose_sample(ddp_model,sample,device_id):
    sample = {k: v.to(device_id) for k, v in sample.items()} 
    # inputs-------------------------------------------------------------------
    intrinsic_layer = sample['intrinsic']
    flow_gt = sample['flow']
        
    flow_input = torch.cat( ( flow_gt, intrinsic_layer ), dim=1 ) 
    # forward------------------------------------------------------------------
    relative_motion = ddp_model(flow_input)


    # loss calculation---------------------------------------------------------
    motions_gt = sample['motion']
    total_loss,trans_loss,rot_loss = calculate_pose_loss(relative_motion, motions_gt,device_id)
    
    return total_loss,trans_loss,rot_loss

def calculate_pose_loss( relative_motion, motions_gt,device_id='cuda:0'):
    
    # Translation loss with normalization
    epsilon = 1e-6
    T_pred = relative_motion[:, :3]
    T_gt = motions_gt[:, :3]
    T_pred_norm = T_pred / torch.max(torch.norm(T_pred, dim=1, keepdim=True), torch.tensor(epsilon).to(device_id))
    T_gt_norm = T_gt / torch.max(torch.norm(T_gt, dim=1, keepdim=True), torch.tensor(epsilon).to(device_id))
    trans_loss = torch.nn.functional.mse_loss(T_pred_norm, T_gt_norm)
    
    # Simple Rotation loss
    R_pred = relative_motion[:, 3:]
    R_gt = motions_gt[:, 3:]
    rot_loss = torch.nn.functional.mse_loss(R_pred, R_gt)

    # Overall motion loss
    pose_loss = trans_loss + rot_loss


    return pose_loss,trans_loss,rot_loss

def test_pose_batch(model, sample):
    model.eval()
    # inputs-------------------------------------------------------------------
    flow_gt = sample['flow']
    motions_gt = sample['motion']
    intrinsic_gt = sample['intrinsic']
    # forward------------------------------------------------------------------
    with torch.no_grad():
        # batch shapes are [batch_size,channels, Height,Width]
        # So concant on dimension 1 not 0
        flow_input = torch.cat((flow_gt, intrinsic_gt), dim=1)
        relative_motion = model(flow_input)
        total_loss,trans_loss,rot_loss = calculate_pose_loss(relative_motion, motions_gt)
       
    relative_motion = relative_motion.cpu().numpy()
    # if 'motion' in sample:
    #     motions_gt = sample['motion'].cpu().numpy()
    #     scale = np.linalg.norm(motions_gt[:,:3], axis=1)
    #     trans_est = relative_motion[:,:3]
    #     trans_est = trans_est/np.linalg.norm(trans_est,axis=1).reshape(-1,1)*scale.reshape(-1,1)
    #     relative_motion[:,:3] = trans_est 
    # else:
    #     print('    scale is not given, using 1 as the default scale value..')

    return relative_motion,total_loss,trans_loss,rot_loss