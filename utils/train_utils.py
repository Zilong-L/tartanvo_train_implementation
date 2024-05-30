import torch
import numpy as np
pose_std = np.array([ 0.13,  0.13,  0.13,  0.013 ,  0.013,  0.013], dtype=np.float32) # the output scale factor
flow_norm = 20 # scale factor for flow

pose_std_tensor = torch.tensor(pose_std).cuda()
flow_norm_tensor = torch.tensor(flow_norm).cuda()


def train_pose_batch(model, optimizer, sample):
    model.train()
    # inputs-------------------------------------------------------------------
    flow_gt = sample['flow'].cuda()
    intrinsic_gt = sample['intrinsic'].cuda()
    
    # forward------------------------------------------------------------------
    optimizer.zero_grad()  # Zero the parameter gradients
    flow_input = torch.cat( ( flow_gt, intrinsic_gt ), dim=1 ) 
    relative_motion = model.module.flowPoseNet(flow_input)

    # normalization------------------------------------------------------------
    relative_motion /= pose_std_tensor

    # loss calculation---------------------------------------------------------
    motions_gt = sample['motion'].cuda()
    # flow_gt is not fed into calculate_loss, so no flow loss for pose only training.
    total_loss,flow_loss,pose_loss,trans_loss,rot_loss = calculate_loss(relative_motion, motions_gt)

    # backpropagation----------------------------------------------------------
    total_loss.backward()
    max_norm = 2.0
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

    optimizer.step()

    return total_loss,flow_loss,pose_loss,trans_loss,rot_loss
def test_batch(model, sample):
    model.eval()
    img0 = sample['img1'].cuda()
    img1 = sample['img2'].cuda()
    intrinsic = sample['intrinsic'].cuda()
    
    # inference----------------------------------------------------------------
    inputs = [img0, img1, intrinsic]

    # forward------------------------------------------------------------------
    with torch.no_grad():
        flow, relative_motion = model(inputs)
        relative_motion *= pose_std_tensor
        # calculate scale from GT posefile
    relative_motion = relative_motion.cpu().numpy()
    # if 'motion' in sample:
    #     motions_gt = sample['motion']
    #     scale = np.linalg.norm(motions_gt[:,:3], axis=1)
    #     trans_est = relative_motion[:,:3]
    #     trans_est = trans_est/np.linalg.norm(trans_est,axis=1).reshape(-1,1)*scale.reshape(-1,1)
    #     relative_motion[:,:3] = trans_est 
    # else:
    #     print('    scale is not given, using 1 as the default scale value..')

    return relative_motion

def test_pose_batch(model, sample):
    model.eval()
    # inputs-------------------------------------------------------------------
    flow_gt = sample['flow'].cuda()
    intrinsic_gt = sample['intrinsic'].cuda()

    # forward------------------------------------------------------------------
    with torch.no_grad():
        flow_input = torch.cat((flow_gt, intrinsic_gt), dim=1)
        relative_motion = model.module.flowPoseNet(flow_input)
        relative_motion *= pose_std_tensor
    relative_motion = relative_motion.cpu().numpy()
    if 'motion' in sample:
        motions_gt = sample['motion']
        scale = np.linalg.norm(motions_gt[:,:3], axis=1)
        trans_est = relative_motion[:,:3]
        trans_est = trans_est/np.linalg.norm(trans_est,axis=1).reshape(-1,1)*scale.reshape(-1,1)
        relative_motion[:,:3] = trans_est 
    else:
        print('    scale is not given, using 1 as the default scale value..')

    return relative_motion


def train_batch(model, optimizer, sample):
    model.train()
    # inputs-------------------------------------------------------------------
    img0 = sample['img1'].cuda()
    img1 = sample['img2'].cuda()
    intrinsic = sample['intrinsic'].cuda()
    
    # inference----------------------------------------------------------------
    optimizer.zero_grad()  # Zero the parameter gradients
    inputs = [img0, img1, intrinsic]
    flow, relative_motion = model(inputs)

    # normalization------------------------------------------------------------
    relative_motion /= pose_std_tensor
    flow /= flow_norm_tensor

    # loss calculation---------------------------------------------------------
    motions_gt = sample['motion'].cuda()
    total_loss,flow_loss,pose_loss,trans_loss,rot_loss = calculate_loss(relative_motion, motions_gt)

    total_loss.backward()
    optimizer.step()

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}.grad: {param.grad}")

    return total_loss,flow_loss,pose_loss,trans_loss,rot_loss

def calculate_loss( relative_motion, motions_gt,flow=None , flow_gt=None, lambda_f=1.0):
    # Optical flow loss
    if flow is None or flow_gt is None:
        flow_loss = 0
    else:
        flow_loss = torch.nn.functional.mse_loss(flow, flow_gt)

    # Translation loss with normalization
    epsilon = 1e-6
    T_pred = relative_motion[:, :3]
    T_gt = motions_gt[:, :3]
    T_pred_norm = T_pred / torch.max(torch.norm(T_pred, dim=1, keepdim=True), torch.tensor(epsilon).cuda())
    T_gt_norm = T_gt / torch.max(torch.norm(T_gt, dim=1, keepdim=True), torch.tensor(epsilon).cuda())
    trans_loss = torch.nn.functional.mse_loss(T_pred_norm, T_gt_norm)
    
    # Simple Rotation loss
    R_pred = relative_motion[:, 3:]
    R_gt = motions_gt[:, 3:]
    rot_loss = torch.nn.functional.mse_loss(R_pred, R_gt)

    # Overall motion loss
    pose_loss = trans_loss + rot_loss

    # Combined loss
    total_loss = lambda_f * flow_loss + pose_loss

    return total_loss,flow_loss,pose_loss,trans_loss,rot_loss



def save_model(model, optimizer, scheduler, epoch, iteration, filepath):
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
        'epoch': epoch,
        'iteration': iteration,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, filepath)
def load_model(model, optimizer, scheduler, filepath):
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
        return 0,0
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(checkpoint['optimizer_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    print(scheduler)
    epoch = checkpoint['epoch']  # 如果epoch没保存，默认值为0
    iteration = checkpoint['iteration']
    return epoch, iteration