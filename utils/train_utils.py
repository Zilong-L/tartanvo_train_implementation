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
        flow_loss = torch.nn.functional.mse_loss(flow, flow_gt,reduction='sum')

    # Translation loss with normalization
    epsilon = 1e-6
    T_pred = relative_motion[:, :3]
    T_gt = motions_gt[:, :3]
    T_pred_norm = T_pred / torch.max(torch.norm(T_pred, dim=1, keepdim=True), torch.tensor(epsilon).cuda())
    T_gt_norm = T_gt / torch.max(torch.norm(T_gt, dim=1, keepdim=True), torch.tensor(epsilon).cuda())
    trans_loss = torch.nn.functional.mse_loss(T_pred_norm, T_gt_norm,reduction='sum')
    
    # Simple Rotation loss
    R_pred = relative_motion[:, 3:]
    R_gt = motions_gt[:, 3:]
    rot_loss = torch.nn.functional.mse_loss(R_pred, R_gt,reduction='sum')

    # Overall motion loss
    pose_loss = trans_loss + rot_loss

    # Combined loss
    total_loss = lambda_f * flow_loss + pose_loss

    return total_loss,flow_loss,pose_loss,trans_loss,rot_loss
