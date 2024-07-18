import torch
from Datasets.utils import ToTensor,  CropCenter, dataset_intrinsics, DownscaleFlow, Compose,RandomCropAndResized,ConsistentRandomResizedCrop
from Datasets.tartanDataset import TartanDataset

from torch.utils.data import DataLoader

def train_pose_batch(model, optimizer, sample):
    model.train()
    # inputs-------------------------------------------------------------------
    flow_gt = sample['flow']
    intrinsic_gt = sample['intrinsic']
    
    # forward------------------------------------------------------------------
    optimizer.zero_grad()  # Zero the parameter gradients
    flow_input = torch.cat( ( flow_gt, intrinsic_gt ), dim=1 ) 
    relative_motion = model(flow_input)

    # loss calculation---------------------------------------------------------
    motions_gt = sample['motion']
    total_loss,trans_loss,rot_loss = calculate_loss(relative_motion, motions_gt)

    # backpropagation----------------------------------------------------------
    total_loss.backward()

    optimizer.step()

    return total_loss,trans_loss,rot_loss
    
def load_from_file(posefile,datastr,height,width,batch_size,worker_num,flow_only=True,rcr_type="NO_RCR"):
    focalx, focaly, centerx, centery = dataset_intrinsics(datastr) 
    transform = Compose([CropCenter((height,width)), DownscaleFlow(), ToTensor()])
    if rcr_type == "RCR":
        transform = Compose([CropCenter((height,width)),DownscaleFlow(), ToTensor(),RandomCropAndResized()])
    elif rcr_type == "CONSISTENT_RCR":
        transform = Compose([CropCenter((height,width)),DownscaleFlow(), ToTensor(),ConsistentRandomResizedCrop()])
        
    dataset = TartanDataset( posefile = posefile, transform=transform, 
                                                focalx=focalx, focaly=focaly, centerx=centerx, centery=centery,flow_only=flow_only)
    dataloader= DataLoader(dataset, batch_size=batch_size,
                                                shuffle=False, num_workers=worker_num)
    return dataloader
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
        total_loss,trans_loss,rot_loss = calculate_loss(relative_motion, motions_gt)
       
    # relative_motion = relative_motion.cpu().numpy()
    # if 'motion' in sample:
    #     motions_gt = sample['motion'].cpu().numpy()
    #     scale = np.linalg.norm(motions_gt[:,:3], axis=1)
    #     trans_est = relative_motion[:,:3]
    #     trans_est = trans_est/np.linalg.norm(trans_est,axis=1).reshape(-1,1)*scale.reshape(-1,1)
    #     relative_motion[:,:3] = trans_est 
    # else:
    #     print('    scale is not given, using 1 as the default scale value..')

    return total_loss,trans_loss,rot_loss



def calculate_loss(relative_motion, motions_gt,flow_prediction,flow_gt,lambda_flow=0.1):
    
    flow_loss = torch.nn.functional.mse_loss(flow_prediction, flow_gt)
    flow_loss = lambda_flow * flow_loss
    
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

    total_loss = flow_loss + pose_loss
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
def load_model(model, optimizer=None, scheduler=None, filepath=""):
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
    state_dict = checkpoint['model_state_dict']
    if any(key.startswith('module.') for key in state_dict.keys()):
        state_dict = {key[7:]: value for key, value in state_dict.items()}
    model.load_state_dict(state_dict)
    if optimizer is not None and 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict'] is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']  # 如果epoch没保存，默认值为0
    iteration = checkpoint['iteration']
    print(f"successfully load model from {filepath}")
    return epoch+1, iteration

def validate(model, testposefiles, device,RCR=False ):
    model.eval()
    N = 0
    total_loss,trans_loss,rot_loss = 0,0,0
    for posefile in testposefiles:
        posefile = posefile.strip()
        testdataiter = load_from_file(posefile, datastr='tartanair', height=448, width=640, batch_size=100, worker_num=16,RCR=RCR)
        for sample in testdataiter:
            sample = {k: v.to(device) for k, v in sample.items()}
            total,trans,rot = test_pose_batch(model, sample )
            N += 1   
            total_loss += total.item()  # Ensure losses are scalars, not tensors
            trans_loss += trans.item()
            rot_loss += rot.item()
            
    return total_loss/N,trans_loss/N,rot_loss/N
            