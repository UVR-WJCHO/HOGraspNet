import os
import sys
sys.path.append(os.environ['HOG_DIR'])
sys.path.append(os.path.join(os.environ['HOG_DIR'], "thirdparty/manopth"))

from HOG_dataloader import HOGDataset
import torch
from torch.utils.data import DataLoader
from config import cfg
from scripts.util.utils_vis import *
import numpy as np
from thirdparty.manopth.manopth.manolayer import ManoLayer



setup = 's0'
split = 'test'
vis_num = 10

save_path = os.path.join(os.environ['HOG_DIR'], "vis")
os.makedirs(save_path, exist_ok=True)

HOG = HOGDataset(setup, split)
HOG_loader = DataLoader(HOG, batch_size=1, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mano_layer = ManoLayer(side='right', mano_root=cfg.mano_path, use_pca=False, flat_hand_mean=True,
                            center_idx=0, ncomps=45, root_rot_mode="axisang", joint_rot_mode="axisang").to(device)
hand_faces_template = mano_layer.th_faces.repeat(1, 1, 1)


default_M = np.eye(4)[:3]
default_K = np.eye(3)
renderer_set = {}
for i in range(4):
    renderer = Renderer(device, 1, default_M, default_K, (1080, 1920))
    renderer_set[cfg._CAMIDSET[i]] = renderer



prev_seq_name = None
for idx, sample in enumerate(HOG_loader):
    if idx > vis_num:
        break

    K = sample['intrinsics']
    M = sample['extrinsics']

    rgb = sample['rgb_data']
    depth = sample['depth_data']
    bbox = sample['bbox']
    anno = sample['anno_data']

    # update renderer if sequence has changed
    cam = sample['camera']
    seq_name = sample['rgb_path'].split('/')[-5]
    trial_name = sample['rgb_path'].split('/')[-4]
    img_name = sample['rgb_path'].split('/')[-1]
    if seq_name != prev_seq_name:
        renderer_set[cam].update_intrinsic(K)
    
    hand_joints = anno['annotations'][0]['data']
    hand_mano_rot = anno['Mesh'][0]['mano_trans']
    hand_mano_pose = anno['Mesh'][0]['mano_pose']
    hand_mano_shape = anno['Mesh'][0]['mano_betas']

    hand_mano_rot = torch.FloatTensor(np.asarray(hand_mano_rot))
    hand_mano_pose = torch.FloatTensor(np.asarray(hand_mano_pose))
    hand_mano_shape = torch.FloatTensor(np.asarray(hand_mano_shape)).to(device)

    mano_param = torch.cat([hand_mano_rot, hand_mano_pose], dim=1).to(device)
    mano_verts, mano_joints = mano_layer(mano_param, hand_mano_shape)

    hand_scale = anno['hand']['mano_scale']
    hand_xyz_root = anno['hand']['mano_xyz_root']

    ## 3D hand verts in world coordinate
    mano_verts = (mano_verts / hand_scale) + torch.Tensor(hand_xyz_root).to(device)


    verts_cam = torch.unsqueeze(mano3DToCam3D(mano_verts, Ms), 0)
    pred_rendered_hand_only = renderer_set[cam].render(verts_cam, hand_faces_template, flag_rgb=True)

    # verts_cam_obj = torch.unsqueeze(mano3DToCam3D(obj_verts_world, Ms), 0)
    # pred_rendered = renderer_set[camIdx].render_meshes([verts_cam, verts_cam_obj], [hand_faces_template, obj_faces_template], flag_rgb=True)

    rgb_mesh = np.squeeze((pred_rendered_hand_only['rgb'][0].cpu().detach().numpy() * 255.0)).astype(np.uint8)

    cv2.imwrite(os.path.join(save_path, "../vis", f"mesh_{seq_name}_{trial}_{img_name}.png"), rgb_mesh)