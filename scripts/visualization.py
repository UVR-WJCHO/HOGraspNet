import os

#### remove ####
os.environ["HOG_DIR"] = "/scratch/NIA/HOGraspNet"

import sys
sys.path.append(os.environ['HOG_DIR'])
sys.path.append(os.path.join(os.environ['HOG_DIR'], "thirdparty/manopth"))

from HOG_dataloader import HOGDataset
import torch
from torch.utils.data import DataLoader
from config import cfg
from scripts.util.renderer import Renderer
from scripts.util.utils_vis import *
import numpy as np
from thirdparty.manopth.manopth.manolayer import ManoLayer
import cv2
from pytorch3d.io import load_obj


def load_object_meshes(model_path, device='cpu'):
    obj_templates = {}
    obj_templates["verts_h"] = {}
    obj_templates["faces"] = {}

    obj_list = os.listdir(model_path)

    print("loading object meshes ...")
    for obj_name in obj_list:
        obj_mesh_path = os.path.join(model_path, obj_name, obj_name + '.obj')
        obj_idx = int(obj_name.split('_')[0])
        obj_scale = cfg._OBJECT_SCALE_FIXED[obj_idx-1]

        obj_verts, obj_faces, _ = load_obj(obj_mesh_path)
        obj_verts_template = (obj_verts * float(obj_scale)).to(device)
        obj_faces_template = torch.unsqueeze(obj_faces.verts_idx, axis=0).to(device)

        h = torch.ones((obj_verts_template.shape[0], 1), device=device)
        obj_verts_template_h = torch.cat((obj_verts_template, h), 1)

        obj_templates["verts_h"][obj_idx] = obj_verts_template_h
        obj_templates["faces"][obj_idx] = obj_faces_template
    print("... done")

    return obj_templates


if __name__ == '__main__':
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
    
    obj_model_path = os.path.join(os.environ['HOG_DIR'], "data", "obj_scanned_models")
    obj_templates = load_object_meshes(model_path=obj_model_path, device=device)

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

        K = torch.squeeze(sample['intrinsics'])
        M = torch.squeeze(sample['extrinsics'])

        rgb = sample['rgb_data']
        depth = sample['depth_data']
        bbox = sample['bbox']
        anno = sample['anno_data']
        rgb_path = sample['rgb_path'][0]
        seq_name = rgb_path.split('/')[-5]
        trial_name = rgb_path.split('/')[-4]
        img_name = rgb_path.split('/')[-1]

        obj_id = int(sample['obj_ids'][0])
        obj_verts_template_h = obj_templates["verts_h"][obj_id]
        obj_faces_template = obj_templates["faces"][obj_id]


        ## update renderer if sequence has changed
        cam = sample['camera'][0]
        if seq_name != prev_seq_name:
            renderer_set[cam].update_intrinsic(K)
        
        ## set hand parameters
        hand_joints = anno['annotations'][0]['data']
        hand_mano_rot = anno['Mesh'][0]['mano_trans']
        hand_mano_pose = anno['Mesh'][0]['mano_pose']
        hand_mano_shape = anno['Mesh'][0]['mano_betas']

        hand_mano_rot = torch.FloatTensor(hand_mano_rot).to(device)
        hand_mano_pose = torch.FloatTensor(hand_mano_pose).to(device)
        hand_mano_shape = torch.FloatTensor(hand_mano_shape).to(device)

        mano_param = torch.cat([hand_mano_rot, hand_mano_pose], dim=1).to(device)
        mano_verts, mano_joints = mano_layer(mano_param, hand_mano_shape)

        hand_scale = anno['hand']['mano_scale']
        hand_xyz_root = anno['hand']['mano_xyz_root']

        mano_verts = (mano_verts / hand_scale.to(device)) + torch.Tensor(hand_xyz_root).to(device)
        verts_cam = torch.unsqueeze(mano3DToCam3D(mano_verts, M), 0)

        ## set object parameters
        obj_mat = torch.FloatTensor(anno['Mesh'][0]['object_mat']).to(device)

        obj_points = obj_verts_template_h @ obj_mat.T
        obj_verts_world = obj_points[:, :3] / obj_points[:, 3:]
        obj_verts_world = obj_verts_world.view(1, -1, 3)
        verts_cam_obj = torch.unsqueeze(mano3DToCam3D(obj_verts_world, M), 0)

        ## render
        # pred_rendered_hand_only = renderer_set[cam].render(verts_cam, hand_faces_template, flag_rgb=True)
        pred_rendered = renderer_set[cam].render_meshes([verts_cam, verts_cam_obj], [hand_faces_template, obj_faces_template], flag_rgb=True)

        rgb_mesh = np.squeeze((pred_rendered['rgb'][0].cpu().detach().numpy() * 255.0)).astype(np.uint8)
        
        bbox_np = np.squeeze(np.asarray(bbox, dtype=int))
        rgb_mesh = rgb_mesh[bbox_np[1]:bbox_np[1]+bbox_np[3], bbox_np[0]:bbox_np[0]+bbox_np[2], :]

        rgb_np = np.squeeze(np.asarray(rgb))
        rgb_mesh = cv2.addWeighted(rgb_mesh, 0.4, rgb_np, 0.6, 0)
        cv2.imwrite(os.path.join(save_path, f"mesh_{seq_name}_{trial_name}_{img_name}"), rgb_mesh)
        print(f"rendered mesh_{seq_name}_{trial_name}_{img_name}")
