import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from pytorch3d.structures import Meshes, join_meshes_as_scene
from pytorch3d.renderer import (
    BlendParams,
    look_at_view_transform,
    FoVPerspectiveCameras, 
    PointLights, 
    AmbientLights,
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    SoftSilhouetteShader,
    SoftPhongShader,
    TexturesVertex,
    PerspectiveCameras
)
import transforms3d as t3d
from scipy.spatial.transform import Rotation as R
import time


def changeCoordtopytorch3D(extrinsic):
    extrinsic_py = np.copy(extrinsic)

    # axis flip rotation value
    r = R.from_euler('z', [180], degrees=True)
    r = np.squeeze(r.as_matrix())
    extrinsic_py = r @ extrinsic_py

    return extrinsic_py


class Renderer(nn.Module):
    def __init__(self, device='cuda', bs=1, extrinsic=None, intrinsic=None, image_size=None, light_loaction=((2.0, 2.0, -2.0),)):
        super().__init__()
        '''
            R : numpy array [3, 3]
            T : numpy array [3]
            Ext : numpy array [4, 4] or [3, 4]
            intrinsics : [3, 3]
        '''
        self.device = device
        self.bs = bs

        focal_l = (intrinsic[0, 0], intrinsic[1, 1])
        principal_p = (intrinsic[0, -1], intrinsic[1, -1])

        self.image_size = image_size
        self.extrinsic_py3D = changeCoordtopytorch3D(extrinsic)

        self.R = torch.unsqueeze(torch.FloatTensor(self.extrinsic_py3D[:, :-1]), 0)
        self.T = torch.unsqueeze(torch.FloatTensor(self.extrinsic_py3D[:, -1]), 0)

        cameras = PerspectiveCameras(device=self.device, image_size=(self.image_size,), focal_length=(focal_l,),
                                     principal_point=(principal_p,), R=self.R, T=self.T, in_ndc=False)

        self.raster_settings = RasterizationSettings(
            image_size=self.image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
            bin_size = None,
            max_faces_per_bin = None
        )

        self.lights = PointLights(device=self.device, location=light_loaction)

        self.renderer_rgb = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=self.raster_settings
            ),
            shader=SoftPhongShader(device=self.device, cameras=cameras, lights=self.lights)
        )

        self.rasterizer_depth = MeshRasterizer(
        cameras=cameras,
        raster_settings=self.raster_settings
        )


    def update_intrinsic(self, intrinsic):
        focal_l = (intrinsic[0, 0], intrinsic[1, 1])
        principal_p = (intrinsic[0, -1], intrinsic[1, -1])

        cameras = PerspectiveCameras(device=self.device, image_size=(self.image_size,), focal_length=(focal_l,),
                                     principal_point=(principal_p,), R=self.R, T=self.T, in_ndc=False)
        self.renderer_rgb = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=self.raster_settings
            ),
            shader=SoftPhongShader(device=self.device, cameras=cameras, lights=self.lights)
        )

        self.rasterizer_depth = MeshRasterizer(
        cameras=cameras,
        raster_settings=self.raster_settings
        )

    def register_seg(self, seg_ref):
        self.register_buffer('seg_ref', seg_ref)

    def register_depth(self, depth_ref):
        self.register_buffer('depth_ref', depth_ref)

    def render(self, verts, faces, flag_rgb=False):
        '''
        verts : [bs, V, 3]
        faces : [bs, F, 3]
        
        -> [bs, H, W, 3], [bs, H, W], [bs, H, W]
        '''
        verts_rgb = torch.ones_like(verts, device=self.device)
        textures = TexturesVertex(verts_features=verts_rgb)
        meshes = Meshes(verts=verts, faces=faces, textures=textures)

        if flag_rgb:
            rgb = self.renderer_rgb(meshes)[..., :3]
        else:
            rgb = None

        depth = self.rasterizer_depth(meshes).zbuf[..., 0]
        # depth map process
        depth[depth == -1] = 0.
        seg = torch.empty_like(depth).copy_(depth)

        # depth = depth * 10.0    # change to mm scale (same as gt)
        depth = depth / 100.0   # change to m scale
        depth[depth == 0] = 10.


        # loss_depth = torch.sum(((depth_rendered - self.depth_ref / self.scale) ** 2).view(self.batch_size, -1),
        #                        -1) * 0.00012498664727900177  # depth scale used in HOnnotate

        self.output = {"rgb":rgb, "depth":depth, "seg":seg}

        return self.output

    def render_meshes(self, verts_list, faces_list, flag_rgb=False):
        '''
        verts : [bs, V, 3]
        faces : [bs, F, 3]

        -> [bs, H, W, 3], [bs, H, W], [bs, H, W]
        '''
        mesh_list = list()
        for verts, faces in zip(verts_list, faces_list):
            verts_rgb = torch.ones_like(verts, device=self.device)
            textures = TexturesVertex(verts_features=verts_rgb)
            meshes = Meshes(verts=verts, faces=faces, textures=textures)
            mesh_list.append(meshes)

        mesh_joined = join_meshes_as_scene(mesh_list)

        if flag_rgb:
            rgb = self.renderer_rgb(mesh_joined)[..., :3]
        else:
            rgb = None

        depth = self.rasterizer_depth(mesh_joined).zbuf

        # depth map process
        depth[depth == -1] = 0.
        depth = depth * 10.0

        seg = torch.empty_like(depth).copy_(depth)

        # loss_depth = torch.sum(((depth_rendered - self.depth_ref / self.scale) ** 2).view(self.batch_size, -1),
        #                        -1) * 0.00012498664727900177  # depth scale used in HOnnotate

        return {"rgb": rgb, "depth": depth[..., 0], "seg":seg[..., 0]}

