import os
import sys
import warnings
warnings.filterwarnings(action='ignore')
import json
import pandas as pd
from pytorch3d.io import load_obj
import torch
from enum import IntEnum
import numpy as np
import cv2
from tqdm import tqdm
import pickle
import math
import random
import matplotlib.pyplot as plt

from config import *
from utils import extractBbox





def grid_maker(i,j,test_dataloader) :

    i_idx = 0
    j_idx = 0
    list_2d = []
    totalnum = 0

    for frame_instance in test_dataloader :
        
        # function calling
        #print(frame_instance['rgb_vis_data'])
        rgb_vis_data = np.asarray(cv2.imread(frame_instance['rgb_vis_data'][0]))
        img1 = np.array(rgb_vis_data)

        # print(img1.shape)

        if img1.shape != (480,640,3) :
            continue


        if i_idx == 0 :
            list_1d = []
            list_1d.append(img1)
            totalnum += 1
            i_idx += 1     
        
        elif i_idx == i :
            list_2d.append(list_1d)
            
            i_idx = 0
            j_idx += 1
            if j_idx == j :
                print(len(list_2d))
                
                img_tile = cv2.vconcat([cv2.hconcat(list_h) for list_h in list_2d]) 
                cv2.imwrite('concat_vh.jpg', img_tile)

                # img_tile_plt = cv2.cvtColor(img_tile, cv2.COLOR_BGR2RGB)
                # plt.imshow(img_tile_plt)
                # plt.savefig("concat_vh.svg", format = 'svg', dpi=300)

                break
        
        else:
            list_1d.append(img1)
            totalnum += 1
            i_idx += 1     

        print(totalnum)

def main():
    from natsort import natsorted
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    baseDir = os.path.join('/scratch/NIA')

    base_source = os.path.join(baseDir, '1_Source_data')
    base_anno = os.path.join(baseDir, '2_Labeling_data')

    seq_list = natsorted(os.listdir(base_anno))
    print("total sequence # : ", len(seq_list))

    from torch.utils.data import DataLoader

    setup = 'travel_all'
    split = 'test'
    print("loading ... ", setup + '_' + split)

    flag_valid_obj = True
    dataset = NIADataset(setup,split,base_anno, base_source, seq_list, baseDir, device, f'{setup}_{split}_validobj_{flag_valid_obj}.pkl', flag_valid_obj=flag_valid_obj)

    list_mapping = dataset.get_mapping()
    txt_name = os.path.join(baseDir, setup+'_'+split+'_validobj_'+str(flag_valid_obj)+'.txt')
    with open(txt_name, "w") as file:
        for m in list_mapping:            
            line = m[0] + '/' + m[1] + '/' + m[2] + '/' + m[4] + '\n'
            file.write(line)
    print("end")

    # test_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    ### GRID 만들때 사용하세요 ###
    
    # grid_maker(15,10,test_dataloader)
    # for frame_instance in test_dataloader :
        
    #     img1 = np.array(frame_instance['rgb_data'][0].cpu())
    #     cv2.imwrite('img2.png',img1)

    #     exit(0)



        



class NIADataset():

    def __init__(self, setup, split, base_anno, base_source, seq_list, baseDir, device, data_pkl_pth, flag_valid_obj=False):

        """Constructor.
        Args:
        setup: Setup name. 's0', 's1', 's2', or 's3'.
        split: Split name. 'train', 'val', or 'test'.
        """

        self.dataset_pkl_name = data_pkl_pth

        self._setup = setup
        self._split = split

        # assert 'DEX_YCB_DIR' in os.environ, "environment variable 'DEX_YCB_DIR' is not set"
        # self._data_dir = os.environ['DEX_YCB_DIR']

        self.base_anno = base_anno
        self.base_source = base_source
        self.device = device
        self.baseDir = baseDir

        self._h = 480
        self._w = 640

        self.camIDset = CAMIDSET

        self.seq_dict_lst = []

        ## if flag_valid_obj is True, filter non-valid sequence from seq_list        
        if flag_valid_obj:            
            csv_valid_obj_path = os.path.join(os.getcwd(), 'csv_valid_obj.csv')
            valid_df = pd.read_csv(csv_valid_obj_path)
            valid_list = valid_df['Sequence'].to_list()
            for seq in seq_list:
                if seq in valid_list:
                    seq_list.remove(seq)
        
            print("sequence # after valid obj: ", len(seq_list))
        else:
            print("no filtering valid obj")

        ## MINING SEQ INFOS

        _SUBJECTS = []
        _OBJ_IDX = []
        _GRASP_IDX = []
        _OBJ_GRASP_PAIR = []
        
        for idx, seq in enumerate(seq_list) :
            
            seq_info = {}

            split = seq.split('_')

            seq_info['idx']  = idx 
            seq_info['date'] = split[0]
            seq_info['subject'] = split[1]
            seq_info['obj_idx'] = split[3]
            seq_info['grasp_idx'] = split[5]
            seq_info['obj_grasp_pair'] = [split[3],split[5]]
            seq_info['seqName'] = seq
            
            if seq_info['subject'] not in _SUBJECTS :
                _SUBJECTS.append(seq_info['subject'])

            if seq_info['obj_idx'] not in _OBJ_IDX :
                _OBJ_IDX.append(seq_info['obj_idx'])

            if seq_info['grasp_idx'] not in _GRASP_IDX :
                _GRASP_IDX.append(seq_info['grasp_idx'])

            if seq_info['obj_grasp_pair'] not in _OBJ_GRASP_PAIR :
                _OBJ_GRASP_PAIR.append(seq_info['obj_grasp_pair'])

            self._SUBJECTS = _SUBJECTS
            self._OBJ_IDX = _OBJ_IDX
            self._GRASP_IDX = _GRASP_IDX
            self._OBJ_GRASP_PAIR = _OBJ_GRASP_PAIR

            self.seq_dict_lst.append(seq_info)     
        
        ## TRAING / TEST SPLIT 

        # UNSEEN TRIAL
        if self._setup == 's0':
            if self._split == 'train':
                subject_ind = self._SUBJECTS
                serial_ind = self.camIDset
                trial_ind = 'train' # 'full', 'test', 'train'
                obj_grasp_pair_ind = self._OBJ_GRASP_PAIR

            if self._split == 'test':
                subject_ind = self._SUBJECTS
                serial_ind = self.camIDset
                trial_ind = 'test' # 'full'
                obj_grasp_pair_ind = self._OBJ_GRASP_PAIR

        # UNSEEN SUBJECTS
        if self._setup == 's1':
            if self._split == 'train':
                subject_ind = self._SUBJECTS[:73]
                serial_ind = self.camIDset
                trial_ind = 'full' # 'full', 'test', 'train'
                obj_grasp_pair_ind = self._OBJ_GRASP_PAIR

            if self._split == 'test':
                subject_ind = self._SUBJECTS[73:]
                serial_ind = self.camIDset
                trial_ind = 'full' # 'full'
                obj_grasp_pair_ind = self._OBJ_GRASP_PAIR

        # UNSEEN CAM
        if self._setup == 's2':
            if self._split == 'train':
                subject_ind = self._SUBJECTS
                serial_ind = self.camIDset[:3]
                trial_ind = 'full' # 'full', 'test', 'train'
                obj_grasp_pair_ind = self._OBJ_GRASP_PAIR

            if self._split == 'test':
                subject_ind = self._SUBJECTS
                serial_ind = self.camIDset[3:]
                trial_ind = 'full' # 'full'
                obj_grasp_pair_ind = self._OBJ_GRASP_PAIR

        # UNSEEN OBJECTS

        if self._setup == 's3':

            test_obj_lst = ['10','02','12','20','04','06','11']

            test_pair = []
            train_pair = []

            for pair in self._OBJ_GRASP_PAIR :

                if pair[0] in test_obj_lst :
                    test_pair.append(pair)

                else :
                    train_pair.append(pair)

            if self._split == 'train':
                subject_ind = self._SUBJECTS
                serial_ind = self.camIDset
                trial_ind = 'full' # 'full', 'test', 'train'
                obj_grasp_pair_ind = train_pair

            if self._split == 'test':
                subject_ind = self._SUBJECTS
                serial_ind = self.camIDset
                trial_ind = 'full' # 'full'
                obj_grasp_pair_ind = test_pair


        # UNSEEN GRASP TAXONOMY

        test_grasp_lst = ['23','25','29','16']

        test_pair = []
        train_pair = []

        for pair in self._OBJ_GRASP_PAIR :

            if pair[1] in test_grasp_lst :
                test_pair.append(pair)

            else :
                train_pair.append(pair)

        if self._setup == 's4':
            if self._split == 'train':
                subject_ind = self._SUBJECTS
                serial_ind = self.camIDset
                trial_ind = 'full' # 'full', 'test', 'train'
                obj_grasp_pair_ind = train_pair

            if self._split == 'test':
                subject_ind = self._SUBJECTS
                serial_ind = self.camIDset
                trial_ind = 'full' # 'full'
                obj_grasp_pair_ind = test_pair

        
        ### ALL ###
        if self._setup == 'travel_all':
            subject_ind = self._SUBJECTS
            serial_ind = self.camIDset
            trial_ind = 'full' # 'full', 'test', 'train'
            obj_grasp_pair_ind = self._OBJ_GRASP_PAIR

        #########################################

        ### Make Mapping ###
       
        self.objModelDir = os.path.join(os.getcwd(), '3_Other')
        self.csv_save_path = os.path.join(os.getcwd(), 'csv_output_filtered_0.4.csv')
        self.filtered_df = pd.read_csv(self.csv_save_path)

        print("filtered csv check : ", len(self.filtered_df))
        ## 여기서부터 작업하기

        total_count = 0

        self.mapping = [] # its location
        
        # for each object has its mapping index which contains s,t,c,f (subject,trial,cam,frame)

        ## SEQ : S
        self.load = False
        ##### 

        self.CameraParm_K_M_dict = {
            # [Seq][Trial][CamID]
        }

        SEQ_DICT_APPEND = {}
       

        for seqIdx, seq in enumerate(tqdm(self.seq_dict_lst)):

            ## if file exist, break ##
            if os.path.isfile(data_pkl_pth) :

                with open(data_pkl_pth, 'rb') as handle:
                    dict_data = pickle.load(handle)

                self.dataset_samples = dict_data['data']
                self.mapping = dict_data['mapping']
                self.CameraParm_K_M_dict = dict_data['camera_info']
            
                self.load = True

                break


            if seq['subject'] not in subject_ind :
                continue

            if seq['obj_grasp_pair'] not in obj_grasp_pair_ind :
                continue

            seqDir = os.path.join(self.base_anno, seq['seqName'])

            ## TRIAL : T

            TRIAL_DICT_APPEND = {}
            CAM_dict_trial = {}

            for trialIdx, trialName in enumerate(sorted(os.listdir(seqDir))):
                
                if trial_ind == 'test' and trialIdx != 0:
                    continue           

                if trial_ind == 'train' and trialIdx == 0:    
                    continue

                ## CAM : C
                valid_cams = []

                for camID in self.camIDset:

                    p = os.path.join(seqDir, trialName, 'annotation', camID)

                    if os.path.exists(p):

                        valid_cams.append(camID)


                anno_base_path = os.path.join(self.base_anno, seq['seqName'], trialName, 'annotation')

                Ks_dict = {}
                Ms_dict = {}

                for camID in self.camIDset:
                    if camID in valid_cams:

                        anno_list = os.listdir(os.path.join(anno_base_path, camID))
                        anno_path = os.path.join(anno_base_path, camID, anno_list[0])

                        with open(anno_path, 'r', encoding='UTF-8 SIG') as file:
                            anno = json.load(file)

                        Ks = torch.FloatTensor(np.squeeze(np.asarray(anno['calibration']['intrinsic']))).to(self.device)

                        Ms = np.squeeze(np.asarray(anno['calibration']['extrinsic']))
                        Ms = np.reshape(Ms, (3, 4))

                        ## will be processed in postprocess, didn't.
                        Ms[:, -1] = Ms[:, -1] / 10.0
                        Ms = torch.Tensor(Ms).to(self.device)

                        Ks_dict[camID] = Ks
                        Ms_dict[camID] = Ms

                    else:
                        Ks_dict[camID] = None
                        Ms_dict[camID] = None

                
                self.Ks_dict = Ks_dict # Camera Intrinsic
                self.Ms_dict = Ms_dict # Camera Extrinsic
                self.valid_cams = valid_cams

                temp_dict = {}

                temp_dict['Ks_dict'] = self.Ks_dict
                temp_dict['Ms_dict'] = self.Ms_dict

                CAM_dict_trial[trialName] = temp_dict

                ###### CAN LOAD OBJ MESH #######

                #self.obj_mesh_data = self.load_obj_mesh() #### 시간때문에 삭제 했음, Rendering이 필요할 때 사용

                #### wj 
                self.anno_dict, rgb_dict = self.load_data(self.base_anno, self.base_source, seq['seqName'], trialName, self.valid_cams)

                ## anno_path, rgb_path 모두 sample 별로 존재하고 있음 
                
                #self.anno_dict, rgb, depth = self.load_data(self.base_anno, self.base_source, seq['seqName'], trialName, self.valid_cams)

                # self.samples = self.set_sample(rgb_path) ## bounding box and rgb image
                # self.samples = self.set_sample(rgb,depth) ## bounding box and rgb image
                
                CAM_DICT_APPEND = {} # set_sample 에서 

                for camIDX, camID in enumerate(valid_cams) :

                    if camID not in serial_ind :
                        
                        continue

                    ##### FRAM : F ######

                    FRAME_DICT_APPEND = {}

                    for frame_idx, frame in enumerate(self.anno_dict[camID]) :

                        sample = {
                            'rgb_path': rgb_dict[camID][frame_idx], ## rgb path로 수정하기
                            # 'bbox' :  None, ## bbox 수정하기 나중에 rgb load 그리고 annotation load 할 때 Crop 하는 알고리듬도 추가하기
                            'label_path': self.anno_dict[camID][frame_idx], ## path 로 주는 것으로 수정하기
                            # 'intrinsics': None, # Ks_dict[camID], ## 중복됨 Seq,Trail에 Dependent 하게 수정하기 
                            # 'extrinsics': None, # Ms_dict[camID], ## 중복됨                                 
                            'obj_ids': seq['obj_idx'],
                            # 'mano_side':  None, ## frame['Mesh'][0]['mano_side'], -> label path 로 load 하기 
                            # 'mano_betas': None, ## frame['Mesh'][0]['mano_betas'],
                            'taxonomy': seq['grasp_idx'],
                        }

                        FRAME_DICT_APPEND[str(frame_idx)] = sample
                        
                        frame_num = self.anno_dict[camID][frame_idx].split('/')[-1].split('_')[-1][:-5]

                        self.mapping.append([seq['seqName'],trialName,camID, frame_idx, str(frame_num)])

                    
                    CAM_DICT_APPEND[camID] = FRAME_DICT_APPEND

                TRIAL_DICT_APPEND[trialName] = CAM_DICT_APPEND
                
            self.CameraParm_K_M_dict[seq['seqName']] = CAM_dict_trial

            SEQ_DICT_APPEND[seq['seqName']] = TRIAL_DICT_APPEND

        if not self.load :        
            self.dataset_samples = SEQ_DICT_APPEND

            dict_data = {}

            dict_data['data'] = self.dataset_samples
            dict_data['mapping'] = self.mapping
            dict_data['camera_info'] = self.CameraParm_K_M_dict

            with open(self.dataset_pkl_name, 'wb') as handle:
                pickle.dump(dict_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def get_mapping(self):
        return self.mapping


    def get_len(self, camID):
        return len(self.anno_dict[camID])
       
    def set_sample(self,rgb_ori):
        samples = {}
        for camIdx, camID in enumerate(self.camIDset):
            if camID in self.valid_cams:
                samples[camID] = []

                for idx in range(self.get_len(camID)):
                    sample = {}

                    rgb = rgb_ori[camID][idx]
                    #depth = depth_ori[camID][idx]

                    anno = self.anno_dict[camID][idx]
                    hand_2d = np.squeeze(np.asarray(anno['hand']['projected_2D_pose_per_cam']))
                    bbox, bbox_s = extractBbox(hand_2d)

                    rgb = rgb[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2])]
                    #depth = depth[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2])]

                    # seg, vis_seg = deepSegPredict(self.segModel, self.transform, rgb, self.decode_fn, self.device)
                    # # vis_seg = np.squeeze(np.asarray(vis_seg))
                    # # hand_mask = np.asarray(vis_seg[:, :, 0] / 128 * 255, dtype=np.uint8)
                    # # cv2.imshow("vis_seg", hand_mask)
                    # # cv2.waitKey(0)

                    # seg = np.asarray(seg)

                    # # seg_hand = np.where(seg == 1, 1, 0)
                    # seg_obj = np.where(seg == 2, 1, 0)

                    # depth_obj = depth.copy()
                    # depth_obj[seg_obj == 0] = 0
                    # depth[seg == 0] = 0

                    # # change depth image to m scale and background value as positive value
                    # depth /= 1000.
                    # depth_obj /= 1000.

                    # depth_obj = np.where(seg != 2, 10, depth)
                    # # depth_hand = np.where(seg != 1, 10, depth)

                    rgb = torch.FloatTensor(rgb).to(self.device)
                    
                    #depth_obj = torch.unsqueeze(torch.FloatTensor(depth_obj), 0).to(self.device)
                    #seg_obj = torch.unsqueeze(torch.FloatTensor(seg_obj), 0).to(self.device)
                    # depth = torch.unsqueeze(torch.FloatTensor(depth), 0).to(self.device)

                    #sample['rgb'], sample['depth_obj'], sample['seg_obj'] = rgb, depth_obj, seg_obj
                    sample['rgb'] = rgb
                    sample['bb'] = [int(bb) for bb in bbox]

                    samples[camID].append(sample)

        return samples

    def extract_bbox_rgb(self,rgb_ori):
        samples = {}
        for camIdx, camID in enumerate(self.camIDset):
            if camID in self.valid_cams:
                samples[camID] = []

                for idx in range(self.get_len(camID)):
                    sample = {}

                    rgb = rgb_ori[camID][idx]

                    anno = self.anno_dict[camID][idx]
                    hand_2d = np.squeeze(np.asarray(anno['hand']['projected_2D_pose_per_cam']))
                    bbox, bbox_s = extractBbox(hand_2d)

                    rgb = rgb[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2])]
                    rgb = torch.FloatTensor(rgb).to(self.device)

                    sample['rgb'] = rgb
                    sample['bb'] = [int(bb) for bb in bbox]

                    samples[camID].append(sample)

        return samples

    def load_hand_mesh(self):
        from manopth.manolayer import ManoLayer
        mano_path = os.path.join(os.getcwd(), 'modules', 'mano', 'models')
        self.mano_layer = ManoLayer(side='right', mano_root=mano_path, use_pca=False, flat_hand_mean=True,
                               center_idx=0, ncomps=45, root_rot_mode="axisang", joint_rot_mode="axisang").to(self.device)
        self.hand_faces_template = self.mano_layer.th_faces.repeat(1, 1, 1)

    def load_obj_mesh(self):
        target_mesh_class = str(self.obj_id).zfill(2) + '_' + str(OBJType(int(self.obj_id)).name)
        self.obj_mesh_name = target_mesh_class + '.obj'

        obj_mesh_path = os.path.join(self.baseDir, self.objModelDir, target_mesh_class, self.obj_mesh_name)
        obj_scale = CFG_OBJECT_SCALE_FIXED[int(self.obj_id) - 1]
        obj_verts, obj_faces, _ = load_obj(obj_mesh_path)
        obj_verts_template = (obj_verts * float(obj_scale)).to(self.device)
        obj_faces_template = torch.unsqueeze(obj_faces.verts_idx, axis=0).to(self.device)

        # h = torch.ones((obj_verts_template.shape[0], 1), device=self.device)
        # self.obj_verts_template_h = torch.cat((obj_verts_template, h), 1)

        obj_mesh_data = {}
        obj_mesh_data['verts'] = obj_verts_template
        obj_mesh_data['faces'] = obj_faces_template

        return obj_mesh_data

    def get_obj_pose(self, camID, idx):
        anno = self.anno_dict[camID][idx]
        obj_mat = np.squeeze(np.asarray(anno['Mesh'][0]['object_mat']))
        return obj_mat

    def load_data(self, base_anno, base_source, seq, trialName, valid_cams):

        df = self.filtered_df.loc[self.filtered_df['Sequence'] == seq]
        df = df.loc[df['Trial'] == trialName]
        filtered_list = np.asarray(df['Frame'])

        anno_base_path = os.path.join(base_anno, seq, trialName, 'annotation')
        rgb_base_path = os.path.join(base_source, seq, trialName, 'rgb')
        #_base_path = os.path.join(base_source, seq, trialName, 'depth')

        anno_dict = {}
        rgb_dict = {}
        #depth_dict = {}
        for camIdx, camID in enumerate(self.camIDset):
            anno_dict[camID] = []
            rgb_dict[camID] = []
            #depth_dict[camID] = []

            if camID in valid_cams:

                anno_list = os.listdir(os.path.join(anno_base_path, camID))

                if self._setup == 's0' and self._split == 'test' :
                    if anno_list == [] :
                        continue

                    number_of_elements = max(1,  int( len(anno_list) / 5) )
                    anno_list = random.sample(anno_list, number_of_elements)


                for anno in anno_list:
                    if anno[:-5] in filtered_list:

                        anno_path = os.path.join(anno_base_path, camID, anno)
                        
                        # with open(anno_path, 'r', encoding='UTF-8 SIG') as file:
                        #     anno_data = json.load(file)
                        # anno_dict[camID].append(anno_data)
                        anno_dict[camID].append(anno_path)


                        rgb_path = os.path.join(rgb_base_path, camID, anno[:-5] + '.jpg')
                        # rgb_data = np.asarray(cv2.imread(rgb_path))
                        # rgb_dict[camID].append(rgb_data)
                        rgb_dict[camID].append(rgb_path)
                        

                        #depth_path = os.path.join(depth_base_path, camID, anno[:-5] + '.png')
                        #depth_data = np.asarray(cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)).astype(float)
                        #depth_dict[camID].append(depth_data)

        return anno_dict, rgb_dict
    

    def __len__(self):
        return len(self.mapping)


    def __getitem__(self, idx):

        # s : subject , c : camera, f: frame
        s, t, c, f = self.mapping[idx]

        sample = self.dataset_samples[s][t][c][f]
        sample['camera']  = c
        ####### anno_load #####

        rgb_path = sample['rgb_path']

        temp_split = rgb_path.split('/')

        temp_rgb = temp_split[12].split('.')[0]

        # /home/awscliv2/HOI_DATA/1_Construction_process_output/2_Final_verification/1.Datasets/2_Labeling_data/231027_S103_obj_24_grasp_31/trial_0/visualization/mas/blend_pred_mas_8.png
        # /home/awscliv2/HOI_DATA/1_Construction_process_output/2_Final_verification/1.Datasets/1_Source_data/231013_S82_obj_22_grasp_12/trial_1/rgb/mas/mas_42.jpg
        # /home/awscliv2/HOI_DATA/1_Construction_process_output/2_Final_verification/1.Datasets/2_Labeling_data/230919_S23_obj_18_grasp_16/trial_1/visualization/sub1/blend_pred_sub1_62.png

        label_path = sample['label_path']

        with open(label_path, 'r', encoding='UTF-8 SIG') as file:
            anno_data = json.load(file)

        rgb_data = np.asarray(cv2.imread(rgb_path))
        

        sample['intrinsics'] = self.CameraParm_K_M_dict[s][t]['Ks_dict'][c]
        sample['extrinsics'] = self.CameraParm_K_M_dict[s][t]['Ms_dict'][c]
        sample['anno_data']  = anno_data

        hand_2d = np.squeeze(np.asarray(anno_data['hand']['projected_2D_pose_per_cam']))
        bbox, bbox_s = extractBbox(hand_2d)

        rgb_data = rgb_data[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2])]


        rgb_vis_path = f'/home/awscliv2/HOI_DATA/1_Construction_process_output/2_Final_verification/1.Datasets/2_Labeling_data/{temp_split[8]}/{temp_split[9]}/visualization/{temp_split[11]}/blend_pred_{temp_rgb}.png'
        sample['rgb_vis_data'] = rgb_vis_path

        sample['rgb_data'] = rgb_data
        sample['bbox'] = bbox

        return sample

if __name__ == '__main__':
    main()