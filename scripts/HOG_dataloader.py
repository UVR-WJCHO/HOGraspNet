
"""HOGraspNet dataset."""

import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import json
import torch
import numpy as np
import cv2
from tqdm import tqdm
import pickle
from config import cfg
from util.utils import extractBbox
# from pytorch3d.io import load_obj

class HOGDataset():
    def __init__(self, setup, split, db_path=None, use_aug=False, load_pkl=True):
        """Constructor.
        Args:
        setup: Setup name. 's0', 's1', 's2', 's3', or 's4'
        split: Split name. 'train', 'val', or 'test'
        use_aug: Use crop&augmented rgb data if exists.
        load_pkl: Use saved pkl if exists.
        """

        self._setup = setup
        self._split = split
        self._use_aug = use_aug

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        assert 'HOG_DIR' in os.environ, "environment variable 'HOG_DIR' is not set"
        self._base_dir = db_path

        self._base_anno = os.path.join(self._base_dir, 'labeling_data')
        self._base_source = os.path.join(self._base_dir, 'source_data')
        self._base_source_aug = os.path.join(self._base_dir, 'source_augmented')

        self._base_extra = os.path.join(self._base_dir, 'extra_data')     
        self._obj_model_dir = os.path.join(self._base_dir, 'obj_scanned_models')
        
        self._h = 480
        self._w = 640

        self.camIDset = cfg._CAMIDSET

        # create pkl once, load if exist.
        self._data_pkl_pth = f'cfg/{setup}_{split}.pkl'
        
        ## CHECK DATA 
        assert os.path.isdir(self._base_anno), "labeling data is not set, we require at least annotation & source(or source_augmented) to run dataloader"
        assert os.path.isdir(self._base_source) or os.path.isdir(self._base_source_aug) , "source data is not set, we require at least annotation & source(or source_augmented) to run dataloader"
        
        ## MINING SEQUENCE INFOS
        self._SUBJECTS, self._OBJ_IDX, self._GRASP_IDX, self._OBJ_GRASP_PAIR = [], [], [], []
        
        seq_list = os.listdir(self._base_anno)
        self._seq_dict_list = []
        for idx, seq in enumerate(seq_list) :            
            seq_info = {}
            seq_split = seq.split('_')

            seq_info['idx']  = idx 
            seq_info['seqName'] = seq
            # seq_info['date'] = seq_split[0]
            seq_info['subject'] = seq_split[1]
            seq_info['obj_idx'] = seq_split[3]
            seq_info['grasp_idx'] = seq_split[5]
            seq_info['obj_grasp_pair'] = [seq_split[3],seq_split[5]]
                        
            if seq_info['subject'] not in self._SUBJECTS :
                self._SUBJECTS.append(seq_info['subject'])

            if seq_info['obj_idx'] not in self._OBJ_IDX :
                self._OBJ_IDX.append(seq_info['obj_idx'])

            if seq_info['grasp_idx'] not in self._GRASP_IDX :
                self._GRASP_IDX.append(seq_info['grasp_idx'])

            if seq_info['obj_grasp_pair'] not in self._OBJ_GRASP_PAIR :
                self._OBJ_GRASP_PAIR.append(seq_info['obj_grasp_pair'])

            self._seq_dict_list.append(seq_info)     
        

        ## TRAIN / TEST / VALID SPLIT 
                
        # ALL
        if self._setup == 'travel_all':
            subject_ind = self._SUBJECTS
            serial_ind = self.camIDset
            obj_grasp_pair_ind = self._OBJ_GRASP_PAIR            
            trial_ind = 'full'      # 'full', 'train', 'val', 'test'   

        # s0 : UNSEEN TRIAL
        if self._setup == 's0':
            subject_ind = self._SUBJECTS
            serial_ind = self.camIDset
            obj_grasp_pair_ind = self._OBJ_GRASP_PAIR

            trial_ind = self._split     # 'full', 'train', 'val', 'test'            

        # s1 : UNSEEN SUBJECTS
        if self._setup == 's1':
            serial_ind = self.camIDset
            obj_grasp_pair_ind = self._OBJ_GRASP_PAIR
            trial_ind = 'full'      # 'full', 'train', 'val', 'test'   

            if self._split == 'train':
                subject_ind = self._SUBJECTS[:73]
            if self._split == 'test':
                subject_ind = self._SUBJECTS[73:]
            if self._split == 'val':
                subject_ind = self._SUBJECTS[:10]

        # s2 : UNSEEN CAM
        if self._setup == 's2':            
            subject_ind = self._SUBJECTS
            obj_grasp_pair_ind = self._OBJ_GRASP_PAIR
            trial_ind = 'full'      # 'full', 'train', 'val', 'test'   

            if self._split == 'train':
                serial_ind = self.camIDset[:-1]
            if self._split == 'test':
                serial_ind = self.camIDset[-1]
            if self._split == 'val':
                serial_ind = self.camIDset[0]

        # s3 : UNSEEN OBJECTS
        if self._setup == 's3':            
            subject_ind = self._SUBJECTS
            serial_ind = self.camIDset
            trial_ind = 'full'      # 'full', 'train', 'val', 'test'   

            train_pair, test_pair = [], []
            for pair in self._OBJ_GRASP_PAIR :
                if pair[0] in cfg._TEST_OBJ_LIST :
                    test_pair.append(pair)
                else :
                    train_pair.append(pair)

            if self._split == 'train':
                obj_grasp_pair_ind = train_pair
            if self._split == 'test':            
                obj_grasp_pair_ind = test_pair
            if self._split == 'valid':            
                obj_grasp_pair_ind = train_pair[:int(len(train_pair)/5)]


        # s4 : UNSEEN GRASP TAXONOMY       
        if self._setup == 's4':            
            subject_ind = self._SUBJECTS
            serial_ind = self.camIDset
            trial_ind = 'full'      # 'full', 'train', 'val', 'test' 
            
            train_pair, test_pair = [], []
            for pair in self._OBJ_GRASP_PAIR :
                if pair[1] in cfg._TEST_GRASP_LIST :
                    test_pair.append(pair)
                else :
                    train_pair.append(pair)

            if self._split == 'train':
                obj_grasp_pair_ind = train_pair
            if self._split == 'test':
                obj_grasp_pair_ind = test_pair
            if self._split == 'valid':
                obj_grasp_pair_ind = train_pair[:int(len(train_pair)/5)]
               

        #########################################

        
        # for each object has its mapping index which contains s,t,c,f (subject,trial,cam,frame)
        total_count = 0
        self.load = False
        self.mapping = [] # its location
        self.cam_param_dict = {}
        
        sample_seq_dict = {}

        ## load pkl if exist
        if os.path.isfile(self._data_pkl_pth) and load_pkl:
            print(f"loading from saved pkl {self._data_pkl_pth}")
            with open(self._data_pkl_pth, 'rb') as handle:
                dict_data = pickle.load(handle)

            self.dataset_samples = dict_data['data']
            self.mapping = dict_data['mapping']
            self.cam_param_dict = dict_data['camera_info']            
        else:
            for seqIdx, seq in enumerate(tqdm(self._seq_dict_list)):
                # skip if not target sequence
                if seq['subject'] not in subject_ind :
                    continue
                if seq['obj_grasp_pair'] not in obj_grasp_pair_ind :
                    continue

                sample_trial_dict = {}
                cam_param_dict_trial = {}

                seqName = seq['seqName']                
                seqDir = os.path.join(self._base_anno, seqName)
                for trialIdx, trialName in enumerate(sorted(os.listdir(seqDir))):               
                    # skip if not target trial
                    if trial_ind == 'train' and trialIdx == 0:    
                        continue
                    if trial_ind == 'test' and trialIdx != 0:
                        continue 
                    if trial_ind == 'valid' and trialIdx != 1:
                        continue 

                    anno_base_path = os.path.join(seqDir, trialName, 'annotation')
                    valid_cams = os.listdir(anno_base_path)

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
                            # Ms[:, -1] = Ms[:, -1] / 10.0
                            Ms = torch.Tensor(Ms).to(self.device)

                            Ks_dict[camID] = Ks
                            Ms_dict[camID] = Ms

                        else:
                            Ks_dict[camID] = None
                            Ms_dict[camID] = None

                    self.valid_cams = valid_cams

                    cam_param_dict_trial[trialName] = {}
                    cam_param_dict_trial[trialName]['Ks'] = Ks_dict
                    cam_param_dict_trial[trialName]['Ms'] = Ms_dict                    

                    self.anno_dict, rgb_dict, depth_dict, flag_crop = self.load_data(seqName, trialName, valid_cams)

                    sample_cam_dict = {}
                    for camIDX, camID in enumerate(valid_cams):
                        if camID not in serial_ind:                            
                            continue

                        sample_idx_dict = {}
                        for anno_idx, anno_path in enumerate(self.anno_dict[camID]) :
                            sample = {
                                'rgb_path': rgb_dict[camID][anno_idx], ## rgb path로 수정하기
                                'depth_path': depth_dict[camID][anno_idx],
                                'label_path': self.anno_dict[camID][anno_idx], ## path 로 주는 것으로 수정하기
                                'obj_ids': seq['obj_idx'],
                                'taxonomy': seq['grasp_idx'],
                                'flag_crop': flag_crop
                            }
                            
                            frame_num = anno_path.split('/')[-1].split('_')[-1][:-5]
                            self.mapping.append([seqName,trialName,camID,str(anno_idx),str(frame_num)])
                            
                            sample_idx_dict[str(anno_idx)] = sample                        
                        sample_cam_dict[camID] = sample_idx_dict
                    sample_trial_dict[trialName] = sample_cam_dict       
                sample_seq_dict[seqName] = sample_trial_dict       

                self.cam_param_dict[seqName] = cam_param_dict_trial

            self.dataset_samples = sample_seq_dict

            ## save pkl
            dict_data = {}
            dict_data['data'] = self.dataset_samples
            dict_data['mapping'] = self.mapping
            dict_data['camera_info'] = self.cam_param_dict
            
            os.makedirs("cfg", exist_ok=True)
            with open(self._data_pkl_pth, 'wb') as handle:
                pickle.dump(dict_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


        assert len(self.mapping) > 0, "downloaded data is not enough for given split"

    def get_mapping(self):
        return self.mapping

    # def load_hand_mesh(self):
    #     from manopth.manolayer import ManoLayer
    #     mano_path = os.path.join(os.getcwd(), 'modules', 'mano', 'models')
    #     self.mano_layer = ManoLayer(side='right', mano_root=mano_path, use_pca=False, flat_hand_mean=True,
    #                            center_idx=0, ncomps=45, root_rot_mode="axisang", joint_rot_mode="axisang").to(self.device)
    #     self.hand_faces_template = self.mano_layer.th_faces.repeat(1, 1, 1)

    # def load_obj_mesh(self):
    #     target_mesh_class = str(self.obj_id).zfill(2) + '_' + str(OBJType(int(self.obj_id)).name)
    #     self.obj_mesh_name = target_mesh_class + '.obj'

    #     obj_mesh_path = os.path.join(self.baseDir, self.objModelDir, target_mesh_class, self.obj_mesh_name)
    #     obj_scale = _OBJECT_SCALE_FIXED[int(self.obj_id) - 1]
    #     obj_verts, obj_faces, _ = load_obj(obj_mesh_path)
    #     obj_verts_template = (obj_verts * float(obj_scale)).to(self.device)
    #     obj_faces_template = torch.unsqueeze(obj_faces.verts_idx, axis=0).to(self.device)

    #     # h = torch.ones((obj_verts_template.shape[0], 1), device=self.device)
    #     # self.obj_verts_template_h = torch.cat((obj_verts_template, h), 1)

    #     obj_mesh_data = {}
    #     obj_mesh_data['verts'] = obj_verts_template
    #     obj_mesh_data['faces'] = obj_faces_template

    #     return obj_mesh_data

    # def get_obj_pose(self, camID, idx):
    #     anno = self.anno_dict[camID][idx]
    #     obj_mat = np.squeeze(np.asarray(anno['Mesh'][0]['object_mat']))
    #     return obj_mat

    def load_data(self, seqName, trialName, valid_cams):

        anno_base_path = os.path.join(self._base_anno, seqName, trialName, 'annotation')
        rgb_base_path = os.path.join(self._base_source, seqName, trialName, 'rgb')
        depth_base_path = os.path.join(self._base_source, seqName, trialName, 'depth')

        # use cropped image if exists. 
        flag_crop = False
        rgb_aug_base_path = os.path.join(self._base_source_aug, seqName, trialName)
        if os.path.isdir(rgb_aug_base_path):
            flag_crop = True
            depth_base_path = os.path.join(rgb_aug_base_path, 'depth_crop')
            if self._use_aug:
                rgb_base_path = os.path.join(rgb_aug_base_path, 'rgb_aug')
            else:
                rgb_base_path = os.path.join(rgb_aug_base_path, 'rgb_crop')


        anno_dict = {}
        rgb_dict = {}
        depth_dict = {}

        for camIdx, camID in enumerate(self.camIDset):
            anno_dict[camID] = []
            rgb_dict[camID] = []
            depth_dict[camID] = []

            if camID in valid_cams:
                anno_list = os.listdir(os.path.join(anno_base_path, camID))

                for anno in anno_list:
                    anno_path = os.path.join(anno_base_path, camID, anno)                    
                    # with open(anno_path, 'r', encoding='UTF-8 SIG') as file:
                    #     anno_data = json.load(file)
                    anno_dict[camID].append(anno_path)

                    rgb_path = os.path.join(rgb_base_path, camID, anno[:-5] + '.jpg')
                    # rgb_data = np.asarray(cv2.imread(rgb_path))
                    rgb_dict[camID].append(rgb_path)
                    
                    depth_path = os.path.join(depth_base_path, camID, anno[:-5] + '.png')
                    #depth_data = np.asarray(cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)).astype(float)
                    depth_dict[camID].append(depth_path)

        return anno_dict, rgb_dict, depth_dict, flag_crop
    
    def __len__(self):
        return len(self.mapping)

    def __getitem__(self, idx):
        # s : subject , t : trial,  c : camera, i : idx, f : framenum
        s, t, c, i, f = self.mapping[idx]

        sample = self.dataset_samples[s][t][c][i]

        ####### load data and set sample #####
        label_path = sample['label_path']
        with open(label_path, 'r', encoding='UTF-8 SIG') as file:
            anno_data = json.load(file)     

        hand_2d = np.squeeze(np.asarray(anno_data['hand']['projected_2D_pose_per_cam']))
        bbox, _ = extractBbox(hand_2d)
        
        rgb_path = sample['rgb_path']
        depth_path = sample['depth_path']

        rgb_data = np.asarray(cv2.imread(rgb_path))
        depth_data = np.asarray(cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)).astype(float)
        
        # crop the image if the source is from origin
        if not sample['flag_crop']:        
            rgb_data = rgb_data[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2])]
            depth_data = depth_data[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2])]

        sample['anno_data']  = anno_data
        sample['rgb_data'] = rgb_data
        sample['depth_data'] = depth_data
        sample['bbox'] = bbox        

        sample['camera']  = c
        sample['intrinsics'] = self.cam_param_dict[s][t]['Ks'][c]
        sample['extrinsics'] = self.cam_param_dict[s][t]['Ms'][c]

        return sample


def main():
    setup = 's2'
    split = 'test'
    print("loading ... ", setup + '_' + split)

    HOG_db = HOGDataset(setup, split)

    print("db len: ", len(HOG_db))
    data = HOG_db[0]

        
if __name__ == '__main__':
    main()