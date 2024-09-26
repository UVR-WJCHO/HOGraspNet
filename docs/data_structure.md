
### Data folder structure
```bash
HOGraspNet_DIR/ # ROOT

HOGraspNet_DIR/data/ # data dir
    zipped/ # downloaded zip files
    source_data/ # source data folder (1920*1080)
        [capture_date]_S[subject_idx]_obj_[object_idx]_grasp_[grasp_idx]/ # sequence name
            trial_[trial_idx]/ # trial idx
                rgb/
                    [cam_name]/[cam_name]_[frame].jpg # rgb image file
                depth/
                    [cam_name]/[cam_name]_[frame].png # depth image file
                ObjectPose/ # optical marker position records(useless)

    source_augmented/ # cropped and augmented source data folder
        [capture_date]_S[subject_idx]_obj_[object_idx]_grasp_[grasp_idx]/ # sequence name
            trial_[trial_idx]/ # trial idx
                depth_crop/[cam_name]/[cam_name]_[frame].png # cropped depth image file (640*480)
                rgb_crop/[cam_name]/[cam_name]_[frame].jpg # cropped depth image file (640*480)
                rgb_aug/[cam_name]/[cam_name]_[frame].jpg # cropped depth image file (640*480)

    labeling_data/ # annotation folder
        [capture_date]_S[subject_idx]_obj_[object_idx]_grasp_[grasp_idx]/ # sequence name
            trial_[trial_idx]/ # trial idx
                annotation/[cam_name]/[cam_name]_[frame].json # full annotation file

    extra_data/ # pseudo mask data from image
        [capture_date]_S[subject_idx]_obj_[object_idx]_grasp_[grasp_idx]/ # sequence name
            trial_[trial_idx]/ # trial idx
                hand_mask/
                    [cam_name]/[cam_name]_[frame].png # binary hand mask 
                obj_mask/
                    [cam_name]/[cam_name]_[frame].png # binary object mask

    obj_scanned_models/ # scanned 3D object models root
        [object_idx]_[object_name]/
            [object_idx]_[object_name].mtl
            [object_idx]_[object_name].obj
            [object_idx]_[object_name]-TPO-T-DIFF.jpg
            [object_idx]_[object_name]-TPO-T-NORM.jpg
            
    bg_samples/ # manual background sample root
        ...         
```

### Hand joint order 
We use the hand joints order identical to that of OpenPose.   
`[Wrist, TMCP,  TPIP, TDIP, TTIP, IMCP, IPIP, IDIP, ITIP, MMCP,  MPIP, MDIP, MTIP, RMCP, RPIP, RDIP, RTIP, PMCP, PPIP, PDIP, PTIP]`
where ’T’, ’I’, ’M’, ’R’, ’P’ denote ’Thumb’, ’Index’, ’Middle’, ’Ring’, ’Pinky’ fingers.

<img src="joint_order.png" alt="drawing" width="300"/>


### Dataloader sample structure
* Each sample from dataloader contains following data packed in a dictionary:
    * `rgb_path`: path to rgb image file (.jpg)
    * `depth_path`: path to depth image file (.png)
    * `label_path`: path to annotation file (.json)
    * `obj_ids`: index of object `1 - 30`
    * `taxonomy`: index of grasp taxonomy `1 ... 33, total 28 classes, refer to Fig.3 in paper`
    * `flag_crop`: True if the image data is loaded from source_augmented(cropped)
    * `rgb_data`: cropped rgb image data (640*480*3)
    * `depth_data`: cropped depth image data (640*480)
    * `bbox`: 2D bounding box of hand `[col_min, row_min, bbox_width, bbox_height]`
    * `camera`: camera name `mas or sub1 or sub2 or sub3`
    * `intrinsics`: camera intrinsic parameter (tensor)
    * `extrinsics`: camera extrinsic parameter (tensor)
    
    * `anno_data`: dictionary of annotation data
        ```bash
        - info
            - name
            - description


        ```

