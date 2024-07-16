# HOGraspNet
This repository contains instructions on getting the data and code of the work `Dense Hand-Object(HO) GraspNet with Full Grasping Taxonomy and Dynamics` presented at ECCV 2024.

Project page : [HOGraspNet](https://hograspnet2024.github.io/)


## Overview
HOGraspNet provides the following data and models:
- `data/Source_data`: Full 1920*1080 size RGB & Depth images ("Source_data/Object Pose" is unnecessary data. It will be removed soon.)
- `data/Labeling_data`: Json files for annotations.
- `data/extra_data`: Binary hand & object mask data for cropped image. (Bounding box is provided through the dataloader module.)
- `data/Source_augmented`: Cropped images around the hand and background augmented RGB images.
- `data/obj_scanned_models`: Manually scanned 3D models for 30 objects utilized in the dataset.

<!-- See [`data_structure.md`](./docs/data_structure.md) for an explanation of the data you will download. -->


## Download HOGraspNet

Please fill this [form](https://forms.gle/UqH15zN2PiBGQDUs7) to download the dataset after reading the [terms and conditions](#terms).

Set provided data asset in
<!-- HOGraspNet/assets/checksum.json -->
```
HOGraspNet/assets/urls/
        images.txt: Full RGB & Depth images
        annotations.txt: annotations
        extra.txt: hand & object segmentation masks(pseudo)
        images_augmented.txt: Cropped & background augmented RGB images

```

**Options**

Depending on your usage of the dataset, we suggest different download options. 


* --type (type: int, default: 0): 
    * 0 : Source_augmented(cropped) + Labeling_data + extra_data(mask)
    * 1 : Source_augmented + Labeling_data + extra_data + Source_data
    * 2 : Source_augmented
    * 3 : Labeling_data
    * 4 : extra_data
    * 5 : Source_data

* --subject (type: string, default: all): 
    * all : subject 1~99
    * small : pre-defined 5 subjects
    * 1 : subject 1
    * 1,2 : subject 1 and 2
    * 1-3 : subject 1 to 3
        
* --objModel (type: bool, default : True): 
    * True : Download the scanned object 3D models
    * False : Skip

⚠️ If the full dataset is not downloaded (e.g., setting the subject option to "small" or a specific subject index), only the s0 split is available in the dataloader.


**Subject info**

Here, we provide a summary of each subject's information included in the dataset. [`HOGraspNet_subject_info`](./assets/HOGraspNet_subject_info.csv)
Please check it if you need data on a specific type of subject.

**Download procedure**

1. Download the dataset with default option: 
- Cropped/Background augmented Images + Annotations + Masks
- All subject (S1~S99)
- Scanned object 3D models

```bash
python scripts/download_data.py
```

1.1 or Download the dataset with maual option (example): 
- Only Cropped/Background augmented Images
- Pre-defined 5 subjects

```bash
python scripts/download_data.py --type 2 --subject small --objModel False
```

2. Unzip them all:

```bash
python scripts/unzip_data.py # unzip downloaded data
```

The raw downloaded data can be found under `data/zipped/`. The unzipped data and models can be found under `data/`. See [`visualization.md`](./docs/visualization.md) for the explanation of how the files can be visualized.


## Dataloader



## Data visualization



## Manual background augmentation



## TODO ##

- video sequence will be update soon.
- HOGraspNet_v1 has lower quality on object pose/contact map quality. Enhanced v2 will be released in a month.

- Visualization code
- HOGraspNet_v2


## Terms and conditions
<a name="terms"></a>
The download and use of the dataset is released for academic research only and it is free to researchers from educational or research institutes for non-commercial purposes. When downloading the dataset you agree to (unless with expressed permission of the authors): not redistribute, modificate, or commercial usage of this dataset in any way or form, either partially or entirely.

If using this dataset, please cite the following paper:

```
@inproceedings{2024graspnet,
        title={Dense Hand-Object(HO) GraspNet with Full Grasping Taxonomy and Dynamics},
        author={Cho, Woojin and Lee, Jihyun and Yi, Minjae and Kim, Minje and Woo, Taeyun and Kim, Donghwan and Ha, Taewook and Lee, Hyokeun and Ryu, Je-Hwan and Woo, Woontack and Kim, Tae-Kyun},
        booktitle={ECCV},
        year={2024}
}
```

## Acknowledgments
이 연구는 과학기술정보통신부의 재원으로 한국지능정보사회진흥원의 지원을 받아 구축된 "물체 조작 손 동작 3D 데이터"을 활용하여 수행된 연구입니다.
본 연구에 활용된 데이터는 AI 허브([aihub.or.kr](http://aihub.or.kr/))에서 다운로드 받으실 수 있습니다.
This research (paper) used datasets from 'The Open AI Dataset Project (AI-Hub, S. Korea)'.
All data information can be accessed through 'AI-Hub ([www.aihub.or.kr](http://www.aihub.or.kr/))'.
