# HOGraspNet
This repository contains instructions on getting the data and code of the work `Dense Hand-Object(HO) GraspNet with Full Grasping Taxonomy and Dynamics` presented at ECCV 2024. For more information on the benchmark please check out [[1]](#refs).

## Overview

So far ARCTIC provides the following data and models:
- `arctic_data/data/images [649G]`: Full 2K-resolution images
- `arctic_data/data/cropped_images [116G]`: loosely cropped version of the original images around the object center for fast image loading
- `arctic_data/data/raw_seqs [215M]`: raw GT sequences in world coordinate (e.g., MANO, SMPLX parameters, egocentric camera trajectory, object poses)
- `arctic_data/data/splits [18G]`: splits created by aggregating processed sequences together based on the requirement of a specific split.
- `arctic_data/data/feat [14G]`: validation set image features needed for LSTM models.
- `arctic_data/data/splits_json [40K]`: json files to define the splits
- `meta [91M]`: camera parameters, object info, subject info, subject personalized vtemplates, object templates.
- `arctic_data/model [6G]`: weights of our CVPR baselines, and the same baseline models re-trained after upgrading dependencies.

See [`data_doc.md`](./data_doc.md) for an explanation of the data you will download, how the files are related to each others, and details on each file type.


## Download HOGraspNet
Please fill this [form](https://forms.gle/UqH15zN2PiBGQDUs7) to download the dataset after reading the [terms and conditions](#terms).

Set provided data asset in
```
HOGraspNet/assets/checksum.json
HOGraspNet/assets/
    urls/
        images.txt: Full RGB & Depth images
        annotations.txt: annotations
        extra.txt: hand & object segmentation masks(pseudo)
        images_augmented.txt: Cropped & background augmented RGB images

```

Depending on your usage of the dataset, we suggest different download protocols. 

options:

--type : all_small(0): image_crop+annotation+mask, all(1): all_small+image_origin, image_crop(2), annotation(3), mask(4), image_origin(5) (default : 0)
--subject : all, 1 or 1,2 or 1-3 or small. (default : all)
--objModel : True, False (default : True)


- ...if the full dataset is not downloaded(ex. setting --subject option to "small" or specific subject indexs), only s0 split is available.

(additional) subjects_info : "subject_info.csv"

Download the dataset with default option: 
- Cropped/Background augmented Images + Annotations + Masks
- All subject (S1~S99)
- Scanned object 3D models

```bash
python scripts/download_data.py
```

Download the dataset with maual option (example): 
- Only Cropped/Background augmented Images
- Pre-defined 5 subjects

```bash
python scripts/download_data.py --type 2 --subject small --objModel False
```

Now, unzip them all:

```bash
python scripts_data/unzip_data.py # unzip downloaded data
```

The raw downloaded data can be found under `data/zipped/`. The unzipped data and models can be found under `data/`. See [`visualization.md`](visualization.md) for the explanation of how the files can be visualized.



## Dataset structure:

The dataset is organized as the f


## Object models
Available objects: 'ju


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

## References
<a name="refs"></a>

[1] *Dense Hand-Object(HO) GraspNet with Full Grasping Taxonomy and Dynamics*, Woojin Cho, Jihyun Lee, Minjae Yi, Minje Kim, Taeyun Woo, Donghwan Kim, Taewook Ha, Hyokeun Lee, Je-Hwan Ryu, Woontack Woo and Tae-Kyun Kim, ECCV 2024.)