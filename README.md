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
Please fill this [form](https://goo.gl/forms/FIsXpYVIUov0j7Wv2) to download the dataset after reading the [terms and conditions](#terms).

Depending on your usage of the dataset, we suggest different download protocols. 

Clean download cache:
```bash
./bash/clean_downloads.sh
```


Download cropped images (optional):

```bash
./bash/download_cropped_images.sh
```

Download full-resolution images (optional):

⚠️ If you just want to train and compete with our CVPR models, you only need the cropped images above. The script below download the full-resolution images which can take a while.

```bash
./bash/download_images.sh
```

Download pre-processed splits (optional):

```bash
./bash/download_splits.sh
```

Download image features used by our CVPR LSTM models (optional):

```bash
./bash/download_feat.sh
```

Download our pre-trained CVPR model weights (optional):

```bash
./bash/download_baselines.sh
```

Download MoCap related data (optional):

```bash
./bash/download_mocap.sh
```

After downloading what you need, you can now verify the checksum for corruption, and unzip them all:

```bash
python scripts_data/checksum.py # verify checksums; this could take a while
python scripts_data/unzip_download.py # unzip downloaded data
```

The raw downloaded data can be found under `downloads`. The unzipped data and models can be found under `unpack`. See [`processing.md`](processing.md) for explanation of how the files are organized and what they represent.



## Dataset structure:

The dataset is organized as the f


## Object models
Available objects: 'ju





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