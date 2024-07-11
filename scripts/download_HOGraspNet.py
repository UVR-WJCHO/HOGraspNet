import argparse
import os
import os.path as op
import warnings

from tqdm import tqdm
import urllib.request
from utils import check_args


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def download_data(url_file, out_folder, dry_run):
    logger.info("Done")


def main():
    ## check if the url is set ##
    if os.path.isdir("assets/url"):
        if len(os.listdir("assets/url")) < 1:
            print(f"Error: no files in assets/url Check README.")
            sys.exit(0)
    else:
        print(f"Error: assets/url not exists. Check README.")
        sys.exit(0)

    ## parse download options ##
    """
    - split : --split s0 (s0~s4)
    - subject range : --subj 1 or --subj 2-5 or --subj 2,5,7 (default : s1-s99)
    - object : --obj 1 or --obj 1,2,3
    - grasp : --grasp 14 or --grasp 14, 27 (error if not exists)
    - objModel : True(default), False (download 3D models)
    """

    ## config ##
    base_url_path = "./assets/url/"
    base_url_set = ["images", "annotations", "extra", "images_augmented"]
    base_obj_models_url = "http://data.uvrlab.org/datasets/HOGraspNet/HOGraspNet_obj_models.zip"

    
    ## parse arguments ##
    parser = argparse.ArgumentParser(description="Download files from a list of URLs")
    parser.add_argument(
        "--split",
        choices=split_types,
        help="Select the split option as 0 to 4. (Total s0 ~ s4) ",
        required=False,
        default=0
    )
    parser.add_argument(
        "--subject",
        type=str,
        help="Select the subject option as 1 or 1,2 or 1-3 or small. (Total S1~S99)",
        required=False,
        default="all"
    )
    parser.add_argument(
        "--object",
        type=str,
        help="Select the object option as 1 or 1,2 or 1-3. (Total 1~30)",
        required=False,
        default="all"
    )
    parser.add_argument(
        "--grasp",
        type=str,
        help="Select the grasp class option. (Check taxonomy index in paper)",
        required=False,
        default="all"
    )

    args = parser.parse_args()

    ## check options ##
    subjects, objects, grasps = check_args(args.subject, args.object, args.grasp)
    
    for url_type in base_url_set:
        url_file = os.path.join(base_url_path, url_type+".txt")
        
        if os.isfile(url_file):
            
        else:
            print(f"There is no file in {url_file}; please download it through the Google form link.")

    # url = "http://data.uvrlab.org/datasets/HOGraspNet/HOGraspNet_obj_models.zip"
    # output_path = "HOGraspNet_obj_models.zip"
    # download_url(url, output_path)


if __name__ == "__main__":
    main()