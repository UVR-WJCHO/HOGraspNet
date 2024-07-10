import argparse
import os
import os.path as op
import warnings

from tqdm import tqdm
import urllib.request


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

    parser = argparse.ArgumentParser(description="Download files from a list of URLs")
    parser.add_argument(
        "--url_file",
        type=str,
        help="Path to file containing list of URLs",
        required=True,
    )
    parser.add_argument(
        "--out_folder",
        type=str,
        help="Path to folder to store downloaded files",
        required=True,
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Select top 5 URLs if enabled and 'images' is in url_file",
    )
    args = parser.parse_args()
    
    # download_data(args.url_file, args.out_folder, args.dry_run)
    url = "http://data.uvrlab.org/datasets/HOGraspNet/HOGraspNet_obj_models.zip"
    output_path = "HOGraspNet_obj_models.zip"
    download_url(url, output_path)


if __name__ == "__main__":
    main()