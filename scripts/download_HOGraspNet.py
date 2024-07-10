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
    ## check if the url is set
    if os.is

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
    if args.dry_run:
        logger.info("Running in dry-run mode")

    # download_data(args.url_file, args.out_folder, args.dry_run)
    url = "http://data.uvrlab.org/datasets/HOGraspNet/HOGraspNet_obj_models.zip"
    output_path = "HOGraspNet_obj_models.zip"
    download_url(url, output_path)


if __name__ == "__main__":
    main()