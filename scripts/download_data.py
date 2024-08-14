import argparse
import os
import os.path as op
import warnings

from tqdm import tqdm
import urllib.request
from util.utils import check_args, download_urls


def main():
    ## config ##
    base_url_path = "./assets/urls/"    
    base_obj_models_url = "https://www.dropbox.com/scl/fi/lpud17kswi3dj6egtihbs/HOGraspNet_obj_models.zip?rlkey=d8hkdwieecp557gx39y0bc6tw&st=jxh42gsc&dl=0"

    
    ## check if the url is set ##
    if os.path.isdir("assets/urls"):
        if len(os.listdir("assets/urls")) < 1:
            print(f"ERROR: no files in assets/urls. Check README.")
            sys.exit(0)
    else:
        print(f"ERROR: assets/urls not exists. Check README.")
        sys.exit(0)


    ## parse arguments ##
    parser = argparse.ArgumentParser(description="Download files from a list of URLs")
    parser.add_argument(
        "--type",
        type=int,
        help="Select the data types(all_small(0): image_crop+annotation+mask, all(1): all_small+image_origin, image_crop(2), annotation(3), mask(4), image_origin(5)) ",
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
        "--objModel",
        type=bool,
        help="Download the scanned object 3D models",
        required=False,
        default=True
    )
    args = parser.parse_args()


    ## check arguments and create dataset directory ##
    target_url_set, subjects = check_args(args.type, args.subject)
    print(f"Target data types : {target_url_set}")
    print(f"Target subject indexs for option - {args.subject} : {subjects}")

    os.makedirs("data/zipped", exist_ok=True)

    ## iterate each target urls ##
    print("Downloading data from urls ...")
    for url_type in target_url_set:        
        download_list = []
        url_file = os.path.join(base_url_path, url_type+".txt")        
        if os.path.isfile(url_file):
            with open(url_file, 'r') as f:                
                url_list = f.read().splitlines()
                for url in url_list:
                    file_name = url.split('/')[-1]
                    names = file_name.split('_')      
                    subject = int(names[1][1:])
                    if subject in subjects:                   
                        download_list.append(url)
        else:
            print(f"ERROR: There is no file in {url_file}; please download it through the Google form link.")
        
        output_path = f"data/zipped/{url_type}"
        os.makedirs(output_path, exist_ok=True)
        download_urls(download_list, output_path)


    if args.objModel:
        print("Downloading scanned object 3D models ...")

        output_path = "data/obj_scanned_models"
        os.makedirs(output_path, exist_ok=True)        
        download_urls([base_obj_models_url], output_path)
        


if __name__ == "__main__":
    # os.chdir('/scratch/NIA/HOGraspNet')   # for debug
    main()