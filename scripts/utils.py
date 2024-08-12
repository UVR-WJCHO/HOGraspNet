import os
import wget
from config import cfg
import requests
from requests.exceptions import ConnectionError
from tqdm.autonotebook import tqdm
import time
import math


def extractBbox(hand_2d, image_rows=1080, image_cols=1920, bbox_w=640, bbox_h=480):
    # consider fixed size bbox
    x_min_ = min(hand_2d[:, 0])
    x_max_ = max(hand_2d[:, 0])
    y_min_ = min(hand_2d[:, 1])
    y_max_ = max(hand_2d[:, 1])

    x_avg = (x_min_ + x_max_) / 2
    y_avg = (y_min_ + y_max_) / 2

    x_min = max(0, x_avg - (bbox_w / 2))
    y_min = max(0, y_avg - (bbox_h / 2))

    if (x_min + bbox_w) > image_cols:
        x_min = image_cols - bbox_w
    if (y_min + bbox_h) > image_rows:
        y_min = image_rows - bbox_h

    bbox = [x_min, y_min, bbox_w, bbox_h]
    return bbox, [x_min_, x_max_, y_min_, y_max_]


def check_args(arg_type, arg_subject):
    try:  
        if arg_type == 0:            
            target_url_set = ["images_augmented", "annotations", "extra"]
        elif arg_type == 1:
            target_url_set = cfg.base_url_set
        elif arg_type == 2:            
            target_url_set = ["images_augmented"]
        elif arg_type == 3:            
            target_url_set = ["annotations"]
        elif arg_type == 4:            
            target_url_set = ["extra"]
        elif arg_type == 5:            
            target_url_set = ["images"]
        else:
            raise Exception("wrong type argument")
    except Exception as e:
        print("ERROR: wrong --type argument format. Please check the --help")

    try:  
        if arg_subject == "all":            
            subjects = cfg.subject_types
        else:
            if arg_subject == "small":
                subjects = [43, 63, 83, 84, 93]
            elif "-" in arg_subject:
                subjects = arg_subject.split("-")
                subjects = list(range(int(subjects[0]), int(subjects[1])+1))
            else:
                subjects = arg_subject.split(",")
                subjects = list(map(int, subjects))
    except Exception as e:
        print("ERROR: wrong --subject argument format. Please check the --help")

    return target_url_set, subjects


def download_urls(urls, output_folder, max_tries=7):
    """
    Download file from a URL to file_name,
    source refer from https://github.com/facebookresearch/ContactPose/blob/main/utilities/networking.py
    """
    for url in urls:
        file_name = url.split('/')[-1].split('?')[0]

        print(f"Downloading file name : {file_name}")

        file_name = output_folder + '/' + file_name
        url = url[:-1] + '1'

        # os.system(f"wget -cO {file_name} \"{url}\"")
        # os.system(f"curl -L -o {file_name} \"{url}\"")

        ## from contactpose
        tries = 0
        while tries < max_tries:
            done = download_url_once(url, file_name, True)
            if done:
                break
            else:
                t = 1
                print('*** Sleeping for {:f} s'.format(t))
                time.sleep(t)
                tries += 1
        if tries == max_tries:
            print('*** Download not complete. Max download tries exceeded')




def download_url_once(url, filename, progress=True):
    try:
        # headers = {'user-agent': 'Wget/1.16 (linux-gnu)'}
        headers = {"User-Agent": "Mozilla/5.0 (X11; CrOS x86_64 12871.102.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.141 Safari/537.36"}

        r = requests.get(url, stream=True, proxies=None, headers=headers)
    except ConnectionError as err:
        print(err)
        return False
    
    total_size = int(r.headers.get('content-length', 0))
    block_size = 102400 #100 Kibibyte
    if progress:
        t=tqdm(total=total_size, unit='iB', unit_scale=True)
    done = True
    datalen = 0
    with open(filename, 'wb') as f:
        itr = r.iter_content(block_size)
        while True:
            try:
                try:
                    data = next(itr)
                except StopIteration:
                    break
                if progress:
                    t.update(len(data))
                datalen += len(data)
                f.write(data)
            except KeyboardInterrupt:
                done = False
                print('Cancelled')
            except ConnectionError as err:
                done = False
                print(err)
    if progress:
        t.close()
    if (not done) or (total_size != 0 and datalen != total_size):
        print("ERROR, something went wrong")
        try:
            os.remove(filename)
        except OSError as e:
            print(e)
        return False
    else:
        return True
