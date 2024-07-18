import os
import os.path as op
import warnings
from tqdm import tqdm
import urllib.request
from config import *

import wget
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
            target_url_set = base_url_set
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
            subjects = subject_types
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


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def bar_custom(current, total, width=80):
    width=30
    avail_dots = width-2
    shaded_dots = int(math.floor(float(current) / total * avail_dots))
    percent_bar = '[' + '■'*shaded_dots + ' '*(avail_dots-shaded_dots) + ']'
    progress = "%d%% %s [%d / %d]" % (current / total * 100, percent_bar, current, total)
    return progress

def download_urls(urls, output_folder):
    # for url in urls:
    #     file_name = url.split('/')[-1]
    #     output_path = os.path.join(output_folder, file_name)

    #     with DownloadProgressBar(unit='B', unit_scale=True,
    #                             miniters=1, desc=file_name) as t:
    #         urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

        # header={"Accept-Encoding": "gzip", "Content-Type":"application/json"}
        # data='{"message":"hello"}'
        # response=requests.get(url)

    for url in urls:
        file_name = url.split('/')[-1]
        output_path = os.path.join(output_folder, file_name)
        os.system(f"wget -c {url} -P {output_folder}")