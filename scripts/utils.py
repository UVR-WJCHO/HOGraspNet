import os
import os.path as op
import warnings

from tqdm import tqdm
import urllib.request


## default setting ##
split_types = list(range(5))
subject_types = list(range(100))
object_types = list(range(31))
del subject_types[0]
del object_types[0]
grasp_types = [1,2,17,18,22,30,3,4,5,19,31,10,11,26,28,16,29,23,20,25,9,24,33,7,12,13,27,14]

base_url_set = ["images_augmented", "annotations", "extra", "images"]

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


def download_urls(urls, output_folder):
    for url in urls:
        file_name = url.split('/')[-1]
        output_path = os.path.join(output_folder, file_name)

        with DownloadProgressBar(unit='B', unit_scale=True,
                                miniters=1, desc=file_name) as t:
            urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)
