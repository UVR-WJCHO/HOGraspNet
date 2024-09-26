import os
import os.path
from glob import glob
from tqdm import tqdm
import zipfile


def unzip(zip_file, output_path):
    os.makedirs(output_path, exist_ok=True)
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(output_path)


def main():
    ## config ##
    base_path = "./data"    

    ## unzip object models if exists
    fname = os.path.join(base_path, "obj_scanned_models", "HOGraspNet_obj_models.zip")
    if os.path.isfile(fname):
        print("HOGraspNet_obj_models.zip exists. Unzip.")
        unzip(fname, os.path.join(base_path, "obj_scanned_models"))
        os.remove(fname) 

    ## unzip dataset
    fnames = glob(os.path.join(base_path, "zipped", "**/*"), recursive=True)

    img_zips = []
    img_cropped_zips = []
    annotation_zips = []
    mask_zips = []

    zips = [[], [], [], []]

    fname_types = ["Labeling_data","extra_data","Source_augmented","Source_data"]
    fname_outs = ["labeling_data","extra_data","source_augmented","source_data"]
    for fname in fnames:
        if not ".zip" in fname:
            continue
        for fname_type, zip_list in zip(fname_types, zips):
            if fname_type in fname:
                zip_list.append(fname)
                break

    for fname_type, zip_list, fname_out in zip(fname_types, zips, fname_outs):
        output_path = os.path.join(base_path, fname_out)
        # os.makedirs(output_path, exist_ok=True)
 
        pbar = tqdm(zip_list)
        for zip_file in pbar:
            pbar.set_description(f"Unzipping {zip_file} to {output_path}")
            unzip(zip_file, output_path)
            os.remove(zip_file)




if __name__ == "__main__":
    main()