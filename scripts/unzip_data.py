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
    
    fnames = glob(os.path.join(base_path, "zipped", "**/*"), recursive=True)

    img_zips = []
    img_cropped_zips = []
    annotation_zips = []
    mask_zips = []

    zips = [[], [], [], []]

    fname_types = ["Labeling_data","extra_data","Source_augmented","Source_data"]
    for fname in fnames:
        if not ".zip" in fname:
            continue
        for fname_type, zip_list in zip(fname_types, zips):
            if fname_type in fname:
                zip_list.append(fname)
                break

    for fname_type, zip_list in zip(fname_types, zips):
        output_path = os.path.join(base_path, fname_type)
        os.makedirs(output_path, exist_ok=True)

        pbar = tqdm(zip_list)
        for zip_file in pbar:
            pbar.set_description(f"Unzipping {zip_file} to {output_path}")
            unzip(zip_file, output_path)


if __name__ == "__main__":
    main()