import os
import sys
sys.path.append(os.environ['HOG_DIR'])
from glob import glob
from tqdm import tqdm
import cv2
import numpy as np
import random


if __name__ == '__main__':
    ## set background images
    bg_list = glob(os.path.join(os.environ['HOG_DIR'], "data", "bg_samples", "*.*"))
    bg_list = [f for f in bg_list if f.endswith('.png') or f.endswith('.jpg')]
    assert len(bg_list) > 0,  "No *.jpg or *.png typed images; modify the code."


    ## source list to augment
    source_path = os.path.join(os.environ['HOG_DIR'], "data", "source_augmented")
    seq_list = os.listdir(source_path)

    save_path = os.path.join(os.environ['HOG_DIR'], "data", "manual_augmented")
    os.makedirs(save_path, exist_ok=True)

    extra_path = os.path.join(os.environ['HOG_DIR'], "data", "extra_data")

    for seq_idx, seq_name in enumerate(tqdm(seq_list)):
        rgb_list = glob(source_path + '/' + seq_name + '/**/rgb_crop/*/*.jpg', recursive=True)

        for rgb_path in rgb_list:
            splits = rgb_path.split('/')
            seq = splits[-5]
            trial = splits[-4]
            cam = splits[-2]
            rgb_name = splits[-1]

            rgb_crop = cv2.imread(rgb_path)

            hand_mask_path = os.path.join(extra_path, seq, trial, 'hand_mask', cam, rgb_name[:-4] + '.png')
            obj_mask_path = os.path.join(extra_path, seq, trial, 'obj_mask', cam, rgb_name[:-4] + '.png')
            hand_mask = cv2.imread(hand_mask_path)
            obj_mask = cv2.imread(obj_mask_path)

            mask = np.zeros_like(hand_mask)
            mask[(hand_mask == 255) | (obj_mask == 255)] = 255

            mask_img = rgb_crop.copy()
            mask_img[mask != 255] = 0
            bg_img_path = random.choice(bg_list)
            bg_img = cv2.imread(bg_img_path)
            bg_img = cv2.resize(bg_img, (mask_img.shape[1], mask_img.shape[0]))
            mask_img[mask_img == 0] = bg_img[mask_img == 0]

            ## save augmented images         
            new_aug_path = os.path.join(save_path, seq, trial, cam)
            os.makedirs(new_aug_path, exist_ok=True)
            new_aug_path = os.path.join(new_aug_path, rgb_name)
            cv2.imwrite(new_aug_path, mask_img)