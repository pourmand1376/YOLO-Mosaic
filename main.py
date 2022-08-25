import random

import cv2

from dataset import dataset
from mosaic import mosaic
from typer import Typer
from pathlib import Path 
import glob
import os


app = Typer(name='Mosaic', add_help_option=False)

OUTPUT_SIZE = (1200, 1200)  # Height, Width
SCALE_RANGE = (0.5, 0.5)
FILTER_TINY_SCALE = 1 / 50  # if height or width lower than this scale, drop it.

ANNO_DIR = '/media/amir/external_1TB/Dataset/Anevrism/output_folder2/labels'
IMG_DIR = '/media/amir/external_1TB/Dataset/Anevrism/output_folder2/images'

# Names of all the classes as they appear in the Pascal VOC Dataset
category_name = ['Anevrism']


def chunker(seq, size):
    res = []
    for el in seq:
        res.append(el)
        if len(res) == size:
            yield res
            res = []
    if res:
        res = res + (size - len(res)) * [res[-1]] 
        yield res

# Change the output path of the imwrite commands to wherever you want to get both the mosaic image and the image with boxes
@app.command()
def main(img_dir:str=None, annotations_dir:str=None, classes:str=None, output_folder: str=None):
    """
    
    """
    output_folder = '/media/amir/external_1TB/Dataset/Anevrism/output_mosaic'
    output_folder = Path(output_folder)
    output_folder_img = output_folder / "images"
    output_folder_img.mkdir(exist_ok=True,parents=True)

    output_folder_img_box = output_folder / "images_box"
    output_folder_img_box.mkdir(exist_ok=True, parents=True)

    output_folder_label = output_folder / "labels"
    output_folder_label.mkdir(exist_ok=True, parents=True)

    # img_paths, annos = dataset(ANNO_DIR, IMG_DIR)
    img_paths = glob.glob(os.path.join(IMG_DIR, '*.png'))

    for idxs in chunker(range(len(img_paths)), 4):
        new_image, new_annos = mosaic(img_paths, ANNO_DIR,
                                    idxs,
                                    OUTPUT_SIZE, SCALE_RANGE,
                                    filter_scale=FILTER_TINY_SCALE)
        
        file_name = img_paths[idxs[0]].split('/')[-1]
        cv2.imwrite(str(output_folder_img / file_name), new_image) #The mosaic image
        for anno in new_annos:
            start_point = (int(anno[1] * OUTPUT_SIZE[1]), int(anno[2] * OUTPUT_SIZE[0]))
            end_point = (int(anno[3] * OUTPUT_SIZE[1]), int(anno[4] * OUTPUT_SIZE[0]))
            cv2.rectangle(new_image, start_point, end_point, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imwrite(str(output_folder_img_box / file_name), new_image) # The mosaic image with the bounding boxes
        
        yolo_anno = []
        
        for anno in new_annos:
            tmp = []
            tmp.append(anno[0])
            tmp.append((anno[3]+anno[1])/2)
            tmp.append((anno[4]+anno[2])/2)
            tmp.append(anno[3]-anno[1])
            tmp.append(anno[4]-anno[2])
            yolo_anno.append(tmp)

        if yolo_anno:
            with open(str(output_folder_label / file_name.replace('.png','.txt')), 'w') as file: # The output annotation file will appear in the output.txt file
                for line in yolo_anno:
                    file.write((' ').join([str(x) for x in line]) + '\n')   

if __name__ == '__main__':
    app()