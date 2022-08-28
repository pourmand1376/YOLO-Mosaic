import glob
from lib2to3.pytree import convert
import os
from pathlib import Path

import cv2
from typer import Typer

from dataset import dataset
from mosaic import mosaic
import yaml

from logger import logger

app = Typer(name="Mosaic")

OUTPUT_SIZE = (1200, 1200)  # Height, Width
SCALE_RANGE = (0.5, 0.5)
FILTER_TINY_SCALE = 0  # if height or width lower than this scale, drop it.


def chunker(seq, size):
    def _get_patient(path):
        return str.join('_',path.split('/')[-1].split('_')[:-1])
        
    def _copy_last_element(res):
        return res + (size - len(res)) * [res[-1]]

    res = []
    previous_patient = None
    for el in seq:
        if previous_patient is None or _get_patient(el) == previous_patient:
            previous_patient = _get_patient(el)
            res.append(el)
            if len(res) == size:
                yield res
                res = []
        else:
            if len(res):
                logger.info(f"Finishing Patient {previous_patient}") 
                res = _copy_last_element(res)
                yield res
            
            previous_patient = _get_patient(el)
            logger.info(f'Starting New patient {previous_patient}')
            res = []
            res.append(el)
    if res:
        res = _copy_last_element(res)
        yield res


@app.command()
def convert_database(database_yaml: str, output_folder :str,):
    """
    This one receives a yaml file and iterates over all folders in that folder

    Args:
        database_yaml (str): path to database yaml file. Yaml file should contain the path to images dir
        output_folder (str): the folder in which you want your data to be stored
    """
    
    database_yaml = Path(database_yaml)
    output_folder = Path(output_folder)
    output_folder.mkdir(exist_ok=True, parents=True)

    database=yaml.safe_load(database_yaml.read_text())
    for item in ['train', 'val', 'test']:
        if item in database:
            folder_path=database[item]
            annotation_path = folder_path.replace('/images','/labels')
            convert_images(str(folder_path), annotation_path, str(output_folder / item) )

    

@app.command()
def convert_images(
    img_dir: str,
    annotation_dir: str,
    output_folder: str,
    output_height: int = 1000,
    output_width: int = 1000,
    draw_bbox: bool = False
):
    """
    This one works with a single directory containning images and labels

    Args:
        img_dir: directory which contains the images
        annotaion_dir: directory containning text files
        output_folder: main directory which data should be extracted to
        output_height and output_width is clear!
    """
    img_dir = Path(img_dir)
    annotation_dir = Path(annotation_dir)
    output_folder = Path(output_folder)

    OUTPUT_SIZE = (output_height, output_width)

    output_folder_img = output_folder / "images"
    output_folder_img.mkdir(exist_ok=True, parents=True)

    if draw_bbox:
        output_folder_img_box = output_folder / "images_box"
        output_folder_img_box.mkdir(exist_ok=True, parents=True)

    output_folder_label = output_folder / "labels"
    output_folder_label.mkdir(exist_ok=True, parents=True)

    # img_paths, annos = dataset(ANNO_DIR, IMG_DIR)
    img_paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))
        
    for idxs in chunker(img_paths, 4):
        logger.info("Starting Chunk")
        logger.info(f"Processing  {idxs[0]}")
        logger.info(f"Processing  {idxs[1]}")
        logger.info(f"Processing  {idxs[2]}")
        logger.info(f"Processing  {idxs[3]}")
         
        new_image, new_annos = mosaic(
            img_paths,
            annotation_dir,
            idxs,
            OUTPUT_SIZE,
            SCALE_RANGE,
            filter_scale=FILTER_TINY_SCALE,
        )

        file_name = idxs[0].split("/")[-1]
        cv2.imwrite(str(output_folder_img / file_name), new_image)  # The mosaic image

        if draw_bbox:
            for anno in new_annos:
                start_point = (int(anno[1] * OUTPUT_SIZE[1]), int(anno[2] * OUTPUT_SIZE[0]))
                end_point = (int(anno[3] * OUTPUT_SIZE[1]), int(anno[4] * OUTPUT_SIZE[0]))
                cv2.rectangle(
                    new_image, start_point, end_point, (0, 255, 0), 1, cv2.LINE_AA
                )
            
            cv2.imwrite(
                str(output_folder_img_box / file_name), new_image
            )  # The mosaic image with the bounding boxes

        yolo_anno = []

        for anno in new_annos:
            tmp = []
            tmp.append(anno[0])
            tmp.append((anno[3] + anno[1]) / 2)
            tmp.append((anno[4] + anno[2]) / 2)
            tmp.append(anno[3] - anno[1])
            tmp.append(anno[4] - anno[2])
            yolo_anno.append(tmp)

        if len(yolo_anno):
            with open(
                str(output_folder_label / file_name.replace(".png", ".txt")), "w"
            ) as file:  # The output annotation file will appear in the output.txt file
                for line in yolo_anno:
                    file.write((" ").join([str(x) for x in line]) + "\n")


if __name__ == "__main__":
    app()
    # for debugging!
    # convert_images('/media/amir/external_1TB/Dataset/Anevrism/output_folder2/images',
    # '/media/amir/external_1TB/Dataset/Anevrism/output_folder2/labels',
    # '/media/amir/external_1TB/Dataset/Anevrism/output_mosaic')
    #main('/media/amir/external_1TB/Dataset/Anevrism/output_folder2/database.yaml', 
    #'/media/amir/external_1TB/Dataset/Anevrism/output_mosaic', 10)
