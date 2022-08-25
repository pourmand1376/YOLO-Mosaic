import glob
import os
from pathlib import Path

import cv2
from typer import Typer

from dataset import dataset
from mosaic import mosaic

from logger import logger

app = Typer(name="Mosaic", add_help_option=False)

OUTPUT_SIZE = (1200, 1200)  # Height, Width
SCALE_RANGE = (0.5, 0.5)
FILTER_TINY_SCALE = 1 / 50  # if height or width lower than this scale, drop it.


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


@app.command()
def main(
    img_dir: str,
    annotation_dir: str,
    output_folder: str,
    output_height: int = 1200,
    output_width: int = 1200,
):
    """
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

    output_folder_img_box = output_folder / "images_box"
    output_folder_img_box.mkdir(exist_ok=True, parents=True)

    output_folder_label = output_folder / "labels"
    output_folder_label.mkdir(exist_ok=True, parents=True)

    # img_paths, annos = dataset(ANNO_DIR, IMG_DIR)
    img_paths = glob.glob(os.path.join(img_dir, "*.png"))
    
    patient_list = [item.split("/")[-1].split("_")[:-1] for item in img_paths]
    
    for idxs in chunker(range(len(img_paths)), 4):
        logger.info(f"Processing  {img_paths[idxs[0]]}")
        logger.info(f"Processing  {img_paths[idxs[1]]}")
        logger.info(f"Processing  {img_paths[idxs[2]]}")
        logger.info(f"Processing  {img_paths[idxs[3]]}")

        new_image, new_annos = mosaic(
            img_paths,
            annotation_dir,
            idxs,
            OUTPUT_SIZE,
            SCALE_RANGE,
            filter_scale=FILTER_TINY_SCALE,
        )

        file_name = img_paths[idxs[0]].split("/")[-1]
        cv2.imwrite(str(output_folder_img / file_name), new_image)  # The mosaic image
        logger.info(f"Image written to {str(output_folder_img / file_name)}")

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

        if yolo_anno:
            with open(
                str(output_folder_label / file_name.replace(".png", ".txt")), "w"
            ) as file:  # The output annotation file will appear in the output.txt file
                for line in yolo_anno:
                    file.write((" ").join([str(x) for x in line]) + "\n")


if __name__ == "__main__":
    app()
