import random

import cv2

from dataset import dataset
from mosaic import mosaic

OUTPUT_SIZE = (2000, 2000)  # Height, Width
SCALE_RANGE = (0.5, 0.5)
FILTER_TINY_SCALE = 1 / 50  # if height or width lower than this scale, drop it.

ANNO_DIR = '/media/amir/external_1TB/Dataset/Anevrism/output_folder/labels'
IMG_DIR = '/media/amir/external_1TB/Dataset/Anevrism/output_folder/images'

# Names of all the classes as they appear in the Pascal VOC Dataset
category_name = ['Anevrism']


# Change the output path of the imwrite commands to wherever you want to get both the mosaic image and the image with boxes
def main():
    img_paths, annos = dataset(ANNO_DIR, IMG_DIR)

    idxs = random.sample(range(len(annos)), 4)

    new_image, new_annos = mosaic(img_paths, annos,
    	                          idxs,
    	                          OUTPUT_SIZE, SCALE_RANGE,
    	                          filter_scale=FILTER_TINY_SCALE)

    cv2.imwrite('output.png', new_image) #The mosaic image
    for anno in new_annos:
        start_point = (int(anno[1] * OUTPUT_SIZE[1]), int(anno[2] * OUTPUT_SIZE[0]))
        end_point = (int(anno[3] * OUTPUT_SIZE[1]), int(anno[4] * OUTPUT_SIZE[0]))
        cv2.rectangle(new_image, start_point, end_point, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.imwrite('output_box.png', new_image) # The mosaic image with the bounding boxes
    
    yolo_anno = []
    
    for anno in new_annos:
      tmp = []
      tmp.append(anno[0])
      tmp.append((anno[3]+anno[1])/2)
      tmp.append((anno[4]+anno[2])/2)
      tmp.append(anno[3]-anno[1])
      tmp.append(anno[4]-anno[2])
      yolo_anno.append(tmp)

    with open('output.txt', 'w') as file: # The output annotation file will appear in the output.txt file
      for line in yolo_anno:
        file.write((' ').join([str(x) for x in line]) + '\n')   

if __name__ == '__main__':
    main()
