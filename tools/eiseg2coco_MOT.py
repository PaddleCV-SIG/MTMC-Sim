import argparse
import glob
import json
import os
import shutil
# follow openMMDet path processing patterns
from pathlib import Path
import numpy as np
import PIL.ImageDraw

from sklearn.model_selection import train_test_split

from mtmc.config import settings

useQtBottomLeft = True  # should be Qt TopLeft

"""
Add MOT configuration, with following fields:
<carema_id> <frame_id><conf><ignore> <visibility/ratio>
By now the given data has no occlusion issues. And all labeling are visible.
Considering that we may need inference result of detection, set conf=1 to match inference
set ignore = 0 and visibility_ratio = 1 to use all annotations
carema_id and frame_id are extracted automatically from folder and filename

In MTMC we may need to consider how multiple caremas associate together.
So we split dataset after merging all annotations and image.
This requires a refactor of the current annotation structures.
"""

class MyEncoder(json.JSONEncoder):
    def default(self, val):
        if isinstance(val, np.integer):
            return int(val)
        elif isinstance(val, np.floating):
            return float(val)
        elif isinstance(val, np.ndarray):
            return val.tolist()
        else:
            return super(MyEncoder, self).default(val)


def merge_coco_annotation(coco_annotation_paths):
    data_coco = {
        "images": [],
        "annotations": []
    }
    coco_image_id = 0
    imgs_info_indexed_by_img_name = {}
    imgs_info_indexed_by_img_idx = {}

    # print('Generating dataset from:', coco_annotation_path)
    for coco_annotation_path in coco_annotation_paths:
        with open(coco_annotation_path) as f:
            data = json.load(f)
            # fetch categroies, info and licenses fields
            # data_coco["categories"] = data["categories"]
            # data_coco["info"] = data["info"]
            # data_coco["licenses"] = data["licenses"]
            # fetch images and annotations
        camera_id = coco_annotation_path.split("/")[-3]
        for idx, img_info in enumerate(data['images']):
            # For image, make sure you update id and filename
            img_name = img_info["file_name"]
            img_info["old_id"] = img_info["id"]
            img_name = camera_id + "_" + img_name
            img_info["file_name"] = img_name
            img_info["id"] = coco_image_id
            if imgs_info_indexed_by_img_name.get(img_name, None) is None:
                imgs_info_indexed_by_img_name[img_name] = {}
            imgs_info_indexed_by_img_name[img_name]["img_info"] = img_info
            imgs_info_indexed_by_img_idx[img_info["id"]] = img_info
            for record in data["annotations"]:
                if "img_id_update" not in record and record["image_id"] == img_info["old_id"]:
                    record["image_id"] = img_info["id"]
                    record["img_id_update"] = True
            coco_image_id += 1

        for annotation in data['annotations']:
            # if you update image_id, then reflect in annotations
            # update image_id and file_name and camera_id
            # rest is fine. Do whatever you want
            # need post visualization on this
            img_idx = annotation["image_id"]
            img_info = imgs_info_indexed_by_img_idx[img_idx]
            img_name = img_info["file_name"]
            print("img:",img_name)
            # annotation polygon
            epsilon = 1e-5
            if 'iscrowd' not in annotation:
                annotation['iscrowd'] = 0
            if annotation['area'] < epsilon:
                annotation['area'] = annotation['bbox'][2] * annotation['bbox'][3]
            if useQtBottomLeft:
                annotation['bbox'][1] = annotation['bbox'][1] - annotation['bbox'][3]

            annotation["carema_id"] = camera_id
            annotation["frame_id"] = img_name.split("_")[-1].split(".")[0]
            annotation["conf"] = 1
            annotation["ignore"] = 0
            annotation["visibility_ratio"] = 1
            imgs_info_indexed_by_img_name[img_name]["img_annotation"] = annotation
    return imgs_info_indexed_by_img_name

def save_coco_annotations(dataset, imgs_info_indexed_by_img_name):
    data_coco = {
        "images": [],
        "annotations": []
    }
    for img_path in dataset:
        img_file = Path(img_path)
        img_name = img_path.split("/")[-3] + "_" + img_file.name
        img_info_with_annotation = imgs_info_indexed_by_img_name[img_name]
        print(img_name)
        data_coco["images"].append(img_info_with_annotation["img_info"])
        data_coco["annotations"].append(img_info_with_annotation["img_annotation"])
    return data_coco


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset_type', default='eiseg', help='type of dataset is supported is EISeg')
    parser.add_argument('--json_input_dir', default='%s/data/%s' % (settings.PROJECT_ROOT, settings.TEST_DATASET),
                        help='input annotation directory')
    parser.add_argument('--image_input_dir', default='%s/data/%s' % (settings.PROJECT_ROOT, settings.TEST_DATASET),
                        help='image directory')
    parser.add_argument('--output_dir',
                        default='%s/data/MOT_labeling' % (settings.PROJECT_ROOT),
                        help='output dataset directory')

    args = parser.parse_args()
    try:
        assert args.dataset_type in ['coco', 'eiseg', 'labelme']
    except AssertionError as e:
        print('Only coco and EISeg dataset is supported for the moment!')
        return -1

    coco_annotation_dir = args.json_input_dir
    camera_available = []
    for root, dirs, files in os.walk(coco_annotation_dir, topdown=True):
        camera_available.append(dirs)
        break
    camera_available = camera_available[0]


    assert (settings.TRAIN_PROPORTION > 0 and settings.TRAIN_PROPORTION < 1)
    assert (settings.TRAIN_PROPORTION + settings.VALIDATION_PROPORTION == 1)

    setattr(settings, 'coco_dataset_path', args.output_dir)
    train_path = os.path.join(settings.coco_dataset_path, "train")
    val_path = os.path.join(settings.coco_dataset_path, "val")
    annotation_path = os.path.join(settings.coco_dataset_path, "annotations")
    if not os.path.exists(settings.coco_dataset_path):
        os.makedirs(train_path)
        os.makedirs(val_path)
        os.makedirs(annotation_path)

    setattr(settings, 'DATA_DIR', args.image_input_dir)
    comb_images = []
    for camera in camera_available:
        raw_images = list(glob.iglob(os.path.join(settings.DATA_DIR, camera, "img", "*.jpg")))
        raw_images = sorted(raw_images, key=lambda x: int(os.path.split(x)[1].split('.')[0]))
        comb_images += raw_images

    train_dataset, val_dataset = train_test_split(comb_images, train_size=settings.TRAIN_PROPORTION)
    # Disk task : move images and annotation files to coco files hierarchy
    # TODO: new_name: add carema id to current filename update image_id and filename accordingly
    for img_path in train_dataset:
        img_name = Path(img_path).name  # similar to boot api
        img_name_to = img_path.split("/")[-3] + "_" + img_name
        shutil.copyfile(
            img_path,
            os.path.join(train_path, img_name_to)
        )

    for img_path in val_dataset:
        img_name = Path(img_path).name  # similar to boot api
        img_name_to = img_path.split("/")[-3] + "_" + img_name
        shutil.copyfile(
            img_path,
            os.path.join(val_path, img_name_to)
        )

    coco_annotation_files = []
    for camera in camera_available:
        # carema -> carema_0001
        coco_annotation_file = os.path.join(coco_annotation_dir, camera, "gt", "coco.json")
        coco_annotation_files.append(coco_annotation_file)

    dataset = merge_coco_annotation(coco_annotation_files)
    train_coco_col = save_coco_annotations(train_dataset, dataset)
    val_coco_col = save_coco_annotations(val_dataset, dataset)

    train_json_path = os.path.join(annotation_path, "train.json")
    json.dump(
        train_coco_col,
        open(train_json_path, 'w'),
        indent=4,
        cls=MyEncoder
    )

    val_json_path = os.path.join(annotation_path, "val.json")
    json.dump(
        val_coco_col,
        open(val_json_path, 'w'),
        indent=4,
        cls=MyEncoder
    )
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main())
