import os
import numpy as np
from ppdet.core.workspace import register, serializable
# replace with installed PPDet DetDataset
from ppdet.data.source.coco import COCODataSet
from data.dataset import ImagesDataset

from ppdet.utils.logger import setup_logger
logger = setup_logger(__name__)

# follow openMMDet path processing patterns
from pathlib import Path

# import Settings singleton
from mtmc.config import settings


@register
@serializable
class LeGOCityDataSet(ImagesDataset, COCODataSet):
    """
    Load dataset with COCO format.

    Args:
        dataset_dir (str): root directory for dataset.
        image_dir (str): directory for images.
        anno_path (str): coco annotation file path.
        data_fields (list): key name of data dictionary, at least have 'image'.
        sample_num (int): number of samples to load, -1 means all.
        load_crowd (bool): whether to load crowded ground-truth.
            False as default
        allow_empty (bool): whether to load empty entry. False as default
        empty_ratio (float): the ratio of empty record number to total
            record's, if empty_ratio is out of [0. ,1.), do not sample the
            records. 1. as default
    """

    def __init__(self,
                 name,
                 augumentor=None,
                 dataset_dir=None,
                 image_dir=None,
                 anno_path=None,
                 data_fields=['image', 'gt_bbox', 'gt_class', 'is_crowd'],
                 sample_num=-1,
                 load_crowd=False,
                 allow_empty=False,
                 empty_ratio=1.):
        # initialize non-cooperative base classes
        ImagesDataset.__init__(self, name, augumentor)
        COCODataSet.__init__(self, dataset_dir, image_dir, anno_path,
                             data_fields, sample_num, load_crowd, allow_empty, empty_ratio)

        # subset of labels used in this task
        self._class_names = {
            0: 'background',
            1: 'town_mini_figure', # 'town_men_mini_figure'
            2: 'space_mini_figure' # 'space_men_mini_figure'
        }

    # ===== paddle interface : @todo TODO
    def __getitem__(self, idx):
        return COCODataSet.__getitem__(self, idx)

    def __len__(self):
        return COCODataSet.__len__(self)

    def __iter__(self):
        return COCODataSet.__iter__(self)

    # ===== PaddleDetection data loader interface

    # original method use relative path to resovle dataset path which is wrong
    def check_or_download_dataset(self):
        root = Path(settings.PROJECT_ROOT).resolve()
        dataset_dir = Path(self.dataset_dir)
        dataset_dir = (root / dataset_dir).resolve()

        if dataset_dir.exists() and dataset_dir.is_dir():
            self.dataset_dir = str(dataset_dir)
        else:
            logger.warning("Config dataset_dir {} does not exits, "
                           "dataset config is not valid".format(dataset_dir))

            # @todo TODO donwload
            logger.info("trying to download dataset ...")
            raise NotImplemented("The donwload method is not implemented yet!")

        # dataset is ready

        # validation of path
        if self.anno_path:
            annotation_path = dataset_dir / Path(self.anno_path)
            if not annotation_path.exists():
                raise ValueError("Config annotation {} does not exit!".format(annotation_path))
            if not annotation_path.is_file():
                raise ValueError("Config annotation {} is not a file!".format(annotation_path))
            # self.anno_path = str(annotation_path)

        if self.image_dir:
            image_dir = dataset_dir / Path(self.image_dir)
            if not image_dir.exists():
                raise ValueError("Config image directory {} does not exit!".format(image_dir))
            if not image_dir.is_dir():
                raise ValueError("Config image directory {} is not a directory!".format(image_dir))
            # self.image_dir = str(image_dir)

    def parse_dataset(self):
        # COCODataSet.parse_dataset(self)
        anno_path = os.path.join(self.dataset_dir, self.anno_path)
        image_dir = os.path.join(self.dataset_dir, self.image_dir)

        assert anno_path.endswith('.json'), \
            'invalid coco annotation file: ' + anno_path
        from pycocotools.coco import COCO
        coco_api = COCO(anno_path)
        img_ids = coco_api.getImgIds()
        img_ids.sort()
        cat_ids = coco_api.getCatIds()
        records = []
        empty_records = []
        ct = 0

        self.catid2clsid = dict({catid: i for i, catid in enumerate(cat_ids)})
        self.cname2cid = dict({
            coco_api.loadCats(catid)[0]['name']: clsid
            for catid, clsid in self.catid2clsid.items()
        })

        if 'annotations' not in coco_api.dataset:
            self.load_image_only = True
            logger.warning('Annotation file: {} does not contains ground truth '
                           'and load image information only.'.format(anno_path))

        for img_id in img_ids:
            img_anno = coco_api.loadImgs([img_id])[0]
            im_fname = img_anno['file_name']
            im_w = float(img_anno['width'])
            im_h = float(img_anno['height'])

            im_path = os.path.join(image_dir,
                                   im_fname) if image_dir else im_fname
            is_empty = False
            if not os.path.exists(im_path):
                logger.warning('Illegal image file: {}, and it will be '
                               'ignored'.format(im_path))
                continue

            if im_w < 0 or im_h < 0:
                logger.warning('Illegal width: {} or height: {} in annotation, '
                               'and im_id: {} will be ignored'.format(
                                   im_w, im_h, img_id))
                continue

            coco_rec = {
                'im_file': im_path,
                'im_id': np.array([img_id]),
                'h': im_h,
                'w': im_w,
            } if 'image' in self.data_fields else {}

            if not self.load_image_only:
                ins_anno_ids = coco_api.getAnnIds(
                    imgIds=[img_id], iscrowd=None if not self.load_crowd else True) # quick fix #1 for coco dataset
                instances = coco_api.loadAnns(ins_anno_ids)

                bboxes = []
                is_rbox_anno = False
                for inst in instances:
                    # check gt bbox
                    if inst.get('ignore', False):
                        continue
                    if 'bbox' not in inst.keys():
                        continue
                    else:
                        if not any(np.array(inst['bbox'])):
                            continue

                    # read rbox anno or not
                    is_rbox_anno = True if len(inst['bbox']) == 5 else False
                    if is_rbox_anno:
                        xc, yc, box_w, box_h, angle = inst['bbox']
                        x1 = xc - box_w / 2.0
                        y1 = yc - box_h / 2.0
                        x2 = x1 + box_w
                        y2 = y1 + box_h
                    else:
                        x1, y1, box_w, box_h = inst['bbox']
                        x2 = x1 + box_w
                        y2 = y1 + box_h
                    eps = 1e-5
                    if inst['area'] > 0 or (x2 - x1 > eps and y2 - y1 > eps): # quick fix #2 for coco dataset
                        if inst['area'] < eps:
                            inst['area'] = (x2 - x1) * (y2 - y1)
                        inst['clean_bbox'] = [
                            round(float(x), 3) for x in [x1, y1, x2, y2]
                        ]
                        if is_rbox_anno:
                            inst['clean_rbox'] = [xc, yc, box_w, box_h, angle]
                        bboxes.append(inst)
                    else:
                        logger.warning(
                            'Found an invalid bbox in annotations: im_id: {}, '
                            'area: {} x1: {}, y1: {}, x2: {}, y2: {}.'.format(
                                img_id, float(inst['area']), x1, y1, x2, y2))

                num_bbox = len(bboxes)
                if num_bbox <= 0 and not self.allow_empty:
                    continue
                elif num_bbox <= 0:
                    is_empty = True

                gt_bbox = np.zeros((num_bbox, 4), dtype=np.float32)
                if is_rbox_anno:
                    gt_rbox = np.zeros((num_bbox, 5), dtype=np.float32)
                gt_theta = np.zeros((num_bbox, 1), dtype=np.int32)
                gt_class = np.zeros((num_bbox, 1), dtype=np.int32)
                is_crowd = np.zeros((num_bbox, 1), dtype=np.int32)
                difficult = np.zeros((num_bbox, 1), dtype=np.int32)
                gt_poly = [None] * num_bbox

                has_segmentation = False
                for i, box in enumerate(bboxes):
                    catid = box['category_id']
                    gt_class[i][0] = self.catid2clsid[catid]
                    gt_bbox[i, :] = box['clean_bbox']
                    # xc, yc, w, h, theta
                    if is_rbox_anno:
                        gt_rbox[i, :] = box['clean_rbox']
                    if box.get('iscrowd', None) is not None: # quick fix #3 for coco dataset
                        is_crowd[i][0] = box['iscrowd']
                    else:
                        is_crowd[i][0] = 0
                    # check RLE format
                    if 'segmentation' in box and 'iscrowd' in box and box['iscrowd'] == 1: # quick fix #3 for coco dataset
                        gt_poly[i] = [[0.0, 0.0], ]
                    elif 'segmentation' in box and box['segmentation']:
                        gt_poly[i] = box['segmentation']
                        has_segmentation = True

                if has_segmentation and not any(
                        gt_poly) and not self.allow_empty:
                    continue

                if is_rbox_anno:
                    gt_rec = {
                        'is_crowd': is_crowd,
                        'gt_class': gt_class,
                        'gt_bbox': gt_bbox,
                        'gt_rbox': gt_rbox,
                        'gt_poly': gt_poly,
                    }
                else:
                    gt_rec = {
                        'is_crowd': is_crowd,
                        'gt_class': gt_class,
                        'gt_bbox': gt_bbox,
                        'gt_poly': gt_poly,
                    }

                for k, v in gt_rec.items():
                    if k in self.data_fields:
                        coco_rec[k] = v

                # TODO: remove load_semantic
                if self.load_semantic and 'semantic' in self.data_fields:
                    seg_path = os.path.join(self.dataset_dir, 'stuffthingmaps',
                                            'train2017', im_fname[:-3] + 'png')
                    coco_rec.update({'semantic': seg_path})

            logger.debug('Load file: {}, im_id: {}, h: {}, w: {}.'.format(
                im_path, img_id, im_h, im_w))
            if is_empty:
                empty_records.append(coco_rec)
            else:
                records.append(coco_rec)
            ct += 1
            if self.sample_num > 0 and ct >= self.sample_num:
                break
        assert ct > 0, 'not found any coco record in %s' % (anno_path)
        logger.debug('{} samples in file {}'.format(ct, anno_path))
        if len(empty_records) > 0:
            empty_records = self._sample_empty(empty_records, len(records))
            records += empty_records
        self.roidbs = records