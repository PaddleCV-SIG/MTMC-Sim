# adpated from PaddleDetection/tools/infer.py
import os
import sys

try:
    from mtmc.lib.sys import add_path
except:
    def add_path(path):
        if path not in sys.path:
            sys.path.append(path)
# add python path of PadleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))

add_path(parent_path)

# ignore warning log
import warnings
warnings.filterwarnings('ignore')
import glob

# load engine
import paddle
# alternatively you can rebuild opencv-python using provided script "scripts/rebuild_opencv_python.sh"
# to avoid errors introduced in Clion
for k, v in os.environ.items():
        if k.startswith("QT_") and "cv2" in v:
            del os.environ[k]

# load config helper
from ppdet.core.workspace import load_config, merge_config
from ppdet.engine import Trainer, init_parallel_env, set_random_seed, init_fleet_env
from ppdet.metrics.coco_utils import json_eval_results

# setup commandline toolkit
import argparse
import ppdet.utils.check as check
from ppdet.utils.logger import setup_logger
logger = setup_logger('mtmc:evaluator')

# load global config
from mtmc.config import settings

# load dataset
from ppdet.data.source.coco import COCODataSet
from ppdet.data.source.dataset import ImageFolder
from data.datasets.lego_city import LeGOCityDataSet
from ppdet.data.reader import TrainReader, EvalReader, TestReader

def parse_args():
    # parse with yaml loader to load config values
    # Note similar to GCC __attributes__, COCODataSet has already been registered with yaml loader
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-c", "--config", help="configuration file to use")
    parser.add_argument(
        "--infer_dir",
        type=str,
        default=None,
        help="Directory for images to perform inference on.")
    parser.add_argument(
        "--infer_img",
        type=str,
        default=None,
        help="Image path, has higher priority over --infer_dir")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory for storing the output visualization files.")
    parser.add_argument(
        "--draw_threshold",
        type=float,
        default=0.09,
        help="Threshold to reserve the result for visualization.")
    parser.add_argument(
        "--slim_config",
        default=None,
        type=str,
        help="Configuration file of slim method.")
    parser.add_argument(
        "--use_vdl",
        type=bool,
        default=True,
        help="Whether to record the data to VisualDL.")
    parser.add_argument(
        '--vdl_log_dir',
        type=str,
        default="vdl_log_dir/image",
        help='VisualDL logging directory for image.')
    parser.add_argument(
        "--save_txt",
        type=bool,
        default=True,
        help="Whether to save inference result in txt.")
    parser.add_argument(
        "--opts", nargs='*', help="set configuration options")
    args = parser.parse_args()
    return args


def get_test_images(infer_dir, infer_img):
    """
    Get image path list in TEST mode
    """
    assert infer_img is not None or infer_dir is not None, \
        "--infer_img or --infer_dir should be set"
    assert infer_img is None or os.path.isfile(infer_img), \
            "{} is not a file".format(infer_img)
    assert infer_dir is None or os.path.isdir(infer_dir), \
            "{} is not a directory".format(infer_dir)

    # infer_img has a higher priority
    if infer_img and os.path.isfile(infer_img):
        return [infer_img]

    images = set()
    infer_dir = os.path.abspath(infer_dir)
    assert os.path.isdir(infer_dir), \
        "infer_dir {} is not a directory".format(infer_dir)
    exts = ['jpg', 'jpeg', 'png', 'bmp']
    exts += [ext.upper() for ext in exts]
    for ext in exts:
        images.update(glob.glob('{}/*.{}'.format(infer_dir, ext)))
    images = list(images)

    assert len(images) > 0, "no image found in {}".format(infer_dir)
    logger.info("Found {} inference images in total.".format(len(images)))

    return images

def main():
    FLAGS = parse_args()
    cfg = load_config(os.path.join(settings.PROJECT_ROOT, FLAGS.config))

    cfg['use_vdl'] = FLAGS.use_vdl
    cfg['vdl_log_dir'] = FLAGS.vdl_log_dir

    # config use_gpu, pretrain_weights key-values
    if FLAGS.opts is not None:
        opt_config = {}
        for s in FLAGS.opts:
            s = s.strip()
            k, v = s.split('=', 1)
            cfg[k] = v

    check.check_config(cfg)
    check.check_gpu(cfg.use_gpu)
    check.check_version()

    # build dataset, data loader and trainer
    trainer = Trainer(cfg, mode='test')

    # load trained weights
    trainer.load_weights(cfg.weights)

    # get inference images
    images = get_test_images(FLAGS.infer_dir, FLAGS.infer_img)

    # inference
    trainer.predict(
        images,
        draw_threshold=FLAGS.draw_threshold,
        output_dir=FLAGS.output_dir,
        save_txt=FLAGS.save_txt
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())