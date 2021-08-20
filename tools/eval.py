# adpated from PaddleDetection/tools/eval.py
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
from data.datasets.lego_city import LeGOCityDataSet
from ppdet.data.reader import TrainReader, EvalReader

def parser_args():
    # parse with yaml loader to load config values
    # Note similar to GCC __attributes__, COCODataSet has already been registered with yaml loader
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-c", "--config", help="configuration file to use")
    parser.add_argument(
        "--output_eval",
        default="%s/tools/output" % settings.PROJECT_ROOT,
        type=str,
        help="Evaluation directory, default is current directory.")
    parser.add_argument(
        '--json_eval',
        action='store_true',
        default=False,
        help='Whether to re eval with already exists bbox.json or mask.json')
    # we don't use slim for the moment
    # parser.add_argument(
    #     "--slim_config",
    #     default=None,
    #     type=str,
    #     help="Configuration file of slim method.")
    parser.add_argument(
        "--bias",
        action="store_true",
        help="whether add bias or not while getting w and h")
    parser.add_argument(
        "--classwise",
        action="store_true",
        help="whether per-category AP and draw P-R Curve or not.")
    parser.add_argument(
        '--save_prediction_only',
        action='store_true',
        default=False,
        help='Whether to save the evaluation results only')
    parser.add_argument(
        "--opts", nargs='*', help="set configuration options")
    args = parser.parse_args()
    return args

def main():
    FLAGS = parser_args()
    # load dataset and configuration files
    cfg = load_config(os.path.join(settings.PROJECT_ROOT, FLAGS.config))

    cfg['bias'] = 1 if FLAGS.bias else 0
    cfg['classwise'] = True if FLAGS.classwise else False
    cfg['output_eval'] = FLAGS.output_eval
    cfg['save_prediction_only'] = FLAGS.save_prediction_only

    # config use_gpu, pretrain_weights key-values
    if FLAGS.opts is not None:
        for s in FLAGS.opts:
            s = s.strip()
            k, v = s.split('=', 1)
            cfg[k] = v

    check.check_config(cfg)
    check.check_gpu(cfg.use_gpu)
    check.check_version()

    if FLAGS.json_eval:
        logger.info(
            "In json_eval mode, PaddleDetection will evaluate json files in "
            "output_eval directly. And proposal.json, bbox.json and mask.json "
            "will be detected by default.")
        json_eval_results(
            cfg.metric,
            json_directory=FLAGS.output_eval,
            dataset=cfg['EvalDataset'])
        return

    # init parallel environment if nranks > 1
    init_parallel_env()

    # build dataset, data loader and trainer
    trainer = Trainer(cfg, mode='eval')

    # load trained weights
    trainer.load_weights(cfg.weights)

    # evaluating
    trainer.evaluate()
    return 0


if __name__ == "__main__":
    sys.exit(main())