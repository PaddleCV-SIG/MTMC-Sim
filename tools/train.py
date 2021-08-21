# adpated from PaddleDetection/tools/train.py
import os
import sys

from mtmc.lib.sys import add_path
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

# setup commandline toolkit
import argparse
import ppdet.utils.check as check
from ppdet.utils.logger import setup_logger
logger = setup_logger('mtmc:trainer')

# load global config
from mtmc.config import settings

# load dataset
from data.datasets.lego_city import LeGOCityDataSet

def parser_args():
    # parse with yaml loader to load config values
    # Note similar to GCC __attributes__, COCODataSet has already been registered with yaml loader
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-c", "--config", help="configuration file to use")
    parser.add_argument(
        "--eval",
        action='store_true',
        default=False,
        help="Whether to perform evaluation in train")
    parser.add_argument(
        "-r", "--resume", default=None, help="weights path for resume")
    # we don't use slim for the moment
    # parser.add_argument(
    #     "--slim_config",
    #     default=None,
    #     type=str,
    #     help="Configuration file of slim method.")
    parser.add_argument(
        "--enable_ce",
        type=bool,
        default=False,
        help="If set True, enable continuous evaluation job."
        "This flag is only used for internal test.")
    parser.add_argument(
        "--fp16",
        action='store_true',
        default=False,
        help="Enable mixed precision training.")
    # we don't use fleet for the moment
    # parser.add_argument(
    #     "--fleet", action='store_true', default=False, help="Use fleet or not")
    parser.add_argument(
        "--use_vdl",
        type=bool,
        default=False,
        help="whether to record the data to VisualDL.")
    parser.add_argument(
        '--vdl_log_dir',
        type=str,
        default="vdl_log_dir/scalar",
        help='VisualDL logging directory for scalar.')
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

    cfg['fp16'] = FLAGS.fp16
    cfg['use_vdl'] = FLAGS.use_vdl
    cfg['vdl_log_dir'] = FLAGS.vdl_log_dir

    # config use_gpu, pretrain_weights key-values
    if FLAGS.opts is not None:
        for s in FLAGS.opts:
            s = s.strip()
            k, v = s.split('=', 1)
            cfg[k] = v

    paddle.set_device('gpu' if cfg.use_gpu else 'cpu')

    check.check_config(cfg)
    check.check_gpu(cfg.use_gpu)
    check.check_version()

    # init parallel environment if nranks > 1
    init_parallel_env()

    if FLAGS.enable_ce:
        set_random_seed(0)

    # build dataset, data loader and trainer
    trainer = Trainer(cfg, mode='train')

    # load weights
    if FLAGS.resume is not None:
        return trainer.resume_weights(FLAGS.resume)
    elif 'pretrain_weights' in cfg and cfg.pretrain_weights:
        trainer.load_weights(cfg.pretrain_weights)

    # trainning
    trainer.train(FLAGS.eval)
    return 0


if __name__ == "__main__":
    sys.exit(main())
