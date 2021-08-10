import os
import sys

from mtmc.lib.sys import add_path
# add python path of PadleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))

add_path(parent_path)

import paddle

from ppdet.core.workspace import load_config, merge_config
from ppdet.engine import Trainer, init_parallel_env, set_random_seed, init_fleet_env

import ppdet.utils.cli as cli
import ppdet.utils.check as check
from ppdet.utils.logger import setup_logger
logger = setup_logger('mtmc:trainer')

from mtmc.config import settings

def parser_args():
    parser = cli.ArgsParser()
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
    args = parser.parse_args()
    return args

def main():
    FLAGS = parser_args()
    # load dataset and configuration files
    cfg = load_config(FLAGS.config)

    return 0

if __name__ == "__main__":
    sys.exit(main())
