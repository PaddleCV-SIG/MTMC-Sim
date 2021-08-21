ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

# single gpu card training
export CUDA_VISIBLE_DEVICES=0

TOOLKIT=$ROOT/tools
DATA=$ROOT/data

# train with pretrained coco model. Note this is model has different parameters and labels (up to 96) with this fine tuned model 
# (only 3, background: 0, space_mini_figure: 1, town_mini_figure: 2)
python $TOOLKIT/train.py \
  -c configs/ppyolo/ppyolo_r50vd_dcn_1x_coco.yml \
  --opts use_gpu=true \
         pretrain_weights=$DATA/ppyolo_r50vd_dcn_1x_coco.pdparams
