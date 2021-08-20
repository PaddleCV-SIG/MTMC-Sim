ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

# use gpu for inference
export CUDA_VISIBLE_DEVICES=0

TOOLKIT=$ROOT/tools
DATA=$ROOT/data

python -m pdb $TOOLKIT/eval.py \
  -c configs/ppyolo/ppyolo_r50vd_dcn_1x_coco.yml \
  --opts use_gpu=true \
         weights=$ROOT/output/ppyolo_r50vd_dcn_1x_coco/191.pdparams
