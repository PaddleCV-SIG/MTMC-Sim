ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

# single gpu card inference
export CUDA_VISIBLE_DEVICES=0

TOOLKIT=$ROOT/tools
DATA=$ROOT/data

# use customer label set
python $TOOLKIT/infer.py \
  -c configs/ppyolo/ppyolo_r50vd_dcn_1x_coco.yml \
  --infer_img=$DATA/lego_labeling_samples/COCODataFmt/val/006.jpg \
  --opts weights=output/ppyolo_r50vd_dcn_1x_coco/143.pdparams \
  --use_vdl=Ture
