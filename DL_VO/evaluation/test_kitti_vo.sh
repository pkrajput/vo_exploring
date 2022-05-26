DATASET_DIR=../../kitti_odom_test/sequences/
OUTPUT_DIR=vo_results/

if [ -d $OUTPUT_DIR ] 
then
    rm -rf $OUTPUT_DIR
else
    mkdir $OUTPUT_DIR
fi

POSE_NET=../checkpoints_/coordconv_allrandom_whole/05-23-20:03/exp_pose_model_best.pth.tar

python3 test_vo.py \
--img-height 256 --img-width 832 \
--sequence 09 \
--pretrained-posenet $POSE_NET --dataset-dir $DATASET_DIR --output-dir $OUTPUT_DIR

python3 test_vo.py \
--img-height 256 --img-width 832 \
--sequence 10 \
--pretrained-posenet $POSE_NET --dataset-dir $DATASET_DIR --output-dir $OUTPUT_DIR

python3 eval_odom.py --result=$OUTPUT_DIR --align='7dof'