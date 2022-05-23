DATA_ROOT=../..
TRAIN_SET=$DATA_ROOT/kitti_vo_256

EPOCH_SIZE=50

python3 ../train.py $TRAIN_SET \
--resnet-layers 50 \
--num-scales 1 \
-b4 -s0.1 -c0.5 --epoch-size $EPOCH_SIZE --sequence-length 3 \
--with-ssim 1 \
--with-mask 1 \
--with-auto-mask 1 \
--with-pretrain 0 \
--pretrained-disp ../checkpoints_pretrained/dispnet_model_best.pth.tar \
--pretrained-pose ../checkpoints_pretrained/exp_pose_model_best.pth.tar \
--epochs 100 \
--lr 1e-5 \
--name coordconv_random_whole \
--with-coord-conv 1 \
--conv1-weight-mode random \
--fine-tune-mode whole \
--unfreeze-epoch None \
--use-scheduler 1 \
--run-id $1 \
--warmup-lr 1e-10 \
--warmup-epoch $(( 20 * $EPOCH_SIZE )) \
--step-size $(( 30 * $EPOCH_SIZE )) \
--gamma-lr .1 \
--min-lr 1e-10 \
--print-freq 1