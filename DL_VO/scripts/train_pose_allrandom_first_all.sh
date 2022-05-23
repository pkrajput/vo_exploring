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
--name coordconv_allrandom_first_all \
--with-coord-conv 1 \
--conv1-weight-mode all_random \
--fine-tune-mode first_then_all \
--unfreeze-epoch 50 \
--use-scheduler 1 \
--run-id $1 \
--warmup-lr 1e-10 \
--warmup-epoch $(( 20 * $EPOCH_SIZE )) \
--step-size $(( 30 * $EPOCH_SIZE )) \
--gamma-lr .1 \
--min-lr 1e-7 \
--print-freq 1