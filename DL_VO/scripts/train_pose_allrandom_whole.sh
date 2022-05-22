DATA_ROOT=../..
TRAIN_SET=$DATA_ROOT/kitti_vo_256

python3 ../train.py $TRAIN_SET \
--resnet-layers 50 \
--num-scales 1 \
-b4 -s0.1 -c0.5 --epoch-size 1000 --sequence-length 3 \
--with-ssim 1 \
--with-mask 1 \
--with-auto-mask 1 \
--with-pretrain 0 \
--pretrained-disp ../checkpoints_/dispnet_model_best.pth.tar \
--pretrained-pose ../checkpoints_/exp_pose_model_best.pth.tar \
--epochs 100 \
--lr 1e-3 \
--name coordconv_allrandom_whole \
--with-coord-conv 1 \
--conv1-weight-mode all_random \
--fine-tune-mode whole \
--unfreeze-epoch None \
--use-scheduler 1 \
--warmup-lr 1e-5 \
--warmup-epoch 70 \
--step-size 10 \
--gamma-lr .1 \
--min-lr 1e-5 \
--run-id $1 \