# vo_exploring
Paper Link: https://drive.google.com/file/d/106B8Qr4Zbf20hP2hwfwPAP-EPJnWlCYG/view?usp=share_link

Deep Learning 2022 course final project.

Vladislav Trifonov, Hekmat Taherinejad, Prateek Rajput, Vladimir Chernyy, Anna Akhmatova, Timur Bayburin

# ORB_SLAM2
Build ORB-SLAM2 with patches

```
git clone https://github.com/scalyvladimir/vo_exploring.git
cd vo_exploring
git submodule update --init --recursive
cd ORB_SLAM2
./build.sh
```

## Monocular Examples
### TUM Dataset

1. Download sequence or preprocessed data from [Google Drive](https://drive.google.com/drive/folders/1QeJDDLN49xqxn-m0-DE3OGkfm-1Q7DKE?usp=sharing) or the original data from [original dataset](http://vision.in.tum.de/data/datasets/rgbd-dataset/download) and uncompress it.
2. Execute the following command. Change ```TUMX.yaml``` to TUM1.yaml,TUM2.yaml or TUM3.yaml for freiburg1, freiburg2 and freiburg3 sequences respectively. Change ```PATH_TO_SEQUENCE_FOLDER``` to the uncompressed sequence folder.

``` ./Examples/Monocular/mono_tum Vocabulary/ORBvoc.txt Examples/Monocular/TUMX.yaml PATH_TO_SEQUENCE_FOLDER ```

### KITTI Dataset

1. Download the dataset with preprocessed data (sequences 08, 09, 10) from [Google Drive](https://drive.google.com/drive/folders/1QeJDDLN49xqxn-m0-DE3OGkfm-1Q7DKE?usp=sharing) or [full original dataset](http://www.cvlibs.net/datasets/kitti/eval_odometry.php)
2. Execute the following command. Change ``` KITTIX.yaml``` by KITTI00-02.yaml, KITTI03.yaml or KITTI04-12.yaml for sequence 0 to 2, 3, and 4 to 12 respectively. Change ``` PATH_TO_DATASET_FOLDER``` to the uncompressed dataset folder. Change ```SEQUENCE_NUMBER``` to 00, 01, 02,.., 11.

``` ./Examples/Monocular/mono_kitti Vocabulary/ORBvoc.txt Examples/Monocular/KITTIX.yaml PATH_TO_DATASET_FOLDER/dataset/sequences/SEQUENCE_NUMBER ```


# PoseNet for Visual Odometry

Our codebase for visual odemetry has following structure:

1. Code base

  - [training directory](DL_VO/scripts) contains all scripts that we used for fine-tuning our PoseNet with CoordConv.
  - [evaluation directory](DL_VO/evaluation) contains all script which test our checkpoints against ground truth poses, saved in [corresponding directory](DL_VO/evaluation/gt_poses).

2. Execution
  - Simply upload [this notebook](DL_VO/run_inference.ipynb) to the Google colab and follow steps provided there.

# SRN-Deblur

Download pretrained models through: ```download_model.sh``` inside ```checkpoints/```.

For running the deblurring process, replace ```<PATH_TO_DATASET>``` with the directory of input rgb sequence folder and replace ```<PATH_TO_OUTPUT_FOLDER>``` with the directory that you want save the deblurred images. Also you can define the gpu id using ```--gpu=``` argument. Otherwise for using CPU, you can set the gpu id to ```--gpu=-1```.
```
cd vo_exploring/SRN-Deblur
python run_model.py --input_path=<PATH_TO_DATASET> --output_path=<PATH_TO_OUTPUT_FOLDER> --gpu=<GPU_ID>
```

# Metrics

Based on [evo - Python package for the evaluation of odometry and SLAM](https://github.com/MichaelGrupp/evo).

### Installation

Make sure you have `evo` folder from repository, it contains supportive scripts. Also install evo using pip:
```
pip3 install evo --upgrade --no-binary evo
```

### Run

You can run metrics using instructions from original repository, sequences are stored in `sequences` folder.

