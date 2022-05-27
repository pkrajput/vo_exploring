# vo_exploring

# ORB_SLAM2
Build ORB-SLAM2 with patches

```
git clone https://github.com/scalyvladimir/vo_exploring.git
git submodule update --init --recursive
cd ORB_SLAM2
./build.sh

```

## Monocular Examples
### TUM Dataset

1. Download a sequence from http://vision.in.tum.de/data/datasets/rgbd-dataset/download and uncompress it.
2. Execute the following command. Change ```TUMX.yaml``` to TUM1.yaml,TUM2.yaml or TUM3.yaml for freiburg1, freiburg2 and freiburg3 sequences respectively. Change ```PATH_TO_SEQUENCE_FOLDER``` to the uncompressed sequence folder.

``` ./Examples/Monocular/mono_tum Vocabulary/ORBvoc.txt Examples/Monocular/TUMX.yaml PATH_TO_SEQUENCE_FOLDER ```

### KITTI Dataset

1. Download the dataset from http://www.cvlibs.net/datasets/kitti/eval_odometry.php
2. Execute the following command. Change ``` KITTIX.yaml``` by KITTI00-02.yaml, KITTI03.yaml or KITTI04-12.yaml for sequence 0 to 2, 3, and 4 to 12 respectively. Change ``` PATH_TO_DATASET_FOLDER``` to the uncompressed dataset folder. Change ```SEQUENCE_NUMBER``` to 00, 01, 02,.., 11.

``` ./Examples/Monocular/mono_kitti Vocabulary/ORBvoc.txt Examples/Monocular/KITTIX.yaml PATH_TO_DATASET_FOLDER/dataset/sequences/SEQUENCE_NUMBER ```
