# colmap



### Install in Ubuntu

To enable CUDA, you must build from source.

```bash
# dependency
sudo apt-get install \
    git \
    cmake \
    build-essential \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libboost-test-dev \
    libeigen3-dev \
    libsuitesparse-dev \
    libfreeimage-dev \
    libmetis-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev
    
# if in ubuntu 16/18:
sudo apt-get install libcgal-qt5-dev

# ceres
sudo apt-get install libatlas-base-dev libsuitesparse-dev
git clone https://ceres-solver.googlesource.com/ceres-solver
cd ceres-solver
# use <= 2.1, ref: https://github.com/colmap/colmap/issues/1482
git checkout f68321e7de8929fbcdb95dd42877531e64f72f66 
mkdir build
cd build
cmake .. -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF
make -j
sudo make install

# if use anaconda, you may meet undefined refernce to libtiff error
# ref: https://github.com/colmap/colmap/issues/188
conda deactivate

# colmap
git clone https://github.com/colmap/colmap.git
cd colmap
mkdir build
cd build
cmake ..
make -j
sudo make install
```



### CLI

```bash
# help
colmap -h
colmap <subcommand> -h

# path/to/project:
# +-- images
#     +-- 0.jpg
#     +-- 1.jpg
#     +-- ...

colmap automatic_reconstructor \
	--workspace_path /path/to/project \
	--image_path /path/to/project/images \
```

If you want more controls:

```bash
# step to step way:
DATASET_PATH=/path/to/dataset

colmap feature_extractor \
   --database_path $DATASET_PATH/database.db \
   --image_path $DATASET_PATH/images

colmap exhaustive_matcher \
   --database_path $DATASET_PATH/database.db

mkdir $DATASET_PATH/sparse

colmap mapper \
    --database_path $DATASET_PATH/database.db \
    --image_path $DATASET_PATH/images \
    --output_path $DATASET_PATH/sparse

mkdir $DATASET_PATH/dense

colmap image_undistorter \
    --image_path $DATASET_PATH/images \
    --input_path $DATASET_PATH/sparse/0 \
    --output_path $DATASET_PATH/dense \
    --output_type COLMAP \
    --max_image_size 2000

colmap patch_match_stereo \
    --workspace_path $DATASET_PATH/dense \
    --workspace_format COLMAP \
    --PatchMatchStereo.geom_consistency true

colmap stereo_fusion \
    --workspace_path $DATASET_PATH/dense \
    --workspace_format COLMAP \
    --input_type geometric \
    --output_path $DATASET_PATH/dense/fused.ply

colmap poisson_mesher \
    --input_path $DATASET_PATH/dense/fused.ply \
    --output_path $DATASET_PATH/dense/meshed-poisson.ply

colmap delaunay_mesher \
    --input_path $DATASET_PATH/dense \
    --output_path $DATASET_PATH/dense/meshed-delaunay.ply
```

To force use CPU:

```
colmap ...
  --SiftExtraction.use_gpu 0 \
  --SiftMatching.use_gpu 0
```





### GUI

```bash
# gui
colmap gui
```

Then just follow the GUI.



### Troubleshooting

* `No good initial image pair found.`

  This means the dataset is too bad. Not all images are registered in this case. 

  Solutions:

  * remove the un-registered images.
  * use better dataset with more images (better poses).

  

