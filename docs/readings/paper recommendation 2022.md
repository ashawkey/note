# Paper Recommendation 2022



### Tips: 

* Check **[Project]/[Blog]** first for a brief understanding of the paper, and watch their introduction video if available.
* Only about 2 papers are chosen for each area. If you find a paper interesting, please use [[Connected Papers]](https://www.connectedpapers.com/) to find related works!



### Backbone

* [2015] **Deep Residual Learning for Image Recognition** [[Paper]](https://arxiv.org/abs/1512.03385)

  Most known paper and most used backbones in CV.

* [2020] **An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale** [[Paper]](https://arxiv.org/abs/2010.11929)

  Transformer is mainly used in NLP tasks, but vision transformer has been a recent trending.

  Further readings:

  * [2021] **CLIP: Connecting Text and Images** [[Blog]](https://openai.com/blog/clip/) [[Code]](https://github.com/openai/CLIP) [[Paper]](https://arxiv.org/abs/2103.00020)



### Perception (Classification, Segmentation, Detection)

#### 2D

* [2014] **Fully Convolutional Networks for Semantic Segmentation** [[Paper]](https://arxiv.org/abs/1411.4038)

* [2015] **You Only Look Once: Unified, Real-Time Object Detection** [[Paper]](https://arxiv.org/abs/1506.02640)

  Further readings:

  * YOLOv5 [[Code]](https://github.com/ultralytics/yolov5)

#### 3D

Different from 2D images, 3D data have many representations (voxel volumes, point clouds, meshes, and implicit functions). Since dense 3D convolutions are generally limited by GPU memory, an inevitable topic in 3D perception is to leverage the **sparsity** of 3D data.

* [2016] **PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation** [[Paper]](https://arxiv.org/abs/1612.00593)

  Point Cloud is an efficient 3D representation widely used in 3D tasks, and this paper first proposed an effective way of processing point clouds.

  Further readings:

  * **PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space** [[Paper]](https://arxiv.org/abs/1706.02413) [[Project]](http://stanford.edu/~rqi/pointnet2/)

* [2019] **Point-Voxel CNN for Efficient 3D Deep Learning** [[Paper]](https://arxiv.org/abs/1907.03739) 

  Sparse 3D convolution is another popular way for handling point clouds.

  Further readings:

  * [2020] **Searching Efficient 3D Architectures with Sparse Point-Voxel Convolution** [[Paper]](https://arxiv.org/abs/2007.16100) [[Code]](https://github.com/mit-han-lab/torchsparse)



### Generation

* [2014] **Generative Adversarial Networks** [[Paper]](https://arxiv.org/abs/1406.2661)

* [2018] **A Style-Based Generator Architecture for Generative Adversarial Networks** [[Paper]](https://arxiv.org/abs/1812.04948) [[code]](https://github.com/NVlabs/stylegan) 

  StyleGAN series are famous for its realistic image generation ability.

  Further readings:

  * [2021] **Alias-Free Generative Adversarial Networks (StyleGAN3)** [[Project]](https://nvlabs.github.io/stylegan3/) [[Paper]](https://arxiv.org/abs/2106.12423) [[Code]](https://github.com/NVlabs/stylegan3)

* [2021] **DALL-E: Zero-Shot Text-to-Image Generation** [[Blog]](https://openai.com/blog/dall-e/) [[Paper]](https://arxiv.org/abs/2102.12092) [[Online dalle-mini]](https://huggingface.co/spaces/dalle-mini/dalle-mini)

  Further readings: DALL-E 2, Imagen, Parti...



### 3D Reconstruction

* [2016] **COLMAP** [[Doc]](https://colmap.github.io/) [[Code]](https://github.com/colmap/colmap)

  A general-purpose Structure-from-Motion (SfM) and Multi-View Stereo (MVS) pipeline still widely used today.

* [2020] **NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis** [[Project]](https://www.matthewtancik.com/nerf) [[Paper]](http://arxiv.org/abs/2003.08934)

  Photo-realistic 3D scene novel view synthesis from only RGB video or images.

  Further reading:

  * [2022] **Instant Neural Graphics Primitives with a Multiresolution Hash Encoding** [[Project]](https://nvlabs.github.io/instant-ngp/) [[Code]](https://github.com/NVlabs/instant-ngp/) [[Paper]](https://nvlabs.github.io/instant-ngp/assets/mueller2022instant.pdf)

    Train a NeRF in 5 seconds.

  * [2022] **Efficient Geometry-aware 3D Generative Adversarial Networks** [[Project]](https://nvlabs.github.io/eg3d/) [[Code]](https://github.com/NVlabs/eg3d)

    High quality 3D-aware image synthesis, by combining StyleGAN and NeRF.

