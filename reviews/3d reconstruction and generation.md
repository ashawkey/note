# Image based 3D Reconstruction and Generation



## Object Level

> Category:
>
> * Inference:
>
>   * Single-view RGB
>   * Multiple-view RGB
> * Train in addition:
>
>   * 3D GT
>   * semantic / instance segmentation
>   * camera pose 
>
> * Output Representation: 
>
>   * Voxels
>   * Point Cloud
>   * Implicit Function
>   * Mesh



#### [ECCV 2016] 3D-R2N2: 3D Recurrent Reconstruction Neural Network

```
@inproceedings{Choy20163DR2N2AU,
  title={3D-R2N2: A Unified Approach for Single and Multi-view 3D Object Reconstruction},
  author={C. Choy and Danfei Xu and JunYoung Gwak and Kevin Chen and S. Savarese},
  booktitle={ECCV},
  year={2016}
}
```

[paper](http://arxiv.org/abs/1604.00449) | [code](https://github.com/chrischoy/3D-R2N2)

##### Problem

* Input: RGB Image (Single or Multiple View), Instance Segmentation + 3D GT
* Output: Voxels

##### Contributions

* LSTM framework for single or multiple view 3d reconstruction.

* Minimal supervision, no segmentation, no camera pose, no class label.

![image-20210401215040718](3d reconstruction and generation.assets/image-20210401215040718.png)



#### [3DV 2017] Hierarchical Surface Prediction for 3D Object Reconstruction

[paper](https://arxiv.org/abs/1704.00710)

##### Problem

* Input: Single-view RGB Image / Depth Image / Partial Volume + 3D GT
* Output: Voxels

![image-20210401215558646](3d reconstruction and generation.assets/image-20210401215558646.png)



#### [ICCVW 2017] 3D Object Reconstruction from a Single Depth View with Adversarial Learning

[paper](https://arxiv.org/abs/1708.07969)



#### [ICCV 2017] Octree Generating Networks: Efficient Convolutional Architectures for High-resolution 3D Outputs

[paper](https://arxiv.org/abs/1703.09438) | [code](https://github.com/lmb-freiburg/ogn)



#### [NIPS 2017] MarrNet: 3D Shape Reconstruction via 2.5D Sketches

[paper](https://arxiv.org/abs/1711.03129) 

![image-20210401220252010](3d reconstruction and generation.assets/image-20210401220252010.png)



#### [CVPR 2017] A point set generation network for 3d object reconstruction from a single image.

[paper](https://arxiv.org/pdf/1612.00603.pdf)

##### Problem

* Input: Single RGB Image, Instance Segmentation, Camera Pose + 3D GT
* Output: Point Cloud



#### [ECCV 2018] Pixel2Mesh: Generating 3D Mesh Models from Single RGB Images

[paper](https://arxiv.org/pdf/1804.01654.pdf) | [code](https://github.com/nywang16/Pixel2Mesh)

##### Problem

* Input: Single RGB Image, Camera Intrinsics + 3D GT
* Output: Mesh

![image-20210401222806763](3d reconstruction and generation.assets/image-20210401222806763.png)



#### [ECCV 2018] Learning Category-Specific Mesh Reconstruction from Image Collections. 

```
@inproceedings{Kanazawa2018LearningCM,
  title={Learning Category-Specific Mesh Reconstruction from Image Collections},
  author={A. Kanazawa and Shubham Tulsiani and Alexei A. Efros and Jitendra Malik},
  booktitle={ECCV},
  year={2018}
}
```

[paper](https://arxiv.org/abs/1803.07549) | [code](https://github.com/akanazawa/cmr) 

##### Problem

* Input: Single RGB Image. Instance Segmentation, Semantic Key-points
* Output: Camera Pose, Mesh, Texture.

##### Contribution

* Deformable Mesh Representation (need categorical mean shape as mesh template)
* Only need Single-view RGB (w/o Multi-view or 3D GT)
* Able to infer texture.

![image-20210401212228981](3d reconstruction and generation.assets/image-20210401212228981.png)



#### [NIPS 2019] DISN: Deep Implicit Surface Network for High-quality Single-view 3D Reconstruction

[paper](https://arxiv.org/pdf/1905.10711.pdf) | [code](https://github.com/laughtervv/DISN)

##### Problem

* Input: Single RGB Image + 3D GT
* Output: Implicit Function



#### [CVPR 2020] From Image Collections to Point Clouds with Self-supervised Shape and Pose Networks

```
@inproceedings{navaneet2020ssl3drecon,
 author = {Navaneet, K L and Mathew, Ansu and Kashyap, Shashank and Hung, Wei-Chih and Jampani, Varun and Babu, R Venkatesh},
 booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
 title = {From Image Collections to Point Clouds with Self-supervised Shape and Pose Networks},
 year = {2020}
}
```

[paper](https://arxiv.org/pdf/2005.01939.pdf) | [code](https://github.com/klnavaneet/ssl_3d_recon)

##### Problem

* Input: Single RGB Image
* Output: Point Cloud

![image-20210401221826288](3d reconstruction and generation.assets/image-20210401221826288.png)

#### 



#### [NIPS 2020] Convolutional Generation of Textured 3D Meshes

```
@article{Pavllo2020ConvolutionalGO,
  title={Convolutional Generation of Textured 3D Meshes},
  author={Dario Pavllo and Graham Spinks and T. Hofmann and Marie-Francine Moens and Aur{\'e}lien Lucchi},
  journal={ArXiv},
  year={2020},
  volume={abs/2006.07660}
}
```

[paper](https://arxiv.org/abs/2006.07660) | [code](https://github.com/dariopavllo/convmesh)

##### Problem

* Input: Single-view RGB Image, class label / caption (for GAN text conditioning), Instance Segmentation, Camera Pose.
* Output: Mesh, Texture

##### Contribution

* Displacement Map for a smooth convolutional mesh representation
* GAN for producing textured mesh.
* Conditional Generation of mesh from text.

![image-20210401213253243](3d reconstruction and generation.assets/image-20210401213253243.png)

![image-20210401213317770](3d reconstruction and generation.assets/image-20210401213317770.png)





#### [Arxiv 2021] Learning Generative Models of Textured 3D Meshes from Real-World Images

```
@inproceedings{Pavllo2021LearningGM,
  title={Learning Generative Models of Textured 3D Meshes from Real-World Images},
  author={Dario Pavllo and J. Kohler and T. Hofmann and Aur{\'e}lien Lucchi},
  year={2021}
}
```

[paper](https://arxiv.org/pdf/2103.15627.pdf)

##### Problem

* Input: 
* Output: 

##### Contributions





#### []

``

[paper]()

##### Problem

* Input: 
* Output: 

##### Contributions





## Scene Level



