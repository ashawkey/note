# Single View 3D Reconstruction and Generation



## Object Level

> Category:
>
> * Inference:
>
>   * Single-view RGB
> * Train in addition:
> 
>  * 3D GT
>   * Multiple-view RGB
>   * semantic / instance segmentation (can be generated from pretrained Mask R-CNN)
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



#### 
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



#### [ECCV 2014] OpenDR: An approximate differentiable renderer.
#### [CVPR 2018] Neural 3D mesh renderer.

#### [ICCV 2019] A differentiable renderer for image-based 3d reasoning.

#### [NIPS 2019] Learning to predict 3d objects with an interpolation-based differentiable renderer

[paper](https://arxiv.org/pdf/1908.01210.pdf) | [code](https://github.com/nv-tlabs/DIB-R)

##### Problem

* Input: Mesh, Texture, Camera Pose
* Output: rendered Image

##### Contribution

* view foreground rasterization as a weighted interpolation of local properties and background rasterization as a distance-based aggregation of global geometry

![image-20210402101531055](3d reconstruction and generation.assets/image-20210402101531055.png)



#### [NIPS 2019] DISN: Deep Implicit Surface Network for High-quality Single-view 3D Reconstruction

[paper](https://arxiv.org/pdf/1905.10711.pdf) | [code](https://github.com/laughtervv/DISN)

##### Problem

* Input: Single RGB Image + 3D GT
* Output: Implicit Function



#### [ECCV 2020] Shape and viewpoint without keypoints.

[paper](https://arxiv.org/abs/2007.10982) | [code](https://github.com/shubham-goel/ucmr)

##### Problem

* Input: Single RGB Image, Instance Segmentation, categorical mesh templates
* Output: Camera Pose, Mesh, Texture

##### Contribution

* keypoint-free, by using a canonical mesh template for each category, and estimate pose by fitting silhouette. (template based)

![image-20210402103603482](3d reconstruction and generation.assets/image-20210402103603482.png)

![image-20210402103514003](3d reconstruction and generation.assets/image-20210402103514003.png)



#### [ECCV 2020] Self-supervised single-view 3d reconstruction via semantic consistency
[paper](https://arxiv.org/pdf/2003.06473.pdf)

##### Problem

* Input: Single RGB Image, Instance Segmentation
* Output: Camera Pose, Mesh, Texture

##### Contribution

* keypoint-free, by using self-supervised part segmentations (SCOPS) to infer a 3d semantic template. (semantic based)
* Leverage the semantic part invariance property of object instances of a category as a deformable parts model
* Learn a category-level 3D shape template from scratch via iterative learning

![image-20210402102613777](3d reconstruction and generation.assets/image-20210402102613777.png)



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
* Output: Camera Pose, Point Cloud

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

* Input: Single view RGB image, categorical mesh template.
* Output: Camera Pose, Mesh, Texture

##### Contributions

* keypoint-free, combines template-based and semantic-based approaches.
* single GAN for all categories.

![image-20210402101050945](3d reconstruction and generation.assets/image-20210402101050945.png)

![image-20210402101126321](3d reconstruction and generation.assets/image-20210402101126321.png)

##### 

#### [Arxiv 2021] NeuTex: Neural Texture Mapping for Volumetric Neural Rendering

[paper](https://arxiv.org/abs/2103.00762)

##### Contribution

* Disentangled NeRF, regress a UV coordinate before predicting RGB.

* Can Extract View-Independent Mesh.

![image-20210402105216141](3d reconstruction and generation.assets/image-20210402105216141.png)





> What can be done:
>
> * Totally avoid using Mesh Template (e.g. topologically different shapes)
> * Extract mesh from implicit functions --> Marching Cubes
>   * How to extract mesh from NeRF? (view-dependent to independent)



## Scene Level

#### [CVPR 2018] ScanComplete: Large-Scale Scene Completion and Semantic Segmentation for 3D Scans

[paper](https://arxiv.org/pdf/1712.10215.pdf)

##### Problem

* Input: 
* Output: 

##### Contributions

![img](https://github.com/angeladai/ScanComplete/raw/master/images/teaser_mesh.jpg)



#### [CVPR 2020] SG-NN: Sparse Generative Neural Networks for Self-Supervised Scene Completion of RGB-D Scans

``

[paper]()

##### Problem

* Input: 
* Output: 

##### Contributions

![img](https://github.com/angeladai/sgnn/raw/master/sgnn.jpg)



#### [Arxiv 2021] SPSG: Self-Supervised Photometric Scene Generation from RGB-D Scans

``

[paper](https://arxiv.org/pdf/2006.14660)

##### Problem

* Input: 
* Output: 

##### Contributions

![img](https://github.com/angeladai/spsg/raw/master/spsg.jpg)



> What can be done:
>
> * Unsupervised SSC
>   * Input: Single view RGB (+D ?), Output: voxels
>   * How: 
>     * Unsupervised Single-view Novel View Synthesis (e.g. MPI Model, Pixel-NeRF)
>     * Use the novel views (and depths) to reconstruct voxels.
> * Mesh Representation
>   * Hard. It's better to divide and conquer, by first detecting objects and layout.





-------



#### []

``

[paper]()

##### Problem

* Input: 
* Output: 

##### Contributions