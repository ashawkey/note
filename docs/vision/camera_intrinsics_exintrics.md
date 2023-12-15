## Camera Pose



Very good reference: https://ksimek.github.io/2012/08/22/extrinsic/

Good demo of look-at camera: https://learnwebgl.brown37.net/07_cameras/camera_lookat/camera_lookat.html



### Homogeneous Coordinates


$$
\text{2D:} \quad [u, v, 1]^T  \\
\text{3D:} \quad [x, y, z, 1]^T  \\
$$


### Projection from 3D to 2D

We usually first transform the 3D points to the camera coordinate system by the camera pose, then project the 3D points to 2D image plane by the camera matrix:

$$
z_c\begin{bmatrix}
u \\ v \\ 1 \\
\end{bmatrix}
=
\mathbf K
\begin{bmatrix}
x_c \\ y_c \\ z_c \\ 1 \\
\end{bmatrix}
=
\mathbf K \begin{bmatrix}\mathbf R & \mathbf T \\ 0&  1\end{bmatrix}
\cdot
\begin{bmatrix}
x_w \\ y_w \\ z_w \\ 1 \\
\end{bmatrix}
$$

3D Point in the world coordinate system: $[x_w, y_w,z_w]^T$ (relative to a defined origin position.)

3D Point in the camera coordinate system: $[x_c, y_c, z_c]^T$ (relative to the camera center position.)

2D Point (Pixel) in the image plane: $[u,v]^T$ (range in $[0, H]\times[0, W]$)

Camera Intrinsic (determined only by the camera itself): $\mathbf K \in \mathbb R^{3 \times 4}$.

Camera Extrinsic (describes the transformation from **world to camera**, inversion of camera pose): $\begin{bmatrix}\mathbf R& \mathbf T \\ 0&  1 \end{bmatrix} \in \mathbb R ^ {4 \times 4}$.



### Intrinsic

A $3 \times 4$ matrix used to **project** 3D points to 2D coordinates:

$$
z_c\begin{bmatrix}
u \\ v \\ 1 \\
\end{bmatrix}
 = 
\mathbf K
\begin{bmatrix}
x_c \\ y_c \\ z_c \\ 1 \\
\end{bmatrix}
 =
\begin{bmatrix}
f_x & \gamma & u_0 & 0 \\
0 & f_y & v_0 &  0 \\
0 & 0 & 1 & 0 \\
\end{bmatrix}
\cdot
\begin{bmatrix}
x_c \\ y_c \\ z_c \\ 1 \\
\end{bmatrix}
$$

$f_x, f_y$ are the **focal length in pixels**, usually $f_x =f_y$.

$\gamma$ is the skew coefficient, usually 0.

$(u_0, v_0)$ are the **principal point (camera center) in pixels**, ideally the center of the image, i.e., $(H/2, W/2)$.

Inversely, we can use the intrinsic to project pixel coordinates to 3D points in the camera's coordinate system.

Since a pixel can be projected to multiple depth planes, so we need to know the depth value $z_c$ in advance.

$$
\begin{cases}
x_c = \frac {(u - u_0)} {f_x} z_c \\
y_c = \frac {(v - v_0)} {f_y} z_c \\
z_c
\end{cases}
$$




### Extrinsic (w2c)

A $4 \times 4$ matrix, a regular **3D transformation from world coordinate system to camera coordinate system**.

$$
\begin{bmatrix}\mathbf R_{3\times3}& \mathbf T_{3\times1} \\ 0_{1\times3}&  1 \end{bmatrix}
=
\begin{bmatrix}\mathbf I& \mathbf T \\ 0&  1 \end{bmatrix}
\begin{bmatrix}\mathbf R& 0 \\ 0&  1 \end{bmatrix}
=
\begin{bmatrix}\mathbf R& 0 \\ 0&  1 \end{bmatrix}
\begin{bmatrix}\mathbf I& -\mathbf C \\ 0&  1 \end{bmatrix}
$$

It can be decomposed as:

* first rotate with $\mathbf R$, then translate with $\mathbf T$, or
* first translate with $-\mathbf C$, then rotate with $\mathbf R$.

$\mathbf R$ is a rotation matrix. (orthogonal, $\mathbf R^T = \mathbf R^{-1}$)

$\mathbf T$ is the position of **the world origin in the camera coordinate system**,

**NOT the camera position in the world coordinate system!** 

instead, the position of the camera center in the world coordinate system, $\mathbf C=[x_0, y_0, z_0]$ should be calculated as:

$$
\begin{bmatrix}
0 \\ 0 \\ 0 \\ 1
\end{bmatrix}
=
\begin{bmatrix}\mathbf R& \mathbf T \\ 0&  1 \end{bmatrix} 
\cdot
\begin{bmatrix}
x_0 \\ y_0 \\ z_0 \\ 1
\end{bmatrix}
$$

thus, $\mathbf C = -\mathbf R^{-1}\mathbf T$

this also gives a way to calculate $\mathbf{T} = -\mathbf{RC}$.



### Pose (c2w)

Also a $4 \times 4$ matrix, but it describes the **3D transformation from camera to world**.

Obviously, **camera pose (c2w) is the inversion of extrinsic (w2c)**.

$$
\begin{bmatrix}\mathbf R_{3\times3}^T& \mathbf C_{3\times1} \\ 0_{1\times3}&  1 \end{bmatrix} = 
\begin{bmatrix}\mathbf R_{3\times3}& \mathbf T_{3\times1} \\ 0_{1\times3}&  1 \end{bmatrix}^{-1}
$$

Note that now the translation vector $\mathbf{C}$ is the camera's position in the world coordinate system now.



### Construct by `LookAt`

The camera pose matrix in OpenGL is defined as:

![img](camera_intrinsics_exintrics.assets/look-at-2.png)

Assuming you know the camera position $\mathbf{C}$, and target position $\mathbf{O}$, note the forward direction is $\mathbf{\overrightarrow{OC}}$.

To construct the **camera pose matrix**, you can calculate the normalized **right, up, and forward vector**, then simply concatenate them：

$$
\begin{bmatrix}
x_w \\ y_w \\ z_w \\ 1
\end{bmatrix}
=
\begin{bmatrix}
\text{right}_x & \text{up}_x & \text{forward}_x & \text{C}_x \\ 
\text{right}_y & \text{up}_y & \text{forward}_y & \text{C}_y \\ 
\text{right}_z & \text{up}_z & \text{forward}_z & \text{C}_z \\ 
0 & 0 & 0 & 1 
\end{bmatrix} 
\cdot
\begin{bmatrix}
x_c \\ y_c \\ z_c \\ 1
\end{bmatrix}
$$

Or the **camera extrinsic/view matrix** similarly:

$$
\begin{bmatrix}
x_c \\ y_c \\ z_c \\ 1
\end{bmatrix}
=
\begin{bmatrix}
\text{right}_x & \text{right}_y & \text{right}_z & 0 \\ 
\text{up}_x & \text{up}_y & \text{up}_z & 0 \\ 
\text{forward}_x & \text{forward}_y & \text{forward}_z & 0 \\ 
0 & 0 & 0 & 1 
\end{bmatrix} 
\cdot
\begin{bmatrix}
1 & 0 & 0 & -C_x \\ 
0 & 1 & 0 & -C_y \\ 
0 & 0 & 1 & -C_z \\ 
0 & 0 & 0 & 1 
\end{bmatrix} 
\cdot
\begin{bmatrix}
x_w \\ y_w \\ z_w \\ 1
\end{bmatrix}
$$



There are different world/camera coordinate conventions, which are really confusing:

```
Four common world coordinate conventions:

   OpenGL          OpenCV           Blender        Unity             
Right-handed       Colmap                        Left-handed  

     +y                +z           +z  +y         +y  +z                                               
     |                /             |  /           |  /                                               
     |               /              | /            | /                                                   
     |______+x      /______+x       |/_____+x      |/_____+x                                          
    /               |                                                                                       
   /                |                                                                                       
  /                 |                                                                                         
 +z                 +y                                                                                       


Two common camera coordinate conventions:

   OpenGL                OpenCV       
   Blender               Colmap       

     up  target          forward & target
     |  /                /         
     | /                /          
     |/_____right      /______right   
    /                  |           
   /                   |           
  /                    |           
forward                up          
```

A common color code: x/right = red, y/up = green, z/forward = blue (XYZ=RGB=RUF)

The camera xyz follows corresponding world coordinate system.
However, the three directions (right, up, forward) can be defined differently:

* forward can be (camera --> target) or (target --> camera).
* up can align with the world-up-axis (y) or world-down-axis (-y).
* right can also be left, depending on it's (up cross forward) or (forward cross up).

But many datasets are just very confusing and combine different conventions together.
You may check a few poses to make sure what the convention they are using... and combine:

* axis switching: `pose[[1, 2]] = pose[[2, 1]]`
* axis inverting: `pose[1] *= -1`
* forward inverting: `pose[:3, 2] *= -1`
* up inverting: `pose[:3, 1] *= -1`
