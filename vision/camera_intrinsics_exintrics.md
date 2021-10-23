# Camera Pose

### Homogeneous Coordinates

$$
\text{2D:} \quad [u, v, 1]^T  \\
\text{3D:} \quad [x, y, z, 1]^T  \\
$$

### Projection

We usually first transform the 3D points to the camera coordinate system by the camera pose, then project the 3D points to 2D image plane by the camera matrix:
$$
z_c\begin{bmatrix}
u \\ v \\ 1 \\
\end{bmatrix}
=
\mathbf K \begin{bmatrix}\mathbf R & \mathbf T \\ 0&  1\end{bmatrix}

\cdot
\begin{bmatrix}
x_w \\ y_w \\ z_w \\ 1 \\
\end{bmatrix}
$$
3D Point in the world coordinate system: $[x_w, y_w,z_w, 1]^T$ (relative to a defined origin position.)

3D Point in the camera coordinate system: $[x_c, y_c, z_c, 1]^T$ (relative to the camera center position.)

2D Point in the image plane: $[u,v,1]^T$

Camera Intrinsics (determined by the camera itself, also called the **camera matrix**): $\mathbf K \in \mathbb R^{3 \times 4}$.

Camera Extrinsics(describes the position of camera in the world, also called the **camera pose**): $\begin{bmatrix}\mathbf R& \mathbf T \\ 0&  1 \end{bmatrix} \in \mathbb R ^ {4 \times 4}$.



### Intrinsics (Camera Matrix)

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
\alpha_x & \gamma & u_0 & 0 \\
0 &\alpha_y & v_0 &  0 \\
0 & 0 & 1 & 0 \\
\end{bmatrix}
\cdot
\begin{bmatrix}
x_c \\ y_c \\ z_c \\ 1 \\
\end{bmatrix}
$$
$\alpha_x, \alpha_y$ are the **focal length in pixels**. (e.g., $\alpha_x = f\cdot m_x$, where $f$ is the focal length in distance, and $m_x$ is the inverse width of a pixel)

$\gamma$ is the skew coefficient, usually 0.

$(u_0, v_0)$ are the **principal point (camera center)**, ideally the center of the image, i.e., $(H/2, W/2)$



### Extrinsics

A $4 \times 4$ matrix, a regular 3D transformation from world coordinate system to camera coordinate system.
$$
\begin{bmatrix}\mathbf R_{3\times3}& \mathbf T_{3\times1} \\ 0_{1\times3}&  1 \end{bmatrix}
$$
$\mathbf R$ is a rotation matrix. (orthogonal, $\mathbf R^T = \mathbf R^{-1}$)

$\mathbf T$ is the position of *the origin of the world coordinate system*, in the camera coordinate system. (NOT the position of camera in the world system!)

> instead, the position of the camera center in the world coordinate system, $\mathbf C=[x_0, y_0, z_0]$ should be calculated as:
> $$
> \begin{bmatrix}
> 0 \\ 0 \\ 0 \\ 1
> \end{bmatrix}
> =
> \begin{bmatrix}\mathbf R& \mathbf T \\ 0&  1 \end{bmatrix} 
> \cdot
> \begin{bmatrix}
> x_0 \\ y_0 \\ z_0 \\ 1
> \end{bmatrix}
> $$
> thus, $\mathbf C = -\mathbf R^{-1}\mathbf T$