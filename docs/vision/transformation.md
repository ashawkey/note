### Transformation

#### 2D

The transformation matrix is **Rotate THEN Translate (and scale)**

$$
\displaylines{
\begin{bmatrix}
\cos\theta & -\sin\theta & t_x \\
\sin\theta &  \cos\theta & t_y \\
0&0&1 \\
\end{bmatrix}
=
\begin{bmatrix}
1 & 0 & t_x \\
0 & 1 & t_y \\
0&0&1 \\
\end{bmatrix}
\begin{bmatrix}
\cos\theta & -\sin\theta & 0 \\
\sin\theta &  \cos\theta & 0 \\
0&0&1 \\
\end{bmatrix}
}
$$


Note: We always use $\mathbf T \mathbf R$ because in this form the translation is applied later and is explicit.

$$
\displaylines{
\mathbf T \mathbf R = \begin{bmatrix}1 & \mathbf t \\ 0 & 1\end{bmatrix} \begin{bmatrix}\mathbf r & 0 \\ 0 & 1\end{bmatrix} = \begin{bmatrix}\mathbf r & \mathbf t \\ 0 & 1\end{bmatrix}
\ne
\begin{bmatrix}\mathbf r & \mathbf {rt} \\ 0 & 1\end{bmatrix} = \begin{bmatrix}\mathbf r & 0 \\ 0 & 1\end{bmatrix}\begin{bmatrix}1 & \mathbf t \\ 0 & 1\end{bmatrix}  =
\mathbf R \mathbf T
}
$$


#### 3D

Main difference from 2D is the three rotation matrices **along three axes**:

$$
\displaylines{
\mathbf R_x(\alpha) = 
\begin{bmatrix}
1&0&0&0\\
0&\cos\alpha & -\sin\alpha &0 \\
0&\sin\alpha & \cos\alpha &0 \\
0&0&0 &1
\end{bmatrix} \\

\mathbf R_y(\alpha) = 
\begin{bmatrix}
\cos\alpha &0 & -\sin\alpha &0 \\
0&1&0&0\\
\sin\alpha &0 & \cos\alpha &0 \\
0&0&0 &1
\end{bmatrix} \\

\mathbf R_z(\alpha) = 
\begin{bmatrix}

\cos\alpha & -\sin\alpha &0&0 \\
\sin\alpha & \cos\alpha &0&0 \\
0&0&1&0 \\
0&0&0&1\\
\end{bmatrix} \\
}
$$

With the final form:

$$
\displaylines{
\mathbf R_{xyz}(\alpha) = \mathbf R_x(\alpha)\mathbf R_y(\alpha)\mathbf R_z(\alpha)
}
$$

Rodrigues' Rotation Formula for rotation along any axis $\mathbf n$:

$$
\displaylines{
\mathbf R(\mathbf n, \alpha) = \cos\alpha\mathbf I + (1 - \cos\alpha)\mathbf n\mathbf n^T + \sin\alpha
\begin{bmatrix}
0 & -n_x &n_y \\
n_z & 0 & -n_x \\
-n_y & n_x & 0
\end{bmatrix}
}
$$


### Decompose 3D transformation

> ref: https://math.stackexchange.com/questions/237369/given-this-transformation-matrix-how-do-i-decompose-it-into-translation-rotati/417813
>
> ref: https://nghiaho.com/?page_id=846

Code in python:

```python
from scipy.spatial.transform import Rotation as sciRot

def decompose(M):
    # M: [4, 4], assuming NO scaling.
    
    # translation 
	T = np.eye(4)
    T[:3, 3] = M[:3, 3]
    
    # rotation at different axes
    rx = np.arctan2(M[2, 1], M[2, 2])
    ry = np.arctan2(-M[2, 0], np.sqrt(M[2, 1]**2 + M[2, 2]**2))
    rz = np.arctan2(M[1, 0], M[0, 0])
	
    R = np.eye(4)
    R[:3, :3] = sciRot.from_euler('xyz', [rx, ry, rz], degrees=False).as_matrix()

    M2 = T @ R
    assert np.allclose(M, M2)
    
    
    
```

