# quaternion



### Basics about complex number

$$
i^2 = -1 \\

z = a + bi = \begin{bmatrix}a &-b \\ b & a\end{bmatrix} \\

||z|| = \sqrt{a^2 + b^2} = \sqrt{z \bar z}
$$



### 2D Rotation

Multiply with a complex number equals **scaling and rotating**.

let $\theta = \arccos \frac{b}{\sqrt{a^2+b^2}}, r=||z||=\sqrt{a^2+b^2}$:


$$
 
z = \begin{bmatrix}a &-b \\ b & a\end{bmatrix} = \sqrt{a^2+b^2} \begin{bmatrix} \frac {a} {\sqrt{a^2+b^2}} & \frac {-b} {\sqrt{a^2+b^2}} \\ \frac{b}{\sqrt{a^2+b^2}} & \frac{a} {\sqrt{a^2+b^2}}\end{bmatrix} = r \begin{bmatrix}\cos \theta  & - \sin \theta \\ \sin \theta & \cos \theta\end{bmatrix} \\ 
= r(\cos\theta + i\sin\theta) \\
= re^{i\theta} 
$$


### 3D Rotation

3D rotation can be represented by three **Euler angles** $(\theta, \phi, \gamma)$, but it relies on the axes system and can lead to Gimbal Lock.


$$
 
\mathbf R_x(\theta) = 
\begin{bmatrix}
1&0&0&0\\
0&\cos\theta & -\sin\theta &0 \\
0&\sin\theta & \cos\theta &0 \\
0&0&0 &1
\end{bmatrix} \\

\mathbf R_y(\phi) = 
\begin{bmatrix}
\cos\phi &0 & -\sin\phi &0 \\
0&1&0&0\\
\sin\phi &0 & \cos\phi &0 \\
0&0&0 &1
\end{bmatrix} \\

\mathbf R_z(\gamma) = 
\begin{bmatrix}
\cos\gamma & -\sin\gamma &0&0 \\
\sin\gamma & \cos\gamma &0&0 \\
0&0&1&0 \\
0&0&0&1\\
\end{bmatrix} \\
$$
Another representation is **axis-angle**, i.e., rotation $\theta$ degree along axis $\textbf {u} = (x, y, z)^T$. ($||\mathbf u|| = 1$ so there are still only 3 Degree of Freedom.)

The Rodrigues' Rotation Formula:
$$
\mathbf v' = \cos\theta\mathbf v + (1 - \cos\theta)(\mathbf u \mathbf v)\mathbf u + \sin\theta(\mathbf u\times\mathbf v)
$$

$$

$$

