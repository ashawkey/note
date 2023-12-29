## Physically-based Rendering

* notations:
  * $\mathbf p$ is the shading point.
  * $\omega_o$ is the outgoing view direction (shading point to eye)
  * $\omega_i$ is the outgoing lighting direction (shading point to light)
  * $\mathbf n$ is the normal vector
  * $\mathbf t = 2(\mathbf n \cdot \omega_o) \mathbf n - \omega_o$ is the reflective direction
  * $\mathbf h=\frac {w_i + w_o} {||w_i + w_o||}$ is the halfway vector (different from $\mathbf n$ !)
  * $m$ is the metalness
  * $\rho$ is the roughness
  * $\mathbf a$ is the ambient color
  * $s(\omega_i)$ is the occlusion probability from the shading point to light.
* rendering Equation:


$$
\displaylines{
\mathbf c(\mathbf \omega_o) = \int_\Omega L(\omega_i)f(\omega_i, \omega_o)(\omega_i \cdot \mathbf n)d \omega_i
}
$$


* micro-facet BRDF (Cook-Torrance 1982):
  $$
  f(\omega_i, \omega_o) = (1 - m) \frac {\mathbf a} {\pi} + \frac {DFG} {4(\omega_i \cdot \mathbf n)(\omega_o \cdot \mathbf n)}
  $$
  The BRDF can be divided into diffuse + specular terms, where the specular term includes:

  * F = Fresnel term

  $$
  F_0 = m * \mathbf a + (1 - m) * 0.04 \\
  F = F_0 + (1 - F_0)(1-(\mathbf h \cdot \omega_o))^5
  $$

  * G = Geometry term (Schlick-GGX)
    $$
    k = \rho^4 / 2 \\
    g(\mathbf v) = \frac {\mathbf n \cdot \mathbf v} {k + (1 - k)\mathbf n \cdot \mathbf v}\\
    G = g(\omega_o) g(\omega_i)
    $$

  * D = Normal Distribution (Trowbridge-Reitz GGX)
    $$
    \alpha = \rho^2 \\
    D = \frac {\alpha^2} {\pi((\mathbf n \cdot \mathbf h)(\alpha^2 - 1) + 1)^2}
    $$

* split-sum approximation (Karis and Games 2013)
  $$
  \int_\Omega L(\omega_i)\frac {DFG} {4(\omega_i \cdot \mathbf n)(\omega_o \cdot \mathbf n)} (\omega_i \cdot \mathbf n)d \omega_i \\ 
  = \int_\Omega L(\omega_i)Dd\omega_i \int_\Omega \frac {DFG} {4(\omega_o \cdot \mathbf n)} d \omega_i \\
  $$
  
* lighting representation
  $$
  L(\omega_i) = (1 - s(\omega_i)) g_\text{direct}(\omega_i) +  s(\omega_i) g_\text{indirect}(\omega_i, \mathbf p) \\
  s(\omega_i) = g_\text{occ}(\omega_i, \mathbf p)
  $$

  * direct term (from light): only dependent on the outgoing light direction.
  * indirect term (from other reflective surfaces): also dependent on the current shading point.
