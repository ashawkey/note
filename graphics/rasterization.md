# Rasterization

### Visibility

When rasterize triangles, we need to solve the visibility/occlusion problem.

##### Z-buffer algorithm

store an extra z-value for each pixel to represent the depth. (convention: positive, the larger the further, in fact it can be viewed as the depth.)