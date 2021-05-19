# Rasterization

### Visibility

When rasterize triangles, we need to solve the visibility/occlusion problem.

##### Z-buffer algorithm

store an extra z-value for each pixel to represent the depth. (convention: positive, the larger the further, in fact it can be viewed as the depth.)



```c++
static bool insideTriangle(int x, int y, const Vector4f* _v){
    Vector3f v[3];
    for(int i=0;i<3;i++)
        v[i] = {_v[i].x(), _v[i].y(), 1.0};
    Vector3f f0,f1,f2;
    f0 = v[1].cross(v[0]);
    f1 = v[2].cross(v[1]);
    f2 = v[0].cross(v[2]);
    Vector3f p(x,y,1.);
    if((p.dot(f0)*f0.dot(v[2])>0) && (p.dot(f1)*f1.dot(v[0])>0) && (p.dot(f2)*f2.dot(v[1])>0))
        return true;
    return false;
}

static std::tuple<float, float, float> computeBarycentric2D(float x, float y, const Vector4f* v){
    float c1 = (x*(v[1].y() - v[2].y()) + (v[2].x() - v[1].x())*y + v[1].x()*v[2].y() - v[2].x()*v[1].y()) / (v[0].x()*(v[1].y() - v[2].y()) + (v[2].x() - v[1].x())*v[0].y() + v[1].x()*v[2].y() - v[2].x()*v[1].y());
    float c2 = (x*(v[2].y() - v[0].y()) + (v[0].x() - v[2].x())*y + v[2].x()*v[0].y() - v[0].x()*v[2].y()) / (v[1].x()*(v[2].y() - v[0].y()) + (v[0].x() - v[2].x())*v[1].y() + v[2].x()*v[0].y() - v[0].x()*v[2].y());
    float c3 = (x*(v[0].y() - v[1].y()) + (v[1].x() - v[0].x())*y + v[0].x()*v[1].y() - v[1].x()*v[0].y()) / (v[2].x()*(v[0].y() - v[1].y()) + (v[1].x() - v[0].x())*v[2].y() + v[0].x()*v[1].y() - v[1].x()*v[0].y());
    return {c1,c2,c3};
}

// triangle rasterization
void rst::rasterizer::rasterize_triangle(const Triangle& t) {
    auto v = t.v;
    
    // Find out the bounding box of current triangle.
    float min_x = std::min(v[0](0), std::min(v[1](0), v[2](0)));
    float max_x = std::max(v[0](0), std::max(v[1](0), v[2](0)));
    
    float min_y = std::min(v[0](1), std::min(v[1](1), v[2](1)));
    float max_y = std::max(v[0](1), std::max(v[1](1), v[2](1)));

    // iterate through the pixel and find if the current pixel is inside the triangle
    for (int x = min_x; x <= max_x; x++) {
        for (int y = min_y; y <= max_y; y++) {
            if (insideTriangle(x, y, t.v)) {
                // barycentric interpolation of depth (z)
                auto[alpha, beta, gamma] = computeBarycentric2D(x, y, t.v);
                float w_reciprocal = 1.0/(alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
                float interpolated_z = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
                interpolated_z *= w_reciprocal;
                                
                // update z_buffer 
                int ind = (height - 1 - y) * width + x;
                if (depth_buf[ind] > z_interpolated) {
                    depth_buf[ind] = z_interpolated;
                    // barycentric interpolate colors / normals / coords...
                    auto interpolated_color = alpha * t.color[0] + beta * t.color[1] + gamma * t.color[2];
                    auto interpolated_texcoords = alpha * t.tex_coords[0] + beta * t.tex_coords[1] + gamma * t.tex_coords[2];
                    auto interpolated_normal = alpha * t.normal[0] + beta * t.normal[1] + gamma * t.normal[2];
                    auto interpolated_v = alpha * t.v[0] + beta * t.v[1] + gamma * t.v[2];
                    Eigen::Vector3f interpolated_shadingcoords = interpolated_v.head(3) / interpolated_v(3);

                    auto pixel_color = shader_get_color(interpolated_color, interpolated_texcoords, interpolated_normal, interpolated_shadingcoords);
                    // update color
                    frame_buf[ind] = pixel_color;
                }
            }
        }
    }
}
```

