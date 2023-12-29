# OpenGL

A definition of a set of APIs to call GPU to render images.

Kudos to [learnopengl](https://learnopengl-cn.github.io/) !


### Versions

* **OpenGL < 3.3**: Legacy, not used now. Use fixed pipeline.
* **OpenGL >= 3.3**: Modern. Use programmable pipeline (shaders).
* **OpenGL ES**: for embedded systems and **webGL**. (1.1 for fixed pipeline, 2.0 for programmable pipeline.)
* **Vulkan** (OpenGL 5, next generation GL.)
* **DirectX** (Not OpenGL, Windows's graphic library.)


### APIs

The way OpenGL works is like a **state machine**. We always call it **Sequentially**.

#### Window

* `glViewport(0, 0, 800, 600)`: init a window of size (800, 600).
* `glClearColor(r, g, b, a)`: set the color used to clear the window (fill with this color after clear)
* `glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)`: clear the window in act.

#### Triangles

* VBO = Vertex Buffer Object, used to save triangle vertices (coordinates).

  ```cpp
  unsigned int VBO;
  // create VBO object (we cannot access the object directly, we only have its id.)
  glGenBuffers(1, &VBO); 
  // bind array buffer (use this VBO buffer as ARRAY_BUFFER)
  glBindBuffer(GL_ARRAY_BUFFER, VBO); 
  // copy data from vertices to buffer. 
  // GL_STATIC_DRAW == const, which means vertices will (almost) never change.
  // GL_DYNAMIC_DRAW/GL_STREAM_DRAW: the data may change frequently, so better write it to faster memory.
  glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW); 
  ```

* Vertex Shader: operate on vertices, should set `gl_Position`.

  Write the shader first.

  ```glsl
  // use OpenGL 3.3 core mode
  #version 330 core 
  
  // define input (keyword `in`)
  // `layout (location = 0)` means aPos is passed in as the first parameter (like in a function)
  layout (location = 0) in vec3 aPos;
  
  // the program
  void main() {
      // the output (MUST!), tell OpenGL the current vertex's position. 
      // `gl_Position` is a keyword. It's vec4, and the last number is used in perspective division (x, y, z, w) -> (x/w, y/w, z/w), usually we just use w = 1.
      gl_Position = vec4(aPos.x, aPos.y, aPos.z, 1.0);
  }
  ```

  Then bind it with OpenGL:

  ```cpp
  unsigned int vertexShader;
  // we create vertex shader.
  vertexShader = glCreateShader(GL_VERTEX_SHADER);
  // pass in the source code.
  glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
  // compile it.
  glCompileShader(vertexShader);
  
  // debug shader compilation:
  int  success;
  char infoLog[512];
  glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
  if(!success) {
      glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
      std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
  }
  ```

* Fragment Shader (or Pixel Shader): handles each pixel (rasterization).

  Write the source:

  ```glsl
  #version 330 core
      
  // define output (keyword `out`)
  // we can define any output! But we need to tell OpenGL how to interpret them later (glVertexAttribPointer)
  // usually we want to output the color of the pixel.    
  out vec4 FragColor;
  
  void main() {
      // just use a constant color. 
      // In GLSL, color range is [0, 1] float.
      FragColor = vec4(1.0f, 0.5f, 0.2f, 1.0f);
  } 
  ```

  Bind with OpenGL:

  ```cpp
  unsigned int fragmentShader;
  fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
  glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
  glCompileShader(fragmentShader);
  ```

* Shader Program: the final program that links all shaders.

  ```cpp
  unsigned int shaderProgram;
  shaderProgram = glCreateProgram();
  // attach the two shaders
  glAttachShader(shaderProgram, vertexShader);
  glAttachShader(shaderProgram, fragmentShader);
  // link them
  glLinkProgram(shaderProgram);
  // delete shaders (since we don't need them now)
  glDeleteShader(vertexShader);
  glDeleteShader(fragmentShader);
  // activate the program
  glUseProgram(shaderProgram);
  
  
  // debug link error
  glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
  if(!success) {
      glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
      ...
  }
  ```


* Link vertex attribute: tell OpenGL the output of our fragment shader.

  ```cpp
  // tell OpenGL how to interpret fragment shader's `out`s.
  // `0` refers to `layout (location = 0)`, so we are telling the first `out`.
  // `3` means size of the attribute, here vec3
  // `GL_FLOAT` means datatype.
  // `GL_FALSE` means we do not normalize the out. (`GL_TRUE` will normalize it to [-1, 1] or [0, 1] if unsigned. but we usually do normalization in the shader manually.)
  // `3 * sizeof(float)` is the stride for current attr. If we have multiple `out`s, this should be the size of all `out`s !
  // `(void*)0` is the offset for current attr. If we have multiple `out`s, this should be the size of previous `out`s.
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
  
  // enable the attr at location 0. by defult it's disabled.
  glEnableVertexAttribArray(0);
  ```

  

* VAO = Vertex Array Object. It saves the vertex attribute pointers, to make it easy to switch between different VBOs.

  Without VAO, we have to define vertex attribute every time before we draw:

  ```cpp
  //// these may run multiple times
  glUseProgram(shaderProgram);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
  glEnableVertexAttribArray(0);
  glDrawArrays(GL_TRIANGLES, 0, 3); // any draw function
  ```

  

  ![img](opengl.assets/vertex_array_objects.png)

  ```cpp
  //// these are run only once at initilization!
  // generate VAO object.
  unsigned int VAO;
  glGenVertexArrays(1, &VAO);
  // bind VAO.
  glBindVertexArray(VAO);
  // bind VBO and copy data.
  glBindBuffer(GL_ARRAY_BUFFER, VBO);
  glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
  // tell vertex attr
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
  glEnableVertexAttribArray(0);
  
  //// these may run multiple times
  glUseProgram(shaderProgram);
  // bind VAO
  glBindVertexArray(VAO);
  glDrawArrays(GL_TRIANGLES, 0, 3); // any draw function
  // unbind VAO
  glBindVertexArray(0);
  ```

  

* EBO/IBO = Element / Index Buffer Object. This is used to avoid drawing repeated vertices (e.g., for a rectangle composed of two triangles, we don't want to draw the diagonal line twice.)

  ```cpp
  float vertices[] = {
      0.5f, 0.5f, 0.0f,   // 右上角
      0.5f, -0.5f, 0.0f,  // 右下角
      -0.5f, -0.5f, 0.0f, // 左下角
      -0.5f, 0.5f, 0.0f   // 左上角
  };
  
  // the faces, in fact.
  unsigned int indices[] = { 
      0, 1, 3, // 第一个三角形
      1, 2, 3  // 第二个三角形
  };
  
  // create EBO object
  unsigned int EBO;
  glGenBuffers(1, &EBO);
  // bind buffer and copy data
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
  // draw
  // `glDrawElements` draw from EBO, so we won't draw repeated vertices.
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
  glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
  ```

  Similarly, EBO can be bound to VAO for easy switching.

  ![img](opengl.assets/vertex_array_objects_ebo.png)

  ```cpp
  //// these are run only once at initilization!
  // generate VAO object.
  unsigned int VAO;
  glGenVertexArrays(1, &VAO);
  // bind VAO.
  glBindVertexArray(VAO);
  // bind VBO and copy data.
  glBindBuffer(GL_ARRAY_BUFFER, VBO);
  glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
  // bind EBO and copy data.
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
  // tell vertex attr
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
  glEnableVertexAttribArray(0);
  
  //// these may run multiple times
  glUseProgram(shaderProgram);
  // bind VAO (also EBO!)
  glBindVertexArray(VAO);
  glDrawArrays(GL_TRIANGLES, 0, 3); // any draw function
  // unbind VAO
  glBindVertexArray(0);
  ```

* wireframe mode.

  ```cpp
  // start wireframe mode
  glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
  // return to default fill mode
  glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
  ```

  

#### Shaders

GLSL (OpenGL Shading Language): Small c-like programs to describe Vertex & Fragment Shaders.

* datatype:

  ```glsl
  //// vectors
  vecn v; // float32, n = 1,2,3,4
  dvecn v; // float64
  ivecn v; // int32
  uvecn v; // uint32
  
  v.x; v.y; v.z; v.w; // get each scalar.
  
  // init
  vec2 v = vec2(0.0, 1.0);
  vec4 w = vec4(v, 2.0, 3.0);
  vec4 u = vec4(v.xyz, 1.0);
  
  // swizzling
  vec2 a;
  vec4 b = a.xyxx;
  vec3 c = b.zyw;
  vec4 d = a.xxxx + c.yxzy;
  
  // color mixing by elementwise multiply
  vec4 a = vec4(1.0, 2.0, 3.0, 4.0);
  vec4 b = vec4(0.1, 0.2, 0.3, 0.4);
  vec4 c = a * b; // vec4(0.1, 0.4, 0.9, 1.6)
  c = clamp(c, 0.0, 1.0); // vec4(0.1, 0.4, 0.9, 1.0)
  
  // other functions
  vec3 nn = normalize(n.xyz);
  vec3 c = normalize(a - b);
  float d = max(dot(dir1, dir2), 0.0);
  
  //// matrices
  matn m; // float32, m = 2,3,4
  
  mat4 m;
  vec4 x = vec4(1.0, 1.0, 1.0, 1.0);
  vec4 y = m * x; // x is treated as a column vector! [4,4] @ [4,1] = [4,1]
  
  //// struct
  struct mat {
      vec3 ambient;
      vec3 diffuse;
      vec3 specular;
      float shininess;
  }
  
  mat m; // new
  m.ambient; // access
  ```

* input & output

  We can pass parameters from Vertex Shader to Fragment Shader:

  ````glsl
  //// vertex shader
  #version 330 core
      
  // input
  layout (location = 0) in vec3 aPos;
  
  // output to frag shader
  out vec4 vertexColor;
  
  void main() {
      gl_Position = vec4(aPos, 1,0);
      // assign value 
      vertexColor = vec4(0.5, 0.0, 0.0, 1.0);
  }
  ````
  
  and receive it with:
  
  ```glsl
  //// fragment shader
  #version 330 core
  
  // recieve input (should have the same type and name !!!)
  // vertexColor is barycentric-interpolated at the current pixel.
  in vec4 vertexColor; 
  // output to further pipeline.
  out vec4 FragColor;
  
  void main() {
      FragColor = vertexColor;
  }
  ```
  
* uniform (global variable passed from OpenGL directly)

  ```glsl
  //// fragment shader
  #version 330 core
  out vec4 FragColor;
  
  // define uniforms, should be set in OpenGL code
  // since we link all shaders, do not use same uniform names at vert & frag shaders !
  uniform vec4 ourColor; 
  
  void main() {
      FragColor = ourColor;
  }
  ```

  and we can set it with:

  ```cpp
  // locate the uniform with name
  int vertexColorLocation = glGetUniformLocation(shaderProgram, "ourColor");
  // must first use program!
  glUseProgram(shaderProgram);
  // set uniform (4f means 4 float values)
  // glUniform[ni, nf, nui, fv]
  glUniform4f(vertexColorLocation, 0.0f, 1.0f, 0.0f, 1.0f);
  
  /// to set uniform for a struct, just use the name "a.b"
  int ambientLocation = glGetUniformLocation(shaderProgram, "material.ambient");
  glUniform3f(vertexColorLocation, 0.1f, 0.1f, 0.1f);
  ```

  > 如果你声明了一个uniform却在GLSL代码中没用过，编译器会静默移除这个变量，导致最后编译出的版本中并不会包含它!

* example of more inputs.

  shaders:

  ```cpp
  //// vertex shader
  #version 330 core
  layout (location = 0) in vec3 aPos;   // pos
  layout (location = 1) in vec3 aColor; // color
  
  out vec3 vertexColor; // output to fragment shader
  
  void main() {
      gl_Position = vec4(aPos, 1.0);
      vertexColor = aColor;
  }
  ```

  ```glsl
  //// fragment shader
  #version 330 core
  out vec4 FragColor;  
  in vec3 vertexColor; // interpolated at current pixel, not the exact vertex color!
  
  void main() {
      FragColor = vec4(vertexColor, 1.0);
  }
  ```

  vertex attributes:

  ```cpp
  // pos, note the stride!
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
  glEnableVertexAttribArray(0);
  // color, note the offset!
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3* sizeof(float)));
  glEnableVertexAttribArray(1);
  ```

  

#### Textures

* Texture can be mapped with UVs. 

  The range of UV is always [0, 1], if we use out of range UV, the default warp behavior is to REPEAT the texture (equal to modulo the UV). But we can set different behavior at each axis (UV axes are called `s,t,r`, corresponding to `x,y,z`):

  ```cpp
  // set different warp behavior
  // `GL_TEXTURE_2D` means we use 2D texture
  // `GL_TEXTURE_WRAP_[S/T]` set the axis.
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT); // default
  
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT);
  
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); 
  
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER); // constant, need to set additional border color.
  float borderColor[] = { 1.0f, 1.0f, 0.0f, 1.0f };
  glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);
  ```

* Texture filtering (interpolation)

  ```cpp
  // set different filtering method
  // `GL_TEXTURE_MIN_FILTER` means the minify behavior
  // `GL_TEXTURE_MAG_FILTER` means the magnify behavior
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST); // default
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  ```

  Mipmap filtering: Mipmap is a image pyramid at multiple resolutions. OpenGL use Mipmap to choose the proper texture resolution, to save memory and improve realism (use low resolution texture at far distance).

  ![img](opengl.assets/mipmaps.png)

  ```cpp
  // best practice in set mipmap filtering.
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR); // for minify, use mipmap for low-resolution.
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR); // for magnify, do not use mipmap since we cannot super-resolution the texture.
  ```

  OpenGL can automatically generate Mipmap with any input texture (see later load texture).

* load texture

  ```cpp
  // gen texture object
  unsigned int texture;
  glGenTextures(1, &texture);
  // bind current texture
  glBindTexture(GL_TEXTURE_2D, texture);
  // copy data
  // `GL_TEXTURE_2D` means we use 2d texture.
  // `0` set the mipmap level, we use 0 for the original resolution.
  // `GL_RGB` is the texture data format. (other possibles are GL_RGBA, ...)
  // `width, height` are the image size.
  // `0` is always 0 (historical reason)
  // `GL_RGB` is the input texture data format.
  // `data` is the image data (in char array).
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
  // generate mipmap
  glGenerateMipmap(GL_TEXTURE_2D);
  // free the data
  free(data);
  ```

  > usually we should **flip y axis** of the image, because OpenGL use (0,0) at left-bottom, while most image loaders use (0,0) at left-top.

* apply texture

  we need another vertex-level input, the texture UVs.

  ```cpp
  float vertices[] = {
  //     ---- 位置 ----       ---- 颜色 ----     - 纹理坐标 -
       0.5f,  0.5f, 0.0f,   1.0f, 0.0f, 0.0f,   1.0f, 1.0f,   // 右上
       0.5f, -0.5f, 0.0f,   0.0f, 1.0f, 0.0f,   1.0f, 0.0f,   // 右下
      -0.5f, -0.5f, 0.0f,   0.0f, 0.0f, 1.0f,   0.0f, 0.0f,   // 左下
      -0.5f,  0.5f, 0.0f,   1.0f, 1.0f, 0.0f,   0.0f, 1.0f    // 左上
  };
  ```

  declare it with:

  ```cpp
  glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));
  glEnableVertexAttribArray(2);
  ```

  input to vertex shader:

  ```glsl
  #version 330 core
  layout (location = 0) in vec3 aPos;
  layout (location = 1) in vec3 aColor;
  layout (location = 2) in vec2 aTexCoord; // UV
  
  out vec3 ourColor;
  out vec2 TexCoord; // pass to frag shader
  
  void main() {
      gl_Position = vec4(aPos, 1.0);
      ourColor = aColor;
      TexCoord = aTexCoord;
  }
  ```

  use it in fragment shader:

  ```glsl
  #version 330 core
  out vec4 FragColor;
  
  in vec3 ourColor;
  in vec2 TexCoord; // catch it (also barycentric interpolated!)
  
  // texture should be a uniform. 
  // sampler[nD] is a built in datatype for texture.
  // it use the default texture automatically (0), so we don't need to glUniform it since we only have one texture.
  uniform sampler2D ourTexture;
  
  void main() {
      // texture() is a build in function. It performs interpolation at TexCoord on ourTexture.
      FragColor = texture(ourTexture, TexCoord);
  }
  ```

  finally, draw it:

  ```cpp
  // bind texture!
  // glActiveTexture(GL_TEXTURE0); // this is run automatically, it assigns GL_TEXTURE0 to the uniform sampler2D.
  glBindTexture(GL_TEXTURE_2D, texture);
  glBindVertexArray(VAO);
  glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
  ```

* texture unit: To use multiple textures in one shader, we need to introduce texture unit.

  The default texture unit is `GL_TEXTURE0`, and OpenGL provides `GL_TEXTURE1, ..., GL_TEXTURE15`.

  We can also use add to get next texture, e.g., `GL_TEXTURE0 + 1 ` equals ` GL_TEXTURE1`.

  examples of using multiple textures:

  ```cpp
  //// fragment shader
  #version 330 core
  ...
  
  uniform sampler2D texture1;
  uniform sampler2D texture2;
  
  void main() {
      // mix(a, b, x) = (1 - x) * a + x * b
      FragColor = mix(texture(texture1, TexCoord), texture(texture2, TexCoord), 0.2);
  }
  ```

  bind texture sequentially:

  ```cpp
  // create texture, etc
  unsigned int texture1, texture2;
  ... 
  //// only run once at initialization    
  // set uniforms (now we have to glUniform it)
  glUniform1i(glGetUniformLocation(shaderProgram, "texture1"), 0);
  glUniform1i(glGetUniformLocation(shaderProgram, "texture2"), 1);
  
  //// should run everytime we draw
  // bind texture    
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, texture1);
  glActiveTexture(GL_TEXTURE1);
  glBindTexture(GL_TEXTURE_2D, texture2);
  // draw
  glBindVertexArray(VAO);
  glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
  ```

  

#### Transformations

* we pass in transformations as `mat4` simply:

  ```glsl
  //// vertexShader
  #version 330 core
  layout (location = 0) in vec3 aPos;
  layout (location = 1) in vec2 aTexCoord;
  
  out vec2 TexCoord;
  
  // the transform
  uniform mat4 transform;
  
  void main() {
      gl_Position = transform * vec4(aPos, 1.0f);
      TexCoord = vec2(aTexCoord.x, 1.0 - aTexCoord.y);
  }
  ```

  assign it:

  ```cpp
  unsigned int transformLoc = glGetUniformLocation(shaderProgram, "transform");
  // `transformLoc` is the id.
  // `1` means we send one matrix.
  // `GL_FALSE` means do not transpose (OpenGL use column-major format! If we pass a numpy array, we should set GL_TRUE, but we prefer to still use GL_FALSE and manually transpose the numpy array instead.)
  // `glm::value_ptr(trans)` is the data, glm is also column-major.
  glUniformMatrix4fv(transformLoc, 1, GL_FALSE, glm::value_ptr(trans));
  ```

  

#### Coordinate system

* OpenGL use Left-hand coordinate system.

* The usual coordinate system transformations:

  ![coordinate_systems](opengl.assets/coordinate_systems.png)

* NDC (Normalized Device Coordinate): range in [-1, 1], coordinates outside will be simply out of screen and not rendered. It is used in the CLIP space where perspective division is performed.

* Projection.

  * Orthogonal.
  * Perspective.

* Z-buffer: display only the nearest triangle if multiple triangles overlapped.

  By default it is disabled, we usually enable it with:

  ```cpp
  // at initialization
  glEnable(GL_DEPTH_TEST);
  
  // need to clear Z-buffer at the beginning of each render loop!
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  ```

  

#### Camera

* Yaw（偏航角）, Pitch（俯仰角）, Roll（滚转角）: 以自己为中心的三转轴。

  ![img](opengl.assets/camera_pitch_yaw_roll.png)


### [A complete program example in C++](https://github.com/JoeyDeVries/LearnOpenGL/tree/master/src/1.getting_started/6.2.coordinate_systems_depth)

```glsl
#version 330 core
out vec4 FragColor;

in vec2 TexCoord;

// texture samplers
uniform sampler2D texture1;
uniform sampler2D texture2;

void main()
{
	// linearly interpolate between both textures (80% container, 20% awesomeface)
	FragColor = mix(texture(texture1, TexCoord), texture(texture2, TexCoord), 0.2);
}
```

```glsl
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTexCoord;

out vec2 TexCoord;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
	gl_Position = projection * view * model * vec4(aPos, 1.0f);
	TexCoord = vec2(aTexCoord.x, aTexCoord.y);
}
```

```cpp
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <stb_image.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <learnopengl/shader_m.h>

#include <iostream>

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow *window);

// settings
const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 600;

int main()
{
    // glfw: initialize and configure
    // ------------------------------
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    // glfw window creation
    // --------------------
    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "LearnOpenGL", NULL, NULL);
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    // glad: load all OpenGL function pointers
    // ---------------------------------------
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    // configure global opengl state
    // -----------------------------
    glEnable(GL_DEPTH_TEST);

    // build and compile our shader zprogram
    // ------------------------------------
    Shader ourShader("6.2.coordinate_systems.vs", "6.2.coordinate_systems.fs");

    // set up vertex data (and buffer(s)) and configure vertex attributes
    // ------------------------------------------------------------------
    float vertices[] = {
        -0.5f, -0.5f, -0.5f,  0.0f, 0.0f,
         0.5f, -0.5f, -0.5f,  1.0f, 0.0f,
         0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
         0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
        -0.5f,  0.5f, -0.5f,  0.0f, 1.0f,
        -0.5f, -0.5f, -0.5f,  0.0f, 0.0f,

        -0.5f, -0.5f,  0.5f,  0.0f, 0.0f,
         0.5f, -0.5f,  0.5f,  1.0f, 0.0f,
         0.5f,  0.5f,  0.5f,  1.0f, 1.0f,
         0.5f,  0.5f,  0.5f,  1.0f, 1.0f,
        -0.5f,  0.5f,  0.5f,  0.0f, 1.0f,
        -0.5f, -0.5f,  0.5f,  0.0f, 0.0f,

        -0.5f,  0.5f,  0.5f,  1.0f, 0.0f,
        -0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
        -0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
        -0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
        -0.5f, -0.5f,  0.5f,  0.0f, 0.0f,
        -0.5f,  0.5f,  0.5f,  1.0f, 0.0f,

         0.5f,  0.5f,  0.5f,  1.0f, 0.0f,
         0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
         0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
         0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
         0.5f, -0.5f,  0.5f,  0.0f, 0.0f,
         0.5f,  0.5f,  0.5f,  1.0f, 0.0f,

        -0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
         0.5f, -0.5f, -0.5f,  1.0f, 1.0f,
         0.5f, -0.5f,  0.5f,  1.0f, 0.0f,
         0.5f, -0.5f,  0.5f,  1.0f, 0.0f,
        -0.5f, -0.5f,  0.5f,  0.0f, 0.0f,
        -0.5f, -0.5f, -0.5f,  0.0f, 1.0f,

        -0.5f,  0.5f, -0.5f,  0.0f, 1.0f,
         0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
         0.5f,  0.5f,  0.5f,  1.0f, 0.0f,
         0.5f,  0.5f,  0.5f,  1.0f, 0.0f,
        -0.5f,  0.5f,  0.5f,  0.0f, 0.0f,
        -0.5f,  0.5f, -0.5f,  0.0f, 1.0f
    };
    unsigned int VBO, VAO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);

    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    // position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    // texture coord attribute
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);


    // load and create a texture 
    // -------------------------
    unsigned int texture1, texture2;
    // texture 1
    // ---------
    glGenTextures(1, &texture1);
    glBindTexture(GL_TEXTURE_2D, texture1);
    // set the texture wrapping parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    // set texture filtering parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    // load image, create texture and generate mipmaps
    int width, height, nrChannels;
    stbi_set_flip_vertically_on_load(true); // tell stb_image.h to flip loaded texture's on the y-axis.
    unsigned char *data = stbi_load(FileSystem::getPath("resources/textures/container.jpg").c_str(), &width, &height, &nrChannels, 0);
    if (data)
    {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);
    }
    else
    {
        std::cout << "Failed to load texture" << std::endl;
    }
    stbi_image_free(data);
    // texture 2
    // ---------
    glGenTextures(1, &texture2);
    glBindTexture(GL_TEXTURE_2D, texture2);
    // set the texture wrapping parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    // set texture filtering parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    // load image, create texture and generate mipmaps
    data = stbi_load(FileSystem::getPath("resources/textures/awesomeface.png").c_str(), &width, &height, &nrChannels, 0);
    if (data)
    {
        // note that the awesomeface.png has transparency and thus an alpha channel, so make sure to tell OpenGL the data type is of GL_RGBA
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);
    }
    else
    {
        std::cout << "Failed to load texture" << std::endl;
    }
    stbi_image_free(data);

    // tell opengl for each sampler to which texture unit it belongs to (only has to be done once)
    // -------------------------------------------------------------------------------------------
    ourShader.use();
    ourShader.setInt("texture1", 0);
    ourShader.setInt("texture2", 1);


    // render loop
    // -----------
    while (!glfwWindowShouldClose(window))
    {
        // input
        // -----
        processInput(window);

        // render
        // ------
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // also clear the depth buffer now!

        // bind textures on corresponding texture units
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texture1);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, texture2);

        // activate shader
        ourShader.use();

        // create transformations
        glm::mat4 model         = glm::mat4(1.0f); // make sure to initialize matrix to identity matrix first
        glm::mat4 view          = glm::mat4(1.0f);
        glm::mat4 projection    = glm::mat4(1.0f);
        model = glm::rotate(model, (float)glfwGetTime(), glm::vec3(0.5f, 1.0f, 0.0f));
        view  = glm::translate(view, glm::vec3(0.0f, 0.0f, -3.0f));
        projection = glm::perspective(glm::radians(45.0f), (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 100.0f);
        // retrieve the matrix uniform locations
        unsigned int modelLoc = glGetUniformLocation(ourShader.ID, "model");
        unsigned int viewLoc  = glGetUniformLocation(ourShader.ID, "view");
        // pass them to the shaders (3 different ways)
        glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
        glUniformMatrix4fv(viewLoc, 1, GL_FALSE, &view[0][0]);
        // note: currently we set the projection matrix each frame, but since the projection matrix rarely changes it's often best practice to set it outside the main loop only once.
        ourShader.setMat4("projection", projection);

        // render box
        glBindVertexArray(VAO);
        glDrawArrays(GL_TRIANGLES, 0, 36);


        // glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
        // -------------------------------------------------------------------------------
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // optional: de-allocate all resources once they've outlived their purpose:
    // ------------------------------------------------------------------------
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);

    // glfw: terminate, clearing all previously allocated GLFW resources.
    // ------------------------------------------------------------------
    glfwTerminate();
    return 0;
}

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow *window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and 
    // height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width, height);
}
```

