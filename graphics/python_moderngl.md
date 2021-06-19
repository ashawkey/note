# moderngl

An alternative for `pyopengl`. At least it is updating.



### OpenGL version

* **OpenGL < 3.3**: Legacy. Use fixed pipeline.
* **OpenGL >= 3.3**: Modern. Use programmable pipeline (shaders).
* **OpenGL ES**: for embedded systems and **webGL**. (1.1 for fixed pipeline, 2.0 for programmable pipeline.)
* **Vulkan** (OpenGL 5, next generation GL.)



`moderngl ` supports OpenGL >= 3.3.



### install

```bash
pip install moderngl # only headless
pip install moderngl-window # for window creation
```





### troubles

version `5.6` just fails: (maybe caused by the new `glcontext`)

```bash
(torch) tang@CIS5:~/projects/nerf_pl$ python  -m moderngl
Traceback (most recent call last):
  File "/home/tang/anaconda3/envs/torch/lib/python3.6/runpy.py", line 193, in _run_module_as_main
    "__main__", mod_spec)
  File "/home/tang/anaconda3/envs/torch/lib/python3.6/runpy.py", line 85, in _run_code
    exec(code, run_globals)
  File "/home/tang/anaconda3/envs/torch/lib/python3.6/site-packages/moderngl/__main__.py", line 50, in <module>
    main()
  File "/home/tang/anaconda3/envs/torch/lib/python3.6/site-packages/moderngl/__main__.py", line 33, in main
    ctx = moderngl.create_standalone_context()
  File "/home/tang/anaconda3/envs/torch/lib/python3.6/site-packages/moderngl/context.py", line 1664, in create_standalone_context
    ctx.mglo, ctx.version_code = mgl.create_context(glversion=require, mode=mode, **settings)
SystemError: <built-in function create_context> returned NULL without setting an error

```

however, version`5.5` is OK.



### example

```python
import struct
import moderngl

ctx = moderngl.create_context(standalone=True)

program = ctx.program(
    vertex_shader="""
    #version 330

    // Output values for the shader. They end up in the buffer.
    out float value;
    out float product;

    void main() {
        // Implicit type conversion from int to float will happen here
        value = gl_VertexID;
        product = gl_VertexID * gl_VertexID;
    }
    """,
    # What out varyings to capture in our buffer!
    varyings=["value", "product"],
)

NUM_VERTICES = 10

# We always need a vertex array in order to execute a shader program.
# Our shader doesn't have any buffer inputs, so we give it an empty array.
vao = ctx.vertex_array(program, [])

# Create a buffer allocating room for 20 32 bit floats
buffer = ctx.buffer(reserve=NUM_VERTICES * 8)

# Start a transform with buffer as the destination.
# We force the vertex shader to run 10 times
vao.transform(buffer, vertices=NUM_VERTICES)

# Unpack the 20 float values from the buffer (copy from graphics memory to system memory).
# Reading from the buffer will cause a sync (the python program stalls until the shader is done)
data = struct.unpack("20f", buffer.read())
for i in range(0, 20, 2):
    print("value = {}, product = {}".format(*data[i:i+2]))
```



#### Context

the GL Context, all the operations are called in this context.

```python
# create
ctx = moderngl.create_context()

# clear the bound framebuffer
ctx.clear(red=0, green=0, blue=0, alpha=0, depth=1, ...)

# viewport
ctx.viewport = (0, 0, 640, 360)

# opengl version
ctx.version_code


```



### Primitive Modes

```python
ctx.POINTS
ctx.LINES
ctx.LINE_LOOP
Context.LINE_STRIP
Context.TRIANGLES
...
```



#### program

`shader` programs.

```python
program = ctx.program(
    vertex_shader="""
    #version 330

    out float value;
    out float product;

    void main() {
        value = gl_VertexID;
        product = gl_VertexID * gl_VertexID;
    }
    """,
    varyings=["value", "product"],
)
```



#### buffer (vbo)

OpenGL objects that stores an array in GPU allocated by the context.

can be used to store vertex, pixel, or framebuffer data.

```python
vbo = ctx.buffer(data=None, reverse=0, dynamic=False)

vbo.write(data, offset=0)
vbo.clear(size=-1, offset=0) # -1 means all
vbo.release()

vbo.size
vbo.ctx
```



#### vertex_array (vao)

describes how `buffer` is read by `shader`. 

```python
# Simple version with a single buffer
vao = ctx.vertex_array(program, buffer, "in_position", "in_normal")

vao.render(mode=None, vertices=-1) # None --> TRIANGLES
vao.release()

vao.mode
vao.program
vao.extra
```



### Texture

texture contains one or more images of the same format, can be the source access from a shader, or a render target.

```python
# create
ctx.texture(size, components, data=None, samples=0, alignment=1, dtype='f1')

# Reading pixel data into a bytearray
data = bytearray(4)
texture = ctx.texture((2, 2), 1)
texture.read_into(data)

# Reading pixel data into a buffer
data = ctx.buffer(reserve=4)
texture = ctx.texture((2, 2), 1)
texture.read_into(data)

# Write data from a moderngl Buffer
data = ctx.buffer(reserve=4)
texture = ctx.texture((2, 2), 1)
texture.write(data)

# Write data from bytes
data = b'ÿÿÿÿ'
texture = ctx.texture((2, 2), 1)
texture.write(data)

# Write to a sub-section of the texture using viewport
texture = ctx.texture((100, 100), 4)
# Fill the lower left 50x50 pixels with new data
texture.write(data, viewport=(0, 0, 50, 50))
```

