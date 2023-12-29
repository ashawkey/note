# CUDA project layout

Project with CUDA implementation + Python bindings.

```bash
include/
	myproject/
		common.h # common functions for CUDA side
		utils.cuh # CUDA header with source
		main.h # CPP side API, must not contain/include any CUDA code.
			
src/
	main.cu # implementation for main.h, may contain/include CUDA code.	
	bindings.cpp # pybind code
	
myproject/
	__init__.py
	main.py # python side API (call the C impl)
	
setup.py # build C side and install python side
```


### Implement Function in CPP

This is much easier. 

We can handle memory allocation in python side, and only pass data pointer to C side.

* define the function API in python side `func.py`, which imports and calls backend functions.
* write a `func.h` to define the backend functions.
* implement backend functions in `func.cu`
* write a `bindings.cpp` to expose the backend.
* write `setup.py` to build everything.


### Implement Class in CPP

We need to let C side hold data structures and maybe allocate memory! This becomes harder...

For example, we'll need to convert data pointer to custom `struct` or `class`.


### Header Source Separation


### Example

