## cmake

To make cross-platform compilation less painful (or maybe more painful ?)

> CMake is used to control the software compilation process using simple platform and compiler independent configuration files, and generate native makefiles and workspaces that can be used in the compiler environment of your choice.

```bash
# check version
cmake --version
```

### `CMakeLists.txt`

The rules to generate makefile.
* commands are not case-sensitive (e.g., `project()` or `PROJECT()`) for historical reason.
* variables are case-sensitive! (e.g., must be `PROJECT_NAME`)
* paths can be string or raw (e.g., `include` or `"include"`)
* command parameters are separated by spaces or line breaks.

### Project structure

A regular C/C++/CUDA project layout:
```bash
Readme.md
CMakeLists.txt
./include
    mylib.h
./src
    mylib.cpp
    main.cpp # it #include "mylib.h"
```

### Build with cmake

Traditional way:
```bash
mkdir build
cd build
cmake ..
make -j8
./binary
```

More modern way:
```bash
cmake -B build . # mkdir build && cd build && cmake ..
cmake --build build -j 8 # invoke make
./build/binary

cmake --build build --config Release -j 8 # set build type
```

### Built in variables

Like shell variables, use with `${CAPITAL_NAME}`
```CMake
PROJECT_NAME # project name defined in project()
PROJECT_SOURCE_DIR # the source dir, usually ./
PROJECT_BINARY_DIR # the target dir, usually ./build/

# define your variables
set(SOURCES
    src/main.cpp
    src/test.cpp
)
```

### Basic example

To include  `mylib`, we can use `target_include_directories` for our target:
```cmake
cmake_minimum_required(VERSION 3.5)
project(test_example) # project name

add_executable(test # output binary name
    # all the sources
    src/main.cpp
    src/mylib.cpp
)

target_include_directories(test PUBLIC # target + qualifier (use public as default)
    include # dir to include
)
```

> what is a `target` in `target_*()`:
> (1) executable by `add_executable()`
> (2) library by `add_library()`, sometimes we need to link library to library.

We can also use global `include_directories` if these headers are globally needed:
```cmake
cmake_minimum_required(VERSION 3.5)
project(test_example) # project name

include_directories(include)

add_executable(test # output binary name
    # all the sources
    src/main.cpp
    src/mylib.cpp
)

```

`include_directories()` equals to the `-I` option in `gcc`, denoting the folders to search for headers. 
It supports nested folders, e.g., the following file structure will still work: 
```bash
./include
    ./mylib
        mylib.h
./src
    main.cpp # change to #include "mylib/mylib.h"
```

We can also (1) create a static library (2) link it to the executable:
```cmake
cmake_minimum_required(VERSION 3.5)
project(test_example) # project name

include_directories(include)

add_library(mylib STATIC # build a static lib, called libmylib.a on linux, or libmylib.lib on windows
    # move all the sources here
    src/mylib.cpp
)

add_executable(test src/main.cpp) # only add the main.cpp

target_link_libraries(test PUBLIC mylib) # link mylib to the executable
```

We can also build a dynamic/shared library by:
```cmake
add_library(mylib SHARED # named libmylib.so on linux , or libmylib.dll on windows
    src/mylib.cpp
)
```

> Difference between static and dynamic libraries:
> (1) static lib will be copied to the binary, leading to a much larger binary size (both in storage and memory cost), but no runtime external dependency.
> (2) dynamic lib is only referenced by the binary, and called in runtime as an external dependency (the dynamic loading from OS will lead to a small additional time cost too).
> (3) In most cases, dynamic lib is preferred. 

For Windows MSVC, we need to add additional things for shared lib to work:
```cmake
if (MSVC)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
    set(BUILD_SHARED_LIBS TRUE)
endif()
```

### Configurations

#### Build types

Built-in build types and their equal command:
* Release: `-O3 -DNDEBUG`
* Debug: `-g`
* RelWithDebInfo: `-O2 -g -DNDEBUG`, 

Explicitly use it by `cmake .. -DCMAKE_BUILD_TYPE=Release`.
We can also detect and set default build type:
```cmake
# Set a default configuration if none was specified
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "No release type specified. Setting to 'Release'.")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo")
endif()
```

#### Compile flags

Set default (global) c++ flags:
```cmake
# set c++ compile flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC") # usually this is enough
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC" CACHE STRING "doc string" FORCE) # force set in CMakeCache.txt... seems not very needed.

# set c++ standard
set(CMAKE_CXX_STANDARD 14)
```
We can also set it with command line:
```bash
cmake .. -DCMAKE_CXX_FLAGS="-fPIC"
```

Set per-target c++ flags:
```cmake

target_compile_definitions(test PUBLIC
    fPIC
)
```

#### Thrid party dependency

`find_package()` can handle about 142 common third party libraries, to automatically handle it for your project.
```cmake
# find and include boost 1.46.1; error out if not found; only find the following components
find_package(Boost 1.46.1 REQUIRED COMPONENTS filesystem system)

# check if the package is found
if(Boost_FOUND)
    message ("Boost found")
    include_directories(${Boost_INCLUDE_DIRS}) # include headers.
else()
    message (FATAL_ERROR "Cannot find Boost")
endif()

# your binary
add_executable(test src/main.cpp)

# link library
target_link_libraries(test PRIVATE Boost::filesystem) # alias target, equals to ${Boost_FILESYSTEM_LIBRARY}
```

#### Build with ninja
> what is `ninja`?
> A small build system (i.e., alternative to `make`) focusing on speed.

Cmake can generate ninja config too:
```bash
cmake -B build -G Ninja . # generate ./build/build.ninja
ninja -C build # build in ./build
```

### Sub-projects

Let say your projects include some sub-projects that contain individual `CMakeLists.txt`:
```bash
Readme.md
CMakeLists.txt
# these dependencies are sub-projedts
./dependencies
    ./package1 # a normal package
        CMakeLists.txt
        ./include
            lib1.h
        ./src
            lib1.cpp
    ./package2 # a header-only package
        CMakeLists.txt
        ./include
            lib2.h
    ...
./include
    mylib.h
./src
    mylib.cpp
    main.cpp # it #include "mylib.h", #include "package1/lib1.h"
```

To use these packages in your binary, use:
```cmake
cmake_minimum_required(VERSION 3.5)
project(test_example) # project name

include_directories(include)

# regular dependency: first process their CMakeLists.txt
add_subdirectory(dependencies/package1)

# header-only dependency: just include it
include_directories(dependencies/package2)

add_library(mylib STATIC # build a static lib, called libmylib.a on linux, or libmylib.lib on windows
    # move all the sources here
    src/mylib.cpp
)

add_executable(test src/main.cpp) # only add the main.cpp

target_link_libraries(test PUBLIC 
    package1 # link package1
    mylib 
)
```


### Misc
* Should I use `#include <mylib/mylib.h>` or `#include "mylib/mylib.h"`?
    * If you use CMakelists.txt to manage the include directories, `<>` is the only thing you need.
    * `<>` only searches the include dirs, while `""` searches the current directory first (e.g., when `.h` are in the same dir as `.cpp`), then the include dirs.
