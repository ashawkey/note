# cmake

To make cross-platform compilation less painful (or maybe more painful ?)

> CMake is used to control the software compilation process using simple platform and compiler independent configuration files, and generate native makefiles and workspaces that can be used in the compiler environment of your choice.

```bash
# check version
cmake --version
```



### `CMakeLists.txt`

The rules to generate makefile. (case-sensitive)

#### Single source file

First, assume we have a cpp source file `main.cpp` to compile:

```c++
#include <iostream>

int main() {
    std::cout << "Hello, World!" << std::endl;
    return 0;
}
```

Then, we write a `CMakeLists.txt`:

```cmake
# CMakeLists ignores case.
# required version
cmake_minimum_required(VERSION 2.8)

# project name. Any string is ok.
project(Test)

# add the executable, (output file, source file)
add_executable(Test1 main.cpp) # generate ./Test1 executable
```

Run it! 

```bash
mkdir build # enter a new dir to avoid messing up.
cd build
cmake .. # generate Makefile
make # compile
```

And you will see the output files:

```bash
-rw-rw-r-- 1 tang tang 1.4K 5月  15 22:07 cmake_install.cmake
-rw-rw-r-- 1 tang tang  12K 5月  15 22:07 CMakeCache.txt
-rw-rw-r-- 1 tang tang 4.8K 5月  15 22:07 Makefile
-rwxrwxr-x 1 tang tang 8.9K 5月  15 22:07 Test
drwxrwxr-x 6 tang tang 4.0K 5月  15 22:07 CMakeFiles
```



####  Multiple source files, same directory

Assume we have:

```bash
main.cpp # in which we includes "lib"
lib.cpp
lib.h
```

The cmakelists:

```cmake
cmake_minimum_required(VERSION 2.8)
project(Test)
add_executable(Test1 main.cpp lib.cpp)
```

Another choice is to let cmake find the `lib.cpp` automatically, useful if we have lots of libs to include.

```cmake
cmake_minimum_required(VERSION 2.8)
project(Test)
aux_source_directory(. DIR_SRCS) # find all source files under '.', assign to variable DIR_SRCS
add_executable(Test1 ${DIR_SRCS}) # DIR_SRCS == main.cpp lib.cpp
```



#### Multiple source files, different directories

Assume we have:

```bash
main.cpp
libs/
  lib.cpp
  lib.h
```

where `main.cpp` includes `#include "libs/lib.h"` and use the functions defined in it.

We should use two cmakelists here:

First is at the current dir:

```cmake
cmake_minimum_required(VERSION 2.8)
project(Test)
add_subdirectory(libs) # add subdir
add_executable(Test1 main.cpp) # executable
target_link_libraries(Test1 libs) # link lib
```

Second at the `libs` dir:

```cmake
aux_source_directory(. DIR_LIB_SRCS)
add_library (libs ${DIR_LIB_SRCS}) # generate lib
```



#### gdb

just add these lines:

```cmake
set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")
```



#### find_package

automatically find the directories to includes in compiling.

```cmake
add_executable(my_bin src/my_bin.cpp)
find_package(OpenCV REQUIRED) # find <pkg>, assign to <pkg>_INCLUDE_DIRS
include_directories(${OpenCV_INCLUDE_DIRS}) # headers
target_link_libraries(my_bin, ${OpenCV_LIBS}) # libraries
```

In particular, it searches the following directories:

```
<package>_DIR
CMAKE_PREFIX_PATH
CMAKE_FRAMEWORK_PATH
CMAKE_APPBUNDLE_PATH
PATH
```

then it looks for

```
<prefix>/(lib/<arch>|lib|share)/cmake/<name>*/          (U)
<prefix>/(lib/<arch>|lib|share)/<name>*/                (U)
<prefix>/(lib/<arch>|lib|share)/<name>*/(cmake|CMake)/  (U)
```





### Checklist

```cmake
# 本CMakeLists.txt的project名称
# 会自动创建两个变量，PROJECT_SOURCE_DIR和PROJECT_NAME
# ${PROJECT_SOURCE_DIR}：本CMakeLists.txt所在的文件夹路径
# ${PROJECT_NAME}：本CMakeLists.txt的project名称
project(xxx)

# 获取路径下所有的.cpp/.c/.cc文件，并赋值给变量中
aux_source_directory(路径 变量)

# 给文件名/路径名或其他字符串起别名，用${变量}获取变量内容
set(变量 文件名/路径/...)
# e.g., set(CMAKE_CXX_STANDARD 17)

# 添加编译选项
add_definitions(编译选项)
# e.g., add_definitions(-O3 -Wall)

# 自动寻找依赖，每个包的用法都不一样
find_package(PkgName [Version] [REQUIRED])
# e.g., find_package(Eigen3 3.3 REQUIRED)

# 打印消息
message(消息)

# 编译子文件夹的CMakeLists.txt
add_subdirectory(子文件夹名称)

# 将.cpp/.c/.cc文件生成.a静态库
# 注意，库文件名称通常为libxxx.so，在这里只要写xxx即可
add_library(库文件名称 STATIC 文件)

# 将.cpp/.c/.cc文件生成可执行文件
add_executable(可执行文件名称 文件) # 自动选择这些文件中唯一的main()为入口

# 规定.h头文件路径
include_directories(路径)

# 规定.so/.a库文件路径
link_directories(路径)

# 对add_library或add_executable生成的文件进行链接操作
# 注意，库文件名称通常为libxxx.so，在这里只要写xxx即可
target_link_libraries(库文件名称/可执行文件名称 链接的库文件名称)
```



### examples

```cmake
cmake_minimum_required(VERSION 2.8)
project(asicp)

set (CMAKE_CXX_STANDARD 17)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(nanoflann)

add_definitions(-Ofast -ftree-vectorize -finline-functions -march=native -flto)

aux_source_directory(src SRCS)
add_executable(asicp ${SRCS})
target_link_libraries(asicp nanoflann::nanoflann)
target_link_libraries(asicp Eigen3::Eigen)

```

