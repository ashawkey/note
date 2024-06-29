cmake_minimum_required(VERSION 3.5)

project(example)

# select build type
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	message(STATUS "No release type specified. Setting to 'Release'.")
	set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
	set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo")
endif()

# for MSVC shared lib
if(MSVC)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
    set(BUILD_SHARED_LIBS TRUE)
endif()

# include mylib
include_directories(include)

# include dependencies
include_directories(dependencies)

# add sub-project
add_subdirectory(dependencies/package1)
include_directories(dependencies/package1/include)

# build library for lib1
add_library(mylib SHARED
    src/lib1.cpp
)

# main exe
add_executable(test src/main.cpp)

# link lib1
target_link_libraries(test PUBLIC mylib)

# link package1
target_link_libraries(test PUBLIC package1)