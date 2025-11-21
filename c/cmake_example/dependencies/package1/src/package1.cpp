#include <iostream>

#include <package1/package1.h>

// header-only dependency, similar usage to mylib/lib2.h
void call_package1() {
    std::cout << "[INFO] call package1" << std::endl;
}