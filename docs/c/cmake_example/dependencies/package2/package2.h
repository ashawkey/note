#pragma once

#include <iostream>

// header-only dependency, similar usage to mylib/lib2.h
void call_package2() {
    std::cout << "[INFO] call package2" << std::endl;
}