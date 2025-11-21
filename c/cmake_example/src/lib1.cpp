#include <iostream>

// include another custom lib
#include <mylib/lib2.h>

void call_lib1() {
    std::cout << "[INFO] call lib1" << std::endl;
}

void call_lib2_from_lib1() {
    std::cout << "[INFO] call lib2 from lib1" << std::endl;
    call_lib2();
}