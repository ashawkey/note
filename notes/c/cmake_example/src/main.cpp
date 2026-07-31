#include <iostream>

#include <package1/package1.h>
#include <package2/package2.h>

#include <mylib/lib1.h>
#include <mylib/lib2.h>

int main() {
    std::cout << "[INFO] start"  << std::endl;

    // call dependency functions
    call_package1();
    call_package2();

    // call mylib functions
    call_lib1();
    call_lib2();
    call_lib2_from_lib1();

    return 0;
}