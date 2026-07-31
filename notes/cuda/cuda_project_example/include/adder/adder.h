#include <cuda_runtime.h>

namespace adder {

// abstract class API
class Adder {
public:
    Adder() {}
    virtual ~Adder() {}

    virtual int call(int x, int y) = 0;
};

Adder* create_adder();

}

