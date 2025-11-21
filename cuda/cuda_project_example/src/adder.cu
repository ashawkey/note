#include <adder/common.h>
#include <adder/adder.h>

namespace adder {

// impl the virtual class
class AdderImpl : public Adder {
public:
    AdderImpl() : Adder() {}
    
    // impl the virtual function
    int call(int x, int y) {
        // call the function defined in common.h

        return add(x, y);
    }

};

Adder* create_adder() {
    return new AdderImpl();
}

}
