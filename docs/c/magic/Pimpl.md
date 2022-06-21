## Pointer to implementation
reference: https://en.cppreference.com/w/cpp/language/pimpl

A technique that **put the implementation of a class in a separate class**, to **construct stable ABI and reduce compile-time dependencies**.
The form is:
```cpp
// interface (widget.h)
class widget {
private:
	struct impl;
	std::experimental::propagate_const<std::unique_ptr<impl>> pImpl;
public:
	//...
};

// implementation (widget.cpp)
struct widget::impl {
	//...
};
```


### example
```cpp
// ----------------------
// interface (widget.hpp)
#include <iostream>
#include <memory>
#include <experimental/propagate_const>
 
class widget
{
    class impl;
    [std::experimental::propagate_const](http://en.cppreference.com/w/cpp/experimental/propagate_const)<[std::unique_ptr](http://en.cppreference.com/w/cpp/memory/unique_ptr)<impl>> pImpl;
public:
    void draw() const; // public API that will be forwarded to the implementation
    void draw();
    bool shown() const { return true; } // public API that implementation has to call
 
    widget(); // even the default ctor needs to be defined in the implementation file
              // Note: calling draw() on default constructed object is UB
    explicit widget(int);
    ~widget(); // defined in the implementation file, where impl is a complete type
    widget(widget&&); // defined in the implementation file
                      // Note: calling draw() on moved-from object is UB
    widget(const widget&) = delete;
    widget& operator=(widget&&); // defined in the implementation file
    widget& operator=(const widget&) = delete;
};
 
// ---------------------------
// implementation (widget.cpp)
#include "widget.hpp"
 
class widget::impl
{
    int n; // private data
public:
    void draw(const widget& w) const
    {
        if(w.shown()) // this call to public member function requires the back-reference 
            [std::cout](http://en.cppreference.com/w/cpp/io/cout) << "drawing a const widget " << n << '\n';
    }
 
    void draw(const widget& w)
    {
        if(w.shown())
            [std::cout](http://en.cppreference.com/w/cpp/io/cout) << "drawing a non-const widget " << n << '\n';
    }
 
    impl(int n) : n(n) {}
};
 
void widget::draw() const { pImpl->draw(*this); }
void widget::draw() { pImpl->draw(*this); }
widget::widget() = default;
widget::widget(int n) : pImpl{[std::make_unique](http://en.cppreference.com/w/cpp/memory/unique_ptr/make_unique)<impl>(n)} {}
widget::widget(widget&&) = default;
widget::~widget() = default;
widget& widget::operator=(widget&&) = default;
 
// ---------------
// user (main.cpp)
#include "widget.hpp"
 
int main()
{
    widget w(7);
    const widget w2(8);
    w.draw(); // drawing a non-const widget 7
    w2.draw(); // drawing a const widget 8
}
```