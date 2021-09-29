# right-value reference

### left-value, right-value

* `lvalue`: have an address (can use `&` operator)
* `rvalue`: no address, a temporary value  (cannot use `&` operator)

```cpp
// a is left, 5 is right
int a = 5;

struct A {};
// a is left, A() is right
A a = A();
```

### left-value reference

The most common reference we use.

Mostly used in lieu of pointer to make code cleaner.

```cpp
int a = 5;
int& r1 = a; // OK! left-value can be referenced.
int& r2 = 5; // Error! right-value cannot be referenced.

const int& r2 = 5; // OK! right-value can be const referenced. This is necessary in cases such as vector::push_back(const T& val), we want both v.push_back(complex_obj) and v.push_back(1) work.
```

### right-value reference

`rvalue` can be further categorized:

* `prvalue`, pure-right-value: the initial value, or expression, or function returns.
* `xvalue`, expiring value: a temporary value instantiated from a `prvalue`.

```cpp
int&& r1 = 5; // OK! standard right-value reference.
r1 = 6; // can be modified (not a const).

int a = 5;
int&& r2 = a; // Error! left-value cannot be right-referenced.
int&& r3 = std::move(a); // OK! std::move cast left value to right value.
int&& r4 = static_cast<int&&>(a); // ditto.

// note &a == &r2 == &r3 == &r4
```

In fact, right-value reference is just a wrapper of left-value reference with a temp variable.

```cpp
int &&r = 5;
// eqauls
int tmp = 5;
int &&r = std::move(tmp);
```

right-value can also be used to pass reference:

```cpp
void f(int& x) { x++; }
void g(int&& x) { x++; } 

int main() {
    int x = 1;
    f(x); // x is now 2
    g(std::move(x)); // x is now 3
    
    // we prefer to use f(x) here, since this is not a useful application of rvalue reference.
}
```

**右值引用既可以是左值也可以是右值，如果有名称则为左值，否则是右值**。

```cpp
int a = 5;
std::move(a); // no name, a right value.
int&& r = std::move(a); // with name, so r is a left value.

void f(int&& r) {}
void g1(int&& r) { f(r); } // wont compile.
void g2(int&& r) { f(std::move(r)); }

f(r); // Error! r is a left value.
f(std::move(r)); // OK! 
f(std::move(a)); // OK! equals last line.

// Once passed in a function and got named, the right value is no longer a right value.
g1(std::move(r)); // Error! (no matching function to call f) 
g2(std::move(r)); // OK!
```

More Examples:

```cpp
// 形参是个右值引用
void change(int&& right_value) {
    right_value = 8;
}
 
int main() {
    int a = 5; // a是个左值
    int &ref_a_left = a; // ref_a_left是个左值引用，ref_a_left本身也是左值
    int &&ref_a_right = std::move(a); // ref_a_right是个右值引用，但ref_a_right本身也是左值
 
    change(a); // 编译不过，a是左值，change参数要求右值
    change(ref_a_left); // 编译不过，左值引用ref_a_left本身也是个左值
    change(ref_a_right); // 编译不过，右值引用ref_a_right本身也是个左值
     
    change(std::move(a)); // 编译通过
    change(std::move(ref_a_right)); // 编译通过
    change(std::move(ref_a_left)); // 编译通过
 
    change(5); // 当然可以直接接右值，编译通过
     
    cout << &a << ' ';
    cout << &ref_a_left << ' ';
    cout << &ref_a_right;
    // 打印这三个左值的地址，都是一样的
}
```



### Applications

右值引用主要在函数传参时**用移动代替拷贝**，即**需要拷贝但被拷贝者之后又不再需要**的场景，例如临时变量。因为`std::move`只改变地址指向，而不会物理上的移动（即拷贝）数据。

需要注意的是，类的默认移动构造函数（`A(A&& a){}`）就是默认拷贝构造函数（`A(A& a){}`）。为了使得自己定义的类能够利用移动语义，我们需要自己定义移动构造函数：

```cpp
// ref: https://stackoverflow.com/questions/11572669/move-with-vectorpush-back
// ref: https://stackoverflow.com/questions/3106110/what-is-move-semantics

class mystring {
    char* data;
// the big THREE to manage raw pointers: destructor, copy constructor, copy assignment operator.    
// from c++11, we have the big FIVE, plus: move constructor, move assignment operator.
// with the copy-and-swap idiom, we only need one assignment operator, so it's also called big 4.5
public: 
    // default constructor
    mystring(const char* p) {
        size_t s = std::strlen(p) + 1;
        data = new char[s];
        std::memcpy(data, p, s);
    }
    // destructor
    ~mystring() {
        delete[] data;
    }
    // copy constructor (deep copy)
    mystring(const mystring& that) {
        size_t s = std::strlen(that.data) + 1;
        data = new char[s];
        std::memcpy(data, that.data, s);
    }
    // move constructor (no copy)
    mystring(mystring&& that) {
        data = that.data;
        that.data = nullptr;
    }
    // (copy and move) assignment operator
    // ref: https://stackoverflow.com/questions/3279543/what-is-the-copy-and-swap-idiom
    // we can use copy-and-swap idiom to avoid duplicated code and self-assignment test.
    // this handles both copy and move, based on whether that is lvalue(copy) or rvalue(move).
    // e.g., assume x, y are both mystring.
    // mystring s = x; // copy, since x is lvalue. the copy constructor is called.
    // mystring s = x + y; // move, since x + y is rvalue (assume we overload operator+). first create a temp object, then the move constructor is called.
    mystring& operator=(mystring that) {
        std::swap(data, that.data);
        return *this;
    }
}
```

STL容器大多实现了各种方法的移动语义，例如

```cpp
// just an illustration
vector(vector&& tmp_vector) {
    data = tmp_vector.data;
    tmp_vector.data = nullptr;
}

void push_back(const T& x) {/* copy semantic */}
void push_back(T&& x) {/* move semantic */}
```

Use case:

```cpp
vector<mystring> v;
mystring s("a_long_string");
v.push_back(s); // copied
v.push_back(move(s)); // moved
v.push_back(mystring("blahblah")); // moved

struct A {
    /* naive flat class, no raw pointer, no move constructor */
    int x;
    A(int _x) : x(_x) {}
};

vector<A> v;
A a(1);
v.push_back(a); // traditional. a is copied. slow!
v.push_back(move(a)); // a is still copied, no difference from the last line.
v.push_back(A(1)); // A(1) is still copied.
```



> 其它语言中的类似情况：
>
> c++默认的对象传值方式是拷贝，例如`vec.push_back(obj); auto vec2 = vec1;`等都会触发拷贝行为。引用传值必须通过`&`显式显式实现，例如`vec.push_back(move(obj)); auto vec2 = &vec1;`
>
> python默认的对象传值方式就是引用，例如`l.append(obj), obj2 = obj1`均为引用。相反，如果需要拷贝，则要通过`deepcopy`显式实现。



### 通用引用 Universal Reference

表现形式为**T && (模板+右值引用)**，利用编译器对模板的推导多义性，可以同时绑定左值与右值。

**引用折叠**：编译器特性，实际语法不存在引用的引用`int& &`，但模板类型推导支持对引用的引用进行折叠，具体规则为：

* `T& &, T& &&, T&& & --> T&`
* `T&& && --> T&&`

利用此特性，通用引用得以实现：

```cpp
template <typename T> void f(T&& x) {}

int x = 1;
f(x); // x is lvalue, T = int won't match, so try T = int&, and use f(int& x) which matches!
f(1); // 1 is rvalue, T = int matches, so use f(int&& x).
```

事实上，`move`的实现就利用了通用引用：

```cpp
template <typename T> 
typename remove_reference<T>::type&& move(T&& t) {
    return static_cast<typename remove_reference<T>::type&&>(t);
}
```



### 完美转发问题

完美转发指的是在函数内部将参数**包括类型原封不动**的传递给内部的另一函数。但由于右值引用传入函数后会变成左值，简单的参数转发是错误的！

```cpp
// inner function
void f(int& x) { x++; }
void g(int&& x) { x++; }

// sender
template<typename F, typename T>
void sender(F f, T t) { f(t); }

int x = 1;
f(x); // x is now 2.
sender(f, x); // not work, x is still 2.
sender(g, x); // wont compile, T = int, t is a lvalue
sender(g, move(x)); // wont compile, T = int&&, but t is a lvalue
```

通用引用只能解决`f(int& x)`的问题：

```cpp
// inner function
void f(int& x) { x++; }
void g(int&& x) { x++; }

// sender
template<typename F, typename T>
void sender(F f, T&& t) { f(t); }

int x = 1;
f(x); // x is now 2.
sender(f, x); // x is now 3. T = int&
sender(g, x); // wont compile, T = int&, but t is still a lvalue
sender(g, move(x)); // wont compile, T = int, but t is still a lvalue
```

从而，我们需要动态判断应该转发左值还是右值。仅仅通过move是不够的，所以我们要使用`std::forward<T>(u)`实现左右值转化：

```cpp
// impl for lvalue
template <typename T> 
T&& forward(typename remove_reference<T>::type& t) noexept {
    return static_cast<T&&>(t);
}
// impl for rvalue
template <typename T> 
T&& forward(typename remove_reference<T>::type&& t) noexept {
    return static_cast<T&&>(t);
}

// examples
std::forward<int>(x); // cast to right-value, equals `std::move(x)`
std::forward<int&&>(x); // also cast to right-value, equals `std::move(x)`

std::forward<int&>(x); // cast to left-value
```

More Examples：

```cpp
#include <iostream>

void change2(int&& ref_r) {}
 
void change3(int& ref_l) {}

void change4(const int& ref) {}
 
void change(int&& ref_r) {

    change2(5);
    // change2(ref_r);
    change2(std::move(ref_r)); 
    change2(std::forward<int>(ref_r));
    change2(std::forward<int &&>(ref_r));  
     
    // change3(5);
    change3(ref_r); 
    change3(std::forward<int &>(ref_r)); 

    change4(5);
	change4(ref_r); 
    change4(std::forward<int &>(ref_r)); 
}
 
int main() {
    int a = 5;
    change(std::move(a));
}
```

完美转发的实现（结合forward与通用引用）：

```cpp
// inner function
void f(int& x) { x++; }
void g(int&& x) { x++; }

// sender
template<typename F, typename T>
void sender(F f, T&& t) { f(forward<T>(t)); }

int x = 1;
f(x); // x is now 2.
sender(f, x); // x is now 3. T = int&, forward cast t to lvalue.
sender(g, move(x)); // x is now 4, T = int, forward cast t to rvalue.

sender(g, x); // wont compile, T = int&, forward cast t to lvalue.
```

