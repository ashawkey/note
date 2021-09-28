# modern c++

### nullptr

To replace `NULL`'s use as a pointer.

> `NULL` is interpreted as `0` or `(void*) 0`, which will lead to confusion in overloaded functions:
>
> ```c++
> void f(char*);
> void f(int);
> f(NULL) // here NULL is expected as a NULL pointer, but will call f(int)
> ```
>
> `nullptr` has the type of `std::nullptr_t` and can be safely casted to any pointer type:
>
> ```c++
> f(0); // f(int)
> f(nullptr); // f(char*)
> ```





### cast

* Implicit (Automatic) conversion

  ```cpp
  int a = 1;
  float b = a;
  ```

* Explicit conversion (C-style)

  ```cpp
  int a = (int) 1.0;
  
  float* p;
  malloc((void*)p, sizeof(float) * 10);
  ```

* `static_cast`

  attempt to convert between two different data types

  ```cpp
  int a = 1;
  float b = static_cast<float>(a);
  
  // wont compile! pointer types not related (int* --> float*).
  float* b = static_cast<float*>(&a);
  ```

  

* `const_cast`

  [do not use!] change the `const`-ness of the pointer.

  ```cpp
  int a = 1;
  // not-const to const
  int* p = &a;
  const int* cp = const_cast<int*>(p);
  
  // const to not-const
  const int* cp = &a;
  int* p = const_cast<int*>(cp);
  ```

* `reinterpret_cast`

  Converts between types by reinterpreting the underlying bit pattern. powerful but use with care.

  ```cpp
  struct S1 { int a; } s1;
  struct S2 { int a; private: int b; } s2; // not standard-layout
  union U { int a; double b; } u = {0};
  int arr[2];
   
  int* p1 = reinterpret_cast<int*>(&s1); // value of p1 is "pointer to s1.a" because s1.a
                                         // and s1 are pointer-interconvertible
   
  int* p2 = reinterpret_cast<int*>(&s2); // value of p2 is unchanged by reinterpret_cast and
                                         // is "pointer to s2". 
   
  int* p3 = reinterpret_cast<int*>(&u);  // value of p3 is "pointer to u.a": u.a and u are
                                         // pointer-interconvertible
   
  double* p4 = reinterpret_cast<double*>(p3); // value of p4 is "pointer to u.b": u.a and u.b
                                              // are pointer-interconvertible because both
                                              // are pointer-interconvertible with u
   
  int* p5 = reinterpret_cast<int*>(&arr); // value of p5 is unchanged by reinterpret_cast and
                                          // is "pointer to arr"
  ```

* `dynamic_cast`

  to convert classes up, down along the inheritance hierarchy.

  ```cpp
  struct V {
      virtual void f() {}  // must be polymorphic to use runtime-checked dynamic_cast
  };
  struct A : virtual V {};
  struct B : A {};
  
  B b;
  A& a = b; // upcast (automatic)
  B& bb = a; // wont compile! 
  
  B& bb = (B&)a; // c-style downcast
  B& bb = dynamic_cast<B&>(a); // or use dynamic_cast to downcast
  
  ```

  



### constexpr

constant expression is different from constant value. 

Length of an array must be a constant expression.

```cpp
const int l = 10;
char a[l]; // Error, though reasonable

constexpr int l2 = l;
char a[l2]; // OK.

// it even supports recursion (though useless)
constexpr int fibonacci(const int n) {
    return n == 1 || n == 2 ? 1 : fibonacci(n-1) + fibonacci(n-2);
}
char a[fibonacci(5)]; // OK
```



### declare variable inside if (c++17)

```cpp
if (size_t n = v.size(); n > 10) {
    // n is only visible inside this block.
}
    
// like in go, and also in python 3.8:
if (n := len(v)) > 10:
	pass // but n will be visible since this block!
```



### initializer list

```cpp
class A {
public:
    std::vector<int> v;
    A(std::initializer_list<int> l) {
        for (int x: l) v.push_basck(x);
    }
};

A a{1,2,3,4,5};
A f() { return {1,2,3}; }
```





### tuple (c++17)

```cpp
#include <tuple>

// instanciate a tuple
std::tuple<int, int> foo_tuple()  {
  return {1, -1};  // Error until c++11:N4387
  return std::tuple<int, int>{1, -1}; // Always works
  return std::make_tuple(1, -1); // Always works
}

// access values
std::tuple<double, char, std::string> get_student(int id)
{
    if (id == 0) return std::make_tuple(3.8, 'A', "Lisa Simpson");
    if (id == 1) return std::make_tuple(2.9, 'C', "Milhouse Van Houten");
    if (id == 2) return std::make_tuple(1.7, 'D', "Ralph Wiggum");
    throw std::invalid_argument("id");
}
 
int main()
{
    // via get<i>(tuple)
    auto student0 = get_student(0);
    std::cout << "ID: 0, "
              << "GPA: " << std::get<0>(student0) << ", "
              << "grade: " << std::get<1>(student0) << ", "
              << "name: " << std::get<2>(student0) << '\n';
 	
    // via tie
    double gpa1;
    char grade1;
    std::string name1;
    std::tie(gpa1, grade1, name1) = get_student(1);
    std::cout << "ID: 1, "
              << "GPA: " << gpa1 << ", "
              << "grade: " << grade1 << ", "
              << "name: " << name1 << '\n';
 
    // via C++17 structured binding: (recommend!)
    auto [gpa2, grade2, name2] = get_student(2);
    std::cout << "ID: 2, "
              << "GPA: " << gpa2 << ", "
              << "grade: " << grade2 << ", "
              << "name: " << name2 << '\n';
}
```

Tricks: `struct` as a tuple.

```c++
struct S { int a, b, c; };

S func(int _a, int _b, int c_) {
    return (S) {_a, _b, _c}; // c++11 list initialization (https://en.cppreference.com/w/cpp/language/list_initialization)
}

S s = func(0, 1, 2);

cout << s.a << endl;
```



### std::optional (c++17)

Any instance of `optional<T>` at any given point in time either *contains a value* or *does not contain a value*.

```c++
#include <string>
#include <functional>
#include <iostream>
#include <optional>
 
// optional can be used as the return type of a factory that may fail
std::optional<std::string> create(bool b) {
    if (b)
        return "Godzilla";
    return {};
}
 
// std::nullopt can be used to create any (empty) std::optional
auto create2(bool b) {
    return b ? std::optional<std::string>{"Godzilla"} : std::nullopt;
}
 
// std::reference_wrapper may be used to return a reference
auto create_ref(bool b) {
    static std::string value = "Godzilla";
    return b ? std::optional<std::reference_wrapper<std::string>>{value}
             : std::nullopt;
}
 
int main()
{
    std::cout << "create(false) returned "
              << create(false).value_or("empty") << '\n';
 
    // optional-returning factory functions are usable as conditions of while and if
    if (auto str = create2(true)) {
        std::cout << "create2(true) returned " << *str << '\n';
    }
 
    if (auto str = create_ref(true)) {
        // using get() to access the reference_wrapper's value
        std::cout << "create_ref(true) returned " << str->get() << '\n';
        str->get() = "Mothra";
        std::cout << "modifying it changed it to " << str->get() << '\n';
    }
}

/*
create(false) returned empty
create2(true) returned Godzilla
create_ref(true) returned Godzilla
modifying it changed it to Mothra
*/
```



### lambda function (c++11)

Lambdas can be used as an anonymous in-place function, or a nested function.

```cpp
#include <iostream>

using namespace std;

// value catching
int main() {
    int a = 0;
    
    // do not catch any local variables
    auto foo = []() { cout << a << endl; }; // error, a not defined.
    
    // catch all local variables, pass by value
    auto foo = [=]() { a = 1; cout << a << endl; }; // 1
    cout << a << endl; // still 0
    
    // catch all local variables, pass by reference
    auto foo = [&]() { a = 1; cout << a << endl; }; // 1
    cout << a << endl; // also 1
    
    // catch spesific variables
    auto foo = [&a]() {}; // only catch a
}
```

Recursive lambdas: must define the type explicitly!

```cpp
#include <iostream>
#include <functional> // to use std::function

using namespace std;

int main() {
    
    // examples to use function
    function<void()> foo = []() {};
    function<bool(int, int)> foo = [](int a, int b) {};
    
    // recursive lambda
    function<bool(int)> dfs = [&](int x) {
        for (int y: graph[x]) {
            dfs(y);
        }
    }
    
    dfs(0);
    
}
```

