# C++



## Tutorial

### C Library

```c++
#include <cstdio>
#include <cstdlib>
#include <cstring> // memset
#include <cmath>
```



### C++ Library

#### #include \<iostream\>

```c++
#include <iostream>
 
using namespace std;
 
int main() {
   char name[50];

   cin >> name;
   cout << "Name: " << name << endl;
 
}
```

#### #include \<string\>

```c++
#include <iostream>
#include <string>
 
using namespace std;
 
int main ()
{
   string str1 = "Hello";
   string str2 = "World";
   string str3;
 
   str3 = str1 + str2;

   int len = str3.size();
}
```

#### #include \<vector\>

```c++
#include <iostream>
#include <vector>
using namespace std;
 
int main()
{
   vector<int> vec; 
 
   for(int i = 0; i < 5; i++){
      vec.push_back(i);
   }
    
   cout << vec.size() << endl;
 
   for(i = 0; i < 5; i++){
      cout << i << " = " << vec[i] << endl;
   }
 
   vector<int>::iterator v = vec.begin();
   while(v != vec.end()) {
      cout << *v << endl;
      v++;
   }
 
   return 0;
}
```



### Class

```c++
class A {
// access modifier (default is private)
public:
    // constructor
    A() {} // default
    A(int _a) { a = _a; } 
    A(int _a): a(_a) {} // init-list
    
    A(const A& other) { if (this != &other) *this = other; } // default copy constructor
    
    // operator overload
    A& operator=(const A& other) {
        a = other.a;
        return *this;
    }
    
    // destructor
    ~ A() {}
    
    // member
    int a;
    
    // method
    void setA(int _a) { a = _a; }
}

A a;
A a(0);

printf("%d", a.a);
a.setA(1);
```

Inherit

```c++
class B: public A {
public:
    void printA() { printf("%d", a); }
}
```



### Template

#### Function

```c++
template <typename T>
T const& max(T const& a, T const& b) {
    return a < b ? b : a;
}
```

#### Class

```c++
template <typename T>
class A {
public:
    vector<T> a;
    void push(T const& x) { a.push_back(x); }
    T top() const { return a.back(); }
}
```



### new & delete

```c++
char* p = new double;
delete p;

char* s = NULL;
s = new char[20];
delete [] s;
```



### Reference

```c++
int i = 0;
int & j = i; // j is an alias of i
```

ref as function return value:

```c++
double vals[] = {10.1, 12.6, 33.1, 24.1, 50.0};
double& setValues(int i) { return vals[i]; }
setValues(0) = 0; // okay
```

ref as function args:

```c++
void swap(int& x, int& y) {
    int tmp = x; x = y; y = tmp;
}
```

