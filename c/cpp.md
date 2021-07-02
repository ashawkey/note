# C++

old name is C with Classes.

Contains 4 parts, each designed with different philosophy.

* C

* Object-oriented programming

* Template (generic programming)

* STL (standard library)

  

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
    s = s + 'c';
   
    int len = str3.size(); // or str3.length()
    
	int idx = str3.find("pattern"); // -1 (string::npos) if not found
    
    char* cstr3 = str3.c_str();
    
    // erase
    s.erase(s.begin() + i); // erase one char
    s.erase(start, num); // erase from start for num chars
    
    // string to int
    // int stoi(string& s, size_t* idx=0, int base=10)
    string s = "123";
    int i = stoi(s); // 123
    
    string s = "123 xyz";
    size_t sz;
    int i = stoi(s, &sz); // 123, sz = 3
    
    string s = "-10010110001";
    int b = stoi(s, nullptr, 2); // -1201, binary.
    
    // string to float
    string s = "12.34";
    float f = stof(s); // 12.34
    
    // similarly, we have `stol, stoll, stod, stold, stoul, stoull`
        
    // int/float/double to string
    string s = to_string(42); // "42"
    string s = to_string(12.23); // "12.23"
    string s = to_string(1e40); // "10000000000000000303786028427003666890752.000000"
}
```

#### #include \<vector\>

```c++
#include <iostream>
#include <vector>
using namespace std;
 
int main() {
    // init
    vector<int> vec; 
    vector<int> vec = {1, 2, 3};
    vector<int> vec(size[, init_value]);
    
    // 2d init: [N, M], -inf
    vector<vector<int>> dp(N, vector<int>(M, -0x3f3f3f3f));
    
 	// push_back
    for(int i = 0; i < 5; i++){
       vec.push_back(i);
    }
    
    // size
    cout << vec.size() << endl;
 	
    // access by index
    for(i = 0; i < 5; i++){
       cout << i << " = " << vec[i] << endl;
    }
 	
    // iterator
    vector<int>::iterator v = vec.begin();
    while(v != vec.end()) {
       cout << *v << endl;
       v++;
    }
    
    // erase i-th element
    v.erase(v.begin() + i);
    
    // copy another vector
    vector<int> v2(v1);
    
    // insert x at i-th position
    v.insert(v.begin() + i, x);
    
    // extend
    v1.insert(v1.end(), v2.begin(), v2.end());
 
    return 0;
}
```

#### #include \<set\>

set implements a binary search tree, so querying existence and order is in `O(log n)`

```c++
#include <iostream>
#include <set>
using namespace std;
 
int main() {
    set<int> s;
    
    // insert element
    s.insert(1);
    
    // count element, can check if an element is in the set.
    cout << s.count(1) << endl;
    bool is_in = s.count(1);
    
    // find the first element, another way to check is in. (faster than count!)
    bool is_in2 = s.find(1) != s.end();
    auto it = s.find(1); // return iterator to the first found element.
	
    // [c++20] contains
    bool is_in3 = s.contains(1);
    
    // delete by value
    s.erase(2); // erase(value)
    
    // size
    cout << s.size() << endl;
    
    // clear
    s.clear();
    
    // is empty
    if (s.empty()) {
        //...
    }
    
    // loop all elements
    for (auto it = s.begin(); it != s.end(); it++) {
        cout << *it << endl;
    }
    
    // custom cmp function
    
}
```

#### #include \<map\>

map implements binary search tree to store keys.

```c++
#include <iostream>
#include <map>
using namespace std;
 
int main() {
    // init
    map<int, string> m;
    map<int, int> n = {{0:1}, {1:2}};
    // insert
    m[0] = "zero"; // m[key] = val
    m.insert(pair<int, string>(1, "one")); // m.insert(pair)
    
    // get
	cout << m[-1] << endl; // if not exist, will insert & initialize it ! (here int --> 0)
    cout << m.at(-1) << endl; // check if exist first, throw an error
    
    // size
    m.size();
    m.empty();
    m.clear();
    
    // iterator
    // auto == std::map<int,string>::iterator == pair<int,string>
    for (auto it = m.begin(); it != m.end(); it++) {
        cout << it->first << ": " << it->second << endl;
    }
    for (auto& it : m) {
        cout << it.first << ": " << it.second << endl;
    }
    
    // delete
    m.erase(0); // erase(key)
    // find
    auto it = m.find(2);
    if (it != m.end()) {
        m.erase(it);
    }
    
    // check if a key is in map
	bool is_in = m.find(0) != m.end();
    bool is_in2 = m.count(0); // also work in map
    
}
```

#### #include \<unordered_set\>

unordered_set implements a hash table, thus faster than ordered set.

API is the same with set.

```c++
#include <iostream>
#include <unordered_set>
using namespace std;
 
int main() {
    unordered_set<int> s;
    s.insert(1);
    s.erase(1);
    s.empty();
    s.clear();
    s.find(1);
    s.count(1);
    
    // regular use to check duplicate
    unordered_set<int> vis;
    while(...) {
    	if (vis.count(x)) return true;
		else vis.insert(x);
    }
    return false;
}
```

#### #include \<unordered_map\>

unordered_map implements a hash table. 

API is the same with map.

#### #include \<queue\>

```c++
#include <iostream>
#include <queue>
using namespace std;

int main() {
    queue<int> q;
    q.push(1);
    while (!q.empty()) {
        int x = q.pop(); // access and remove
		int y = q.front(); // just access
        int z = q.back(); // just access
        cout << q.size() << endl;
    }
}
```

#### #include \<stack\>

```c++
#include <iostream>
#include <stack>
using namespace std;

int main() {
    stack<int> s;
    s.push(2);
	int x = s.pop();
    s.empty();
}
```

#### #include \<priority_queue\>

```c++
#include <iostream>
#include <priority_queue>
using namespace std;

// <表示大的在前。
struct cmp {
    bool operator()(int a, int b) {
        return a < b;
    }
};

int main() {
    // heap, large-first
    priority_queue<int> pq;
    pq.push(1);
    pq.push(2);
    pq.push(0);
    p.top(); // 2
    
    // smaller-first
    priority_queue<int, vector<int>, greater<int>> pq;
    
    
    // custom cmp function
	set<int, cmp> s;
	priority_queue<int, vector<int>, cmp> pq;
}
```

#### #include \<algorithm\>

```c++
//// quick sort
vector<int> v = {0, 2, 1, 3};
sort(v.begin(), v.end()); // small -> large

// reverse sort
sort(v.begin(), v.end(), greater<int>());  // large -> small, c++14
// or in two steps
sort(v.begin(), v.end());
reverse(v.begin(), v.end()); // large -> small


int v[4] = {4, 2, 3, 1};
sort(v, v + 4);

// sort with custom funciton / object.
bool cmp(int i, int j) {return i < j;} // True -> [i, j]
struct cmp {
    bool operator < (int i, int j) {
        return i < j;
    }
}
sort(v, v + 4, cmp); // same as default, small -> large
    
//// merge sort
stable_sort(v, v + 4);

//// swap
swap(x, y);

//// min/max_element
cout << *min_element(v, v+4) << " ~ " << *max_element(v, v+4) << endl; // return pointer/iterator

//// nth_element, O(n) quick sort
nth_element(v, v+2, v+4); // rearrange v[2] as the second element if ordered.
cout << v[2] << endl;

//// prev/next_permutation
int xs[3] = {1, 2, 3};
do {
    for (int i=0; i<3; i++) cout << xs[i] << " ";
    cout << endl;
} while (next_permutation(xs, xs+3));

//// unique
auto end = unique(v.begin(), v.end());
int new_size = end - v.begin();
v.erase(end, v.end());

int new_size = unique(v, v+4) - v;

//// lower/upper_bound
// check out of bound conditions!
int l = lower_bound(v.begin(), v.end(), 10) - v.begin();
int r = upper_bound(v.begin(), v.end(), 10) - v.begin();

//// reverse
string s = "string";
reverse(s.begin(), s.end()); // inplace
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
};

A a;
A a(0);

printf("%d", a.a);
a.setA(1);
```

#### Inheritance

```c++
class B: public A {
public:
    void printA() { printf("%d", a); } // member a is inherited from class A
};
```

#### operator overload

```c++
class A {
public:
    int x;
    // constructor 
    A (int x):x(x) {}
    
    // A = 1
    void operator= (int rhs) {
        x = rhs;
    }
    
    // ++A
    A& operator++ () {
        x++;
        return this;
    }
    // A++, use a fake parameter
    A operator++ (int) {
        A tmp(*this);
        x++;
        return tmp;
    }
    
    // -A
    A operator- () {
        return A(-x);
    }
    
    // A + A
    A operator+ (const A& rhs) {
        return A(x + rhs.x);
    }
    // A + 1
    A operator+ (const int rhs) {
        return A(x + rhs);
    }
    // 1 + A
    friend A operator+ (const int lhs, const A& rhs) {
        return A(lhs + rhs.x);
    }
    
    // A < 1
    bool operator< (const int rhs) {
        return x < rhs;
    }
    
    // A[i]
    int& operator[] (int i) {
        return arr[i];
    }
    
    // A(a)
    void operator() (int a) {
		return a + x;
    }
    
    // cout << A;
    friend ostream &operator << (ostream& output, const A& a) {
        output << a.x;
        return output;
    }
    
    // cin >> A;
    friend istream &operator >> (istream& input, A& a) {
        input >> a.x;
        return input;
    }
    
};
```

#### polymorphism

```c++

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



### shared_ptr

a smart pointer class, that manages reference count of the object and automatically deconstruct the object if reference count is 0.

```c++
#include <memory>

std::shared_ptr<ClassName> p(new ClassName());
std::shared_ptr<ClassName> p = std::make_shared<ClassName>();

p.reset(new ClassName()); // the first object is deconstructed.
```





### Reference

less powerful but safer than pointer. it works like an alias.

* reference must be initialized as soon as created.

* reference cannot be made to refer to another object once created. (cannot be re-initialized)

  > in fact it is impossible to do so.

* reference cannot be null. (thus there is no container of references.)

###### ref as an alias:

```c++
int i = 0;
int & j = i; // j is an alias of i
j = 1; // i == 1 is true

int & k; // error, not initialized !

int f() { return 42 ; };
int (& rf)() = f; // reference to a func
```

###### ref as function return value:

```c++
double vals[] = {10.1, 12.6, 33.1, 24.1, 50.0};
double& setValues(int i) { return vals[i]; }
setValues(0) = 0; // okay
```

###### ref as function args:

```c++
void swap(int& x, int& y) {
    int tmp = x; x = y; y = tmp;
}
```



### Memory Layout

![img](cpp.assets/memoryLayoutC.jpg)

* Text Segment: the executable program itself. often read-only.
* Initialized Data Segment (DS, data segment): initialized global variables, static variables.
* Uninitialized Data Segment (BSS, block started by symbol): uninitialized global variables, static variables.
* Stack: for normal memory allocation (mostly int x[size]), also contains the program stack, to enable recursive functions.
* Heap: for dynamic memory allocation.

```cpp
// C++ code -- non-vector parts are also true for C 
 
char* s = "hello world!"; // DS, read-write
const int debug = 1; // DS, read-only

int arr1[100000]; // BSS 
int N; // BSS
vector<int> arr2; // HEAP 
 
struct DumbStruct { 
    int someArr[10000]; 
}; 
 
int main () { 
    int arr3[100000]; // STACK
    static int arr7[100000]; // BSS 
    
    vector<int> arr4; // HEAP 
    
    int* arr5 = new int[100000]; // HEAP 
    int* arr6 = (int*) malloc(100000 * sizeof(int)); // HEAP 

    DumbStruct struct; // STACK 
    
    DumbStruct* struct2 = new DumbStruct(); // HEAP 
    vector<DumbStruct> structarr; // HEAP 
    
    int n; 
    scanf("%d", &n); 
    int arr8[n]; // STACK (assuming C99 -- this does not compile in C++) 
}
```





### Tricks

* [Nested functions](https://stackoverflow.com/questions/4324763/can-we-have-functions-inside-functions-in-c)

  We can use lambdas to nest functions. (c++11)

  ```cpp
  int main() {
      // This declares a lambda, which can be called just like a function
      auto print_message = [](std::string message) 
      { 
          std::cout << message << "\n"; 
      };
  
      // Prints "Hello!" 10 times
      for(int i = 0; i < 10; i++) {
          print_message("Hello!"); 
      }
  }
  ```

  
