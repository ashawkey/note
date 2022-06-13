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
    
    // reserve v.s. resize
    vec.reserve(10); // only increase capacity, will not push empty elements! vec.size() == 0
    vec.resize(10); // push 10 empty elements into vec. vec.size() == 10
    
 	// push_back
    for(int i = 0; i < 5; i++){
       vec.push_back(i);
    }
    
    // clear
    v.clear();
        
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
    
    // extend another vector (also by insert)
    v1.insert(v1.end(), v2.begin(), v2.end());
    
    // emplace_back: in-place push_back  (cpp11, but not supported in msvc10)
    // use condition: push a complicated object to a vector, but don't want to create a temporary object and then copy it to the vector.
    // assume we have a Complicated class initialized by Complicated(x, y, z)
    vec<Complicated> v;
    v.push_back(Complicated(x, y, z)); // create temporary then copy, slow.
    v.emplace_back(x, y, z); // no temporary, fast.

    // emplace: in-place insert.
    v.emplace(v.end(), x, y, z);
    
    // slice subvector
    vector<int> v2(v.begin(), v.begin() + 5); // select and copy [0, 5)
    v.resize(5); // inplace trick, keep first 5 elements
    
    // iterator
	auto it = v.begin();
    auto next = it + 1;
    auto nnnnext = it + 4; // vector is sequence container, so we can add an int to it.
    int x = *it; 
    v.erase(next);
    v.insert(next, 0);
    
 	
    return 0;
}

```

Note: `vector<bool>` is not a STL container! It optimizes space and use bit to save each bool variable, so many STL operations are not supported. Use `bitset` or `deque<bool>` instead.

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
    
    // loop all elements (increasing order)
    for (auto it = s.begin(); it != s.end(); it++) {
        cout << *it << endl;
    }
    // loop with index (enumerate)
    for (auto [it, i] = tuple{s.begin(), 0}; it != s.end(); it++, i++) m[*it] = i;
    
	// get the smallest (first) or largest (last) element
    int mn = *s.begin();
    int mx = *s.end();
    
    // get the k-th smallest element
    auto i = s.begin();
	cout << *i << endl; // s[0]
    auto next = ++i; // s[1], DO NOT USE i++, then next is still s[0]
    
    advance(i, k); // in-place modification of i (since set is an associative container, i + k will not work)
    
    auto i = s.begin();
	auto i2 = next(i, 1); // not in-place ver
    auto i3 = prev(i2, 1);
    
    int mmn = *next(s.begin(), 1);
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
    for (auto& [k, v]: m) { // c++17
        cout << k << ": " << v << endl;
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
    
    // binary search keys
    // assume the keys are {0,1,3,5,7}
    auto it = m.lower_bound(3); // will get iterator to 3
    auto it = m.lower_bound(2); // will get iterator to 3
    auto it = m.lower_bound(-1); // will get iterator to 0
    auto it = m.lower_bound(8); // will get map.end(), which has `it->first = map.size(); it->second = 0` (注意不要用key来判断是否达到了map.end()!这里key等于map中的元素数量。要用it == map.end()。)
    
    auto it = m.upper_bound(3); // will get iterator to 5
    auto it = m.upper_bound(7); // will get map.end()

    
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
    q.push(1); // not push_back
    while (!q.empty()) {
		int a = q.front(); 
        int b = q.back(); 
        q.pop(); // return void
        cout << q.size() << endl;
    }
}

// a trick in BFS to get each layer of a tree (only use one queue!)
bool BFS_leveling(TreeNode* root) {
    // bfs leveling
    queue<TreeNode*> q;
    q.push(root);
    int layer = 0;
    while (!q.empty()) {
        // record current size
        int cur_size = q.size();
        // lnodes in current layer
        for (int i = 0; i < cur_size; i++) {
            TreeNode* p = q.front(); q.pop();
            // do something...
            if (p->left) q.push(p->left);
            if (p->right) q.push(p->right);
        }
        layer++;
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
    s.push(2); // not push_back
	int x = s.top();
    s.pop(); // return void
    s.empty();
}
```

#### #include \<priority_queue\>

```c++
#include <iostream>
#include <queue> // priority_queue is define
using namespace std;

// < means larger-first
struct cmp {
    bool operator()(int a, int b) {
        return a < b;
    }
};

int main() {
    // heap, default is larger-first
    priority_queue<int> pq;
    pq.push(1);
    pq.push(2);
    pq.push(0);
    p.top(); // 2, yes it's top, not front.
    
    // smaller-first
    priority_queue<int, vector<int>, greater<int>> pq;
    
    
    // custom cmp function by struct
	set<int, cmp> s;
	priority_queue<int, vector<int>, cmp> pq;
    
    // custom cmp function by lambda
    auto cmp = [&](const pair<int, int>& x, const pair<int, int>& y) {
        return arr[x.first] * arr[y.second] > arr[x.second] * arr[y.first];
    };
    priority_queue<pair<int, int>, vector<pair<int, int>>, decltype(cmp)> q(cmp);
}

// you may need to store a pair in priority queue and sort by a key:
typedef pair<int,int> pii;
priority_queue<pii, vector<pii>, greater<pii>> pq; // smaller first
// unfortunately, you cannot update the pq in-place, if you would like to modify it, you have to pop and re-push.
auto [k, v] = pq.top(); pq.pop();
pq.push(pii(k))
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
sort(v, v + 4, [](int i, int j) { return i < j; }) // lambda version
    
//// merge sort O(nlogn)
stable_sort(v, v + 4);

//// partial sort / topk, in O(klogn)
partial_sort(v, v + 2, v + 4); // only sort the first two elements

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
v.erase(end, v.end());
int new_size = v.size();

int new_size = unique(v, v+4) - v;

//// lower/upper_bound
// check out of bound conditions!
// lower_bound(begin,end,num)：从数组的begin位置到end-1位置二分查找第一个大于或等于num的数字，找到返回该数字的地址，不存在则返回end。通过返回的地址减去起始地址begin,得到找到数字在数组中的下标。
// upper_bound(begin,end,num)：从数组的begin位置到end-1位置二分查找第一个大于num的数字，找到返回该数字的地址，不存在则返回end。通过返回的地址减去起始地址begin,得到找到数字在数组中的下标。
// lower/upper_bound([1,2,3], 4) --> 3
// lower/upper_bound([1,2,3], 0) --> 0

// the first position >= target
int l = lower_bound(v.begin(), v.end(), target) - v.begin();
// the first position > target
int r = upper_bound(v.begin(), v.end(), target) - v.begin();

// border condition 
if (l == v.size()) {
	// v[l-1] < target, non-exist case
} else {
    // v[l] >= target
    if (v[l] == target) {}
    else {}
}

// find the last position <= target
if (l == v.size()) return l-1;
else if (l == 0) {
    if (v[l] == target) return l;
    else return -1; // non-exist case
}
else {
    if (v[l] == target) return l;
    else return l-1;
}


//// reverse
string s = "string";
reverse(s.begin(), s.end()); // inplace

//// argsort (handy implementation)
vector<pair<int, int>> v2;
for (int i = 0; i < v.size(); i++) {
    v2.push_back({v[i], i}); // pair value with indice
}
sort(v2.begin(), v2.end(), [](pair<int, int> a, pair<int, int> b) {
    if (a.first < b.first || (a.first == b.first && a.second < b.second)) return true;
    else return false;
});
vector<int> idx;
for (int i = 0; i < v2.size(); i++) {
    idx.push_back(v2[i].second);
}
```

#### #include \<bitset\>

```cpp
#include <bitset>

bitset<1000> bs; // 1000 bits

bitset<32> bs{10}; // binary form of x (only support unsigned int/long long)
bitset<8> bs{0b00001111}; // just 00001111
bitset<6> bs{"010101"}; // just 010101

cout << bs << endl; // bit format

// access pos bit, NOTE: the order is reversed! e.g., bs[0] = 1 for "010101"
// out-of-bound is undefined!
bool x = bs[pos]; 

bs.all();
bs.any();
bs.none();

int cnt = bs.count(); // count 1
int l = bs.size();

bs.set(); // set all to 1
bs.set(pos, val=true); // set pos to 1 or 0
bs.reset(); // set all to 0
bs.reset(pos); // set pos to 0, equals bs.set(pos, false);
bs.flip(); // flip all
bs.flip(pos); // flip pos

string s = bs.to_string();
unsigned long x = bs.to_ulo
```





### Class

```c++
class A {
// access modifier (default is private)
private:
	int p; // private variable
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
A a = A(0);
A* pa = new A;
A* pa = new A(0);

printf("%d", a.a);
a.setA(1);
```

Struct is like an default-to-public Class:

```c++
struct A {
    int a; // public
    void setA(int _a) { a = _a; }
};
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

