# c pitfalls

* NEVER call `top()/front()` when a  (priority) queue is empty, it is an undefined behavior, and will not cause runtime error:

  ```cpp
  #include <iostream>
  #include <queue>
  
  using namespace std;
  
  int main() {
      priority_queue<int> q;
      //cout << q.empty() << " " << q.top() << endl; // undefined, will destroy the priority queue and nothing is printed...
      q.push(1);
      cout << q.empty() << " " << q.top() << endl; // 0 1
      q.pop();
      cout << q.empty() << " " << q.top() << endl; // 1 1 (undefined behavior, seems still the last top element.)
  }
  ```

  ```cpp
  #include <iostream>
  #include <queue>
  
  using namespace std;
  
  int main() {
      queue<int> q;
      cout << q.empty() << " " << q.front() << endl; // 1 0 (undefined behavior, seems default to 0)
      q.push(1);
      cout << q.empty() << " " << q.front() << endl; // 0 1
      q.pop();
      cout << q.empty() << " " << q.front() << endl; // 1 0 (undefined behavior)
  }
  ```

  
* `unordered_map<pair<int, int>, int>` throws error like `deleted implicit constructor`:

  this is because there is no built in `hash` function for `pair<>`. [see here.](https://stackoverflow.com/questions/62869571/call-to-implicitly-deleted-default-constructor-of-unordered-set-vectorint)

  Unfortunately there is no perfect solution:

  * custom hash, like `p.first * MAX_SECOND + p.second`.
  * use `map<pair<int,int>, int>`.

  
* `std::map::operator[]` will initialize the value if key doesn't exist. (instead of throw an error like `at()`)

  ```c++
  cout << m[-1] << endl; // if not exist, will insert & initialize it ! (here int --> 0)
  cout << m.at(-1) << endl; // check if exist first, throw an error
  ```

* `string::operator[] & string::at` behaves similarly.

  ```c++
  string s = "a";
  
  // [] won't check out of range!
  cout << "s[1] = " << s[1] << endl; // \0
  cout << "s[2] = " << s[2] << endl; // undefined char
  cout << "s = " << s << endl; // a
  cout << "s.size() = " << s.size() << endl; // 1
  
  // even if you modify it, it still won't throw an error.
  s[1] = 'b'; // dangerous!
  s[2] = 'c';
  cout << "s[1] = " << s[1] << endl; // b 
  cout << "s[2] = " << s[2] << endl; // c
  // but s is not changed.
  cout << "s = " << s << endl; // a
  cout << "s.size() = " << s.size() << endl; // 1
  
  // at() checks out of range!
  cout << "s.at(1) = " << s.at(1) << endl; // error
  ```

  
* `%` modulo operator will return **signed** value

  ```c++
  cout << -1 % 3 << endl; // -1
  
  // to get a positive modulo (like % in python), we need:
  cout << (x % N + N) % N << endl;
  ```

* `.0f` for `float 0`, not `0f`

* `0x80000000` default type is `unsigned (int)`, while `0x7fffffff` is `int`.

  ```cpp
  #include <iostream>
  #include <typeinfo>
  
  using namespace std;
  
  auto x = 0x80000000;
  cout << typeid(x).name() << " = " << x << endl; // j (unsigned) = 2147483648 
  
  int x = 0x80000000;
  cout << typeid(x).name() << " = " << x << endl; // i (int) = -2147483648
  
  long long x = 0x80000000; // unsigned -> long long
  cout << typeid(x).name() << " = " << x << endl; // x (long long) = 2147483648
  
  auto x = 0x7fffffff;
  cout << typeid(x).name() << " = " << x << endl; // i (int) = 2147483647
  ```

  So if you want to get a number smaller than `INT_MIN`, you should do this:

  ```cpp
  long long x = (long long)(int)0x80000000 - 1;
  
  // equals
  long long x = -2147483649;
  ```

* `str.erase`

  ```cpp
  string s = "abc";
  
  // str.erase(int pos, int len = npos);
  s.erase(1); // a
  s.erase(1, 1); // ac
  
  // str.erase(str::iterator it);
  s.erase(s.begin() + 1); // ac
  
  // str.erase(str::iterator begin, str::iterator end);
  s.erase(s.begin() + 1, s.end()); // a
  ```

* [alternative operators](https://en.cppreference.com/w/cpp/language/operator_alternative)

  though amazing, you can use `and` , `or` in c++.

  ```
  // can be directly used in c++
  // to use in c, need to include <iso646.h>
  &&	and
  &=	and_eq
  &	bitand
  |	bitor
  ~	compl
  !	not
  !=	not_eq
  ||	or
  |=	or_eq
  ^	xor
  ^=	xor_eq
  {	<%
  }	%>
  [	<:
  ]	:>
  #	%:
  ##	%:%:
  ```

* Structure binding of `tuple` cannot be used to assign value!
  
    ```cpp
    // structure binding
    tuple<int, int> t = {1, 2};
    auto [a, b] = t; // will CREATE and assign a, b to 1, 2
    
    // however, if you already have a, b, you cannot assign value to them.
    int a = 0, b = 0;
    auto [a, b] = t; // error: conflicting declaration!

    // in certain cases, it will not give error but silently go wrong...
    int a = 0, b = 0;
    { // any new scope
        auto [a, b] = t; // a, b are local variables, not the a, b outside.
    }
    cout << a << " " << b << endl; // 0 0
    ```    

  
