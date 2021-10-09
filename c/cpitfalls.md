# c pitfalls

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

  
