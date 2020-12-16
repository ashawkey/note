# c pitfalls

* `std::map::operator[]` will initialize the value if key doesn't exist. (instead of throw an error like `at()`)

  ```c++
  cout << m[-1] << endl; // if not exist, will insert & initialize it ! (here int --> 0)
  cout << m.at(-1) << endl; // check if exist first, throw an error
  ```

* `%` modulo operator will return **signed** value

  ```c++
  cout << -1 % 3 << endl; // -1
  
  // to get a positive modulo (like % in python), we need:
  cout << (x % N + N) % N << endl;
  ```

  