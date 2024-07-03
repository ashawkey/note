## c tricks

### [do ... while (0)](https://bruceblinn.com/linuxinfo/DoWhile.html#:~:text=You%20may%20see%20a%20do,single%20statement%20can%20be%20used.)

A trick (idiom) used to correctly define macros with multiple lines / containing `if`.

```cpp
// wrong
#define FOO statement1; statement2; 

// we want both statements are controlled by if, but this will left statement2 always be run.
if (cond) FOO; // if (cond) statement1; statement2;

// correct way with do...while(0)
#define FOO do {statement1; statement2;} while(0);
```

Just using `{...}` is also wrong, but `({...})` is OK.

```cpp
// another wrong way
#define FOO {statement1; statement2;}

// note the last ;, it breaks the if-else and will throw compile error!
if (cond) FOO; // if (cind) {statement1; statement2;};
else ...;

// another correct way (recommended!)
#define FOO ({statement1; statement2;})
```


### Playing with #define

```cpp
// variable
#define X 100

// func
#define ADD(x, y) (x + y)

// stringfy: #
#define PQSR(x) printf("square of" #x "is %d\n", (x) * (x))
int y = 2;
PQSR(y); // "square of y is 4"

// concat: ##
#define XN(n) x##n
#define PXN(n) printf("x"#n" = %d\n", x##n)

int XN(1) = 1; // int x1 = 1;
PXN(1); // printf("x1 = %d\n", x1);

// multi-line: do while(0)
#define M(x, y) do { /
stmt1; /
stmt2; /
} while(0)

// example
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
#define _REG_FUNC(funname) m.def(#funname, &funname)
  _REG_FUNC(sample_grid);
  _REG_FUNC(sample_grid_backward);
#undef _REG_FUNC
}
```


### static

General meaning: **only initialized once**.

* Static global variable / function:

  Only visible inside the file it is declared in.

* Static class variable / function:

  The member belongs to the class itself, instead of its instantiations.

  ```cpp
  class A {
  public:
      static int x = 0;
  }
  
  int main() {
      A a, b;
      a.x = 1;
      cout << b.x << endl; // 1
  }
  ```

* Static local variable inside function:

  Retains value between function calls.

  ```cpp
  int foo() {
      static int x = 0; // preserved in all function calls
      cout << x << endl;
  }
  
  int main() {
      for(int i = 0; i < 3; i++) foo(); // 0, 1, 2
  }
  ```

  
