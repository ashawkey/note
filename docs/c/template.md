# template tutorial

### basics

* 输入变量的自动类型推导

  ```cpp
  #include <iostream>
  using namespace std;
  
  template <typename T> T abs(T x) { return x >= 0 ? x : -x; }
  
  int main() {
    int a = -1;
    cout << abs<int>(a) << endl; // standard call, OK.
    cout << abs(a) << endl; // auto deduce, also OK.
  }
  ```

  * 返回值不能用于自动类型推导！

    ```cpp
    template <typename T> T cast(int x) { return static_cast<T>(x); }
    
    float x = cast(1); // cannot compile (no matching function to call)
    float x = cast<float>(1); // OK, must be explicit
    ```

  * 部分自动类型推导：必须手动指定的在左，可自动推导的在右（类似于默认参数在右）

    ```cpp
    int x = 1;
    
    // wrong
    template <typename F, typename T> T cast(F x) { return static_cast<T>(x); }
    float x = cast<int>(x); // cannot compile
    float x = cast<int, float>(x); // standard, OK
    
    // correct
    template <typename T, typename F> T cast(F x) { return static_cast<T>(x); }
    float x = cast<float>(x); // OK, F = int is deduced from input.
    float x = cast<float, int>(x); // standard, OK
    ```

* 整型模板参数

  泛整型（int, unsigned int, long long, unsigned long long, short, unsigned short, bool）也可以作为模板参数，通常用作常数。

  ```cpp
  template <typename T, int size> struct A {
      T data[size];
  };
  
  A<int, 10> a;
  ```



### meta/generic programming

功能类似于C中的MACRO，各种动态语言的duck typing，其目的是实现静态类型的抽象。

* 特化与偏特化（Specialization & Partial Specialization）

  用于实现对不同类型的输入调用不同方法：

  ```python
  def f(x):
      if isinstance(i, int): return -x;
      else if isinstance(i, float): return 1/x;
      else: return x;
  ```

  注意c++没有判断类型的机制，我们只能通过重载实现这种功能。特化在编译器实现类型判断并分流。

  ```cpp
  #include <iostream>
  using namespace std;
  
  // the generic template
  template <typename T> T f(T x) { return x; }
  
  // specialization for int: negation
  template <> int f<int>(int x) { return -x; }
  
  // specialization for float: reciprocal
  template <> float f<float>(float x) { return 1/x; }
  
  int main() {
      int x = 1;
      float y = 2;
      cout << f(x) << endl;
      cout << f(y) << endl;
  }
  ```

  Application: Recursively Remove Pointer type:

  ```cpp
  template <typename T>
  class RemovePointer<T*> {
  public:
  	// `typename A<T>::t` means `t` is a type, instead of a member of `A<T>`.
      // ref: https://stackoverflow.com/questions/610245/where-and-why-do-i-have-to-put-the-template-and-typename-keywords
      typedef typename RemovePointer<T>::Result Result;
  };
  ```

* 双阶段名称查找（Two phase name lookup）

  问题：编译时的模板有时并不能确定代码是否正确，只有实例化确定参数类型之后才能。

  ```cpp
  // we don't know if x really has a member called `a`.
  template <typename T> void f(T& x) {
      x.a = 1;
  }
  ```

  因此，名称查找会在模板定义（编译）和实例化（运行）时各进行一次。编译阶段无法确定的地方也会假设是正确的。

  ```cpp
  template <typename T> struct X {};
  template <typename T> struct Y {
      typedef X<T> type1; // pass compilation, can find X.
      typedef typename X<T>::noMember type2; // pass compilation, X<T> is dependent, leave it to runtime.
      typedef noMember type3; // cannot pass compilation, cannot find noMember.
  };
  ```

  `typedef typename` 的应用规则：

  * 如果编译器能在出现的时候知道它是一个类型，那么就不需要`typename` 。

  * 如果必须要到实例化的时候才能知道它是不是合法，那么定义的时候就把这个名称作为变量而不是类型，此时我们需要用`typename`显式声明这是一个类型。

  ```cpp
  template <typename T> void meow() {
  	T::a * b; // will be explained as an expression!
  }
  ```

  More use cases:

  ```cpp
  struct A;
  template <typename T> struct B;
  template <typename T> struct X {
      typedef X<T> _A; // 编译器当然知道 X<T> 是一个类型。
      typedef X    _B; // X 等价于 X<T> 的缩写
      typedef T    _C; // T 不是一个类型还玩毛
      
      // ！！！注意我要变形了！！！
      class Y {
          typedef X<T>     _D;          // X 的内部，既然外部高枕无忧，内部更不用说了
          typedef X<T>::Y  _E;          // 嗯，这里也没问题，编译器知道Y就是当前的类型，
                                        // 这里在VS2015上会有错，需要添加 typename，
                                        // Clang 上顺利通过。
          typedef typename X<T*>::Y _F; // 这个居然要加 typename！
                                        // 因为，X<T*>和X<T>不一样哦，
                                        // 它可能会在实例化的时候被别的偏特化给抢过去实现了。
      };
      
      typedef A _G;                   // 嗯，没问题，A在外面声明啦
      typedef B<T> _H;                // B<T>也是一个类型
      typedef typename B<T>::type _I; // 嗯，因为不知道B<T>::type的信息，
                                      // 所以需要typename
      typedef B<int>::type _J;        // B<int> 不依赖模板参数，
                                      // 所以编译器直接就实例化（instantiate）了
                                      // 但是这个时候，B并没有被实现，所以就出错了
  };
  ```

* 重载与偏特化的区别

  偏特化必须与原型相容（模板参数的数量相同），但函数重载可以任意改变参数列表。

  如果一次调用同时匹配多个特化，则会报错（ambiguous partial specialization）。

  ```cpp
  template <typename T, typename U> struct X            ;    // 0，原型
  template <typename T>             struct X<T,  T  > {};    // 1，特化两个模板参数相同，仍接受一个参数
  template <typename T>             struct X<T*, T  > {};    // 2
  template <typename T>             struct X<T,  T* > {};    // 3
  template <typename U>             struct X<U,  int> {};    // 4，只特化第二个参数，仍接受一个参数
  template <typename U>             struct X<U*, int> {};    // 5
  template <typename U, typename T> struct X<U*, T* > {};    // 6，仍接受两个参数，但均使用其指针类型
  template <typename U, typename T> struct X<U,  T* > {};    // 7
  
  template <typename T>             struct X<unique_ptr<T>, shared_ptr<T>>; // 8
  
  // 以下特化，分别对应哪个偏特化的实例？
  // 此时偏特化中的T或U分别是什么类型？
  
  X<float*,  int>      v0; // 5
  X<double*, int>      v1; // 5                  
  X<double,  double>   v2; // 1                     
  X<float*,  double*>  v3; // 6
  X<float*,  float*>   v4; // wont compile            
  X<double,  float*>   v5; // 7                     
  X<int,     double*>  v6; // 7                      
  X<int*,    int>      v7; // wont compile       
  X<double*, double>   v8; // 2
  ```

* 默认模板参数

  类似于默认参数，可以用于模拟变长模板参数列表。

  ```cpp
  template <typename T0, typename T1 = void> struct X {
      static void call(T0& p0, T1& p1); // 0
  };
  
  template <typename T0> struct X<T0> {
      static void call(T0& p0); // 1
  };
  
  template <> struct X<double> {...};
  template <> struct X<double, double> {...};
  
  int main() {
      X<int>::call(5); // 1
      X<int, float>::call(5, 0.1f) // 0
  }
  ```

  使用表达式的默认实参：

  ```cpp
  #include <complex>
  #include <type_traits>
  
  template <typename T> T CustomDiv(T lhs, T rhs) {
      T v;
      // Custom Div的实现
      return v;
  }
  
  // 原型
  template <typename T, typename Enabled = std::true_type> struct SafeDivide {
      static T Do(T lhs, T rhs) {
          return CustomDiv(lhs, rhs);
      }
  };
  
  // 偏特化A
  template <typename T> struct SafeDivide<T, typename std::is_floating_point<T>::type>{    
      static T Do(T lhs, T rhs){
          return lhs/rhs;
      }
  };
  
  // 偏特化B
  template <typename T> struct SafeDivide<T, typename std::is_integral<T>::type>{   
      static T Do(T lhs, T rhs){
          return rhs == 0 ? 0 : lhs/rhs;
      }
  };
  
  void foo(){
      SafeDivide<float>::Do(1.0f, 2.0f);	// 调用偏特化A
      // 先匹配原型，得到默认实际参数<float, true_type>；再匹配特化，A为<float, true_type>，B为<float, false_type>，故进一步匹配A。
      SafeDivide<int>::Do(1, 2);          // 调用偏特化B
      SafeDivide<std::complex<float>>::Do({1.f, 2.f}, {1.f, -2.f}); // 调用原型CustomDiv
  }
  ```

  

* 变参模板（Variadic Template）

  c++11引入的原生变长模板参数支持。

  类似于函数的变长参数，只能放在参数列表的最后。

  ```cpp
  template <typename... Ts> class X {}; // OK
  template <typename... Ts, typename U> class X {}; // Wrong! Ts should be at last.
  
  template <typename... Ts, typename U> class X<U, Ts...> {}; // OK, this is a specialization of X, the real parameter order is <U, Ts, ...>
  template <typename... Ts, typename U> class X<Ts..., U> {}; // Wrong specialization.
  ```

  

* SFINAE：替换失败不是错误（Substitution Failure Is Not An Error）

  出现在模板参数套娃的情况，例如`typename T::T2`使用了未知的`T`内定义的未知的`T2`。

  替换指模板实例化时，即`T`被指定时，上述套娃模板`T::T2`被替换为具体类型的过程。

  SFINAE描述了只要有任意一个特化可以成功完成替换，代码就不会出错的行为。

  > SFINAE是c++对于缺乏introspection机制（从实例获取类信息）的补丁。
  
  ```cpp
  struct X { typedef int type; };
  struct Y { typedef int type2; };
  
  template <typename T> void foo(typename T::type x); // 0
  template <typename T> void foo(typename T::type2 x); // 1
  template <typename T> void foo(T x); // 2
  
  // all can pass compilation!
  int main() {
      foo<X>(5);   // 0 = OK,     1 = Failed, 2 = Failed
      foo<Y>(5);   // 0 = Failed, 1 = OK,     2 = Failed
      foo<int>(5); // 0 = Failed, 1 = Failed, 2 = OK
  }
  ```

  Example: implement a type-transparent `inc_counter()`.
  
  ```cpp
  // boost enable_if
  #include <type_traits>
  #include <utility>
  #include <cstdint>
  
  struct ICounter {}; // just a placeholder
  struct Counter: public ICounter {
      void increase() {
          // impl
      }
  };
  
  template <typename T> void inc_counter(T& counterObj, typename std::enable_if<std::is_base_of<ICounter, T>::value>::type* = nullptr ){
      counterObj.increase();  
  }
  
  template <typename T> void inc_counter(T& counterInt, typename std::enable_if<std::is_integral<T>::value>::type* = nullptr ){
      ++counterInt;
  }
    
  void doSomething() {
      Counter cntObj;
      uint32_t cntUI32;
  
      // as expected
      inc_counter(cntObj);
      inc_counter(cntUI32);
  }
  ```
  
  ```cpp
  // requires c++11
  struct Counter {
      void increase() {
          // Implements
      }
  };
  
  template <typename T>
  void inc_counter(T& counterInt, std::decay_t<decltype(++counterInt)>* = nullptr) {
      ++counterInt;
  }
  
  template <typename T>
  void inc_counter(T& counterObj, std::decay_t<decltype(counterObj.increase())>* = nullptr) {
      counterObj.increase();
  }
  
  void doSomething() {
      Counter cntObj;
      uint32_t cntUI32;
  
      // as expected
      inc_counter(cntObj);
      inc_counter(cntUI32);
  }
  ```

  
### 其他

* 反射（Reflection）：通过字符串的名称来调用函数、类等。

  ```python
  class Foo:
      def __init__(self):
  		self.x = 1
      
     	def call(self):
          print('foo!', self.x)
  
  # reflection from class name
  foo = globals()['Foo']() # foo = Foo()
  
  # reflection from method name
  getattr(foo, 'call')() # foo.call()
  ```

* 自省（Introspection）：通过一个实例获取它所属的类的任意方法或特性。

  ```python
  for name in dir(foo):
      print(name, getattr(foo, name))
      
  # or use inspect
  import inspect
  for name, val in inspect.getmembers(foo):
      print(name, val) 
  ```

  