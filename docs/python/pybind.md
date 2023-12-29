# pybind11

### install

```bash
pip install pybind11
```


### example

`example.cpp`:

```cpp
// include
#include <pybind11/pybind11.h>
namespace py = pybind11;

// bind a simple function
int add(int i, int j) {
    return i + j;
}

PYBIND11_MODULE(example, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring
    m.def("add", &add, "A function which adds two numbers");
}
```

compile with:

```bash
c++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) example.cpp -o example$(python3-config --extension-suffix)
```

use in `main.py`:

```python
import example
example.add(1, 2)
```


### binding basics

* functions

  ```cpp
  int add(int i, int j) {
      return i + j;
  }
  
  PYBIND11_MODULE(example, m) {
      m.def("add", &add, "help");
  }
  ```

  ```python
  import example
  example.add(1, 2)
  ```

  * keyword parameters

    ```cpp
    // regular notation
    m.def("add1", &add, "help", py::arg("i"), py::arg("j"));
    
    // shorthand
    using namespace pybind11::literals;
    m.def("add2", &add, "help", "i"_a, "j"_a);
    ```

    ```python
    import example
    example.add(i=1, j=2)
    ```

  * default parameters

    ```cpp
    // regular notation
    m.def("add1", &add, "help", py::arg("i") = 1, py::arg("j") = 2);
    
    // shorthand
    using namespace pybind11::literals;
    m.def("add2", &add, "help", "i"_a=1, "j"_a=2);
    ```

* variables

  ```cpp
  PYBIND11_MODULE(example, m) {
      m.attr("the_answer") = 42;
      py::object world = py::cast("World");
      m.attr("what") = world;
  }
  ```

  ```python
  import example
  example.the_answer
  example.what
  ```

* class

  ```cpp
  // definition
  struct Pet {
      Pet(const std::string &name) : name(name) { }
      void setName(const std::string &name_) { name = name_; }
      const std::string &getName() const { return name; }
      
      std::string name;
  };
  
  // binding
  #include <pybind11/pybind11.h>
  namespace py = pybind11;
  
  PYBIND11_MODULE(example, m) {
      py::class_<Pet>(m, "Pet") // "Pet" is the class name in python
          .def(py::init<const std::string &>()) // init method
          .def("setName", &Pet::setName)
          .def("getName", &Pet::getName);
  }
  ```

  ```python
  import example
  p = example.Pet('Molly')
  print(p)
  p.getName()
  p.setName('Charly')
  p.getName()
  ```

  * binding lambda functions

    ```cpp
    py::class_<Pet>(m, "Pet")
        .def(py::init<const std::string &>())
        .def("setName", &Pet::setName)
        .def("getName", &Pet::getName)
        // the __repr__ using a lambda function
        .def("__repr__",
            [](const Pet &a) {
                return "<example.Pet named '" + a.name + "'>";
            }
        );
    ```

  * binding attributes

    ```cpp
    // public attribute
    py::class_<Pet>(m, "Pet")
        .def(py::init<const std::string &>())
        .def_readwrite("name", &Pet::name)
        ...
    ```

    ```python
    import example
    p = example.Pet('Molly')
    p.name
    p.name = "Charly"
    ```

    ```cpp
    // private attribute
    class Pet {
    public:
        Pet(const std::string &name) : name(name) { }
        void setName(const std::string &name_) { name = name_; }
        const std::string &getName() const { return name; }
    private:
        std::string name;
    };
    
    py::class_<Pet>(m, "Pet")
        .def(py::init<const std::string &>())
        .def_property("name", &Pet::getName, &Pet::setName)
        ...
    ```

    ```python
    import example
    p = example.Pet('Molly')
    p.name
    p.name = "Charly" # Error!
    ```

  * dynamic attribute

    By default, `pybind11` class doesn't support dynamic attribute like python class:

    ```python
    import example
    p = example.Pet('Molly')
    p.age = 2 # Attribute Error!
    ```

    To enable it, use `py::dynamic_attr()`:

    ```cpp
    py::class_<Pet>(m, "Pet", py::dynamic_attr())
        .def(py::init<>())
        .def_readwrite("name", &Pet::name);
        ...
    ```

    ```python
    import example
    p = example.Pet('Molly')
    p.age = 2 # OK!
    ```

  * Inheritance

    ```cpp
    // TBD...
    ```

  * Overloaded method

    ```cpp
    // TBD...
    ```


### Misc

* Binding template functions. 

```cpp
template <typename T>
void set(T t);
```
You cannot bind a template function directly, but you can instantiate it first:
```cpp
// overload 
m.def("set", &set<int>);
m.def("set", &set<std::string>);

// or explicitly bind to different name
m.def("setInt", &set<int>);
m.def("setString", &set<std::string>);

```

### Type Conversion

#### Basic types
This includes `int, float, bool, ...`
Copy is made when conversion between.

#### python str <--> std::string
Also quite straight-forward, copy is made.
```cpp
m.def("utf8_test",
    [](const std::string &s) {
        cout << "utf-8 is icing on the cake.\n";
        cout << s;
    }
);
```

```python
utf8_test("🎂")
```

#### list/tuple <--> std::vector/deque/array
list and tuple are not distinguished, since there must be a COPY at every conversion.

```cpp
#include <pybind11/stl.h>

m.def("cast_vector", []() { return std::vector<int>{1}; });
m.def("load_vector", [](const std::vector<int> &v) { return v.at(0) == 1 && v.at(1) == 2; });
```

```python
lst = m.cast_vector()
assert lst == [1]

lst.append(2)
assert m.load_vector(lst)
assert m.load_vector(tuple(lst))
```

#### dict <--> std::map
key and value types are automatically handled.
```cpp
#include <pybind11/stl.h>

m.def("cast_map", []() { return std::map<std::string, std::string>{{"key", "value"}}; });
m.def("load_map", [](const std::map<std::string, std::string> &map) {
	return map.at("key") == "value" && map.at("key2") == "value2";
});
```

```python
d = m.cast_map()
assert d == {"key": "value"}
assert "key" in d
d["key2"] = "value2"
assert "key2" in d
assert m.load_map(d)
```

#### set <--> std::set
```cpp
m.def("cast_set", []() { return std::set<std::string>{"key1", "key2"}; });
m.def("load_set", [](const std::set<std::string> &set) {
	return (set.count("key1") != 0u) && (set.count("key2") != 0u) && (set.count("key3") != 0u);
});
```

```python
s = m.cast_set()
assert s == {"key1", "key2"}
s.add("key3")
assert m.load_set(s)
```

#### numpy ndarray <--> Eigen Matrix
Pass-by-value are supported between `np.ndarray` and `Eigen::MatrixXd`. A copy is made for each conversion, which maybe unwanted.
Pass-by-reference can be achieved through `Eigen::Ref<T>`. 

TODO: https://github.com/pybind/pybind11/blob/master/tests/test_eigen.cpp

##### numpy ndarray --> Eigen Matrix

```cpp

```

##### Eigen Matrix --> numpy ndarray


### Building

* `Setuptools`

  ```python
  
  ```

  
* `Cmake`

  Start by adding `pybind11` as a sub-module:

  ```bash
  git init # must be a repo to add submodule
  git submodule add -b stable https://github.com/pybind/pybind11.git pybind11
  git submodule update --init
  ```

  Then, modify the `CMakeLists.txt`:

  ````cmake
  cmake_minimum_required(VERSION 3.4...3.18)
  project(example)
  
  # the original dependencies
  find_package(Eigen3 3.3 REQUIRED NO_MODULE)
  
  # add pybind11
  add_subdirectory(pybind11)
  
  # compile the library
  pybind11_add_module(example example.cpp)
  
  # must use PRIVATE target_link_libraries !!!
  # ref: https://github.com/pybind/pybind11/issues/387
  target_link_libraries(example PRIVATE Eigen3::Eigen)
  
  ````

  