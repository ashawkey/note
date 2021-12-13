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
      py::class_<Pet>(m, "Pet")
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

  