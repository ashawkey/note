# Compile



### Header `(.h/hpp)` vs Implementation `(.c/cpp)`

##### Concepts

* Header(interface): only class and function declarations .
* Implementation: all the detailed definition.

##### How c++ compiles ?

* compile `cpp` implementation into `objects`. 

  This only uses the `cpp` and the included `cpps/libs` files.

* linking all the `objects` to create the final binary executable / library.

##### what does #include do ?

Copy and paste code from the library to the current `cpp` file.

##### WHY ?

* Historical Reason. 
  * the final executable code does not carry any symbol information.
  * compiler is unable to search for symbol declarations alone.
  * So we have to use header (interface) and `#include`.
  * Java's `import` can automatically recognize identifiers from implementation and dynamic library symbols.

* Reduce compilation time: 

  * avoid re-compilation.

  

### Libraries

A collection of resources to provide **re-usable** code.

Generally, any code you include can be viewed as a library.



##### Using at Compiling

This is the case when you **have the source code of the library**.

You have both `libxxx.cpp` and `libxxx.h`.

```bash
# the simplest example
# all files are assumed to be in the current dir.
g++ main.cpp libxxx.cpp
```

* Header-only library 

  the full definitions and classes are visible to the compiler in a header file. (just `hpp`, no `cpp`)

  Do not need to create a binary library file, and do not need to copy the implementation here.

  * [+] don't need to be compiled. Just include it.
  * [+] thus, compiler can better optimize it with the source code available.
  * [-] changes to the library will cause re-compilation.
  * [-] longer compilation time.

  To compile:

  ```bash
  g++ main.cpp -I /path/to/include/ 
  # -I: directory that contains the header file.
  ```

  

##### Using at Linking

This is the case when the library is **pre-compiled**.

You only have `libxxx.h` , and a compiled library file. (static or dynamic)

* Static Library

  When you include a static library, all the used library code are copied to your executable .

  * [+] faster.

  * [-] make the executable large.
  * [-] changes to the code need re-linking and re-compiling.

  Created by simply archive the object files.

  ```bash
  # Create the object files (only one here)
  g++ -c unuseful.cpp # --> unuseful.o
  # Create the archive (insert the lib prefix)
  ar rcs libunuseful.a unuseful.o # --> libunuseful.a
  ```

  

* Dynamic Library

  When you include a dynamic library, your executable will load the used library code only at run time.

  * [+] won't make executable large.
  * [+] don't need re-compilation if your code is changed.
  * [-] slower.
  * [-] may fail to execute if the library is not found / corrupt.

  Created by using `-shared` flag.

  ```bash
  # Create the object file with Position Independent Code [PIC]
  g++ -fPIC -c unuseful.cpp # --> unuseful.o
  # Crate the shared library (insert the lib prefix)
  g++ -shared -o libunuseful.so unuseful.o # --> libunuseful.so
  ```



Both kind of library should be compiled & linked by:

```bash
g++ main.cpp -I /path/to/include -L /path/to/lib -lunuseful

# -I tells where to find the header (.h)
# -L tells where to search the library (.so/.a)

# -lunuseful tells the name of the libraries to be used.
# -l:foo = foo.so
# -lfoo = libfoo.so
```



##### use`pkg-config` to auto-find `-l<libname>`

```bash
g++ main.cpp `pkg-config <libname> --cflags --libs`
```

`pkg-config --libs <library>` outputs the link arguments for library.

`pkg-config --cflags <library>` outputs the include arguments and any other needed compile flags.



### gcc / g++

Used to compile **Single** source code.

> However, it is tedious to compile multiple source files with gcc one-by-one.

```bash
gcc main.c # default output is a.out

### 4 step:
# preprocess: main.i 
# compile: main.s (assembly code)
# assemble: main.o (binary)
# link: a.out

# generate objects
gcc -c main.c # main.o

# generate asm
gcc -S main.c

# specify output file name
gcc main.c -o main

# include file
gcc main.c -include ./include/head.h 
# this is the same to add #include <include/head.h> in main.c

# include dir
gcc main.c -I ./include/
# this makes it work for #include <head.h>
#include "file"： 会先在当前目录查找你所制定的头文件, 如果没有, 回到默认的头文件目录找。如果使用-I制定了目录，会先在此目录查找, 然后再按常规的顺序去找。
#include <file>： 会到-I制定的目录查找, 查找不到, 然后将到系统的默认的头文件目录查找 。
# the equivalent ENV: C_INCLUDE_PATH, CPP_INCLUDE_PATH

# library dir
gcc main.c -L ./lib/ -l<libname>
# the equivalent ENV: LD_LIBRARY_PATH

# optimize
gcc main.c -O3 # O0 / O1(default) / O2 / O3

# warning
gcc main.c -w # no warning
gcc main.c -Wall # sall warning

# gdb debug information
gcc main.c -g
```

And `g++ ` is for c++.

```bash
# specify standard
g++ -std=c++11 main.cpp
```



### make

Batch-compile for **multiple** source files.

The behavior is determined by a `makefile`.

> However, it is still tedious to write `makefile` for large projects.

```bash
make # find makefile in current dir, and run g++
make clean # remove intermediate files
make install # install to where ?
```



### makefile

##### example

```makefile
### grammar (use Tab for all indent !!!)
# target ... : pre-requisites ...
# 		command
#       ...

# declare variable
objects = main.o kbd.o command.o display.o insert.o search.o files.o utils.o
# the first target is the final output !
# make will find dependencies automatically, so this order is OK.
edit : $(objects)
    gcc -o edit $(objects)

### this is short for
# edit : main.o kbd.o command.o display.o \
#        insert.o search.o files.o utils.o
#    gcc -o edit main.o kbd.o command.o display.o \
#       insert.o search.o files.o utils.o

main.o : defs.h
### this is short for
# main.o : main.c defs.h
# 	gcc -c main.c

kbd.o : defs.h command.h
command.o : defs.h command.h
display.o : defs.h buffer.h
insert.o : defs.h buffer.h
search.o : defs.h buffer.h
files.o : defs.h buffer.h command.h
utils.o : defs.h

.PHONY : clean
clean :
    rm edit $(objects)
```

>  `.PHONY` 表示 `clean` 是一个“伪目标”。而在 `rm` 命令前面加了一个小减号的意思就是，也许某些文件出现问题，但不要管，继续做后面的事。

even simpler

```makefile
objects = main.o kbd.o command.o display.o \
    insert.o search.o files.o utils.o

edit : $(objects)
    cc -o edit $(objects)

# all needs defs.h
$(objects) : defs.h
# else:
kbd.o command.o files.o : command.h
display.o insert.o search.o files.o : buffer.h

.PHONY : clean
clean :
    rm edit $(objects)
```



### compilers

* **Visual C++**: GUI compiler, mainly used in Windows.

* **gcc/g++**: GNU Compiler Collection.

  Standard compiler for Linux C/C++/Fortran/..., with a long history from 1987.

* **clang/clang++**: or Low Level Virtual Machine (LLVM). 

  More modern general compiler, first released in 2003.

  Mainly used to provide (slightly?) better performance than gcc.

  LLVM is also used as the compilation framework for many new languages (Julia, Rust, Swift...).

  Clang is developed by Apple Inc. to replace GCC for better support of LLVM and other new features.

