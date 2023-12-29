# Golang

### CLI

```bash
go build test.go # generate binary at ./
go install test.go # generate binary at $GOBIN/bin & cache dependencies.
go run test.go # build and run
```


### Go Proxy

Correct way to download modules.

```bash
# powershell
$env:GOPROXY = "https://goproxy.io"
$env:GO111MODULE = "on"

go get <pkg>
```


### Go Modules

```bash
# install packages
go get <pkg> # to $GOPATH

# start a project
go mod init <name> # create go.mod, necessary for using installed modules !!!
vim main.go # edit your code, import installed modules

# run
go run main.go # run, create go.sum (validation purpose)
```

Example: `go.mod`

```go
module test

go 1.14

require github.com/gin-gonic/gin v1.6.3
```


### Defer

```go
func a() {
    defer b() // called after function returns
    return 0
}

// use:
f, err = os.Open(filename)
defer f.close() // don't need to put it in the last

// LIFO order
defer func() { fmt.Println("1") }()
defer func() { fmt.Println("2") }()
defer func() { fmt.Println("3") }()
// => 321


```


### The empty Interface

`interface{}` can hold values of any type. 

It is used as a dynamic type in golang.

```go
// this func accepts any param.
func describe(i interface{}) {
	fmt.Printf("(%v, %T)\n", i, i)
}

func main() {
	var i interface{} // this is a nil object.
	describe(i) // (<nil>, <nil>)

	i = 42
    describe(i) // (42, int)

	i = "hello"
	describe(i) // (hello, string)
}
```


### Overview

```go
package main

import "fmt"

func main() {
	/* comment */
	// also comment
	fmt.Println("hello, world")
}
```

```bash
go run test.go
```


### Types

* Pass-by-Value

  ```
  bool int float32 float64 string
  ```

  
* Pass-by-Reference

  ```
  Pointer struct Channel interface
  ```

  
### Variables

```go
var name type // default value is 0/nil
var name [type] = value // auto
name := value // auto, usually used inside function!

var a b type
var a,b = value1, value2
var (a type1 b type2)

const name [type] = value

// itoa: a counter
const (
            a = iota   //0
            b          //1
            c          //2
            d = "ha"   //iota += 1
            e          //"ha"   iota += 1
            f = 100    //iota +=1
            g          //100  iota +=1
            h = iota   //7
            i          //8
    )
```

* Local variable must be used in the same block (Declared but not used Error)
* Global variable may not be used.


### Control

* `select`

* `goto`

* `for...range`

  ```go
  // array
  nums := []int{2, 3, 4}
  for idx, num := range nums {
      fmt.Println("index:", idx, " value:", num)
  }
  
  // map
  kvs := map[string]string{"a": "apple", "b": "banana"}
  for k, v := range kvs {
      fmt.Printf("%s -> %s\n", k, v)
  }
  ```

  
### Function

```go
func name([param [type1], param2 [type2]]) [return type] {}
```


### Array

```go
var name [size]type
var name [size]type{a, b ,c}
var name [...]type{a, b ,c} // auto

// pass to function
func foo(arr [] int, size int) int {}
```


### Pointer

```go
var ptr *type // nil
var ptr = &a
var ptr [size]*type

// pass to function
func swap (x* int, y* int) {
    var temp int
    temp = *x
    *x = *y
    *y = temp
}
```


### OOP

* `struct`

  ```go
  type name struct {
      member type
      ...
  }
  
  var tmp = name{val, ...}
  var tmp = new(name)
  
  tmp.member
  
  var ptr *name = &tmp
  ptr.member // no ->
  ```

* `interface`

  ```go
  type name interface {
      method [return_type]
      ...
  }
  ```
```
  
* define method

  ```go
  type Circle struct {
    radius float64
  }
  
  func main() {
    var c1 Circle
    c1.radius = 10.00
    fmt.Println("Area of Circle(c1) = ", c1.getArea())
  }
  
  func (c Circle) getArea() float64 {
    return 3.14 * c.radius * c.radius
  }
  
  // c is a tmp instance 
```


### Slice (dynamic array)

```go
var slc []type
var slc []type = make([]type, len)
var slc []type{a,b,c}

slc = append(slc, d, e)

```


### Map

```go
// declare
var m map [key_type] val_type
m := make(map [key_type] val_type)

// assign
m[key] = val

// access
val, ok = m[key]

// del
delete(m, key)

```


### Go Routine

In-born multi-threading support.

* `go`

  ```go
  func loop() {
      for i:=0; i<10; i++ {
          fmt.Printf("%d ", i)
      }
  }
  
  func main() {
      go loop()
      go loop()
      time.Sleep(time.Second)
  }
  ```

* `chan`

  channels are like pipes for message passing.

  ```go
  ch := make(chan int)
  var ch chan int = make(chan int)
  
  ch <- message
  out, ok = <- ch
  
  // example
  var complete chan int = make(chan int)
  
  func loop() {
      for i := 0; i < 10; i++ {
          fmt.Printf("%d ", i)
      }
      complete <- 0
  }
  
  func main() {
      go loop()
      <- complete // 直到线程跑完, 取到消息. main在此阻塞住
  }
  
  ```

* `select`


