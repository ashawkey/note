## Rust basics

### Install

```bash
# install rustup
curl --proto '=https' --tlsv1.2 https://sh.rustup.rs -sSf | sh

# update
rustup update

# uninstall
rustup self uninstall

# rust version
rustc --version

# local doc
rustup doc
```


### Hello World

```rust
/* The main function is the entry of any program.
BTW this is a block comment.
*/
fn main() {
    // macro call, BTW this is a line comment.
    println!("Hello World!");
}
```

compile and run:

```bash
# rust compiler
rustc hello.rs
./hello
```


### Package manager: Cargo
```bash
# check cargo version
cargo --version

# create a project
cargo new hello_cargo
cd hello_cargo
```

It creates a `Cargo.toml` config file:

```toml
[package] 
name = "hello_cargo" 
version = "0.1.0" 
edition = "2018" 

[dependencies]
```

and a `src` folder with `main.rs` in it.
To build and run a cargo project, go to the project directory and:

```bash
# first build then manually run
cargo build
# by default, it builds into debug mode
./target/debug/hello_cargo

# Or two steps in one:
cargo run

# just check compilation process, do not generate executable
# much faster for debugging!
cargo check 

# build for release
cargo build --release
./target/release/hello_cargo
```

To add a dependency in `Cargo.toml`:

```toml
[dependencies] 
rand = "0.8.3"
```

Then `cargo build` will automatically download and build it.
A `Cargo.lock` is generated to keep track of the **exact** dependency versions.
To update all dependencies:

```bash
# only update PATCH version! e.g., 0.8.3 --> 0.8.4
cargo update

# to update MAJOR or MINOR version, you must change the toml manually.
```

### Variables
Rust is a **statically and strongly typed** language.
However, rust can infer the data type from code, so you can declare variables without annotating data type.

#### Scalar types
- integers
	- signed: `i8, i16, i32, i64,  i128, isize` (isize means arch-dependent) 
	- unsigned: `u8, u16, u32, u64, u128, usize`
	* `i32` is the default.
```rust
let x = 0; // i32 by default
let x: u8 = 0; // u8
let x = 0u8; // u8
let x = 1_000; // i32, 1000
let x = 0xff // i32, hex
let x = 0b1111_0000 // i32, binary
```
- float
	- `f32, f64`
	- `f64` is the default.

```rust
let x = 2.0; // f64
let x: f32 = 2.0; // f32
```
* boolean
	* `bool`

```rust
let t = true; // bool
let f: bool = false; // bool
```

* character
	* `char`, but it is **4-byte** for unicode encoding. (not 1-byte ASCII as in `c`)

```rust
let c = 'z'; // char
let c = '😻'; // char, supports unicode emoji
```

#### Compound types
* tuple
	* A general way of grouping a number of values with any type.
	* fixed-length.
```rust
let t = (1, 2.0, 'c'); // simple tuple
let t: (i32, f64, char) = (1, 2.0, 'c'); // with type annotaiton

let x = t.0; // indexing
let y = t.1;
let z = t.2;

let (x, y, z) = t; // destructuring

let u = (); // empty tuple, or the unit value. (default value for empty expression)

// tuples can be useful in function 
fn foo(s: String) -> (String, u32) {
	(s, s.len())
}
```

* array
	- can only hold values of the same data type.
	- also fixed-length! (instead, use `Vec` for python-like list)
```rust
let a = [0, 1, 2]; // simple i32 array
let a: [i32; 3] = [0, 1, 2]; // type annotation [dtype; length]
let a = [0; 3]; // equals let a = [0, 0, 0];

println!("{:?}", a); // debug print for array

let x = a[0]; // indexing

// slicing
let arr = [1, 2, 3, 4, 5];
let s1: &[i32] = &arr; // full slice
let s2 = &arr[0..2] // [1, 2], partial slice
let s3 = &arr[1..] // [2, 3, 4, 5]

// iterator
for i in arr {}
for i in arr.iter() {} // same
for i in &arr {} // same


// pass array by reference to function
// to handle any-length arrays, we use slice of array, noted by &[dtype]
// thankfully, array slice knows its length, unlike the array pointer in c.
fn sum(xs: &[i32]) -> i32 {
	let mut res = 0;
	for i in 0..xs.len() {
		res += xs[i];
	}
	res
}

let arr = [1, 2, 3];
let res = sum(&arr); // 6
```

#### Mutability

```rust
fn main() {

	let x = 0; // immutable, infered as i32
	// x = 1; // compilation error

	let mut y = 0; // mutable
	y = 1; // Ok

	const z = 0; // constant

	let x = x + 1; // shadowing, OK. (reassigned the value)
	{
		let x = 2; // shadowing, only in current scope
		println!("x = {}", x); // x = 2
	}
	println!("x = {}", x); // x = 1
}

```


`const` are more than immutable variables:
- must be declared with data type.
- can be declared in any scope (e.g., the global scope)
- can only be set to a constant expression, not a value computed at runtime. (`3*6` is OK, but `3*x` is not.)
- (convention) name should be UPPER_CASE connected by underscores.


### Functions
rust will find the definition of function automatically, so you can define it anywhere like python and not like c.
* use `fn` to define a function
* must declare the parameter data type.
* if you return some value, must declare the return data type too.
* the last statement without semicolon will be returned implicitly, else it returns the unit value `()`.

```rust
// MUST declare the data type for parameters!
fn foo(x: i32) {
	println!("x = {}", x);
}

// return value as if the function is an expression
// MUST declare the data type for return value! if not declared, it is default to ()
fn bar() -> i32 {
	let y = { // start an expression block
		let x = 1;
		x + 1 // WITHOUT semicolon to serve as return value!!! if with semicolon, this expr returns () implicitly
	}; // y == 2
	y + 1 // equals `return y + 1;`
}

let x = bar(); // x == 3

// fibonacci example
fn fibonacci(x: i32) -> i32 {
	if x == 0 {
		1
	} else {
		x * fibonacci(x - 1)
	}
}
```

by default, parameters are passed by value in function.
To pass parameters by reference, we need reference operator `&` and dereference operator `*`.

```rust

let x = 1;
// pass by ref, and return increased value
fn inc(x: &i32) -> i32 {
	*x + 1
}
let x = inc(x);  // x == 2

// reference can be modified inplace
let mut x = 1;
// pass by ref, do not forget the mut
fn inplace_inc(x: &mut i32) {
	*x += 1;
}
inc(x); // x == 2

```

### Controls
#### condition

```rust
let x = 3;

// if <bool> {}
if x != 0 { // cannot use `if x {}`, no implicit casting!
	// do sth
} else if x % 2 == 0 {
	// do sth
} else {
	// do sth
}

// if in statement
let cond = true;
let x = if cond {5} else {6}; // data type must be the same in two branches.


```

#### loop

```rust
// pure loop
loop { println!("again!"); }

// break nested & labeled loop
let mut i = 10;
'flag1: loop {
	let mut j = 10;
	loop {
		if foo(i, j) == 0 {
			break; // break j loop
		} else if foo(i, j) == 1 {
			break 'flag1; // break i loop
		}
		j -= 1;
		if j == 0 {break;}
	}
	i -= 1;
	if i == 0 {break;}
}

// return value with break
let mut counter = 0; 
let result = loop { 
	counter += 1; 
	if counter == 10 { break counter * 2; } 
}; // result == 20


// while loop
let mut x = 3;
while x != 0 {
	x -= 1;
}

// for loop
let a = [1, 2, 3];
for x in a {
	println!("{}", x);
}

for x in 0..5 {} // for x in [0, 1, 2, 3, 4]

for x in (0..5).rev() {} // for x in [4, 3, 2, 1, 0]

```


### Strings

```rust
// &str (string literals)
// it create on stack a hardcoded string literals, and return its immutable slice reference.
let sl = "hello"; // hardcoded, immutable, fast, on stack.

// TODO: is there mutable string literals? like `let mut sl = "hello";`

// String
let mut s = String::from("hello"); // mutable, slower, on heap.
s.push_str(", world"); // s == "hello, world"
```


### Ownership
Rules of Ownership:
- Each value has a variable called its owner.
* There can only be one owner at a time.
* When the owner goes out of scope, the value will be dropped.

The move semantic:

```rust
// for scalar types (stack-only), there is no moving.
let x = 5; 
let y = x; // y == 5, and x is still 5 (copied x to y)

// for complex type like String (symbol on stack, data on heap), there is moving.
let x = String::from("hello");
let y = x; // y == "hello", but x has been invalid! (copied x's symbol to y, moved x's data to y, and dropped x's symbol)

// if you really want to copy data on heap
let x = String::from("hello");
let y = x.clone(); // x and y are both valid, each with its own data allocated on heap

// passing to function also invokes moving
fn foo(s: String) {}
foo(x); // x moved to foo(), and become invalid.

// returning from function too
fn foo(s: String) -> String {
	s;
}
let x = foo(x); // x is moved to foo(), then returned to x.

```

To avoid moving in function parameters, we need references & borrowing.
Rule of references:
* At any given time, you can have _either_ one mutable reference _or_ any number of immutable references.
- References must always be valid.

```rust

// &String means **reference** of String
fn foo(s: &String) -> u32 {
	s.len()
}
let s = String::from("hello");
let l = foo(&s); // s is still valid, we only **borrow** s in foo by using `&s`

// by default, reference is immutable.
// but we can also declare mutable reference:
fn foo(s: &mut String) {
	s.push_str(", world");
}
let mut s = String::from("hello"); // the variable also should be mutable
foo(&mut s); // mutable borrow, s is now "hello, world"

// however, we can only have one mutable reference for one variable at a time:
let r1 = &mut s;
let r2 = &mut s; // Error

// we also cannot use both mutable and immutable reference at a time:
let r1 = &s;
let r2 = &mut s; // Error if both r1 and r2 are used later:
println!("{}, {}", r1, r2);

let r1 = &s;
println!("{}", r1);
let r2 = &mut s; // OK
println!("{}", r2);

// rust can prevent dangling references：
fn dangle() -> &String { let s = String::from("hello"); &s }
let d = dangle(); // Error
```

A special type of reference is **slice**, which only reference to a continuous part of a collection (e.g., String, array).

```rust
let s = String::from("hello world"); 
let hello: &str = &s[..5]; // a reference to "hello" (point to 0, record length 5)
let world = &s[6..];
let ss = &s[..]; // equals to `&s`

// example
// note we use parameter type &str, instead of &String.
// this utilizes "defef coercion".
fn first_word(s: &str) -> &str { 
	let bytes = s.as_bytes(); 
	for (i, &item) in bytes.iter().enumerate() { 
		if item == b' ' { 
			return &s[0..i]; 
		} 
	} 
	&s[..] 
}

let sl = "hello, world";
let s = String::from(sl);
let w = first_word(&s);
let w = first_word(&s[0..6]);
let w = first_word(sl);
let w = first_word(&sl); // slice of slice is still slice...

```