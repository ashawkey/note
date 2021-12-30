## Rust basics

### Install

```bash
curl https://sh.rustup.rs -sSf | sh
```



### Hello World

```rust
fn main() {
    // macro call
    println!("Hello World!");
}
```

compile and run:

```bash
rustc hello.rs
./hello
```

another example:

```rust
fn main() {
    let answer = 42;
    assert_eq!(answer,42);
}

fn main() {
    // 0 <= i <= 4
    for i in 0..5 {
		let even_odd = if i % 2 == 0 {"even"} else {"odd"};
        println!("{} {}", even_odd, i);
    }
}
```



