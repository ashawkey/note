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



