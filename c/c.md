# C



## Concepts

* ANSI C  / ISO C / Standard C

  eg. C99, C11, C17

  supported by GCC, MSCV.

  `gcc -std=c11 program.c`

* GNU C

  non-standard setup that GCC uses.

  `gcc program.c`

  

## Example

### Simple program

```c
// hello.c
#include <stdio.h>

int main() {
    printf("Hello, World!\n");
    return 0;
}
```

compile it with `gcc`:

```bash
gcc -o hello.out hello.c
./hello.out
```



## Tutorial

### Operator Precedence

|                          Precedence                          | Operator           | Description                                                  | Associativity |
| :----------------------------------------------------------: | :----------------- | :----------------------------------------------------------- | :------------ |
|                              1                               | `++` `--`          | Suffix/postfix increment and decrement                       | Left-to-right |
|                                                              | `()`               | Function call                                                |               |
|                                                              | `[]`               | Array subscripting                                           |               |
|                                                              | `.`                | Structure and union member access                            |               |
|                                                              | `->`               | Structure and union member access through pointer            |               |
|                                                              | `(*type*){*list*}` | Compound literal(C99)                                        |               |
|                              2                               | `++` `--`          | Prefix increment and decrement[[note 1\]](https://en.cppreference.com/w/c/language/operator_precedence#cite_note-1) | Right-to-left |
|                                                              | `+` `-`            | Unary plus and minus                                         |               |
|                                                              | `!` `~`            | Logical NOT and bitwise NOT                                  |               |
|                                                              | `(*type*)`         | Cast                                                         |               |
|                                                              | `*`                | Indirection (dereference)                                    |               |
|                                                              | `&`                | Address-of                                                   |               |
|                                                              | `sizeof`           | Size-of[[note 2\]](https://en.cppreference.com/w/c/language/operator_precedence#cite_note-2) |               |
|                                                              | `_Alignof`         | Alignment requirement(C11)                                   |               |
|                              3                               | `*` `/` `%`        | Multiplication, division, and remainder                      | Left-to-right |
|                              4                               | `+` `-`            | Addition and subtraction                                     |               |
|                              5                               | `<<` `>>`          | Bitwise left shift and right shift                           |               |
|                              6                               | `<` `<=`           | For relational operators < and ≤ respectively                |               |
|                                                              | `>` `>=`           | For relational operators > and ≥ respectively                |               |
|                              7                               | `==` `!=`          | For relational = and ≠ respectively                          |               |
|                              8                               | `&`                | Bitwise AND                                                  |               |
|                              9                               | `^`                | Bitwise XOR (exclusive or)                                   |               |
|                              10                              | `|`                | Bitwise OR (inclusive or)                                    |               |
|                              11                              | `&&`               | Logical AND                                                  |               |
|                              12                              | `||`               | Logical OR                                                   |               |
|                              13                              | `?:`               | Ternary conditional[[note 3\]](https://en.cppreference.com/w/c/language/operator_precedence#cite_note-3) | Right-to-left |
| 14[[note 4\]](https://en.cppreference.com/w/c/language/operator_precedence#cite_note-4) | `=`                | Simple assignment                                            |               |
|                                                              | `+=` `-=`          | Assignment by sum and difference                             |               |
|                                                              | `*=` `/=` `%=`     | Assignment by product, quotient, and remainder               |               |
|                                                              | `<<=` `>>=`        | Assignment by bitwise left shift and right shift             |               |
|                                                              | `&=` `^=` `|=`     | Assignment by bitwise AND, XOR, and OR                       |               |
|                              15                              | `,`                | Comma                                                        | Left-to-right |

* `%` in c keeps the sign:

  ```c++
  -1 % 3 = -1
  -2 % 3 = -2
  ```

  (different from python, which always return a positive integer)

* bit operators are even slower than compare operator !

  ```c++
  a|b == c; // a | (b == c)
  (a|b) == c; // (a|b) == c    
  ```

  

### Implicit type conversion

```
bool -> char -> short int -> int -> 
unsigned int -> long -> unsigned -> 
long long -> float -> double -> long double
```



### Pointer

```c
int x = 0;
int * ptr = &x;
int y = *ptr;
ptr = NULL;

int **pptr = &ptr;
```



### Array

```c
int a[5]; // a is a ptr to the first element in this array.
int* pa = &a[0]; // *pa == *a == a[0]

int b[3][3]; // b is a ptr to ptr
int** pb = &b[0][0]; // **pb == **b == b[0]

// initalize
int x[] = {1,2,3};  // x has type int[3] and holds 1,2,3
int y[5] = {1,2,3}; // y has type int[5] and holds 1,2,3,0,0
int z[3] = {0};     // z has type int[3] and holds all zeroes

int z[2][2] = {0, 1, 2, 3}; // {{0, 1}, {2, 3}}

int y[4][3] = { // array of 4 arrays of 3 ints each (4x3 matrix)
    { 1 },      // row 0 initialized to {1, 0, 0}
    { 0, 1 },   // row 1 initialized to {0, 1, 0}
    { [2]=1 },  // row 2 initialized to {0, 0, 1}
};              // row 3 initialized to {0, 0, 0}
```

##### Size of c arrays:

```c++

// pitfall: sizeof
// sizeof only works if the array is on the stack (declared in the same namespace)
// if the array is passed to a function, it will be recognized as a pointer, thus losing size information.

#include <stdio.h>
#include <stdlib.h>

void printSizeOf(int intArray[]);
void printLength(int intArray[]);

int main(int argc, char* argv[])
{
    int array[] = { 0, 1, 2, 3, 4, 5, 6 };
	
    printf("sizeof of array: %d\n", (int) sizeof(array)); 
    // sizeof of array: 28
    printSizeOf(array); 
    // sizeof of parameter: 8

    printf("Length of array: %d\n", (int)( sizeof(array) / sizeof(array[0]) )); 
    // Length of array: 7
    printLength(array); 
    // Length of parameter: 2
}

void printSizeOf(int intArray[])
{
    printf("sizeof of parameter: %d\n", (int) sizeof(intArray));
}

void printLength(int intArray[])
{
    printf("Length of parameter: %d\n", (int)( sizeof(intArray) / sizeof(intArray[0]) ));
}

```

##### Pass & Return array in c function:

* pass: pointer + size is the most recommended way.

* return: can only return dynamically allocated array's pointer.

  instead of return a pointer, it's better to pass the return value as parameter and **return void**.

* 2d array can be folded as 1d, and fix index liike `i * N + j`

```c++
const int M = 3;
const int N = 3;

// pass 1d
void f(int* arr, int size) {} // dynamic, recommended
void f(int arr[], int size) {} // same
void f(int arr[2]) {} // must be a const size

// return 1d
int* f(int size) {
    // malloc 1d array
    int* res = (int*)malloc(size * sizeof(int)); // don't forget free(res)
    return res; 
}

// wrong! never declare stack variable in function
int* f(int size) {
    int res[size];
    return res; // buggy, res is only valid inside the function!
}

// pass 2d
void f(int** arr, int m, int n) { // dynamic, recommended.
    // ...
}

const int M = 10;
const int N = 10;
void f(int arr[M][N]) { // must use two consts
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            // ...
        }
    }
}

const int N = 10;
void f(int arr[][N], int m) { // the second dim is necessary!
	// ...    
} 

// return 2d
int** f(int h, int w) {
    // ...
}


int main() {
    // declare 2d array in stack
    int a[M][N];
    
    // malloc 2d array in heap
    int** b;
    b = (int**)malloc(M * sizeof(int*));
    for (int i = 0; i < M; i++) 
        b[i] = (int*)malloc(N * sizeof(int));
    
}
```



### Chars

```c
char* s;
s = "string"; 

char s[] = "string";

// char to int
char c = '1';
int ic = c; // 49, ascii code of '1'
int i = c - '0'; // 1

// int to char
int i = 1;
char c = i + '0'; // '1'

// chars to int, do not recommend, use c++ string.
char s[] = "12";
int i = atoi(s); // 12
```



### pass-by-reference in function args

```c
// this doesn't work, because calls in C are pass-by-value.
void inc(int x) { x += 1; }
// this works, by pass-by-reference
void inc(int &x) { x += 1; }
// this also works, by pointer, but need to pass in ptr.
void int(int *px) { *px += 1; }

```



### CMD args

```c
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
    while(argc--) printf("%s\n", *argv++);
    return 0;
}
```



### Struct

```c
// basic
struct mystr {
    int len;
    char * s;
};
struct mystr s1;

// typedef
typedef struct mystr {
    int len;
} mystr;
mystr s1;

int len = s1.len;

mystr* ps1 = &s1;
int len = ps1->len;


// annonymous
struct {
    int len;
} s1;
```



### typedef

```c
typedef unsigned char BYTE;
BYTE b;
```



### Macro

```c
#define ONE 1
#undef ONE

#ifdef 
#ifndef
#endif

#pragma
```



### IO

```c
#include <stdio.h>

int main() {
    int x = 0;
    printf("%d", x);
    char* s = "string";
    printf("%s", s);
    
    char str[100];
    int i;
    scanf("%s %d", str, &i);
    printf( "\nYou entered: %s %d \n", str, i);
}
```



### C Library

#### #include <stdio.c>

```c
typedef long long unsigned int size_t // x64
    
FILE *stdin = (FILE *) &_IO_2_1_stdin_;
FILE *stdout = (FILE *) &_IO_2_1_stdout_;
FILE *stderr = (FILE *) &_IO_2_1_stderr_;

FILE *fopen(const char* filename, const char* mode);
int getchar(void);
int putchar(int char);
int printf(const char *format, ...);
int scanf(const char *format, ...);
```

#### #include <math.c>

```c
double exp(double x);
double log(double x); // log_e 
double log2(double x);
double log10(double x);
double pow(double x, double y);
double sqrt(double x);
double fabs(double x);
double floor(double x);
double ceil(double x);
```

#### #include <stdlib.c>

```c
double atof(const char *str);
int atoi(const char *str);

double strtod(const char *str, char **endptr);
long int strtol(const char *str, char **endptr, int base);

void *calloc(size_t nitems, size_t size);
void *malloc(size_t size);
void free(void *ptr);

void exit(int status);

int abs(int x);
void qsort(void *base, size_t nitems, size_t size, int (*compar)(const void *, const void*));

int rand(void);
void srand(unsigned int seed);
```

```c
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
 
int main() {
   char *str;
 
   str = (char *) malloc(15 * sizeof(char));
   strcpy(str, "runoob");
   printf("String = %s,  Address = %u\n", str, str);
   free(str);
 
   return 0;
}
```



#### #include <string.c>

```c
void *memcpy(void *dest, const void *src, size_t n);
void *memset(void *str, int c, size_t n);
int memcmp(const void *str1, const void *str2, size_t n);

char *strcat(char *dest, const char *src);
int strcmp(const char *str1, const char *str2);
char *strcpy(char *dest, const char *src);
size_t strlen(const char *str);
char *strstr(const char *haystack, const char *needle);
```

