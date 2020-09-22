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
int** pb = &b[0][0];

// initalize
int x[] = {1,2,3};  // x has type int[3] and holds 1,2,3
int y[5] = {1,2,3}; // y has type int[5] and holds 1,2,3,0,0
int z[3] = {0};     // z has type int[3] and holds all zeroes
```



### Strings

```
char* s;
s = "string";

char s[] = "string";
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
double log(double x);
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

