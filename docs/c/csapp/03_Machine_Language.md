## Machine Language

### Compile

```bash
# -Og: do not optimize, generate assembly code according to the program
# -S: generate assembly
gcc -Og -S test.c # test.s
```



#### Registers

* Caller-saved / Callee-saved Register: different strategies to save register context at calling functions.

  ![image-20221202173128425](03_Machine_Language.assets/image-20221202173128425.png)

* Size of data type

  ![image-20221202173302728](03_Machine_Language.assets/image-20221202173302728.png)