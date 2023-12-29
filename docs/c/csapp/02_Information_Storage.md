## Information Storage


### Integer

![image-20221202105118312](02_Information_Storage.assets/image-20221202105118312.png)

Unsigned Encodings:

![image-20221202105322088](02_Information_Storage.assets/image-20221202105322088.png)

Signed Encodings (Two's complement):

![image-20221202105340672](02_Information_Storage.assets/image-20221202105340672.png)

Conversion between signed and unsigned:

![image-20221202105603669](02_Information_Storage.assets/image-20221202105603669.png)

WARNING: C implicitly cast signed to unsigned.

```cpp
int a = -1;
unsigned int b = 0;
assert(a > b); // true, signed -1 --> unsigned 2^{32}-1 > 0
```


### Integer Arithmetic

Overflow:

* unsigned:

  ![image-20221202110344958](02_Information_Storage.assets/image-20221202110344958.png)

* signed

  ![image-20221202110308769](02_Information_Storage.assets/image-20221202110308769.png)


### Floating Point

![image-20221202113856519](02_Information_Storage.assets/image-20221202113856519.png)

* Normalized (most numbers):

  ![image-20221202114105317](02_Information_Storage.assets/image-20221202114105317.png)

  ![image-20221202114156987](02_Information_Storage.assets/image-20221202114156987.png)

  

* Denormalized (zero, and near-zero small numbers)

  ![image-20221202114625408](02_Information_Storage.assets/image-20221202114625408.png)

  

* Special (Infinity and NaN)

  ![image-20221202114331717](02_Information_Storage.assets/image-20221202114331717.png)

  ![image-20221202114349031](02_Information_Storage.assets/image-20221202114349031.png)

​	