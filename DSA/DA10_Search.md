# Search

### 搜索

* 找到关键码**符合特定约束条件**的记录集合。

* **效率**十分重要。

  平均检索长度：$AVL = \sum_{i=1}^np_ic_i$

* How to improve searching efficiency:

  **sorting, indexing, hashing.**


### 线性检索

#### 顺序检索

设置下标0处为哨岗，从尾部检索到此处仍未发现目标元素，意味着检索失败。
$$
\frac {n+1} 2 < ASL < n+1
$$

* 检索成功: 假设 $p_i$ is $\frac 1 n$:
  $$
  \sum_{i=1}^n\frac 1 n(n-i+1) = \frac {n+1} 2
  $$

* 检索失败: $n+1$


#### 二分检索

$$
ASL = \frac 1 n (\sum_{i=1}^{lg\ n}i\cdot 2^{i-1}) \\
\sim O(lg\ n)
$$

* Need Sorting first

* Success or fail: $\sim O(lg\ n)$

* 检索成功最多$ceil(lg(n+1))$次，最少1次。

* 检索失败时，最少比较$floor(lg(n+1))$次 ，最多比较$ceil(lg(n+1))$次。

  eg. 22个元素，失败时最少也要4次比较，最多5次。


#### 分块检索

分块有序，先检索在哪一块中（二分检索），后在块内检索（顺序检索）。

![1543372011178](C:\Users\hawke\AppData\Roaming\Typora\typora-user-images\1543372011178.png)
$$
ASL_{succ} = ASL_b + ASL_s \\
\approx lg\ b + s/2 \\
\approx lg(\frac n s + 1) + s/2
$$


### 散列检索

* 思想

  Inspired by $O(1)$ complexity of Array indexing.

  `Hash function(Key) = Sequential list index.`


* 概念
  * 负载因子 $\alpha$：**已有节点数/散列表空间大小**
  * 冲突：不同关键码映射到同一散列地址。
  * 同义词：冲突的关键码互称同义词。

#### 散列函数构造

**先把关键码转换为整数。**

* 除余法

  $h(x) = x\ mod\ M$

  M一般选择非2的素数。

  Drawbacks: Continuous keys will be mapped to Continuous addresses. This may lead to bad performance.

  **最佳除余选择：**已知关键码序列长M，负载因子a，则最佳余数d为小于表长N=M/a的最大素数。

  eg.	给定关键码序列26, 25, 20, 33, 21, 24, 45, 204, 42, 38, 29, 31，用散列法进行存储(本题采用闭散列方法解决冲突)，规定负载因子α=0.6。则最合理的除余法的散列函数：$H(k) = k \% 19$

* 除余取整法

  $h(x) = floor(n*((A*x)\%1)$

  $0 < A < 1$

* **平方取中法**

  平方扩大差别，之后取其中（二进制）的几位数的组合作为散列地址。

  **统计上最接近于随机化。**

* 数字分析法

  选取关键词中分布均匀的几位作为散列地址。

  某一位上各个符号出现的均匀度可以用**方差**衡量，选取方差较小的几位。

  Drawbacks：需要事先知道Key的分布。

* 基数转换法
  $$
  x_{(a)} \rightarrow x_{(b)} = y_{(a)}
  $$
  其中a，b为两个互素的基数，一般选择b大于a。

  选取 y 的几位作为address。

* 折叠法

  把key分为位数相同的几块，之后叠加分块后的数字。一般不进位。

  移位叠加: 12 34 56 -> 12 + 34 + 56 = 92

  分界叠加: 12 34 56 -> 12 + 43 + 56 = 01


#### 冲突解决

冲突不可避免。

##### 开散列（拉链法）

同义词存储在同一地址的链表中。

**负载因子可以大于1，但一般选择小于等于1。**

* 拉链法（内存）

  ![1543375280320](C:\Users\hawke\AppData\Roaming\Typora\typora-user-images\1543375280320.png)

  实现简单，最为常用。

* 桶式散列（外存）

  减少检索同义词时，**同义词在不同页块时**的外存访问时间。

  把外存中的记录分桶存放，每个桶包含指针连接的同义词页块（外存），每个页块可能包含多个记录。

  桶目录本身也可以放在外存。


##### 闭散列（开地址法）

把冲突的关键码根据探查函数**后移**到另一个空地址内。

更加复杂，非同义词也可能争夺同一地址（纠缠）。

基地址：$d_0 = h(K)$

探查序列：$d_1, d_2, ...$

探查函数p：$d_i=d_0+p(K, i)$

* 线性探查

  $p(K, i) = i$

  探查序列：$ d_0+1, d_0+2, .., M, 0, 1, ..., d_0-1$

  * 优点：所有位置都可以作为候选。

  * 缺点：**聚集**现象严重（散列地址不同的记录争夺同一后继探查地址）。探查序列过长，则接近顺序检索。

  Example：

  `(26,36,41,38,44,15,68,12,06,51,25)`

  `M = 15, h(x) = x % 13, p(x, i) = i`

  生成的散列表为：

  ![1543377053041](C:\Users\hawke\AppData\Roaming\Typora\typora-user-images\1543377053041.png)

  下一条记录被放到11中的概率：2/13 （10， 11）

  ​	注意插入只会映射到前13个槽，分母不是M。

  下一条记录被放到7中的概率：9/13 （0~7，12）
  $$
  ASL_{succ} = \frac 1 {11} (1+5+1+2+2+1+1+1+1+2+3) = 20/11 \\
  ASL_{fail} = \frac 1 {13} (8+7+6+5+4+3+2+1+1+1+2+1+11) = 4
  $$







  ​	改进：`p(K, i)=i*c`（仍然会纠缠）

​	

  * 二次探查
    $$
    p(K, 2i-1) = i^2 \\
    p(K, 2i) = -i^2 
    $$
    基本消除聚集。

  * 伪随机数序列探查
    $$
    p(K, i) = perm[i-1]
    $$
    perm为[1, M-1]的伪随机序列。

    **基本消除聚集。**

  * 双散列探查

    基本聚集（已解决）：基地址不同的关键码，探查序列某些段重叠在一起。

    **二级聚集**：两个关键码散列到同一个基地址，得到的探查序列相同所产生的聚集。

    （二次聚集的原因：之前的三种方法得到的探查序列只是基地址的函数，而不是关键码的函数）

    **双散列探查**：使用两个散列函数$h1, h2$，如果$h1(key) = d$ 冲突，则计算$h2(key)$，根据这个值计算探查序列$(d + k * h2(key)) \% M $ 。

    $h2(key)​$必须与M互素。

$$
d = h_1(key) \\
d_i = (d + p(key, i)) \% M \\
p(key, i) = i*h_2(key)
$$


* Implementation：Dictionary

  ```c++
  Elem* HT;
  int M;
  int current;
  Elem EMPTY;
  Elem TOMB;
  Elem temp;
  
  bool hashInsert(const Elem& e){
      int home = h(getkey(e));
      int i = 0, pos = home;
      while(HT[pos] != EMPTY){
          if(e == HT[pos]) return false; // e exists 
          i++;
          pos = (home + p(getkey(e), i)) % M;
      }
      HT[pos] = e;
      return true;
  }
  
  bool hashSearch(const Key& K, Elem& e){
      int i=0, pos = home = h(K);
      while(HT[pos] != EMPTY){
          if(K == getkey(HT[pos])){
              e = HT[pos];
              return true;
          }
          i++;
          pos = (home + p(K, i)) % M;
      }
      return false;
  }
  
  // delete is difficult
  // 只有开散列才能真正删除。闭散列只能做标记，以免影响后续检索的正确性。
  Elem hashDelete(const Key& K){
      int i=0, pos = home = h(K);
      while(HT[pos] != EMPTY){
          if(K == getkey(HT[pos])){
              temp = HT[pos];
              HT[pos] = TOMB;
              return temp;
          }
          i++;
          pos = (home + p(K, i)) % M;
      }
      return EMPTY;
  }
  
  // modified tomb insert 
  // 有墓碑先插入墓碑，但由于要避免插入两个相同的元素，检索过程仍然要检索完全（排除e exists的情况）
  bool hashInsert(const Elem& e){
      int home = h(getkey(e));
      int i = 0, pos = home;
      int insplace;
      bool tomb = false;
      while(HT[pos] != EMPTY){
          if(e == HT[pos]) return false; // e exists 
          else if(TOMB == HT[pos] && !tomb){
              insplace = pos;
              tomb = true;
          }
          i++;
          pos = (home + p(getkey(e), i)) % M;
      }
      if(!tomb) insplace = pos;
      HT[pos] = e;
      return true;
  }
  ```





### 效率分析

![1543980144853](C:\Users\hawke\AppData\Roaming\Typora\typora-user-images\1543980144853.png)



经验表明，负载因子alpha小于0.5时，大部分操作预期代价均小于2，远比二分检索优秀，但是alpha大于0.5时性能急剧下降。

插入与删除频繁的散列表效率会降低（负载因子增大，同义词表变长，墓碑变多），可以通过定期重新散列来解决（清除墓碑，把访问最频繁的记录向前移动到基地址）





#### Exercises

![1544104068419](C:\Users\hawke\AppData\Roaming\Typora\typora-user-images\1544104068419.png)
$$
ASL = 0.6*1 + 0.4*(0.6*2+0.4*(0.6*3+...)) = (1-a)\sum_i^{N}(i*a^i)
$$
