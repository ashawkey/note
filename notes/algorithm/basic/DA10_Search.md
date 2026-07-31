# Search

## 搜索

* 找到关键码**符合特定约束条件**的记录集合。
* **效率**十分重要。

  平均检索长度：$ASL = \sum_{i=1}^{n} p_i c_i$

* How to improve searching efficiency: **sorting, indexing, hashing.**

## 线性检索

### 顺序检索

设置下标 0 处为哨岗，从尾部检索到此处仍未发现目标元素，意味着检索失败。

$$
\frac{n+1}{2} < ASL < n+1
$$

* 检索成功：假设 $p_i = \frac 1 n$：

$$
\sum_{i=1}^{n} \frac 1 n (n - i + 1) = \frac{n+1}{2}
$$

* 检索失败：$n+1$

### 二分检索

$$
ASL = \frac 1 n \left( \sum_{i=1}^{\lg n} i \cdot 2^{i-1} \right) \sim O(\lg n)
$$

* 需要先排序。
* 成功或失败：$\sim O(\lg n)$
* 检索成功最多 $\lceil \lg(n+1) \rceil$ 次，最少 1 次。
* 检索失败时，最少比较 $\lfloor \lg(n+1) \rfloor$ 次，最多比较 $\lceil \lg(n+1) \rceil$ 次。

  eg. 22 个元素，失败时最少也要 4 次比较，最多 5 次。

### 分块检索

分块有序：先检索在哪一块中（二分检索），后在块内检索（顺序检索）。

$$
ASL_{succ} = ASL_b + ASL_s \approx \lg b + s/2 \approx \lg\left(\frac n s + 1\right) + s/2
$$

## 散列检索

* 思想

  受数组 $O(1)$ 寻址的启发：`Hash function(Key) = Sequential list index.`

* 概念

  * 负载因子 $\alpha$：**已有节点数 / 散列表空间大小**
  * 冲突：不同关键码映射到同一散列地址。
  * 同义词：冲突的关键码互称同义词。

### 散列函数构造

**先把关键码转换为整数。**

* 除余法

  $h(x) = x \bmod M$

  M 一般选择素数（至少不是 2 的幂）。连续关键码会映射到连续地址，可能导致性能不佳。

  **最佳除余选择**：已知关键码序列长 M，负载因子 a，则最佳余数 d 为小于表长 $N = M/a$ 的最大素数。

  eg. 给定关键码序列 26, 25, 20, 33, 21, 24, 45, 204, 42, 38, 29, 31，用散列法进行存储（本题采用闭散列方法解决冲突），规定负载因子 α = 0.6。则最合理的除余法的散列函数：$H(k) = k \% 19$（表长 $12/0.6 = 20$，小于 20 的最大素数为 19）。

* 除余取整法

  $h(x) = \lfloor n \cdot ((A \cdot x) \bmod 1) \rfloor$，$0 < A < 1$

* **平方取中法**

  平方扩大差别，之后取其中（二进制）的几位数的组合作为散列地址。

  **统计上最接近于随机化。**

* 数字分析法

  选取关键码中分布均匀的几位作为散列地址。某一位上各个符号出现的均匀度可以用**方差**衡量，选取方差较小的几位。

  缺点：需要事先知道 Key 的分布。

* 基数转换法

$$
x_{(a)} \rightarrow x_{(b)} = y_{(a)}
$$

  其中 a、b 为两个互素的基数，一般选择 b 大于 a。选取 y 的几位作为 address。

* 折叠法

  把 key 分为位数相同的几块，之后叠加分块后的数字，一般不进位。

  移位叠加：12 34 56 -> 12 + 34 + 56 = 92

  分界叠加：12 34 56 -> 12 + 43 + 56 = 01

### 冲突解决

冲突不可避免。

#### 开散列（拉链法）

同义词存储在同一地址的链表中。

**负载因子可以大于 1，但一般选择小于等于 1。**

* 拉链法（内存）：实现简单，最为常用。

* 桶式散列（外存）

  减少检索同义词时**同义词在不同页块**的外存访问时间。把外存中的记录分桶存放，每个桶包含指针连接的同义词页块（外存），每个页块可能包含多个记录。桶目录本身也可以放在外存。

#### 闭散列（开地址法）

把冲突的关键码根据探查函数**后移**到另一个空地址内。

更加复杂，非同义词也可能争夺同一地址（纠缠）。

* 基地址：$d_0 = h(K)$
* 探查序列：$d_1, d_2, \cdots$
* 探查函数 p：$d_i = d_0 + p(K, i)$

##### 线性探查

$p(K, i) = i$

探查序列：$d_0 + 1, d_0 + 2, \cdots, M, 0, 1, \cdots, d_0 - 1$

* 优点：所有位置都可以作为候选。
* 缺点：**聚集**现象严重（散列地址不同的记录争夺同一后继探查地址）。探查序列过长则接近顺序检索。

Example：

`(26,36,41,38,44,15,68,12,06,51,25)`

`M = 15, h(x) = x % 13, p(x, i) = i`

生成的散列表为：

| 0    | 1    | 2    | 3    | 4    | 5    | 6    | 7    | 8    | 9    | 10   | 11   | 12   | 13   | 14   |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| 26   | 25   | 41   | 15   | 68   | 44   | 06   |      |      |      | 36   |      | 38   | 12   | 51   |

下一条记录被放到 11 中的概率：2/13（基地址 10、11）

> 注意：插入只会映射到前 13 个槽，分母不是 M。

下一条记录被放到 7 中的概率：9/13（基地址 0~7、12）

$$
ASL_{succ} = \frac 1 {11} (1+1+1+1+1+2+2+2+1+3+5) = 20/11 \\
ASL_{fail} = \frac 1 {13} (8+7+6+5+4+3+2+1+1+1+2+1+11) = 4
$$

改进：`p(K, i) = i * c`（仍然会纠缠）。

##### 二次探查

$$
p(K, 2i-1) = i^2 \\
p(K, 2i) = -i^2
$$

基本消除聚集。

##### 伪随机数序列探查

$$
p(K, i) = perm[i-1]
$$

perm 为 [1, M-1] 的伪随机序列。**基本消除聚集。**

##### 双散列探查

* 基本聚集（已解决）：基地址不同的关键码，探查序列某些段重叠在一起。
* **二级聚集**：两个关键码散列到同一个基地址，得到的探查序列相同所产生的聚集。

  （二次聚集的原因：之前的三种方法得到的探查序列只是基地址的函数，而不是关键码的函数。）

* **双散列探查**：使用两个散列函数 $h_1, h_2$，如果 $h_1(key) = d$ 冲突，则计算 $h_2(key)$，根据这个值计算探查序列 $(d + k \cdot h_2(key)) \bmod M$。$h_2(key)$ 必须与 M 互素。

$$
d = h_1(key) \\
d_i = (d + p(key, i)) \bmod M \\
p(key, i) = i \cdot h_2(key)
$$

##### Implementation: Dictionary

```c++
Elem* HT;
int M;
int current;
Elem EMPTY;
Elem TOMB;
Elem temp;

bool hashInsert(const Elem& e) {
    int home = h(getkey(e));
    int i = 0, pos = home;
    while (HT[pos] != EMPTY) {
        if (e == HT[pos]) return false; // e exists
        i++;
        pos = (home + p(getkey(e), i)) % M;
    }
    HT[pos] = e;
    return true;
}

bool hashSearch(const Key& K, Elem& e) {
    int i = 0, pos = home = h(K);
    while (HT[pos] != EMPTY) {
        if (K == getkey(HT[pos])) {
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
Elem hashDelete(const Key& K) {
    int i = 0, pos = home = h(K);
    while (HT[pos] != EMPTY) {
        if (K == getkey(HT[pos])) {
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
// 有墓碑先插入墓碑，但由于要避免插入两个相同的元素，检索过程仍然要检索完全（排除 e exists 的情况）
bool hashInsert(const Elem& e) {
    int home = h(getkey(e));
    int i = 0, pos = home;
    int insplace;
    bool tomb = false;
    while (HT[pos] != EMPTY) {
        if (e == HT[pos]) return false; // e exists
        else if (TOMB == HT[pos] && !tomb) {
            insplace = pos;
            tomb = true;
        }
        i++;
        pos = (home + p(getkey(e), i)) % M;
    }
    if (!tomb) insplace = pos;
    HT[pos] = e;
    return true;
}
```

## 效率分析

经验表明，负载因子 α 小于 0.5 时，大部分操作预期代价均小于 2，远比二分检索优秀；但 α 大于 0.5 时性能急剧下降。

插入与删除频繁的散列表效率会降低（负载因子增大、同义词表变长、墓碑变多），可以通过定期重新散列来解决（清除墓碑，把访问最频繁的记录向前移动到基地址）。

#### Exercises

$$
ASL = 0.6 \cdot 1 + 0.4 \cdot (0.6 \cdot 2 + 0.4 \cdot (0.6 \cdot 3 + \cdots)) = (1-a) \sum_{i}^{N} (i \cdot a^i)
$$
