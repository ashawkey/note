# Introduction

### Data Structure

* **Logical Structure**

  $(K,R)$ : $K$ is the set of Nodes(basic datatype or complex structure), $R$ is the set of binary Relationships.

  Classification by $R$:

  * set structure

  * linear structure

    前驱关系。关系有向，满足全序性（两两节点可以比较先后）、单索性（每个节点存在唯一一个前驱和后继结点）

  * tree structure

    层次结构。直接前驱唯一，但可以有多个直接后继。

    根节点无前驱。

  * graph structure

    网络结构。没有任何关系约束。

* **Storage/Physical Structure**

  内存：空间相邻，随机访问。

  存储结构实现逻辑到物理的映射（节点映射、**逻辑关系映射**）

  * 顺序

    数组。紧凑存储结构（不存储任何附加信息）

  * 链接

    链表。易于增删，难于定位。

  * 索引

    顺序存储的推广。构造索引函数$Y:Z\rightarrow D$. 使得整数域映射到存储地址。

  * 散列

    索引方法的推广。构造散列函数$h:S\rightarrow Z$.  $S$ 为关键码。

* ADT(Abstract Data Type)

  ADT = {数据对象D，数据关系S，数据操作P}

  eg. c++类模板。

### Algorithm

**Data Structure + Algorithm = Program**

* Classification

  * Enumeration
  * Searching(BFS, DFS), Back Tracking
  * Greedy
  * Recursive, Divide and Conquer
  * Dynamic Programming

* Analysis

  **Relationship between Algorithm Complexity and Input Scale.**

  Absolute time is also required but not the most important.

  * $O()$ expression
    $$
    \exist C,N_0 \gt 0 \ \ s.t. \ \forall n \ge N_0:\\
    f(n) \le C \cdot g(n) \\
    \Rightarrow f(n) \in O(g(n))
    $$
    We should choose the **tightest** $g(n)$ .

    eg. $f(x) = n^2 + 2n$ is $O(n^2)$, but we can also say it's $O(n^3)$ or larger.

  * $\Omega()$ expression
    $$
    \exist C,N_0 \gt 0 \ \ s.t. \ \forall n \ge N_0:\\
    f(n) \ge C \cdot g(n) \\
    \Rightarrow f(n) \in \Omega(g(n))
    $$
    Choose the tightest.

  * $\Theta()$ expression
    $$
    f(n) \in O(g(n)) \ and \ f(n) \in \Omega(g(n)) \\
    \Rightarrow f(n) \in \Theta(g(n))
    $$


  Best, Worst and Average Complexity analysis: For most Algorithms, the difference is only in the constants and coefficients.

  Trade-off between Time and Space.
