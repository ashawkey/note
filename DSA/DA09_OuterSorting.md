# Outer Sorting

### 流程

**置换选择**：将外存数据划分为尽可能长的顺串（已排序的外存子串）

**归并**：归并所有顺串。

### 时间构成

* 产生初始顺串的内排序时间
* 初始化顺串与归并的IO时间
* 归并时间

### 置换选择排序

把外存数据转换为多个长度不等的顺串。每次算法结束时，对RAM中剩余的无法处理元素重新建立最小堆，继续生成顺串。最短顺串长度为堆的大小M，平均长度2M。

```c++
template<class elem>;
void ReplacementSelection(elem *a, int n){
    elem mn, tmp;
    readFromInput(a, n); // read n elem to a
    buildMinHeap(heap, a); // build heap from a
    for(int last=n-1; last>=0;){
        elem mn = heap[0];
        sendToOutputBuffer(mn);
        readFromInput(tmp, 1);
        if(tmp >= mn) heap[0] = tmp;
        else{
            heap[0] = heap[last];
            heap[last] = tmp;
            last--;
        }
        if(last) siftdown(0);
    }
}
```

* 读入M个元素，建立最小堆。
* 输出堆顶元素A。
* 读入新元素B：
  * B大于A，则插入堆顶并sift down
  * B小于A，则与堆底交换，堆Size--
* 重复上两步至堆Size==0，重新建堆，开始下一个顺串输出。

### 归并

减少归并次数的角度：

* 减少初始顺串个数m。
* 增加同时归并的顺串数量k。

**如何提高k路归并每次找最小值的效率？**（naive: $O(k-1)$）

#### 胜者树

完全二叉树存储，叶节点为待归并的顺串，每个非叶节点存储其两个子节点中胜利的那个。

![1543030801537](C:\Users\hawke\AppData\Roaming\Typora\typora-user-images\1543030801537.png)

内部节点树深度s，最底层外部节点个数`LowExt`（篮框），`LowExt`以外的节点数`offset`（红框，满完全二叉树）。则L[i]与对应的父节点B[p]的关系为：
$$
s = ceil(log_2n)-1\\
LowExt = 2(n-2^s)\\
offset = 2^{s+1}-1 \\
p = \left\{
	\begin{array}{lr}
	(i+offset)/2, & i \le LowExt \\
	 (i-LowExt+n-1)/2, & i>LowExt
	\end{array}
\right.
$$


* 重构：

  每次移除最小值后，根节点的顺串首值改变，此时需要重构胜者树。

  只需要`siftdown`根节点即可，每次与兄弟比较，更改父节点。



#### 败者树

胜者树的变体，简化重构。

完全二叉树存储，叶节点为待归并的顺串，每个非叶节点存储其两个子节点中**失败**的那个。

新增根节点的父节点`B[0]`记录最终胜者。

```c++
struct T{}; // seq

int winner(T* A, int a, int b);
int loser(T*A, int a, int b);

struct LoserTree{
    int n, LowExt, offset;
    int *B;
    T* L;
    // p is B[p], lc/rc is L[lc/rc]
    void play(int p, int lc, int rc){
        B[p] = loser(L, lc, rc);
        int tmp1, tmp2;
        tmp1 = winner(L, lc, rc);
        while(p>1 && p%2){
            tmp2 = winner(L, tmp1, B[p/2]);
            B[p/2] = loser(L, tmp1, B[p/2]);
            tmp1 = tmp2;
            p/=2;
        }
        B[p/2] = tmp1;
    }
    void init(T* A, int size){
        n = size;
        L = A;
        int i,s;
        for(s=1; 2*s<=n-1; s+=s); // s
        LowExt = 2*(n-s);
        offset = 2*s - 1;
        //初始化内部节点树最底层的内部节点
        for(i=2; i<=LowExt; i+=2)
            play((offset+i)/2, i-1, i);
        //奇数还需要生成一个混合内部节点
        if(n%2){
            play(n/2, B[(n-1)/2], LowExt+1);
            i = LowExt+3;
        }
        else i=LowExt+2;
        for(;i<=n;i+=2){
            play((i-LowExt+n-1)/2, i-1, i);
        }
	}
    int winner(){
        return B[0];
    }
    void replay(int i){
        int p;
        if(i<=LowExt) p = (i+offset)/2;
        else p = (i-LowExt+n-1)/2;
        B[0] = winner(L, i, B[p]);
        B[p] = loser(L, i, B[p]);
        for(; (p/2)>=1; p/=2){
            int tmp;
            tmp = winner(L, B[p/2], B[0]);
            B[p/2] = loser(L, B[p/2], B[0]);
            B[0] = tmp;
        }
    }
};
```

复杂度分析：

初始化$O(k)$，每次操作$O(logk)$

生成n长的顺串总时间$O(k+nlogk)$

（naive需要$O(nk)$）



