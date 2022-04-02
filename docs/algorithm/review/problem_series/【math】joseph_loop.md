### [Joseph Loop](https://leetcode-cn.com/problems/yuan-quan-zhong-zui-hou-sheng-xia-de-shu-zi-lcof/)

> 0,1,···,n-1这n个数字排成一个圆圈，从数字0开始，每次从这个圆圈里删除第m个数字（删除后从下一个数字开始计数）。求出这个圆圈里剩下的最后一个数字。例如，0、1、2、3、4这5个数字组成一个圆圈，从数字0开始每次删除第3个数字，则删除的前4个数字依次是2、0、4、1，因此最后剩下的数字是3。

并不用模拟，是有递推公式的。

$f(n, m)$ 代表此问题最后剩下的人的编号。

$$

f(n, m) = f(n-1, m) + m \mod n

$$


```cpp
class Solution {
public:
    int lastRemaining(int n, int m) {
        if (n == 1) return 0;
        int x = lastRemaining(n - 1, m);
        return (m + x) % n;
    }
};
```

递归转换为迭代：

```cpp
class Solution {
public:
    int lastRemaining(int n, int m) {
        int ans = 0;
        for (int i = 2; i <= n; i++) {
            ans = (ans + m) % i;
        }
        return ans;
    }
};
```



### [消除游戏](https://leetcode-cn.com/problems/elimination-game/)

> 列表 arr 由在范围 [1, n] 中的所有整数组成，并按严格递增排序。请你对 arr 应用下述算法：
>
> 从左到右，删除第一个数字，然后每隔一个数字删除一个，直到到达列表末尾。
> 重复上面的步骤，但这次是从右到左。也就是，删除最右侧的数字，然后剩下的数字每隔一个删除一个。
> 不断重复这两步，从左到右和从右到左交替进行，直到只剩下一个数字。
> 给你整数 n ，返回 arr 最后剩下的数字。

类似，设结果为$f(n)$，即先从左向右删除，再从右向左，...，直到剩下最后一个数字。

考虑对称过程$f'(n)$，即先从右向左删除，再从左向右，...，直到剩下最后一个数字。

```
# f(n) 表示从左到右剩下的数字的结果, f'(n) 表示从右到左删除的结果
# 对称性: f(n) + f'(n) = n + 1
# 递归性: f(n) = 2 * f'(n/2)
# 初始条件: f(1) = f'(1) = 1

# 根据以上条件可得: f(2 * n)/2 + f(n) = n + 1
# f(n)/2 + f(n/2) = n/2 + 1
# f(n) = (n/2 + 1 - f(n/2)) * 2
```

递归求解：

```cpp
class Solution {
public:
    int lastRemaining(int n) {
        return n == 1 ? 1 : 2 * (n / 2 + 1 - lastRemaining(n / 2));
    }
};
```

