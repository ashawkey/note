# [k inverse pairs array](https://leetcode-cn.com/problems/k-inverse-pairs-array/)

> 给出两个整数 `n` 和 `k`，找出所有包含从 `1` 到 `n` 的数字，且恰好拥有 `k` 个逆序对的不同的数组的个数。

令$f(i, j)$代表前$i$个数组成的序列中含有$j$个逆序对的个数。

假设我们知道$f(i-1, *)$, 考虑将$i$插入到$k=0 \rightarrow i-1$的位置处**新**得到的逆序对数量为$i-1-k$，从而：
$$
f(i, j) = \sum_{k=0}^{i-1}f(i-1, j-(i-1-k))=\sum_{k=0}^{i-1}f(i-1, j-k)
$$
展开可发现：
$$
\begin{align}
f(i,j)    &= f(i-1,j) + &f(i-1, j-1) + \cdots + &f(i-1, j-i+1) \\
f(i, j-1) &=            &f(i-1, j-1) + \cdots + &f(i-1, j-i+1) + f(i-1, j-i) \\
\end{align}
$$
从而：
$$
f(i,j) = f(i-1,j) + f(i, j-1)-f(i-1,j-i)
$$
边界条件：
$$
f(*, 0) = 1 \quad \text{specifically, } f(1, 0) = 1\\
f(1, j) = 0 \quad \text{if} ~ j > 0\\
f(*, j) = 0 \quad \text{if} ~ j < 0 \\
$$
动态规划：

```cpp
class Solution {
public:
    const static int M = 1e9 + 7;
    int kInversePairs(int n, int k) {
        vector<vector<long long>> dp(n+1, vector<long long>(k+1, 0));
        for (int i = 1; i <= n; i++) {
            for (int j = 0; j <= k; j++) {
                if (j == 0) dp[i][j] = 1;
                else if (i == 1) dp[i][j] = 0;
                else {
                    dp[i][j] = (dp[i-1][j] + dp[i][j-1]) % M;
                    if (j - i >= 0) dp[i][j] = (dp[i][j] - dp[i-1][j-i] + M) % M;
                }

            }
        }
        return dp[n][k];
    }
};
```

空间优化：

```cpp
class Solution {
public:
    static const int mod = 1000000007;
    int kInversePairs(int n, int k) {
        int f[2][k + 1];
        memset(f, 0, sizeof(f));
        
        f[1][0] = 1;
        for (int i = 2; i <= n; ++i) {
            int flip = i & 1;
            int sum = 0;
            for (int j = 0; j <= k; ++j) {
                sum += f[1 - flip][j];
                if (j >= i) sum -= f[1 - flip][j - i];
                if (sum < 0) sum += mod;
                if (sum >= mod) sum -= mod;
                f[flip][j] = sum;
            }
        }
        return f[n & 1][k];
    }
};
```



