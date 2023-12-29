### [n个骰子投出x的概率](https://leetcode-cn.com/problems/nge-tou-zi-de-dian-shu-lcof/)

从数学上推导公式十分复杂，但满足动态规划的条件，具有递推性质。

```cpp
class Solution {
public:
    vector<double> dicesProbability(int n) {
        map<pair<int, int>, int> dp;
        for (int i = 1; i <= 6; i++) dp[{1, i}] = 1;
        for (int i = 2; i <= n; i++) {
            for (int x = i; x <= 6 * i; x++) {
                for (int j = 1; j <= 6; j++) {
                    dp[{i, x}] += dp[{i-1, x - j}];
                }
            }
        }
        vector<double> ans;
        for (int x = n; x <= 6 * n; x++) {
            ans.push_back((double)dp[{n, x}] / pow(6, n));
        }
        return ans;
    }
};
```

数学公式：

出现$k$的概率为$x^k$的系数。

$$
\displaylines{
(x+x^2+x^3+x^4+x^5+x^6)^n / 6^n
}
$$

可以通过扩展二项式定理展开：

![](https://wikimedia.org/api/rest_v1/media/math/render/svg/7e4dd69e04f849cc627775308fffd170e9472024)

然而展开的系数公式过于复杂，所以没法用。

