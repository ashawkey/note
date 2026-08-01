# 背包 (Knapsack)

## 核心思想

背包是动态规划最经典的入门模型。所有背包的共性：

- 状态：$f[i][j]$ 表示**使用前 $i$ 个物品、容量为 $j$ 时**的最优值；
- 转移：对每个物品决策"选/不选"（或选几个）；
- 优化：滚动数组把二维压成一维，**循环方向决定是 0/1 还是完全**。

## 模板

### 0/1 背包（每个物品最多选一次）

最大容量 N，共 M 个物品，每个物品重量 $w[i]$、价值 $v[i]$。

状态方程：

$$
f[i][j] = \max(f[i-1][j],\ f[i-1][j-w[i]] + v[i])
$$

边界：$f[0][j] = f[i][0] = 0$。

```c++
const int M = 100 + 1; // object
const int N = 1000 + 1; // space

int ws[M];
int vs[M];

int dp[M][N];

int main() {
    int n, m;
    cin >> n >> m;
    for (int i = 1; i <= m; i++) {
        cin >> ws[i] >> vs[i];
    }
    memset(dp, 0, sizeof(dp));
    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            dp[i][j] = dp[i - 1][j];
            if (j >= ws[i]) dp[i][j] = max(dp[i][j], dp[i - 1][j - ws[i]] + vs[i]);
        }
    }
    cout << dp[m][n] << endl;
    return 0;
}
```

空间优化（**逆序循环**！$dp[i][j]$ 只用 $dp[i-1][k]$，而 $k < j$，逆序保证用到的是上一轮的值）：

```c++
int dp[N];

int main() {
    int n, m;
    cin >> n >> m;
    for (int i = 1; i <= m; i++) {
        cin >> ws[i] >> vs[i];
    }
    memset(dp, 0, sizeof(dp));
    for (int i = 1; i <= m; i++) {
        // reversed order!
        for (int j = n; j >= 1; j--) {
            if (j >= ws[i]) dp[j] = max(dp[j], dp[j - ws[i]] + vs[i]);
        }
    }
    cout << dp[n] << endl;
    return 0;
}
```

### 完全背包（每个物品可以选任意次）

$$
f[i][j] = \max_{0 \le k \le K}(f[i-1][j-kw[i]] + kv[i]) = \max(f[i-1][j],\ f[i][j-w[i]] + v[i])
$$

即同一物品可以重复选，转移用到**本行**信息（选了一个 i 之后还能再选 i）。

```c++
int dp[N];

int main() {
    int n, m;
    cin >> n >> m;
    for (int i = 1; i <= m; i++) {
        cin >> ws[i] >> vs[i];
    }
    memset(dp, 0, sizeof(dp));
    for (int i = 1; i <= m; i++) {
        // ordered! dp[i][j] can use dp[i][j-ws[i]]
        for (int j = 1; j <= n; j++) {
            if (j >= ws[i]) dp[j] = max(dp[j], dp[j - ws[i]] + vs[i]);
        }
    }
    cout << dp[n] << endl;
    return 0;
}
```

> **记住一句话**：0/1 背包内层**逆序**，完全背包内层**正序**。区别就在能不能重复选同一个物品。

### 贪心即可的情况

0/1 背包但**所有物品价值相同**：直接按重量从小到大排序，从轻到重装到装不下为止。

```c++
class Solution {
public:
    int maxIceCream(vector<int>& costs, int coins) {
        sort(costs.begin(), costs.end());
        int res = 0;
        while (res < costs.size() && (coins -= costs[res]) >= 0) {
            res++;
        }
        return res;
    }
};
```

## 复杂度与易错点

- 时间 $O(MN)$、空间可压到 $O(N)$。N 很大（如 $10^9$）时背包不可行，考虑贪心或其他模型。
- 记清"求最大价值（max）"与"求方案数（累加）"、"恰好装满"与"不超过容量"的初始化差异：
  - 求**恰好装满**的最值：`dp[0]=0`，其余 `-inf`；
  - 求**方案数**：`dp[0]=1`，其余 0，转移累加。
- 0/1 与完全背包的**唯一区别是内层循环方向**，别记反。

## 例题

### [416 分割等和子集](https://leetcode-cn.com/problems/partition-equal-subset-sum/)

判断数组能否分成两个和相等的子集 → 是否存在和为 `sum/2` 的子集（0/1 背包判定）。

```c++
class Solution {
public:  
    bool canPartition(vector<int>& nums) {
        int N = nums.size();
        int sum = 0;
        for(int i=0; i<N; i++) sum+=nums[i];
        
        if(sum%2!=0) return false;
        sum /= 2;
        
        vector<int> dp(sum+1, 0);
        dp[0] = 1;
        for(int i=1; i<=N; i++){
            // reversely, since the formula use left information.
            for(int j=sum; j>=1; j--){
                if(j >= nums[i-1] && dp[j-nums[i-1]]) dp[j]=1;
            }
        }
        
        return dp[sum];
    }
};
```

### [494 目标和](https://leetcode-cn.com/problems/target-sum/)

给每个数前加 `+`/`-`，求凑出 target 的方案数。数学转化：设正号子集为 P、负号子集为 N，

$$
sum(P)-sum(N) = S,\quad sum(P)+sum(N) = sum(nums) \Rightarrow 2\cdot sum(P) = S + sum(nums)
$$

转化为 0/1 背包**计数**：凑出 $(S+sum)/2$ 的方案数。

```c++
class Solution {
public:
    // 求数组凑出 target 的子集种类数（一维 0/1 计数背包）
    int solve(vector<int>& nums, int target){
        vector<int> dp(target+1, 0);
        dp[0] = 1;
        for(int i=0; i<nums.size(); i++)
            for(int j=target; j>=nums[i]; j--) // 逆序
                dp[j] += dp[j-nums[i]];
        return dp[target];
    }

    int findTargetSumWays(vector<int>& nums, int S) {
        int sum = accumulate(nums.begin(), nums.end(), 0);
        if(sum<S || (sum+S)%2!=0) return 0;
        else return solve(nums, (sum+S)/2);
    }
};
```

### [474 一和零](https://leetcode-cn.com/problems/ones-and-zeroes/)

每个字符串消耗 0 的个数和 1 的个数两种容量 → **二维背包**。转移用到左上信息，内层两维都要逆序。

```c++
class Solution {
public:
    int cnt(string& s, char c){
        int res = 0;
        for(char i:s) if(i==c) res++;
        return res;
    }
    
    int findMaxForm(vector<string>& strs, int m, int n) {
        vector<vector<int>> dp(m+1, vector<int>(n+1, 0));
        for(int l=0; l<strs.size(); l++){
            int cnt0 = cnt(strs[l], '0');
            int cnt1 = cnt(strs[l], '1');
            // go from right-bottom to left-top, to avoid using duplicated information from this iteration.
            for(int i=m; i>=cnt0; i--){
                for(int j=n; j>=cnt1; j--){
                    dp[i][j] = max(dp[i][j], 1 + dp[i-cnt0][j-cnt1]);
                }
            }
        }
        return dp[m][n];
    }
};
```

### [322 零钱兑换](https://leetcode-cn.com/problems/coin-change/)

完全背包求最少硬币数（可以选任意次）。

```c++
class Solution {
public:
    #define inf 0x3f3f3f3f
    int coinChange(vector<int>& coins, int amount) {
        // infinite knapsack
        int len = coins.size();
        vector<int> dp(amount+1, inf);
        dp[0] = 0;
        for(int j=0; j<=amount; j++){
            for(int i=0; i<len; i++){
                if(j>=coins[i]) dp[j] = min(dp[j], dp[j-coins[i]]+1);
            }
        }
        
        return dp[amount]==inf?-1:dp[amount];
    }
};
```

### [279 完全平方数](https://leetcode-cn.com/problems/perfect-squares/)

和为 n 的最少完全平方数个数——完全背包，物品是 $i^2$，价值 1。

```c++
class Solution {
public:
    int numSquares(int n) {
        // complete knapsack
        int m = sqrt(n) + 1;
        vector<int> dp(n + 1, 0x3f3f3f3f); // dp[1~n] should be inf
        dp[0] = 0;
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (j - i * i >= 0) dp[j] = min(dp[j], dp[j - i * i] + 1);
            }
        }
        return dp[n];
    }
};
```
