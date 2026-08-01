# 动态规划 (Dynamic Programming)

## 前提

1. **最优子结构**：问题的解可以通过组合子问题的解得到，即可以写出状态转移方程；
2. **重叠子问题**：求解过程中子问题大量重叠（否则退化为分治/递归）；
3. 通常是求**最值/计数**问题。

## 状态设计方法论

- **`dp[i]` / `dp[i][j]` 的含义**决定一切：想清楚"下标代表什么"、"存的值是什么"。
- **一维不够就加维度**：缺信息是状态定义的问题，不是转移的问题。典型例子：Remove Boxes 需要 `dp[i][j][k]`；股票问题需要维度记录交易次数/持仓状态。
- **初始条件与循环顺序**：`dp[0]`/`dp[0][*]` 必须正确；循环顺序由"转移依赖谁"决定（区间 DP 依赖更短区间 → 按长度循环；用未来信息 → 倒序）。
- **记忆化搜索 = 递归版 DP**：先写暴搜，加 `memo` 表即可，天然避免转移顺序的思考。

## 线性 DP

### 最长上升子序列 LIS — [300](https://leetcode-cn.com/problems/longest-increasing-subsequence/)

`dp[i]`：以 `nums[i]` 结尾的 LIS 长度。$O(n^2)$：

```cpp
class Solution {
public:
    int lengthOfLIS(vector<int>& nums) {
        int n = nums.size();
        vector<int> dp(n, 1); // LIS ending at nums[i]
        int ans = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < i; j++) {
                if (nums[j] < nums[i]) {
                    dp[i] = max(dp[i], dp[j] + 1);
                }
            }
            ans = max(ans, dp[i]);
        }
        return ans;
    }
};
```

贪心 + 二分 $O(n\log n)$：维护"当前长度下最优结尾"数组，`lower_bound` 替换：

```cpp
class Solution {
public:
    int lengthOfLIS(vector<int>& nums) {
        int n = nums.size();
        vector<int> v; // keep track of the best LIS
        for (int i = 0; i < n; i++) {
            int idx = lower_bound(v.begin(), v.end(), nums[i]) - v.begin();
            if (idx == v.size()) v.push_back(nums[i]);
            else v[idx] = nums[i];
        }
        return v.size();
    }
};
```

#### LIS 的数量 — [673](https://leetcode-cn.com/problems/number-of-longest-increasing-subsequence/)

额外记录"以 i 结尾的 LIS 数量"，两个计数数组同步更新：

```c++
class Solution {
public:
    int findNumberOfLIS(vector<int>& nums) {
        int n = nums.size(), maxLen = 0, ans = 0;
        vector<int> dp(n), cnt(n);
        for (int i = 0; i < n; ++i) {
            dp[i] = 1;
            cnt[i] = 1;
            for (int j = 0; j < i; ++j) {
                if (nums[i] > nums[j]) {
                    if (dp[j] + 1 > dp[i]) {
                        dp[i] = dp[j] + 1;
                        cnt[i] = cnt[j]; // 重置计数
                    } else if (dp[j] + 1 == dp[i]) {
                        cnt[i] += cnt[j];
                    }
                }
            }
            if (dp[i] > maxLen) {
                maxLen = dp[i];
                ans = cnt[i]; // 重置计数
            } else if (dp[i] == maxLen) {
                ans += cnt[i];
            }
        }
        return ans;
    }
};
```

### 最长公共子序列 LCS — [1143](https://leetcode-cn.com/problems/longest-common-subsequence/)

**子序列可以不连续！** `dp[i][j]`：前 `i` 个字符与前 `j` 个字符的 LCS。$O(n^2)$：

```c++
class Solution {
public:
    int longestCommonSubsequence(string text1, string text2) {
        int dp[1005][1005];
        // start from 1, to avoid border detection.
        for (int i = 1; i <= text1.size(); i++) {
            for (int j = 1; j <= text2.size(); j++) {
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1]);
                if (text1[i - 1] == text2[j - 1]) dp[i][j] = max(dp[i][j], dp[i - 1][j - 1] + 1);
            }
        }
        return dp[text1.size()][text2.size()];
    }
};
```

#### 可以转化为 LIS 的 LCS — [1713](https://leetcode-cn.com/problems/minimum-operations-to-make-a-subsequence/)

若两个序列**有一个元素各不相同**，可把它视为排序关系，把 LCS 转化为另一个序列的 LIS，$O(n\log n)$：

```cpp
class Solution {
public:
    int minOperations(vector<int>& target, vector<int>& arr) {
        int m = target.size(), n = arr.size();
        // build an order map
        map<int, int> M;
        for (int i = 0; i < m; i++) M[target[i]] = i;
        // LIS
        vector<int> v;
        for (int i = 0; i < n; i++) {
            // ignore those unmet elements!
            if (M.count(arr[i])) {
                int idx = lower_bound(v.begin(), v.end(), M[arr[i]]) - v.begin();
                if (idx == v.size()) v.push_back(M[arr[i]]);
                else v[idx] = M[arr[i]];
            }
        }
        return m - v.size();
    }
};
```

### 最长公共子串 — [718](https://leetcode-cn.com/problems/maximum-length-of-repeated-subarray/)

**子串必须连续！** `dp[i][j]`：以第一个串第 `i` 个字符、第二个串第 `j` 个字符**结尾**的最长公共子串长度，需要额外变量维护全局最大值：

```c++
class Solution {
public:
    int findLength(vector<int>& nums1, vector<int>& nums2) {
        int dp[1005][1005];
        int res = 0;
        for (int i = 1; i <= nums1.size(); i++) {
            for (int j = 1; j <= nums2.size(); j++) {
                if (nums1[i - 1] == nums2[j - 1]) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                    res = max(res, dp[i][j]);
                }
                else dp[i][j] = 0;
            }
        }
        return res;
    }
};
```

### Kadane：最大子数组 — [53](https://leetcode-cn.com/problems/maximum-subarray/)

`dp[i]` = 以 `nums[i]` 结尾的最大子数组和；`dp[i] = max(dp[i-1] + nums[i], nums[i])`。空间优化后：

```cpp
class Solution {
public:
    int maxSubArray(vector<int>& nums) {
        int ans = nums[0];
        int dp = nums[0]; // space optimization
        for (int i = 1; i < nums.size(); i++) {
            dp = max(dp + nums[i], nums[i]);
            ans = max(ans, dp);
        }
        return ans;
    }
};
```

#### 乘积最大子数组 — [152](https://leetcode-cn.com/problems/maximum-product-subarray/)

有负数时**再记录一个最小值**（偶数个负数相乘仍为正）：

```cpp
class Solution {
public:
    int maxProduct(vector<int>& nums) {
        int ans = nums[0];
        int mx = nums[0], mn = nums[0];
        for (int i = 1; i < nums.size(); i++) {
            int old_mx = mx;
            mx = max(mx * nums[i], max(mn * nums[i], nums[i]));
            mn = min(mn * nums[i], min(old_mx * nums[i], nums[i]));
            ans = max(ans, mx);
        }
        return ans;
    }
};
```

#### 最大子矩阵 — [面试题 17.24](https://leetcode-cn.com/problems/max-submatrix-lcci/)

2D 拓展：列前缀和 + 枚举上下边界，对"压扁后的行"做 1D Kadane。$O(n^2m)$。注意还要记录坐标。

```cpp
class Solution {
public:
    vector<int> getMaxMatrix(vector<vector<int>>& matrix) {
        int m = matrix.size();
        int n = matrix[0].size();
        vector<vector<int>> s(m + 1, vector<int>(n, 0)); // column-wise prefix sum
        for (int i = 1; i <= m; i++) {
            for (int j = 0; j < n; j++) {
                s[i][j] = s[i-1][j] + matrix[i - 1][j];
            }
        }
        int mx = matrix[0][0];
        vector<int> ans(4, 0);
        // row start/end, O(n^2)
        for (int r1 = 0; r1 < m; r1++) {
            for (int r2 = r1 + 1; r2 <= m; r2++) {
                // kadane on each row, O(m)
                int dp = s[r2][0] - s[r1][0];
                int start = 0; // start index of max-sum-subarray
                if (dp > mx) mx = dp, ans = {r1, 0, r2 - 1, 0};
                for (int c = 1; c < n; c++) {
                    int v = s[r2][c] - s[r1][c];
                    if (dp + v > v) {
                        dp = dp + v;
                    } else {
                        dp = v;
                        start = c;
                    }
                    if (dp > mx) mx = dp, ans = {r1, start, r2 - 1, c};
                }
            }
        }
        return ans;
    }
};
```

### 解决智力问题 — [2140](https://leetcode-cn.com/problems/solving-questions-with-brainpower/)

`f(i)`：从第 `i` 题开始能拿到的最高分。用到未来信息 → 倒序；转移"做/跳过"两种选择。正序版本则是在做的时候**更新未来下标**（跳跃更新技巧）。

```cpp
// 倒序 DP
class Solution {
public:
    long long mostPoints(vector<vector<int>>& questions) {
        int n = questions.size();
        vector<long long> dp(n + 1, 0);
        for (int i = n - 1; i >= 0; --i) {
            dp[i] = max(dp[i + 1], questions[i][0] + dp[min(n, i + questions[i][1] + 1)]);
        }
        return dp[0];
    }
};
```

```cpp
// 正序 DP：每次循环不只更新当前下标，还更新未来的下标
class Solution {
public:
    long long mostPoints(vector<vector<int>> &questions) {
        int n = questions.size();
        vector<long long> dp(n + 1);
        for (int i = 0; i < n; ++i) {
            dp[i + 1] = max(dp[i + 1], dp[i]);
            int j = min(n, i + questions[i][1] + 1);
            dp[j] = max(dp[j], dp[i] + questions[i][0]);
        }
        return dp[n];
    }
};
```

### 买卖股票系列 — [121](https://leetcode.com/problems/best-time-to-buy-and-sell-stock/) / [122](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/) / [123](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iii/) / [188](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iv/)

- **Ⅰ（一笔交易）**：贪心——记录最低点，每天计算"当天卖出"的利润，取最大。
  $\text{profit}[i] = \max(\text{profit}[i-1],\ \text{prices}[i]-\min_{1\le j\le i}\{\text{prices}[j]\})$
- **Ⅱ（无限笔）**：贪心——只要价格比前一天高就"赚差价"。
  $\text{profit}[i] = \text{profit}[i-1] + \max(0,\ \text{prices}[i] - \text{prices}[i-1])$
- **Ⅲ（最多两笔）**：遍历分界点，左右各调用一次问题 Ⅰ。
- **Ⅳ（最多 K 笔）**：动态规划，状态加一维"已交易次数"。

## 区间 DP

模式：`dp[i][j]` 表示区间 `[i, j]` 的答案；转移枚举**分割点**或**端点相等**；按**区间长度**从小到大的顺序计算（或记忆化）。

### 猜数字大小 II — [375](https://leetcode-cn.com/problems/guess-number-higher-or-lower-ii/)

递归结构显然，但分割点不能贪心确定，只能枚举并记忆化。`dp[i][j]`：范围为 `[i, j]` 时保证获胜的最小金额。

```cpp
class Solution {
private:
    int dp[201][201];
public:
    int dfs(int start, int end) {
        if (dp[start][end] != 0) return dp[start][end];
        if (start == end) return dp[start][end] = 0;
        if (start + 1 == end) return dp[start][end] = start;
        int ans = 0x7fffffff;
        for (int k = end - 1; k > start; k -= 2) {
            int cur = max(dfs(start, k - 1), dfs(k + 1, end)) + k;
            ans = min(ans, cur);
        }
        return dp[start][end] = ans;
    }

    int getMoneyAmount(int n) {
        memset(dp, 0, sizeof(dp));
        return dfs(1, n);
    }
};
```

### 戳气球 — [312](https://leetcode-cn.com/problems/burst-balloons/)

`dp[i][j]`：区间 `[i, j]` 的气球爆炸到只剩 i、j 时的最高得分；**枚举最后爆炸的那一个气球 m**：

```c++
// Top-Down
class Solution {
public:
    int dp[505][505];
    vector<int> ns;
    
    int solve(int i, int j){
        if(dp[i][j]!=-1) return dp[i][j];
        if(i+1 >= j) return dp[i][j] = 0;
        int res = 0;
        for(int m=i+1; m<j; m++){
            res = max(res, solve(i, m)+solve(m, j)+ns[i]*ns[m]*ns[j]);
        }
        return dp[i][j] = res;
    }
    
    int maxCoins(vector<int>& nums) {
        ns.push_back(1);
        for(int x:nums) ns.push_back(x);
        ns.push_back(1);
        memset(dp, -1, sizeof(dp));
        return solve(0, ns.size()-1);
    }
};

// Bottom-up
class Solution {
public:
    int maxCoins(vector<int>& nums) {
        vector<int> ns;
        ns.push_back(1);
        for(int x:nums) ns.push_back(x);
        ns.push_back(1);
        int N = ns.size();
        int dp[505][505];
        memset(dp, 0, sizeof(dp));
        
        for(int l=2; l<N; l++){          // 按区间长度循环
            for(int j=l; j<N; j++){
                int i = j-l;
                for(int m=i+1; m<j; m++){
                    dp[i][j] = max(dp[i][j], ns[i]*ns[m]*ns[j]+dp[i][m]+dp[m][j]);
                }
            }
        }
        
        return dp[0][N-1];
    }
};
```

### 最长回文子序列 — [516](https://leetcode-cn.com/problems/longest-palindromic-subsequence/)

两端字符相等则 `+2`，否则取删掉任一端的结果（注意 516 是子序列，可以不连续）：

```c++
class Solution {
public:
    int longestPalindromeSubseq(string s) {
        int len = s.size();
        if(len==0) return 0;
        vector<vector<int>> dp(len, vector<int>(len, 0));
        
        for(int l=1; l<=len; l++){
            for(int i=0; i<=len-l; i++){
                int j = i+l-1;
                if(j==i) dp[i][j] = 1;
                else if(j==i+1){
                    if(s[i]==s[j]) dp[i][j] = 2;
                    else dp[i][j] = 1;
                }
                else{
                    if(s[i]==s[j]) dp[i][j] = dp[i+1][j-1]+2;
                    else dp[i][j] = max(dp[i+1][j], dp[i][j-1]);
                }
            }
        }
        
        return dp[0][len-1];
    }
};
```

### 移除盒子 — [546](https://leetcode-cn.com/problems/remove-boxes/)

二维端点 DP 缺信息（不知道区间外左侧有多少同色盒子），**拓展到三维** `dp[i][j][k]`，k 为区间 `[i, j]` **左侧**（区间外）与 `i` 同色的方块数：

```c++
// Top-Down DP (memorization)
class Solution {
public:
    int dp[105][105][105];
    vector<int> boxes;
    
    int solve(int i, int j, int k){
        if(dp[i][j][k]!=-1) return dp[i][j][k];
        if(j<i) return 0;
        int res = (k+1)*(k+1) + solve(i+1, j, 0);
        for(int m=i+1; m<=j; m++){
            if(boxes[i] == boxes[m]){
                res = max(res, solve(i+1, m-1, 0)+solve(m, j, k+1));
            }
        }
        return dp[i][j][k]=res;
    }
    
    int removeBoxes(vector<int>& Boxes) {
        boxes = Boxes;
        memset(dp, -1, sizeof(dp));
        return solve(0, boxes.size()-1, 0);
    }
};
```

## 状态压缩 DP

序列（长度 10~30）可以用一个 `int` 的二进制位表示状态。

### 最美子字符串 — [1915](https://leetcode-cn.com/problems/number-of-wonderful-substrings/)

至多一个字母出现奇数次 → 前缀奇偶状态（10 位掩码）+ 哈希计数。子串问题若直接枚举首尾是 $O(n^2)$，用前缀状态压缩到 $O(n\cdot 10)$：

```c++
class Solution {
public:
    long long wonderfulSubstrings(string word) {
        unordered_map<int, int> cnt;
        cnt[0] = 1;
        long long ans = 0;
        int mask = 0; // 10 digits <= 2^10, can be represented with an integer.
        for (int i = 0; i < word.size(); i++) {
            int c = word[i] - 'a';
            mask ^= (1 << c);
            // 0 odd: the same prefix mask
            ans += cnt[mask];
            // 1 odd: 10 types of suitable prefixes, add each count to ans.
            for (int j = 0; j < 10; j++) {
                int prefix = (mask ^ (1 << j));
                ans += cnt[prefix];
            }
            cnt[mask]++;
        }
        return ans;
    }
};
```

### 贴纸拼词 — [691](https://leetcode-cn.com/problems/stickers-to-spell-word/)

目标串作为状态（每个字符的剩余需求），记忆化搜索；把字符串转成 26 维计数向量便于减法：

```c++
class Solution {
public:
    map<vector<int>, int> dp;
    vector<vector<int>> Stickers;
    
    vector<int> str2vec(string s){
        vector<int> cnt(26, 0);
        for(char c:s) cnt[c-'a']++;
        return cnt;
    }
    
    bool allNeg(vector<int>& v){
        for(int i:v) if(i>0) return false;
        return true;
    }
    
    bool allEq(vector<int>& a, vector<int>& b){
        for(int i=0; i<26; i++) if(a[i]!=b[i]) return false;
        return true;
    }
    
    vector<int> allSub(vector<int>& a, vector<int>& b){
        vector<int> ans;
        for(int i=0; i<a.size(); i++)
            ans.push_back((a[i]-b[i]<=0)?0:a[i]-b[i]);
        return ans;
    }
    
    int solve(vector<int> s){
        if(dp.count(s)) return dp[s];
        if(allNeg(s)) return dp[s]=0;
        int res = 0x3f3f3f3f;
        for(vector<int> p:Stickers){
            vector<int> q = allSub(s, p);
            if(!allEq(s, q))   // 必须真的减少需求
                res = min(res, 1+solve(q));
        }
        return dp[s]=res;
    }
    
    int minStickers(vector<string>& stickers, string target) {
        for(string s:stickers)
            Stickers.push_back(str2vec(s));
        int ans = solve(str2vec(target));
        return ans==0x3f3f3f3f?-1:ans;
    }
};
```

## 数位 DP

### 不含连续一的非负整数 — [600](https://leetcode-cn.com/problems/non-negative-integers-without-consecutive-ones/)

规律：$n = 2^k$ 时（不含 0）答案是斐波那契 $f(2^k)=f(2^{k-1})+f(2^{k-2})$；任意 n 从高位扫，遇到连续 1 即终止。通用性差但值得记录：

```cpp
class Solution {
public:
    int findIntegers(int n) {
        // fibonacci
        vector<int> f, b;
        f.push_back(1);
        while (n) {
            if (n & 1) b.push_back(1);
            else b.push_back(0);
            if (f.size() >= 2) f.push_back(f[f.size() - 1] + f[f.size() - 2]);
            else f.push_back(1);
            n /= 2;
        }
        bool flag = false;
        int ans = 1; // 0 also counts
        for (int i = b.size() - 1; i >= 0; i--) {
            if (b[i]) {
                if (!flag) {
                    flag = true;
                    ans += f[i+1];
                } else {
                    ans += f[i+1] - 1; // self do not count, so -1.
                    break;
                }
            } else {
                flag = false;
            }
        }
        return ans;
    }
};
```

## 计数 DP（递推计数）

特征是"求方案数"，转移用累加，注意取模与去重。

### 逆序对为 k 的排列个数 — [629](https://leetcode-cn.com/problems/k-inverse-pairs-array/)

$f(i,j)$：前 $i$ 个数、恰有 $j$ 个逆序对的排列数。把 $i$ 插入到 $k=0..i-1$ 的位置会**新增** $i-1-k$ 个逆序对：

$$
f(i,j) = \sum_{k=0}^{i-1} f(i-1, j-k)
$$

暴力是 $O(nk^2)$，展开发现相邻项可相消，得到**前缀和优化**的递推（$O(nk)$）：

$$
f(i,j) = f(i-1,j) + f(i,j-1) - f(i-1,j-i)
$$

边界：$f(*,0)=1$（$f(1,0)=1$）；$j<0$ 时为 0。

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

滚动数组版：

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

### n 个骰子投出 x 的概率 — [剑指 Offer 60](https://leetcode-cn.com/problems/nge-tou-zi-de-dian-shu-lcof/)

递推：$dp(i, x) = \sum_{j=1}^{6} dp(i-1, x-j)$。数学上等于 $(x+x^2+\cdots+x^6)^n/6^n$ 的展开系数，但系数公式过于复杂，用 DP 即可：

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

### 学生出勤记录 II — [552](https://leetcode-cn.com/problems/student-attendance-record-ii/)

按"结尾状态"拆状态机：P（无 A，结尾 P）、L（无 A，结尾单个 L）、LL（无 A，结尾 LL）、A、AP、AL、ALL，逐个递推：

```c++
class Solution {
public:
    int checkRecord(int n) {
        int m = 1e9+7;
        int* P = new int[n+1]; // no A
        int* L = new int[n+1]; // no A, single L
        int* LL = new int[n+1];
        int* A = new int[n+1];
        int* AP = new int[n+1];
        int* AL = new int[n+1];
        int* ALL = new int[n+1];
        
        P[1] = L[1] = A[1] = 1;
        LL[1] = AP[1] = AL[1] = ALL[1] = 0;
        
        for(int i=2; i<=n; i++){
            P[i] = ((P[i-1] + L[i-1])%m + LL[i-1])%m;
            L[i] = P[i-1];
            LL[i] = L[i-1];
            A[i] = ((P[i-1] + L[i-1])%m + LL[i-1])%m;
            AL[i] = (AP[i-1] + A[i-1])%m;
            AP[i] = (((A[i-1] + AL[i-1])%m + ALL[i-1])%m + AP[i-1])%m;
            ALL[i] = AL[i-1];
        }
        
        return ((((((P[n]+L[n])%m+A[n])%m+LL[n])%m+AP[n])%m+AL[n])%m+ALL[n])%m;
    }
};
```

### 骑士拨号器 — [935](https://leetcode-cn.com/problems/knight-dialer/)

类似计数递推；N 很大时用**矩阵快速幂**（转移矩阵的 N 次方），见数学笔记。

```c++
class Solution {
public:
    int knightDialer(int N) {
        int m = 1e9+7;
        vector<vector<int>> dp(N+1, vector<int>(10,0));
        for(int i=0; i<10; i++) dp[1][i]=1;
        for(int i=2; i<=N; i++){
            dp[i][1] = (dp[i-1][6] + dp[i-1][8])%m;
            dp[i][2] = (dp[i-1][7] + dp[i-1][9])%m;
            dp[i][3] = (dp[i-1][4] + dp[i-1][8])%m;
            dp[i][4] = ((dp[i-1][3] + dp[i-1][9])%m + dp[i-1][0])%m;
            dp[i][5] = 0;
            dp[i][6] = ((dp[i-1][1] + dp[i-1][7])%m + dp[i-1][0])%m;
            dp[i][7] = (dp[i-1][2] + dp[i-1][6])%m;
            dp[i][8] = (dp[i-1][1] + dp[i-1][3])%m;
            dp[i][9] = (dp[i-1][2] + dp[i-1][4])%m;
            dp[i][0] = (dp[i-1][4] + dp[i-1][6])%m;
        }
        int ans = 0;
        for(int i=0; i<10; i++) ans = (ans + dp[N][i])%m;
        return ans;
    }
};
```

### 不同的子序列 — [115](https://leetcode-cn.com/problems/distinct-subsequences/) 与 [940](https://leetcode-cn.com/problems/distinct-subsequences-ii/)

- 115：`s` 中等于 `t` 的子序列个数，`dp[i+1][j+1] = dp[i+1][j] + (t[i]==s[j] ? dp[i][j] : 0)`。
- 940：`s` 的所有不同子序列个数，转移公式 `dp[i] = 2*dp[i-1] - dp[last[s[i]]-1]`（减去重复，`last` 记录上次出现位置）：

```c++
class Solution {
public:
    #define ll long long
    int distinctSubseqII(string S) {
        ll m = 1e9 + 7;
        int len = S.size();
        vector<ll> dp(len+1, 0);
        map<char, int> last;
        dp[0] = 1; // empty string, subtracted at the end
        for(int i=0; i<len; i++){
            dp[i+1] = (2*dp[i])%m;
            if(last.count(S[i])) dp[i+1] -= dp[last[S[i]]];
            dp[i+1] %= m;
            last[S[i]] = i;
        }
        return (dp[len]-1+m)%m; // minus the empty string
    }
};
```

### 最长斐波那契式子序列 — [873](https://leetcode-cn.com/problems/length-of-longest-fibonacci-subsequence/)

**用 map 定位唯一特定值而不是循环检验**，是降低复杂度的重要技巧。`dp[i][j]`：以 `A[i], A[j]` 结尾的最长斐波那契子序列长度：

```c++
class Solution {
public:
    int lenLongestFibSubseq(vector<int>& A) {
        int len = A.size();
        map<int, int> m;
        for(int i=0; i<len; i++) m[A[i]]=i;
        // dp[i][j]: LFSL of FS ending with A[i], A[j].
        vector<vector<int>> dp(len, vector<int>(len, 2));
        int ans = 0;
        for(int i=0; i<len; i++){
            for(int j=i; j<len; j++){
                // m[A[j]-A[i]]<i is necessary
                if(m.count(A[j]-A[i]) && m[A[j]-A[i]]<i)
                    dp[i][j] = max(dp[i][j], dp[m[A[j]-A[i]]][i]+1);
                ans = max(ans, dp[i][j]);
            }
        }
        return ans>2 ? ans : 0;
    }
};
```

### 丑数 II — [264](https://leetcode-cn.com/problems/ugly-number-ii/)

三指针多路归并：每个丑数分别乘 2/3/5，指针指向"最小可用"的因子来源（堆解法见 `12_heap`）：

```cpp
class Solution {
public:
    int nthUglyNumber(int n) {
        vector<int> nums = {1};
        int p2 = 0, p3 = 0, p5 = 0;
        for (int i = 1; i < n; i++) {
            int x2 = nums[p2] * 2;
            int x3 = nums[p3] * 3;
            int x5 = nums[p5] * 5;
            
            int x = min(x2, min(x3, x5));
            if (x == x2) p2++;
            if (x == x3) p3++;
            if (x == x5) p5++;
            
            nums.push_back(x);
        }
        return nums.back();
    }
};
```

### 比特位计数 — [338](https://leetcode-cn.com/problems/counting-bits/)

有趣的递推：`ans[i] = ans[i>>1] + (i&1)`。

```c++
class Solution {
public:
    vector<int> countBits(int num) {
        vector<int> ans(num+1, 0);        
        for(int i=0; i<=num; i++) ans[i] = ans[i>>1] + (i&1);
        return ans;
    }
};
```

## 字符串 DP

### 正则表达式匹配 — [10](https://leetcode-cn.com/problems/zheng-ze-biao-da-shi-pi-pei-lcof/)

`f[i][j]`：`s` 前 i 个与 `p` 前 j 个是否匹配。`*` 分"匹配 0 次（`f[i][j-2]`）"与"匹配多次（`f[i-1][j]`）"两类：

```cpp
class Solution {
public:
    bool isMatch(string s, string p) {
        int m = s.size();
        int n = p.size();

        auto matches = [&](int i, int j) {
            if (i == 0) {
                return false;
            }
            if (p[j - 1] == '.') {
                return true;
            }
            return s[i - 1] == p[j - 1];
        };

        vector<vector<int>> f(m + 1, vector<int>(n + 1));
        f[0][0] = true;
        for (int i = 0; i <= m; ++i) {
            for (int j = 1; j <= n; ++j) {
                if (p[j - 1] == '*') {
                    f[i][j] |= f[i][j - 2];
                    if (matches(i, j - 1)) {
                        f[i][j] |= f[i - 1][j];
                    }
                }
                else {
                    if (matches(i, j)) {
                        f[i][j] |= f[i - 1][j - 1];
                    }
                }
            }
        }
        return f[m][n];
    }
};
```

## 树形 DP

### 附近的奶牛（POJ 树形计数）

树上每个点统计**距离不超过 K** 的邻居权值和：`M[i][k] = sum(M[邻居][k-1])`，再减去被重复计算的 `M[i][k-2]*(deg-1)`（k=1 时单独处理）。

```c++
#include <iostream>
#include <cstdio>
#include <algorithm>
#include <vector>
using namespace std;

const int maxn = 100005;
const int maxk = 25;
int N, K;
int M[maxn][maxk];

struct edge {
	int f, t;
	edge() {}
	edge(int f, int t) :f(f), t(t) {}
};

vector<edge> G[maxn];

void solve() {
    // aggregating information from k-nearest neighbours.
    for (int k = 1; k <= K; k++) {
		for (int i = 1; i <= N; i++) {
			int deg = G[i].size();
			for (int j = 0; j < deg; j++) {
				edge& e = G[i][j];
				M[i][k] += M[e.t][k - 1];
			}
			if (k == 1) M[i][k] += M[i][0];
			else M[i][k] -= M[i][k - 2] * (deg - 1);
		}
	}
}

int x, y;
int main() {
	cin >> N >> K;
	for (int i = 1; i < N; i++) {
		cin >> x >> y;
		G[x].push_back(edge(x, y));
		G[y].push_back(edge(y, x));
	}
	for (int i = 1; i <= N; i++) cin >> M[i][0];
	solve();
	for (int i = 1; i <= N; i++) cout << M[i][K] << endl;
}
```

## 常用优化技巧

- **滚动数组**：只依赖上一行 → 一维数组 + 正确循环方向（0/1 逆序、完全正序，见 `06_knapsack`）。
- **前缀和优化**：转移是"一段区间求和"时（如 629），用前缀和把 $O(nk^2)$ 降到 $O(nk)$。
- **贪心 + 二分**：LIS 的 $O(n\log n)$ 解法。
- **Map 加速查找**：把"循环找某个值"换成"哈希定位"（如 873）。
- **记忆化搜索**：想不清循环顺序时先写递归 + memo。
