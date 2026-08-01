# 回溯 (Backtracking)

## 核心思想

回溯 = DFS + **状态撤销**：枚举所有选择，走不通（或遍历完）就撤销刚才的选择回到上一步。适用于：

- 排列、组合、子集枚举；
- "解是否存在/有多少个"且允许路径重叠的问题（此时 BFS 的全局 vis 不可用）；
- 配合**剪枝**（提前排除不可能的分支）才能高效。

```cpp
void dfs(状态) {
    if (是目标) { 记录答案; return; }
    for (每个可选分支) {
        if (剪枝条件) continue;
        做选择;
        dfs(新状态);
        撤销选择;
    }
}
```

> 与 DP 的区别：回溯枚举的是"路径"本身，通常需要完整方案；只求最值且子问题重叠时，回溯 + 记忆化就变成了 DP。

## 模板

### 全排列（递归构造）

```c++
class Solution {
public:
    vector<vector<int>> permute(vector<int>& nums) {
        vector<vector<int>> ans;
        if (nums.empty()) {
            ans.push_back(vector<int>());
            return ans;
        }
        for (int i = 0; i < nums.size(); i++) {
            vector<int> others(nums);
            others.erase(others.begin() + i); // 去掉当前元素
            auto p = permute(others);
            for (auto v: p) {
                v.insert(v.begin(), nums[i]);
                ans.push_back(v);
            }
        }
        return ans;
    }
};
```

### 组合（选/不选）

```cpp
int ans = 0;
void dfs(vector<int>& v, int target, int idx, int cur) {
    if (idx == v.size()) {
        if (cur == target) ans++;
        return;
    }
    dfs(v, target, idx + 1, cur);            // do not use v[idx]
    dfs(v, target, idx + 1, cur + v[idx]);   // use v[idx]
}
```

### N 皇后

```c++
class Solution {
public:
    bool safe(vector<int> cur, int i) {
        int s = cur.size();
        for (int j = 0; j < s; j++) {
            if (cur[j] == i || j - cur[j] == s - i || j + cur[j] == i + s) return false;
        }
        return true;
    }

    void solve(vector<vector<int>>& ans, vector<int> cur, int n) {
        if (cur.size() == n) {
            ans.push_back(cur);
            return;
        }
        for (int i = 0; i < n; i++) {
            if (safe(cur, i)) {
                auto next = cur;
                next.push_back(i);
                solve(ans, next, n);
            }
        }
    }
    // solveNQueens: 生成字符串棋盘
};
```

## 例题

### [37 解数独](https://leetcode-cn.com/problems/sudoku-solver/)

DFS 回溯，`check` 检查行/列/九宫格；找到解后直接结束（剪枝）：

```cpp
class Solution {
public:
    bool check(vector<vector<char>>& board, int i, int j, char c) {
        int ii = i / 3 * 3, jj = j / 3 * 3;
        for (int k = 0; k < 9; k++) {
            if (board[i][k] == c) return false;
            if (board[k][j] == c) return false;
            if (board[ii + k % 3][jj + k / 3] == c) return false;
        }
        return true;
    }
    bool finished = false;
    void dfs(vector<vector<char>>& board, queue<pair<int,int>> q) {
        if (q.empty()) {
            finished = true;
            return;
        }
        auto [i, j] = q.front(); q.pop();
        for (char c = '1'; c <= '9'; c++) {
            if (check(board, i, j, c)) {
                board[i][j] = c;
                dfs(board, q);
                if (finished) return;
                board[i][j] = '.'; // 撤销
            }
        }
    }
    void solveSudoku(vector<vector<char>>& board) {
        queue<pair<int,int>> q;
        for (int i = 0; i < 9; i++)
            for (int j = 0; j < 9; j++)
                if (board[i][j] == '.') q.emplace(i, j);
        finished = false;
        dfs(board, q);
    }
};
```

### [282 给表达式添加运算符](https://leetcode-cn.com/problems/expression-add-operators/)

DFS 维护"当前值 now"与"上一个因子 last"（乘号优先，`cur_val = now * last`），并剪枝（后缀最大值不够就不搜）：

```cpp
class Solution {
public:
    vector<string> addOperators(string num, int target) {
        if (num.empty()) return result;
        for (int i = 0; i < num.size(); i++) num_after.emplace_back(stoll(num.substr(i)));
        exp.resize(num.size() * 2);
        dfs(num, target, 0, 0, 0, 1);
        return result;
    }
private:
    string exp;
    vector<string> result;
    vector<long long> num_after;

    int dfs(string& num, long long target, int exp_p, int pos, long long now, long long last) {
        now = now * 10 + num[pos] - '0';
        exp[exp_p++] = num[pos];
        long long cur_val = now * last;
        if (pos == num.size() - 1) {
            if (target == cur_val) result.emplace_back(exp.substr(0, exp_p));
            return 0;
        }
        exp[exp_p] = '*';
        dfs(num, target, exp_p + 1, pos + 1, 0, cur_val);
        if (num_after[pos + 1] >= abs(target - cur_val)) { // 剪枝：后缀最大值够不够
            exp[exp_p] = '+';
            dfs(num, target - cur_val, exp_p + 1, pos + 1, 0, 1);
            exp[exp_p] = '-';
            dfs(num, target - cur_val, exp_p + 1, pos + 1, 0, -1);
        }
        if (now) dfs(num, target, exp_p, pos + 1, now, last); // 数字拼接（无运算符）
        return 0;
    }
};
```

### [2305 公平分发饼干](https://leetcode.cn/problems/fair-distribution-of-cookies/)

"把集合划分成 K 组，使各组和的最大值最小"——**贪心是错的，只能回溯**（n ≤ 8，$n^k$ 可行）。排序降序可大幅剪枝：

```cpp
class Solution {
public:
    int distributeCookies(vector<int>& cookies, int k) {
        int n = cookies.size();
        sort(cookies.begin(), cookies.end(), greater<int>());
        int ans = INT_MAX;
        vector<int> v(k, 0);
        function<void(int, int)> dfs = [&](int i, int mx) {
            if (i >= n) {
                ans = min(ans, mx);
                return;
            }
            for (int j = 0; j < k; j++) {
                v[j] += cookies[i];
                dfs(i + 1, max(mx, v[j]));
                v[j] -= cookies[i];
            }
        };
        dfs(0, 0);
        return ans;
    }
};
```

答案具有单调性，还可以**二分加速**（判定函数仍是回溯，剪枝 `v[j] + cookies[i] > m` 跳过）：

```cpp
class Solution {
public:
    int distributeCookies(vector<int>& cookies, int k) {
        int n = cookies.size();
        sort(cookies.begin(), cookies.end(), greater<int>());
        auto test = [&](int m) {
            vector<int> v(k, 0);
            bool flag = false;
            function<void(int)> dfs = [&](int i) {
                if (flag) return;
                if (i >= n) {
                    for (int j = 0; j < k; j++) if (v[j] > m) return;
                    flag = true;
                    return;
                }
                for (int j = 0; j < k; j++) {
                    if (v[j] + cookies[i] > m) continue; // important pruning!
                    v[j] += cookies[i];
                    dfs(i + 1);
                    v[j] -= cookies[i];
                }
            };
            dfs(0);
            return flag;
        };
        int l = 1, r = 1e9 + 1;
        while (l <= r) {
            int m = (l + r) / 2;
            if (test(m)) r = m - 1;
            else l = m + 1;
        }
        return l;
    }
};
```

## 相关

- 矩阵/网格中的路径搜索（DFS + vis 撤销 + 剪枝）：`09_bfs_dfs`（矩阵中的路径）。
- 状态空间可用位掩码表示时见 `10_bit_manipulation`。
