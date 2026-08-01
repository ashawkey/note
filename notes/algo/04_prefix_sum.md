# 前缀和 / 差分 (Prefix Sum / Difference Array)

## 核心思想

把"区间操作"变成"端点操作"，是**静态区间求和/区间修改**的最简单工具：

- **前缀和**：预处理 $O(n)$，之后任意区间求和 $O(1)$。适合"静态数组 + 多次区间查询"。
- **差分**：区间增量修改 $O(1)$（只改两个端点），最后求前缀和还原。适合"先做完全部区间修改、再一次性查询"。
- 如果修改和查询**交替进行**，则只能上树状数组/线段树（`16_segment_tree`、`17_binary_indexed_tree`）。

## 模板

### 1D 前缀和

```cpp
// build: sums[i] = sums[i - 1] + arr[i];
// query: Q[i, j] = sums[j] - sums[i - 1]
```

```cpp
class NumArray {  // LeetCode 303
public:
    vector<int> sums;

    NumArray(vector<int>& nums) {
        int n = nums.size();
        sums.resize(n + 1); // note: sums[0] is reserved for easy border handling.
        for (int i = 0; i < n; i++) {
            sums[i + 1] = sums[i] + nums[i];
        }
    }

    int sumRange(int i, int j) {
        return sums[j + 1] - sums[i];
    }
};
```

### 2D 前缀和

用 `s[i+1][j+1]` 表示**左上子矩阵 `x[0:i+1][0:j+1]`** 的和（正向循环计算，边界自动处理）：

```cpp
class NumMatrix {  // LeetCode 304
public:
    vector<vector<int>> sums;

    NumMatrix(vector<vector<int>>& matrix) {
        int m = matrix.size();
        int n = matrix[0].size();
        sums.resize(m + 1, vector<int>(n + 1));
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                sums[i + 1][j + 1] = sums[i][j + 1] + sums[i + 1][j] - sums[i][j] + matrix[i][j];
            }
        }
    }

    int sumRegion(int row1, int col1, int row2, int col2) {
        return sums[row2 + 1][col2 + 1] - sums[row1][col2 + 1] - sums[row2 + 1][col1] + sums[row1][col1];
    }
};
```

### 1D 差分

```cpp
// build: diff[i] = arr[i] - arr[i - 1];
// modify: add x to [i, j] -> diff[i] += x; diff[j + 1] -= x;
// query (must be sequential): arr[i] = diff[i] + arr[i - 1]
```

```cpp
class Solution {  // LeetCode 1109 航班预订统计
public:
    vector<int> corpFlightBookings(vector<vector<int>>& bookings, int n) {
        vector<int> nums(n, 0);
        for (auto& v : bookings) {
            nums[v[0] - 1] += v[2];
            if (v[1] < n) nums[v[1]] -= v[2];
        }
        for (int i = 1; i < n; i++) {
            nums[i] += nums[i - 1];
        }
        return nums;
    }
};
```

区间范围远大于区间数量时，用 `map` 做**稀疏差分**（离散化，每次查询扫描全表 O(n)）：

```cpp
class MyCalendarThree {  // LeetCode 732
public:
    map<int, int> m;
    MyCalendarThree() {}
    int book(int start, int end) {
        m[start]++;
        m[end]--;
        // query max value
        int ans = 0, val = 0;
        for (auto [k, v] : m) { // ordered map
            val += v; // reconstruct
            ans = max(ans, val);
        }
        return ans;
    }
};
```

### 2D 差分

用 `s[i][j]` 表示**右下子矩阵 `x[i:][j:]`** 的增量（正向循环求原始矩阵值）：

```cpp
// init as 0
vector<vector<int>> diff(m + 1, vector<int>(n + 1, 0));

// modify: add x to mat[i:i+ii][j:j+jj]
diff[i][j] += x;
diff[i][j + jj] -= x;
diff[i + ii][j] -= x;
diff[i + ii][j + jj] += x;

// query: saved to res[i+1][j+1] for easy border condition
vector<vector<int>> res(m + 1, vector<int>(n + 1, 0));
for (int i = 1; i <= m; i++) {
    for (int j = 1; j <= n; j++) {
        res[i][j] = res[i][j - 1] + res[i - 1][j] - res[i - 1][j - 1] + diff[i - 1][j - 1];
    }
}
```

## 例题

### [2132 用邮票贴满网格图](https://leetcode-cn.com/problems/stamping-the-grid/)

核心操作两个：① O(1) 判断某个 H×W 窗口是否全空（2D 前缀和）；② O(1) 标记窗口内所有格子被覆盖（2D 差分）：

```cpp
class Solution {
public:
    bool possibleToStamp(vector<vector<int>>& grid, int H, int W) {
        int m = grid.size();
        int n = grid[0].size();
        vector<vector<int>> s(m + 1, vector<int>(n + 1, 0));
        vector<vector<int>> d(m + 1, vector<int>(n + 1, 0));
        // prefix sum
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                s[i][j] = grid[i - 1][j - 1] + s[i - 1][j] + s[i][j - 1] - s[i - 1][j - 1];
            }
        }
        // loop window: 检查是否整个窗口都不被占据
        for (int i = 0; i <= m - H; i++) {
            for (int j = 0; j <= n - W; j++) {
                if (s[i + H][j + W] - s[i][j + W] - s[i + H][j] + s[i][j] == 0) {
                    // update difference
                    d[i][j] += 1;
                    d[i][j + W] -= 1;
                    d[i + H][j] -= 1;
                    d[i + H][j + W] += 1;
                }
            }
        }
        // query final value: 未被覆盖的空格子 -> false
        vector<vector<int>> res(m + 1, vector<int>(n + 1, 0));
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                res[i][j] = res[i][j - 1] + res[i - 1][j] - res[i - 1][j - 1] + d[i - 1][j - 1];
                if (res[i][j] == 0 && grid[i - 1][j - 1] == 0) return false;
            }
        }
        return true;
    }
};
```

### [437 路径总和 III](https://leetcode-cn.com/problems/path-sum-iii/)（树上前缀和）

字典记录**当前节点到根路径上的前缀和**的出现次数（DFS 进入时加、回溯时减）：

```cpp
class Solution {
public:
    int pathSum(TreeNode* root, int targetSum) {
        int ans = 0;
        unordered_map<int, int> m;
        m[0] = 1;

        function<void(TreeNode*, int)> find = [&](TreeNode* r, int s) {
            if (r == nullptr) return;
            int ss = s + r->val;
            ans += m[ss - targetSum];
            m[ss]++;
            find(r->left, ss);
            find(r->right, ss);
            m[ss]--;
        };

        find(root, 0);
        return ans;
    }
};
```

### [560 和为 K 的子数组](https://leetcode-cn.com/problems/subarray-sum-equals-k/)

前缀和 + **哈希计数**把 $O(n^2)$ 降到 $O(n)$（详见 `13_hash`）：

```cpp
class Solution {
public:
    int subarraySum(vector<int>& nums, int k) {
        map<int, int> m;
        m[0] = 1;
        int sum = 0, ans = 0;
        for(int i = 0; i < nums.size(); i++){
            sum += nums[i];
            if (m.count(sum - k)) ans += m[sum-k];
            if (m.count(sum)) m[sum]++;
            else m[sum] = 1;
        }
        return ans;
    }
};
```

### 环形子数组的最大和（前缀和 + 单调队列）

见 `11_stack_queue`（918 题）。

## 相关

- 修改与查询交替进行时改用树状数组/线段树：`17_binary_indexed_tree`、`16_segment_tree`。
- 适龄朋友的前缀和计数解法见 `02_two_pointers`。
