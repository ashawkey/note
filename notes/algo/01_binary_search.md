# 二分法 (Binary Search)

## 核心思想

**搜索空间单调**是应用二分法的唯一前提：

$$
i < j \Rightarrow test(i) \Rightarrow test(j)
$$

（即 `test` 是一个单调布尔函数：前一段为 false、后一段为 true，或相反。）

二分法把"查找"问题转化为"**判定**"问题：不直接找答案，而是枚举一个答案并用 $O(n)$ 的判定函数验证，从而把 $O(n)$ 的查找降到 $O(\log n)$ 次判定。因此关键永远是**写出单调的 `test(m)`**，而不是纠结边界写法。

## 模板

### 1. 闭区间 `[l, r]` 三件套（按目标分类）

单值查找：

```c++
int binary_search(vector<int>& nums, int target) {
    int left = 0, right = nums.size() - 1;
    while (left <= right) {
        int mid = left + (right - left) / 2;  // 防溢出
        if (nums[mid] < target)      left = mid + 1;
        else if (nums[mid] > target) right = mid - 1;
        else return mid;                       // 直接返回
    }
    return -1;
}
```

左边界（第一个满足 `test` 的下标）：

```c++
bool test(int m) {
    // ... 1 2 [ 3 4 ...
    //   false [ true
    return OK(m);
}

int left_bound() {
    int l = L, r = R;
    while (l <= r) {
        int m = (l + r) / 2;
        if (test(m)) r = m - 1;
        else l = m + 1;
    }
    return l;
}
```

右边界（最后一个满足 `test` 的下标）：

```c++
bool test(int m) {
    // ... 1 2 ] 3 4 ...
    //    true ] false
    return OK(m);
}

int right_bound() {
    int l = L, r = R;
    while (l <= r) {
        int m = (l + r) / 2;
        if (test(m)) l = m + 1;
        else r = m - 1;
    }
    return r;
}
```

> 记忆：左边界返回 `l`，右边界返回 `r`。`l`/`r` 最后必然停在"边界外"一侧。

### 2. 答案二分

当答案是数值（半径、长度、金额…）且**可行性随答案单调**时，直接对答案区间二分，用 `test(m)` 判定可行性：

- **最小值最大化**（求最小的可行答案）→ 左边界模板；
- **最大值最小化**（求最大的可行答案）→ 右边界模板。

典型例子：供暖器半径（见例题）、分割数组最大值（见例题）。

### 3. STL 函数

```c++
#include <algorithm>

// lower_bound(begin, end, num): 第一个 >= num 的位置（地址），不存在返回 end
// upper_bound(begin, end, num): 第一个 >  num 的位置（地址），不存在返回 end
// 用返回值减去 begin 得到下标。

// 升序
sort(v.begin(), v.end());
int l = lower_bound(v.begin(), v.end(), x) - v.begin();
int r = upper_bound(v.begin(), v.end(), x) - v.begin();

// 降序（加比较器）
sort(v.begin(), v.end(), greater<int>());
int l = lower_bound(v.begin(), v.end(), x, greater<int>()) - v.begin();
int r = upper_bound(v.begin(), v.end(), x, greater<int>()) - v.begin();

// 记得检查越界！例如找 v 中最接近 x 的数：
int ans;
int i = lower_bound(v.begin(), v.end(), x) - v.begin();
if (i == 0) {
    ans = v[i];
} else if (i == v.size()) {
    ans = v[i - 1];
} else {
    if (abs(v[i] - x) < abs(v[i - 1] - x)) ans = v[i];
    else ans = v[i - 1];
}
```

### 4. 浮点二分

指定精度 eps，边界不用 +1/-1：

```cpp
float mysqrt(float x, float eps = 1e-4) {
    float l = 0, r = x;
    while (r - l >= eps) {
        float m = (l + r) / 2;
        if (m * m - x > eps) r = m;
        else l = m;
    }
    return l;
}
```

## 复杂度与易错点

- 时间复杂度 $O(\log R \cdot T)$，$R$ 为搜索空间大小，$T$ 为一次 `test` 的代价；空间 $O(1)$。
- **必须先确认单调性**，且 `test` 与"最优答案"是同向的还是反向的要想清楚（决定用左边界还是右边界模板）。
- 闭区间写法 `while (l <= r)` 与开区间写法容易混，**全程只用一个写法**。
- `mid = (l + r) / 2` 可能溢出，用 `l + (r - l) / 2`。
- 二分答案结束后常需处理"答案是否存在/是否越界"（STL 找不到返回 `end`）。
- **并不是能二分就一定二分**：单调性只是必要条件，若存在更简单的算法（如供暖器用贪心），应选更简单的。

## 例题

### A. 数组上的二分

#### [34 在排序数组中查找元素的第一个和最后一个位置](https://leetcode-cn.com/problems/find-first-and-last-position-of-element-in-sorted-array/)

注意 STL 找不到时返回 `end`（即 `size()`），要做边界检查。

```c++
class Solution {
public:
    vector<int> searchRange(vector<int>& nums, int target) {
        if(nums.empty()) return vector<int>(2, -1);
        vector<int> ans;
        int a = lower_bound(nums.begin(), nums.end(), target) - nums.begin();
        int b = upper_bound(nums.begin(), nums.end(), target) - nums.begin();
        // not found exception
        if(a>=nums.size() || nums[a] != target) ans.push_back(-1);
        else ans.push_back(a);
        if(b-1>=nums.size() || nums[b-1] != target) ans.push_back(-1);
        else ans.push_back(b-1);
        return ans;
    }
};
```

#### [33 搜索旋转排序数组](https://leetcode-cn.com/problems/search-in-rotated-sorted-array/)

先二分找旋转点（pivot），再对两段分别 `lower_bound`：

```c++
class Solution {
public:
    int search(vector<int>& nums, int target) {
        // find pivot
        int l=0, r=nums.size()-1;
        while(l<r){
            int m = l + (r - l) / 2;
            if(nums[m] > nums[r]) l = m + 1;
            else r = m;
        }
        // normal binary search
        int idx;
        idx = lower_bound(nums.begin(), nums.begin()+l, target) - nums.begin();
        if(idx < l && nums[idx] == target) return idx;
        idx = lower_bound(nums.begin()+l, nums.end(), target) - (nums.begin()+l);
        if(l+idx < nums.size() && nums[l+idx] == target) return idx+l;
        return -1;
    }
};
```

#### [154 寻找旋转排序数组中的最小值 II（有重复）](https://leetcode-cn.com/problems/find-minimum-in-rotated-sorted-array-ii/)

有重复时无法二分比较 `nums[m]` 与 `nums[r]`，先剔除重复：

```c++
class Solution {
public:
    int findMin(vector<int>& nums) {
        int N = nums.size();
        if(!N) return false;
        // find pivot
        int l=0, r=N-1;
        while(l<r){
            int m = l+(r-l)/2;
            if(nums[l] == nums[r]) l++; // remove duplication
            else if(nums[m] > nums[r]) l = m+1;
            else r = m;
        }
        return min(nums[0], nums[l]);
    }
};
```

#### [162 寻找峰值](https://leetcode-cn.com/problems/find-peak-element/)

边界判定麻烦但二分本身是通用的：`nums` 的"坡度"方向单调。

```c++
class Solution {
public:
    int findPeakElement(vector<int>& nums) {
        // size == 1
        if (nums.size() == 1) return 0;
        int l = 0, r = nums.size() - 1;
        while (l <= r) {
            int m = l + (r - l) / 2;
            // border
            if (m == 0) return nums[m] > nums[m + 1] ? m : m + 1;
            if (m == nums.size() - 1) return nums[m] > nums[m - 1] ? m : m - 1;
            // middle
            if (nums[m] > nums[m - 1] && nums[m] > nums[m + 1]) return m;
            else if (nums[m] <= nums[m - 1]) r = m - 1;
            else if (nums[m] <= nums[m + 1]) l = m + 1;
        }
        return -1; // never reach here.
    }
};
```

#### [剑指 Offer 04 二维数组中的查找（不是二分！）](https://leetcode-cn.com/problems/er-wei-shu-zu-zhong-de-cha-zhao-lcof/)

**没有"二维二分"这种东西**：行列各自有序，但"同时按行和列二分"会失败。正确做法是从右上角出发的 $O(n+m)$ 遍历——每一步都能确定性地排除一整行或一整列。

```
[[1, 3, 5],
 [2, 4, 6],
 [11,13,15],
 [12,14,16]]
both row-col and col-row binary search for 13 will fail!
```

```cpp
class Solution {
public:
    bool findNumberIn2DArray(vector<vector<int>>& matrix, int target) {
        if (matrix.empty() || matrix[0].empty()) return false;
        int H = matrix.size(), W = matrix[0].size();
        int i = 0, j = W - 1;
        while (i < H && j >= 0) {
            if (matrix[i][j] == target) return true;
            else if (matrix[i][j] < target) i++;
            else j--;
        }
        return false;
    }
};
```

### B. 答案二分

#### [410 分割数组的最大值](https://leetcode-cn.com/problems/split-array-largest-sum/)

"最大值最小化"经典题：二分最大和，判定"能否分成不超过 m 段"。

```c++
class Solution {
  public:
    #define ll long long
    bool check(vector<int>& nums, ll mid, int m){
        int cnt = 0;
        ll sum = 0;
        for(int i=0; i<nums.size(); i++){
            sum += nums[i];
            if(sum > mid){
                sum = nums[i];
                cnt++;
                if(cnt >= m) return true;
            }
        }
        return false;
    }
    
    int splitArray(vector<int>& nums, int m) {
        ll sum=0;
        int mx = 0;
        for(int i=0; i<nums.size(); i++){
            sum += nums[i];
            mx = max(mx, nums[i]);
        }
        ll left = mx, right = sum;      // 答案必在 [max, sum] 中
        while(left<right){
            ll mid = (left + right) / 2;
            if(check(nums, mid, m)) left = mid + 1;
            else right = mid;
        }
        return int(left);
    }
};
```

#### [174 地下城游戏](https://leetcode-cn.com/problems/dungeon-game/)

正推 DP 不满足最优子结构（既要最大生命又要最小生命），但**给定初始生命时正推可判定能否存活**，且存活随初始生命单调 → 对左边界（最小初始生命）二分。搜索上界 `+inf`，问题保证有解，故不需要越界检查。

```c++
class Solution {
public:
    bool test(vector<vector<int>>& dungeon, int m) {
        int M = dungeon.size();
        int N = dungeon[0].size();
        vector<vector<int>> dp(M + 1, vector<int>(N + 1, -0x3f3f3f));
        dp[0][1] = dp[1][0] = m;
        for (int i = 1; i <= M; i++) {
            for (int j = 1; j <= N; j++) {
                int val = max(dp[i - 1][j], dp[i][j - 1]) + dungeon[i - 1][j - 1];
                if (val <= 0) continue;
                else dp[i][j] = val;
            }
        }
        return dp[M][N] > 0;
    }
    int calculateMinimumHP(vector<vector<int>>& dungeon) {
        // [l, r] binary search, left bound
        int l = 1, r = 0x3f3f3f3f;
        while (l <= r) {
            int m = l + (r - l) / 2;
            if (test(dungeon, m)) r = m - 1;
            else l = m + 1;
        }
        return l;
    }
};
```

#### [1292 元素和小于等于阈值的正方形的最大边长](https://leetcode-cn.com/problems/maximum-side-length-of-a-square-with-sum-less-than-or-equal-to-threshold/)

前缀和求子矩阵和 + 右边界搜索（边长越大越不可能 ≤ threshold）：

```c++
class Solution {
public:
    bool test(vector<vector<int>>& s, int k, int threshold) {
        int m = s.size();
        int n = s[0].size();
        for (int i = 0; i < m - k; i++) {
            for (int j = 0; j < n - k; j++) {
                if (s[i + k][j + k] - s[i + k][j] - s[i][j + k] + s[i][j] <= threshold) {
                    return true;
                }
            }
        }
        return false;
    }

    int maxSideLength(vector<vector<int>>& mat, int threshold) {
        int m = mat.size();
        if (m == 0) return 0;
        int n = mat[0].size();
        if (n == 0) return 0;
        // prefix sum
        vector<vector<int>> s(m + 1, vector<int>(n + 1, 0));
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                s[i][j] = s[i - 1][j] + s[i][j - 1] - s[i - 1][j - 1] + mat[i - 1][j - 1];
            }
        }
        // bs right bound []
        int l = 1, r = max(m, n);
        while (l <= r) {
            int m = (l + r) / 2;
            if (test(s, m, threshold)) l = m + 1;
            else r = m - 1;
        }
        return r;
    }
};
```

#### [1898 可移除字符的最大数目](https://leetcode-cn.com/problems/maximum-number-of-removable-characters/)

单调性：可移除数量越多，`p` 越难成为 `s` 的子序列。子序列判定用双指针 $O(n)$。

```cpp
class Solution {
public:
    // lr pointer for O(m+n) subsequence check
    bool check(int m, string& s, string& p, vector<int>& v) {
        int i = 0, j = 0;
        while (i < s.size() && j < p.size()) {
            if (s[i] == p[j] && v[i] > m) {
                i++; j++;
            } else {
                i++;
            }
        }
        return (j == p.size());
    }
    int maximumRemovals(string s, string p, vector<int>& removable) {
        // for O(1) check if a char is removed at time t.
        vector<int> v(s.size(), 0x7fffffff);
        for (int i = 0; i < removable.size(); i++) {
            v[removable[i]] = i;
        }
        // binary search for left border
        int l = 0, r = removable.size() - 1;
        while (l <= r) {
            int m = (l + r) / 2;
            if (check(m, s, p, v)) l = m + 1;
            else r = m - 1;
        }
        return r + 1;
    }
};
```

#### [786 第 K 个最小的素数分数](https://leetcode-cn.com/problems/k-th-smallest-prime-fraction/)

二分答案 + 双指针计数：小于 $x$ 的分数数量随 $x$ 单调。$O(N\log C)$，$C = \max_{v\in arr} v$。堆解法（多路归并）见 `12_heap`。

```cpp
class Solution {
public:
    vector<int> kthSmallestPrimeFraction(vector<int>& arr, int k) {
        int n = arr.size();
        double left = 0.0, right = 1.0;
        while (true) {
            double mid = (left + right) / 2;
            
            int i = -1, count = 0;
            int x = 0, y = 1;
            
            // count how many elements are smaller than mid.
            for (int j = 1; j < n; ++j) {
                while ((double)arr[i + 1] / arr[j] < mid) {
                    ++i;
                    if (arr[i] * y > arr[j] * x) {
                        x = arr[i];
                        y = arr[j];
                    }
                }
                count += i + 1;
            }

            if (count == k) {
                return {x, y};
            }
            if (count < k) {
                left = mid;
            }
            else {
                right = mid;
            }
        }
    }
};
```

#### [475 供暖器](https://leetcode-cn.com/problems/heaters/)

半径的可行性单调，可以二分；但**贪心更简单**（记录每个房子的最近供暖器，$O(n\log n)$，见 `05_greedy`）。"能二分"不代表"该二分"。

```cpp
class Solution {
public:
    int findRadius(vector<int>& houses, vector<int>& heaters) {
        sort(houses.begin(), houses.end());
        sort(heaters.begin(), heaters.end());
        // O(nlogn)
        auto check = [&](int m) {
            int next = 0;
            for (int i = 0; i < heaters.size(); i++) {
                int h = heaters[i];
                if (h - m > houses[next]) return false;
                next = upper_bound(houses.begin(), houses.end(), h + m) - houses.begin();
                if (next == houses.size()) return true;
            }
            return false;
        };
        // O(nlog^2n)
        int l = 0, r = max(houses.back(), heaters.back()) - houses[0] + 1;
        while (l <= r) {
            int m = (l + r) / 2;
            if (check(m)) r = m - 1;
            else l = m + 1;
        }
        return l;
    }
};
```
