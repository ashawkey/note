# 复杂度与优化 (Complexity & Optimization)

## 复杂度分析要点

- 大 O 只关心**增长趋势**，忽略常数与低阶项；刷题常用速查：$O(1) < O(\log n) < O(n) < O(n\log n) < O(n^2) < O(2^n) < O(n!)$。
- $n \le 10^5$ 左右接受 $O(n\log n)$；$n \le 10^3$ 接受 $O(n^2)$；$n \le 20$ 左右才考虑 $O(2^n)$ 状压。
- **摊还分析**：单调队列、并查集、双指针"每个元素进出一次"的结构，单步可能 $O(n)$，摊还 $O(1)$（如并查集路径压缩 < 4N）。

### 主定理（分治复杂度）

对 $T(n) = aT(n/b) + O(n^d)$：

$$
T(n) = \begin{cases}
O(n^d) & a < b^d \\
O(n^d \log n) & a = b^d \\
O(n^{\log_b a}) & a > b^d
\end{cases}
$$

例：归并 $T(n) = 2T(n/2) + O(n)$ → $O(n\log n)$；快排平均也是。

## 逆序对计数

### 树状数组法（$O(n\log n)$，需离散化）

思路：创建值域的 BIT，从左到右插入，插入时统计"已出现的比它大的数"：

```c++
int lowbit(int x) { return x & (-x); }
void add(int n, int x) {
    for (; n <= N; n += lowbit(n)) bit[n] += x;
}
int getsum(int n) {
    int res = 0;
    for (; n > 0; n -= lowbit(n)) res += bit[n];
    return res;
}

// main:
// 1) 离散化：sort(arr2) 后 map 每个值 -> 下标
// 2) for i in [0, N): add(m[arr[i]] + 1, 1); ans += i + 1 - getsum(m[arr[i]] + 1);
```

### 归并排序法（$O(n\log n)$）

归并过程中，左边元素大于右边元素时，**左边剩余的所有元素都与它构成逆序对**：

```c++
long long ans = 0;
void merge(int l, int r, int *a, int *b) {
    if (r == l) return;
    int m = (l + r) / 2;
    merge(l, m, a, b);
    merge(m + 1, r, a, b);
    int i = l, j = m + 1, k = l;
    while (i <= m && j <= r) {
        if (a[i] <= a[j]) b[k++] = a[i++];
        else {
            ans += m + 1 - i; // 左边剩余全部构成逆序对
            b[k++] = a[j++];
        }
    }
    while (i <= m) b[k++] = a[i++];
    while (j <= r) b[k++] = a[j++];
    for (int i = l; i <= r; i++) a[i] = b[i];
}
```

### 变体：[493 翻转对](https://leetcode-cn.com/problems/reverse-pairs/)

`i < j` 且 `a[i] > 2*a[j]`。归并中**独立于排序做一次 O(n) 计数**（注意 `2*a[j]` 溢出，用 long long）：

```c++
// 在归并两段前单独计数：
int i = l, j = m + 1;
while (i <= m && j <= r) {
    if (a[i] > 2 * a[j]) {   // 用 long long 比较
        ans += m - i + 1;
        j++;
    } else i++;
}
```

## 常数优化与常用小技巧

- **二分**：`l + (r - l) / 2` 防溢出；循环内多次二分可改双指针（见 `02_two_pointers`）。
- **循环里多次二分 → 双指针**：两边界单调时（适龄朋友题）。
- **把循环查找换成哈希定位**：$O(n)$ → 常数（见 `13_hash`、`07` 的 873 题）。
- **Python 自定义排序**：`functools.cmp_to_key`：

```python
def cmp(a, b):
    a, b = int(a + b), int(b + a)
    if a == b: return 0
    elif a < b: return -1
    else: return 1

nums = sorted(nums, key=functools.cmp_to_key(cmp))
```

C++ 对应（[179 最大数](https://leetcode-cn.com/problems/largest-number/)）：

```c++
class Solution {
public:
    static bool cmp(const string& a, const string& b) {
        return stoll(a + b) > stoll(b + a);
    }
    string largestNumber(vector<int>& nums) {
        vector<string> snums;
        for (int x : nums) snums.push_back(to_string(x));
        sort(snums.begin(), snums.end(), cmp);
        string ans;
        for (auto& s : snums) ans += s;
        while (ans[0] == '0' && ans.size() > 1) ans = ans.substr(1);
        return ans;
    }
};
```

### O(1) 空间技巧

- 找唯一出现一次的数：异或（见 `10_bit_manipulation`）。
- 找重复数（[287](https://leetcode-cn.com/problems/find-the-duplicate-number/)）：值域 `[1, n]` 视为链表，**Floyd 判环**：

```c++
class Solution {
public:
    int findDuplicate(vector<int>& nums) {
        int slow = 0, fast = 0;
        do {
            slow = nums[slow];
            fast = nums[nums[fast]];
        } while (slow != fast);
        slow = 0;
        while (slow != fast) {
            slow = nums[slow];
            fast = nums[fast];
        }
        return slow;
    }
};
```

## 相关

- 二分模板见 `01_binary_search`；树状数组见 `17_binary_indexed_tree`；快排分治见 `51_sorting`。
