# 堆 (Heap / Priority Queue)

## 核心思想

堆是一个**局部有序**的完全二叉树（与 BST 的全局有序不同）：

- 最大值堆：每个节点 ≥ 左右孩子；最小值堆：每个节点 ≤ 左右孩子。
- 同一组数据可以构建出许多不同的堆。
- 一般用静态数组存储（完全二叉树编号，根为 0，节点 i 的孩子为 $2i+1$、$2i+2$）。

**典型用途**：动态取最值（topK、数据流中位数）、多路归并（用堆同时管理多个递增序列的头）、贪心的"每次取当前最优"。

## 模板（手写堆，最小值堆）

```c++
const int maxN = 1000;
int N;
int arr[maxN];

// sink i until it is smaller than its children
void siftdown(int i) {
    int tmp = arr[i];
    int j = 2 * i + 1;
    while (j < N) {
        if (j < N - 1 && arr[j] > arr[j + 1]) j++;
        if (tmp > arr[j]) {
            arr[i] = arr[j];
            i = j;
            j = 2 * j + 1;
        }
        else break;
    }
    arr[i] = tmp;
}

// lift i until its parent is smaller than it (or to top).
void siftup(int i) {
    int tmp = arr[i];
    while (i > 0 && arr[(i - 1) / 2] > tmp) {
        arr[i] = arr[(i - 1) / 2];
        i = (i - 1) / 2;
    }
    arr[i] = tmp;
}

// change arr into a min heap, O(n)
void build() {
    // N/2-1 is the last father.
    for (int i = N / 2 - 1; i >= 0; i--) {
        siftdown(i);
    }
}

// O(logn)
bool insert(int d) {
    if (N == maxN) return false;
    arr[N] = d;  // add new data to the bottom
    siftup(N);
    N++;
    return true;
}

// O(logn)
int pop() {
    if (N == 0) return -1;
    swap(arr[0], arr[--N]); // swap to bottom (out of heap)
    if (N > 1) siftdown(0);
    return arr[N];
}
```

- 建堆 $O(n)$（不是 $O(n\log n)$）：从最后一个非叶节点开始逐个 `siftdown`，用**错位相减**可证 $\sum_{i=0}^{\log n} 2^i (\log n - i) < 2n$。
- 竞赛直接用 STL：`priority_queue<T>`（大根堆）、`priority_queue<T, vector<T>, greater<T>>`（小根堆）。

## 例题

### [剑指 Offer 41 数据流的中位数](https://leetcode-cn.com/problems/find-median-from-data-stream/)

**双堆**：大根堆存较小一半、小根堆存较大一半，保持两堆大小差 ≤ 1，插入时可能需要跨堆交换。插入 $O(\log N)$、查询 $O(1)$：

```cpp
class MedianFinder {
public:
    priority_queue<int> mx;   // smaller half (max-heap)
    priority_queue<int, vector<int>, greater<int>> mn; // larger half (min-heap)

    MedianFinder() {}

    void addNum(int num) {
        if (mx.size() == mn.size()) {
            if (!mn.empty() && num > mn.top()) {
                mx.push(mn.top()); mn.pop();
                mn.push(num);
            } else {
                mx.push(num);
            }
        } else {
            if (num < mx.top()) {
                mn.push(mx.top()); mx.pop();
                mx.push(num);
            } else {
                mn.push(num);
            }
        }
    }

    double findMedian() {
        if (mn.size() == mx.size()) {
            return (mn.top() + mx.top()) * 0.5;
        } else {
            return mx.top();
        }
    }
};
```

另一个思路：`multiset` 本身就是红黑树，插入 $O(\log N)$，关键在于**维护指向中位的迭代器**实现 $O(1)$ 查询：

```cpp
class MedianFinder {
    multiset<int> data;
    multiset<int>::iterator mid;

public:
    MedianFinder() : mid(data.end()) {}

    void addNum(int num)
    {
        int n = data.size();
        data.insert(num);

        if (!n)                                 // first element inserted
            mid = data.begin();
        else if (num < *mid)                    // median is decreased
            mid = (n & 1 ? mid : prev(mid));
        else                                    // median is increased
            mid = (n & 1 ? next(mid) : mid);
    }

    double findMedian()
    {
         int n = data.size();
        return (*mid + *next(mid, n % 2 - 1)) * 0.5;
    }
};
```

### [786 第 K 个最小的素数分数](https://leetcode-cn.com/problems/k-th-smallest-prime-fraction/)

堆解法本质是**多路归并**：把每个分母 j 看作一路，第 1 小是 `arr[0]/arr[j]`，弹出后补同路的下一项。$O(k\log N)$。（二分计数解法见 `01_binary_search`。）

```cpp
class Solution {
public:
    vector<int> kthSmallestPrimeFraction(vector<int>& arr, int k) {
        int n = arr.size();
        auto cmp = [&](const pair<int, int>& x, const pair<int, int>& y) {
            return arr[x.first] * arr[y.second] > arr[x.second] * arr[y.first];
        };
        priority_queue<pair<int, int>, vector<pair<int, int>>, decltype(cmp)> q(cmp);
        for (int j = 1; j < n; ++j) {
            q.emplace(0, j);
        }
        for (int _ = 1; _ < k; ++_) {
            auto [i, j] = q.top();
            q.pop();
            if (i + 1 < j) {
                q.emplace(i + 1, j);
            }
        }
        return {arr[q.top().first], arr[q.top().second]};
    }
};
```

#### 类似题：[373 查找和最小的 K 对数字](https://leetcode-cn.com/problems/find-k-pairs-with-smallest-sums/)

固定一个数时，和对另一个数单调 → 同样用多路归并：

```cpp
class Solution {
public:
    vector<vector<int>> kSmallestPairs(vector<int>& nums1, vector<int>& nums2, int k) {
        auto cmp = [&](const pair<int, int>&x, const pair<int, int>&y) {
            return nums1[x.first] + nums2[x.second] > nums1[y.first] + nums2[y.second];
        };
        priority_queue<pair<int, int>, vector<pair<int, int>>, decltype(cmp)> q(cmp);
        for (int i = 0; i < nums2.size(); i++) {
            q.emplace(0, i);
        }
        vector<vector<int>> ans;
        for (int i = 0; i < k; i++) {
            if (q.empty()) break; // when k > nums1.size() * nums2.size()
            auto [m, n] = q.top(); q.pop();
            ans.push_back({nums1[m], nums2[n]});
            if (m + 1 < nums1.size()) {
                q.emplace(m + 1, n);
            }
        }
        return ans;
    }
};
```

### [264 丑数 II](https://leetcode-cn.com/problems/ugly-number-ii/)

堆版：每次弹出最小的丑数，把它 ×2/×3/×5 入堆（去重）。$O(n\log n)$；三指针的 $O(n)$ 解法见 `07_dynamic_programming`。

```cpp
class Solution {
public:
    int nthUglyNumber(int n) {
        long long ans;
        priority_queue<long long, vector<long long>, greater<long long>> q;
        q.push(1);
        for (int i = 0; i < n; i++) {
            ans = q.top(); q.pop();
            while (!q.empty() && q.top() == ans) q.pop();
            q.push(ans * 2);
            q.push(ans * 3);
            q.push(ans * 5);
        }
        return ans;
    }
};
```

### topK 通用套路

- 找**前 K 大**：维护大小为 K 的小根堆，新元素比堆顶大就替换，$O(n\log K)$；
- 找**前 K 小**：维护大小为 K 的大根堆；
- 第 K 大/小同理（堆顶即答案）。
