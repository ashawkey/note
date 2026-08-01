# 树状数组 (Binary Indexed Tree / Fenwick Tree)

## 核心思想

用 $O(\log N)$ 支持**单点更新**与**区间求和**。

- 定义：$C[i] = a[i - lowbit(i) + 1] + \cdots + a[i]$，其中 `lowbit(x) = x & (-x)` 保留二进制最低位的 1。

  ```
  e.g. x = 00001101, -x = 11110011, lowbit(x) = 00000001
  x2 = x + lowbit(x) = 00001110, lowbit(x2) = 00000010
  x3 = x2 + lowbit(x2) = 00010000
  ```

- **求和**：$\sum_i^j = sum(j) - sum(i-1)$，其中 $sum(k) = C[n_1] + \cdots + C[k]$，$k - lowbit(k)$ 每次消去 k 的最低位 1，至多 $\log N$ 项。
- **更新**：`a[i]` 变化连锁更新 $C[i], C[i+lowbit(i)], \cdots$，同样至多 $\log N$ 项。
- **建树 $O(N)$**：$C[k] = sum(k) - sum(k - lowbit(k))$。

> **下标从 1 开始！！！** 0 号位是哨兵。

## 模板

```c++
int bit[maxn];
int N;

int lowbit(int x) { return x & (-x); }

void add(int i, int v) {
    for (; i <= N; i += lowbit(i)) bit[i] += v;
}

// sum of arr[1, i]
int getsum(int i) {
    int res = 0;
    for (; i > 0; i -= lowbit(i)) res += bit[i];
    return res;
}
```

反向变体（求后缀和）：

```c++
void add(int i, int v) {
    for (; i > 0; i -= lowbit(i)) bit[i] += v;
}

// sum of arr[i, N]
int getsum(int i) {
    int res = 0;
    for (; i <= N; i += lowbit(i)) res += bit[i];
    return res;
}
```

支持"修改 + 区间求和"的类封装（注意 `update` 是**替换**不是增量）：

```cpp
class NumArray {
public:
    int lowbit(int x) { return x & (-x); }
    vector<int> bit, arr; // arr is to record the single point value.

    // arr[i] += v
    void add(int i, int v) {
        for (; i < bit.size(); i += lowbit(i)) bit[i] += v;
    }

    // prefix sum of arr[1, ..., i]
    int query(int i) {
        int res = 0;
        for (; i > 0; i -= lowbit(i)) res += bit[i];
        return res;
    }

    NumArray(vector<int>& nums) {
        bit.resize(nums.size() + 1);
        arr.resize(nums.size() + 1);
        for (int i = 0; i < nums.size(); i++) {
            add(i + 1, nums[i]);
            arr[i + 1] = nums[i];
        }
    }

    // note: this is modify, not add!
    void update(int index, int val) {
        add(index + 1, val - arr[index + 1]);
        arr[index + 1] = val;
    }

    int sumRange(int left, int right) {
        return query(right + 1) - query(left);
    }
};
```

## 例题

### [POJ 3321 Apple Tree](https://www.luogu.com.cn/problem/UVA12235)（DFS 时间戳 + BIT）

关键点：如何排列节点使**每棵子树落在不交的连续区间**——做一次 DFS，用进入/离开时间给节点编号（`S[n]`~`E[n]` 即子树区间）：

```c++
const int maxN = 100000 + 5;
int N, M;
vector<int> t[maxN];
int S[maxN], E[maxN];

int arr[maxN];
int bit[maxN];

int lowbit(int x) { return x & (-x); }

void modify(int i, int v) {
    while (i <= N) {
        bit[i] += v;
        i += lowbit(i);
    }
}

int getsum(int i) {
    int res = 0;
    while (i > 0) {
        res += bit[i];
        i -= lowbit(i);
    }
    return res;
}

int ti = 1;
void dfs(int n) {
    S[n] = ti;
    for (int to : t[n]) {
        ti++;
        dfs(to);
    }
    E[n] = ti;
}

int main() {
    memset(bit, 0, sizeof(bit));
    cin >> N;
    for (int i = 1; i <= N; i++) arr[i] = 1;
    for (int i = 1; i <= N; i++) modify(i, 1);
    for (int i = 1; i < N; i++) {
        int a, b;
        cin >> a >> b;
        t[a].push_back(b);
    }
    dfs(1);
    cin >> M;
    for (int i = 0; i < M; i++) {
        char q;
        int a;
        cin >> q >> a;
        if (q == 'Q') {
            int res = getsum(E[a]) - getsum(S[a] - 1);
            cout << res << endl;
        }
        else {
            int value = arr[a] == 0 ? 1 : -1;
            arr[a] += value;
            modify(S[a], value);
        }
    }
}
```

### [POJ 1195 Mobile phones](https://www.luogu.com.cn/problem/UVA1470)（二维树状数组）

二维 BIT 就是两层循环的嵌套（树结构见下），更新/查询 $O(\log^2 N)$：

```c++
const int maxN = 1024 + 5;
int N;

int bit[maxN][maxN];

int lowbit(int x) { return x & (-x); }

void modify(int x, int y, int v) {
    for (int i = x; i <= N; i += lowbit(i)) {
        for (int j = y; j <= N; j += lowbit(j)) {
            bit[i][j] += v;
        }
    }
}

int getsum(int x, int y) {
    int res = 0;
    for (int i = x; i > 0; i -= lowbit(i)) {
        for (int j = y; j > 0; j -= lowbit(j)) {
            res += bit[i][j];
        }
    }
    return res;
}

int getsquare(int x1, int y1, int x2, int y2) {
    return getsum(x2, y2) - getsum(x2, y1 - 1) - getsum(x1 - 1, y2) + getsum(x1 - 1, y1 - 1);
}
```

（10×10 的二维树结构：每个位置的值是"横纵两个 lowbit 链"的交织，外层循环走 x 的 lowbit 链、内层走 y 的。）

## 局限

- **区间最值**：BIT 可以存 max/min，但复杂度 $O(\log^2 n)$、实现繁琐，直接用线段树更省心。
- **区间增量 + 区间求和**：需要差分 BIT（两个 BIT），不如线段树直观。
- 需要"区间替换"或任意区间操作时，用线段树（`16_segment_tree`）。

## 相关

- Lost Cows（线段树/BIT 找第 k 个未使用位置）见 `16_segment_tree` 的例题部分。
- 差分数组见 `04_prefix_sum`。
