# 线段树 (Segment Tree)

## 核心思想

$O(\log N)$ 完成**区间更新**与**区间查询**。

- 根节点是区间 $[a, b]$，左孩子 $[a, (a+b)/2]$，右孩子 $[(a+b)/2+1, b]$，叶子是单个元素。深度 $\lceil \log_2(b-a+1)\rceil + 1$。
- **区间分解**：递归从根开始，把任意查询/修改区间分解成若干终止节点（每层最多两个），即"区间查询本质就是区间分解"。
- **lazy 更新**：区间更新时先只标记、不真正下传；只有需要深入时才把标记下推。复杂度因此保持 $O(\log N)$。
- 数组存储：虽然不是完全二叉树但近似，$2\cdot 2^{\lceil\log_2 n\rceil}-1 \le 4n-1$，所以开 `[4*maxn]` 安全，用 $2i+1, 2i+2$ 代替指针。

## 模板

### 静态区间最值（Balanced Lineup, POJ 3264）

```c++
const int maxn = 50005;
const int inf = 0x7fffffff;
int N, Q;

struct node {
    int L, R;
    int ma, mi;
    int mid() { return (L + R) / 2; }
};

node tree[4 * maxn];
int arr[maxn];

void build(int root, int l, int r) {
    tree[root].L = l;
    tree[root].R = r;
    if (l == r) {
        tree[root].mi = tree[root].ma = arr[l];
        return;
    }
    build(2 * root + 1, l, (l + r) / 2);
    build(2 * root + 2, (l + r) / 2 + 1, r);
    tree[root].mi = min(tree[2 * root + 1].mi, tree[2 * root + 2].mi);
    tree[root].ma = max(tree[2 * root + 1].ma, tree[2 * root + 2].ma);
}

int MIN = inf, MAX = 0;
void query(int root, int l, int r) {
    if (tree[root].L == l && tree[root].R == r) {
        MIN = min(MIN, tree[root].mi);
        MAX = max(MAX, tree[root].ma);
        return;
    }
    int mid = tree[root].mid();
    if (r <= mid) query(2 * root + 1, l, r);
    else if (l > mid) query(2 * root + 2, l, r);
    else {
        query(2 * root + 1, l, mid);
        query(2 * root + 2, mid + 1, r);
    }
}
```

### 区间加 + 区间求和（lazy 为 int 型增量）

```c++
const int maxn = 100005;
#define ll long long

struct node {
    int l, r;
    ll sum, inc;
    int mid() { return (l + r) / 2; }
} tr[maxn << 2];

#define lc 2*rt+1
#define rc 2*rt+2

void pushup(int rt) {
    tr[rt].sum = tr[lc].sum + tr[rc].sum;
}

void pushdown(int rt) {
    if (tr[rt].inc) {
        tr[rt].sum += (tr[rt].r - tr[rt].l + 1) * tr[rt].inc;
        tr[lc].inc += tr[rt].inc;
        tr[rc].inc += tr[rt].inc;
        tr[rt].inc = 0;
    }
}

void build(int rt, int l, int r) {
    tr[rt].l = l;
    tr[rt].r = r;
    tr[rt].inc = 0;
    if (l == r) {
        tr[rt].sum = arr[l];   // 若初始全 0 可不赋值
        return;
    }
    int m = (l + r) / 2;
    build(lc, l, m);
    build(rc, m + 1, r);
    pushup(rt);
}

void add(int rt, int l, int r, int v) {
    if (tr[rt].l == l && tr[rt].r == r) {
        tr[rt].inc += v;       // 打 lazy 标记，不真正下传
        return;
    }
    tr[rt].sum += (r - l + 1) * v;
    int m = tr[rt].mid();
    if (l > m) add(rc, l, r, v);
    else if (r <= m) add(lc, l, r, v);
    else {
        add(lc, l, m, v);
        add(rc, m + 1, r, v);
    }
}

ll query(int rt, int l, int r) {
    if (tr[rt].l == l && tr[rt].r == r) return tr[rt].sum + tr[rt].inc * (r - l + 1);
    pushdown(rt);
    int m = tr[rt].mid();
    if (l > m) return query(rc, l, r);
    else if (r <= m) return query(lc, l, r);
    else return query(lc, l, m) + query(rc, m + 1, r);
}
```

> 记忆：`add` 不打到底（只加 sum 和 lazy），`query` 前先 `pushdown`，`build`/`modify` 后 `pushup`。

### 区间替换最值（lazy 为 bool 型"是否被替换"）

替换与累加不同：替换用 `=`、lazy 是 bool；modify 最后要 `pushup`：

```c++
const static int maxn = 10005;

struct node {
    int l, r;
    int mx, lazy;
    int m() { return (l + r) / 2; }
} seg[4 * maxn];

void build(int rt, int l, int r) {
    seg[rt].l = l;
    seg[rt].r = r;
    seg[rt].mx = 0;
    seg[rt].lazy = 0;
    if (l == r) return;
    build(2 * rt + 1, l, (l + r) / 2);
    build(2 * rt + 2, (l + r) / 2 + 1, r);
}

void pushup(int rt) {
    seg[rt].mx = max(seg[2 * rt + 1].mx, seg[2 * rt + 2].mx);
}

void pushdown(int rt) {
    if (seg[rt].lazy) {
        seg[2 * rt + 1].mx = seg[rt].mx;
        seg[2 * rt + 2].mx = seg[rt].mx;
        seg[2 * rt + 2].lazy = seg[2 * rt + 1].lazy = 1;
        seg[rt].lazy = 0;
    }
}

int query(int rt, int l, int r) {
    if (l == seg[rt].l && r == seg[rt].r)
        return seg[rt].mx;
    pushdown(rt);
    int m = seg[rt].m();
    if (r <= m) return query(2 * rt + 1, l, r);
    else if (l > m) return query(2 * rt + 2, l, r);
    else return max(query(2 * rt + 1, l, m), query(2 * rt + 2, m + 1, r));
}

void modify(int rt, int l, int r, int v) {
    if (l == seg[rt].l && r == seg[rt].r) {
        seg[rt].mx = v; // set to v, not add v.
        seg[rt].lazy = 1;
        return;
    }
    pushdown(rt);
    int m = seg[rt].m();
    if (r <= m) modify(2 * rt + 1, l, r, v);
    else if (l > m) modify(2 * rt + 2, l, r, v);
    else {
        modify(2 * rt + 1, l, m, v);
        modify(2 * rt + 2, m + 1, r, v);
    }
    pushup(rt);
}
```

## 动态开点（区间范围大，如 1e9）

只在访问到某节点时才创建它。**Range Module**（[LeetCode 715](https://leetcode-cn.com/problems/range-module/)）的完整例子：

```cpp
class RangeModule {
public:
    // dynamically allocated segment tree
    struct node {
        int l, r, v, z;
        node *ll, *rr;
        node(int _l, int _r, int _v = 0): l(_l), r(_r), v(_v), z(0), ll(nullptr), rr(nullptr) {}
        int m() { return (l+r)/2; }
    };

    void pushdown(node* n) {
        if (n->z) {
            n->ll->v = n->v;
            n->ll->z = 1;
            n->rr->v = n->v;
            n->rr->z = 1;
            n->z = 0;
        }
    }

    void pushup(node* n) {
        n->v = n->ll->v && n->rr->v;
    }

    void modify(node* n, int l, int r, int v) {
        if (n->l == l && n->r == r) {
            n->v = v;
            n->z = 1;
            return;
        }
        int nm = n->m();
        if (n->ll == nullptr) n->ll = new node(n->l, nm, n->v);
        if (n->rr == nullptr) n->rr = new node(nm + 1, n->r, n->v);
        pushdown(n);
        if (r <= nm) modify(n->ll, l, r, v);
        else if (l > nm) modify(n->rr, l, r, v);
        else {
            modify(n->ll, l, nm, v);
            modify(n->rr, nm + 1, r, v);
        }
        pushup(n);
    }

    bool query(node* n, int l, int r) {
        if (n->l == l && n->r == r) {
            return n->v;
        }
        int nm = n->m();
        if (n->ll == nullptr) n->ll = new node(n->l, nm, n->v);
        if (n->rr == nullptr) n->rr = new node(nm + 1, n->r, n->v);
        pushdown(n);
        if (r <= nm) return query(n->ll, l, r);
        else if (l > nm) return query(n->rr, l, r);
        else return query(n->ll, l, nm) && query(n->rr, nm + 1, r);
    }

    node* root = new node(0, 1e9, 0);

    RangeModule() {}

    void addRange(int left, int right) {
        modify(root, left, right - 1, 1);
    }

    bool queryRange(int left, int right) {
        return query(root, left, right - 1);
    }

    void removeRange(int left, int right) {
        modify(root, left, right - 1, 0);
    }
};
```

（同题也可用 `set<pair<int,int>>` 维护有序区间集合，见下方 SummaryRanges 一题——"在线区间问题"不一定都要线段树。）

## 变体

### 归并树：区间第 K 小（K-th Number, POJ 2104）

每个节点存**有序的区间序列**（类似归并排序），查询时二分答案 + 统计"≤ v 的元素个数"：

```c++
const int maxn = 100005;
int N, M;

int arr[maxn];

struct node {
    int l, r;
    vector<int> v;
} t[maxn << 2];

void merge(vector<int>& a, vector<int>& b, vector<int>& c) {
    int la = a.size();
    int lb = b.size();
    int i = 0, j = 0;
    while (i < la && j < lb) {
        if (a[i] <= b[j]) c.push_back(a[i++]);
        else c.push_back(b[j++]);
    }
    while (i < la) c.push_back(a[i++]);
    while (j < lb) c.push_back(b[j++]);
}

void build(int rt, int l, int r) {
    t[rt].l = l;
    t[rt].r = r;
    if (l == r) {
        t[rt].v.push_back(arr[l]);
        return;
    }
    build(2 * rt + 1, l, (l + r) / 2);
    build(2 * rt + 2, (l + r) / 2 + 1, r);
    merge(t[2 * rt + 1].v, t[2 * rt + 2].v, t[rt].v);
}

int lt(int rt, int l, int r, int v) {  // count of elements <= v in [l, r]
    if (t[rt].l == l && t[rt].r == r) {
        int num = upper_bound(t[rt].v.begin(), t[rt].v.end(), v) - t[rt].v.begin();
        return num;
    }
    int mid = (t[rt].l + t[rt].r) / 2;
    if (r <= mid) return lt(2 * rt + 1, l, r, v);
    else if (l > mid) return lt(2 * rt + 2, l, r, v);
    else return lt(2 * rt + 1, l, mid, v) + lt(2 * rt + 2, mid + 1, r, v);
}

int query(int l, int r, int K) {
    // binary search the answer
    int L = -1e9 - 1, R = 1e9 + 1;  // max+1, min-1
    while (L + 1 < R) {
        int M = (L + R) / 2;
        int n = lt(0, l, r, M);
        if (n < K) L = M;
        else R = M;
    }
    return R;
}
```

### 扫描线：矩形面积并（Atlantis, POJ 1151）

把矩形按 x 排序拆成竖线，线段树维护 y 方向被覆盖的长度。注意叶子区间是 `[l, l+1]`（区间长度而非点），`cover` 计数而非 lazy：

```c++
const int maxn = 105;

struct node {
    int l, r;
    double len;
    int cover;
} T[maxn << 3];

struct line {
    double x, y1, y2;
    int flag;  // is_left
    bool operator< (const line& b) const { return x < b.x; }
} L[maxn << 1];

double ys[maxn << 1];

void build(int rt, int l, int r) {
    T[rt].l = l;
    T[rt].r = r;
    if (l + 1 == r) return;
    build(2 * rt + 1, l, (l + r) / 2);
    build(2 * rt + 2, (l + r) / 2, r);
}

void getlen(int rt) {
    if (T[rt].cover) T[rt].len = ys[T[rt].r] - ys[T[rt].l];
    else if (T[rt].l + 1 == T[rt].r) T[rt].len = 0;  // required!
    else T[rt].len = T[2 * rt + 1].len + T[2 * rt + 2].len;
}

void update(int rt, line& ln) {
    if (ys[T[rt].l] >= ln.y1 && ys[T[rt].r] <= ln.y2) {  // node's region is covered by line
        T[rt].cover += ln.flag;  // left +1, right -1
        getlen(rt);
        return;
    }
    if (T[rt].l + 1 == T[rt].r) return;
    int mid = (T[rt].r + T[rt].l) / 2;
    if (ln.y1 <= ys[mid]) update(2 * rt + 1, ln);
    if (ln.y2 > ys[mid]) update(2 * rt + 2, ln);
    getlen(rt);
}

double solve(int N) {
    int cnt = 0, ycnt = 0;
    double X1, Y1, X2, Y2;
    for (int i = 0; i < N; i++) {
        scanf("%lf%lf%lf%lf", &X1, &Y1, &X2, &Y2);
        L[cnt].x = X1; L[cnt].y1 = Y1; L[cnt].y2 = Y2; L[cnt++].flag = 1;
        L[cnt].x = X2; L[cnt].y1 = Y1; L[cnt].y2 = Y2; L[cnt++].flag = -1;
        ys[ycnt++] = Y1;
        ys[ycnt++] = Y2;
    }
    sort(L, L + cnt);
    sort(ys, ys + ycnt);
    ycnt = unique(ys, ys + ycnt) - ys;
    build(0, 0, ycnt - 1);
    double ans = 0;
    update(0, L[0]);
    for (int i = 1; i < 2 * N; i++) {
        ans += (L[i].x - L[i - 1].x) * T[0].len;
        update(0, L[i]);
    }
    return ans;
}
```

> 周长并（Picture, POJ 1177）思路类似，但还需维护"区间内覆盖线段数 count"与"左右端点是否被覆盖"，且相同 x 的边要**入边在前**。这里不再展开。

### 二维线段树（Matrix, HDU 1823）

外层 x 线段树每个节点再挂一棵内层 y 线段树：

```c++
int tree[maxn << 2][maxn << 2];
// UpDateY(xnow, ...) 在 xnow 的内层树上做 y 区间更新
// UpDateX(now, ...) 先定位外层区间，再调用 UpDateY
// QueryX / QueryY 类似，查询时把路径上所有内层节点累加/异或
```

## 例题

### [699 掉落的方块](https://leetcode-cn.com/problems/falling-squares/)

离散化 + "区间替换（取 max）"线段树：

```c++
class Solution {
public:
    const static int maxn = 10005;

    struct node {
        int l, r;
        int mx, lazy;
        int m() { return (l + r) / 2; }
    } seg[4 * maxn];

    void build(int rt, int l, int r) { /* 同"区间替换最值"模板 */ }

    void pushdown(int rt) { /* 同模板 */ }

    int query(int rt, int l, int r) { /* 同模板 */ }

    void modify(int rt, int l, int r, int v) { /* 同模板 */ }

    // 俄罗斯方块堆叠式的更新：先查当前高度再替换
    void blockmodify(int rt, int l, int r, int v) {
        modify(rt, l, r, v + query(rt, l, r));
    }

    vector<int> fallingSquares(vector<pair<int, int>>& positions) {
        // 离散化
        vector<int> xs;
        for (int i = 0; i < positions.size(); i++) {
            xs.push_back(positions[i].first);
            xs.push_back(positions[i].first + positions[i].second - 1);
        }
        sort(xs.begin(), xs.end());
        int uN = unique(xs.begin(), xs.end()) - xs.begin();
        map<int, int> m;
        for (int i = 0; i < uN; i++) m[xs[i]] = i;
        // 线段树
        build(0, 0, uN);
        vector<int> ans;
        for (int i = 0; i < positions.size(); i++) {
            blockmodify(0,
                        m[positions[i].first],
                        m[positions[i].first + positions[i].second - 1],
                        positions[i].second
            );
            ans.push_back(query(0, 0, uN));
        }
        return ans;
    }
};
```

### SummaryRanges（LeetCode 352，有序结构替代线段树）

在线区间合并：用有序 `map<int,int>` 按起点存区间，插入时检查前后邻居五种情况。不需要线段树：

```cpp
class SummaryRanges {
private:
    map<int, int> intervals;

public:
    SummaryRanges() {}

    void addNum(int val) {
        // l1 最小的且满足 l1 > val 的区间 interval1；interval0 是它前一个区间
        auto interval1 = intervals.upper_bound(val);
        auto interval0 = (interval1 == intervals.begin() ? intervals.end() : prev(interval1));

        if (interval0 != intervals.end() && interval0->first <= val && val <= interval0->second) {
            // 情况一：已在某个区间内
            return;
        }
        else {
            bool left_aside = (interval0 != intervals.end() && interval0->second + 1 == val);
            bool right_aside = (interval1 != intervals.end() && interval1->first - 1 == val);
            if (left_aside && right_aside) {
                // 情况四：两边都相邻，合并
                int left = interval0->first, right = interval1->second;
                intervals.erase(interval0);
                intervals.erase(interval1);
                intervals.emplace(left, right);
            }
            else if (left_aside) {
                // 情况二：只与左区间相邻
                ++interval0->second;
            }
            else if (right_aside) {
                // 情况三：只与右区间相邻
                int right = interval1->second;
                intervals.erase(interval1);
                intervals.emplace(val, right);
            }
            else {
                // 情况五：独立成段
                intervals.emplace(val, val);
            }
        }
    }

    vector<vector<int>> getIntervals() {
        vector<vector<int>> ans;
        for (const auto& [left, right]: intervals) {
            ans.push_back({left, right});
        }
        return ans;
    }
};
```

## 何时用更简单的结构

| 数组修改 | 查询 | 最佳方案（前为最简单） |
| --- | --- | --- |
| 静态 | 区间求和 | **前缀和**，树状数组，线段树 |
| 单点增量 | 区间求和 | **树状数组**，线段树 |
| 区间增量 | 单点 | **差分**，线段树 |
| 区间增量 | 区间求和 | 线段树 |
| 静态 | 区间最值 | ST 表，线段树 |
| 单点替换 | 区间最值 | 线段树，树状数组 |

线段树最通用（区间改 + 区间查），但更限定的场景用前缀和/差分/树状数组更简单（见 `04_prefix_sum`、`17_binary_indexed_tree`）。
