# 并查集 (Union-Find / Disjoint-Set)

## 核心思想

维护"元素之间的集合归属"：

- `merge(A, B)`：合并 A、B 所在集合；
- `query(A, B)`：查询 A、B 是否同集合。

**关键理解**："在同一集合（同一根）"只意味着**它们之间存在某种关系**，不意味着它们属于同一个分组——带权并查集正是靠"权"编码这种关系（距离、相对大小、吃与被吃等）。

## 模板

### 朴素 + 路径压缩

```c++
int parent[N];

void init(int n) {
    for (int i = 0; i < n; i++) {
        parent[i] = i;
    }
}

int getRoot(int a) {
    if (parent[a] != a)
        parent[a] = getRoot(parent[a]);
    return parent[a];
}

void merge(int a, int b) {
    parent[getRoot(b)] = getRoot(a);
}

bool query(int a, int b) {
    return getRoot(a) == getRoot(b);
}
```

路径压缩：**查询时把路径上每个节点直接指向根**（单次可能 $O(n)$，平摊后接近常数，$N$ 不大时不超过 $4N$）。

### 带权并查集

权重表示**该节点与根节点**之间的关系（注意：不是与父节点的关系），压缩时把父链上的权累加：

```c++
int p[N];
int w[N];

void init(int n) {
    for (int i = 0; i < n; i++) p[i] = i, w[i] = 0;
}

int par(int x) {
    if (x == p[x]) return x;
    int fx = p[x];
    p[x] = par(fx);
    w[x] = (w[x] + w[fx]) % 3; // iteratively compress w[]
    return p[x];
}

void merge(int x, int y) {
    int fx = par(x), fy = par(y);
    p[fy] = fx;  // always only change parents
    w[fy] = (w[x] - w[y] + 3) % 3;
}
```

## 例题

### [POJ 1182 食物链](https://www.luogu.com.cn/problem/P2024)（带权并查集）

对象之间的关系通常构成一个**环**，用向量推导 merge 的公式。三类关系（同类/吃/被吃）模 3 编码：

```bash
fx <- ??? - fy
^           ^|
|w[x]   w[y]||-w[y]     ===> ??? = (w[x] - w[y] + d - 1 + 3) % 3
|           |v                          // +3 for safety
x  <-(d-1)-- y          // d=1 means the same, d=2 means x eats y so w'[y] is 1.
```

```c++
#include <iostream>
#include <cstring>
using namespace std;

const int maxN = 50005;

int par[maxN];
int dis[maxN];

void init(int n) {
    for (int i = 0; i <= n; i++)
        par[i] = i;
}

int parent(int x) {
    if (par[x] == x) return par[x];
    int f = par[x];
    par[x] = parent(f);
    dis[x] = (dis[x] + dis[f]) % 3;
    return par[x];
}

void merge(int d, int x, int y) {
    int fx = parent(x);
    int fy = parent(y);
    par[fy] = fx;
    dis[fy] = (dis[x] - dis[y] + d + 2) % 3;
}

bool query(int x, int y) {
    return parent(x) == parent(y);
}

int main() {
    int N, K, D, X, Y;
    int res = 0;
    memset(dis, 0, sizeof(dis));
    cin >> N >> K;
    init(N);
    for (int i = 0; i < K; i++) {
        cin >> D >> X >> Y;
        if (X > N || Y > N) {
            res++;
            continue;
        }
        if (!query(X, Y)) {
            merge(D, X, Y);
        }
        else if (D == 1) {
            if (dis[X] != dis[Y]) {
                res++;
            }
        }
        else if (D == 2) {
            if (dis[Y] != (dis[X] + 1) % 3) {
                res++;
            }
        }
    }
    cout << res << endl;
}
```

### [POJ 2492 A Bug's Life](https://www.luogu.com.cn/problem/SP8093)（模 2 带权）

与食物链同理，但模 2（同组/异组），"发现矛盾"即存在同性恋配对：

```c++
#include <iostream>
#include <cstring>
using namespace std;

const int maxN = 2005;

int p[maxN];
int w[maxN];

void init(int n) {
    for (int i = 1; i <= n; i++) {
        p[i] = i;
        w[i] = 0;
    }
}

int parent(int x) {
    if (p[x] == x) return p[x];
    int f = p[x];
    p[x] = parent(f);
    w[x] = (w[x] + w[f]) % 2;
    return p[x];
}

void merge(int x, int y) {
    int fx = parent(x);
    int fy = parent(y);
    p[fy] = fx;
    w[fy] = (w[x] - w[y] + 1) % 2;
}

bool query(int x, int y) {
    return parent(x) == parent(y);
}

int main() {
    int S, N, I, a, b;
    bool flag = false;
    cin >> S;
    for (int s = 1; s <= S; s++) {
        cin >> N >> I;
        init(N);
        flag = false;
        if (s != 1) cout << endl;
        for (int i = 0; i < I; i++) {
            cin >> a >> b;
            if (flag) continue;
            // in the same tree, maybe suspicious
            if (query(a, b)) {
                if (w[a] == w[b]) {
                    flag = true;
                }
            }
            else merge(a, b);
        }
        cout << "Scenario #" << s << ":" << endl;
        if (flag) cout << "Suspicious bugs found!" << endl;
        else cout << "No suspicious bugs found!" << endl;
    }
}
```

### [POJ 1988 Cube Stacking](https://www.luogu.com.cn/problem/UVA12232)（维护子树大小与"下方方块数"）

`sum[N]` 记栈大小、`under[N]` 记该方块下方方块数。**只修改（原）根节点的 under**，查询时沿父链累加并压缩：

```c++
#include <iostream>
using namespace std;

const int maxn = 30005;

int N, M;
int p[maxn], sum[maxn], under[maxn];

void init(int N) {
    for (int i = 1; i <= N; i++) {
        p[i] = i;
        sum[i] = 1;
        under[i] = 0;
    }
}

int par(int x) {
    if (p[x] == x) return x;
    int f = p[x];
    p[x] = par(f);
    under[x] += under[f];
    return p[x];
}

void merge(int x, int y) {  // put stack y on top of stack x
    int fx = par(x);
    int fy = par(y);
    p[fy] = fx;
    under[fy] = sum[fx];
    sum[fx] += sum[fy];
}

char c;
int x, y;

int main() {
    init(maxn);
    cin >> M;
    for (int i = 0; i < M; i++) {
        cin >> c;
        if (c == 'C') {
            cin >> x;
            par(x);
            cout << under[x] << endl;
        }
        else {
            cin >> x >> y;
            merge(y, x);
        }
    }
}
```

### [POJ 2528 Mayor's Posters](https://www.luogu.com.cn/problem/UVA10587)（把并查集当链表）

奇妙用法：**不 merge**，只把 `par[j]` 指向 `j+1` 模拟"跳过已被占据的位置"，从而 $O(1)$ 描述区间性质（是否被占据），功能相当于最简化的线段树。倒序贴海报（后贴的优先级高）：

```c++
#include <iostream>
#include <cstdio>
#include <algorithm>
#include <vector>
using namespace std;

const int maxn = 20005; // 2*maxPosters, start+end
int N;

int par[maxn];

void init(int n) {
    for (int i = 0; i <= n; i++) par[i] = i;
}
int findpar(int n) {
    if (n == par[n]) return n;
    par[n] = findpar(par[n]);
    return par[n];
}

// discretization
vector<int> order;
vector<pair<int, int>> posters;

int getindex(vector<int>& order, int key) {
    return lower_bound(order.begin(), order.end(), key) - order.begin();
}

int main() {
    int cas, x, y;
    cin >> cas;
    while (cas--) {
        cin >> N;
        order.clear();
        posters.clear();
        for (int i = 0; i < N; i++) {
            cin >> x >> y;
            order.push_back(x);
            order.push_back(y);
            posters.push_back(pair<int, int>(x, y));
        }
        sort(order.begin(), order.end());
        order.erase(unique(order.begin(), order.end()), order.end());
        for (int i = 0; i < N; i++) {
            posters[i].first = getindex(order, posters[i].first);
            posters[i].second = getindex(order, posters[i].second);
        }
        init(order.size());
        int res = 0;
        for (int i = N - 1; i >= 0; i--) {
            bool flag = false;
            for (int j = posters[i].first; j <= posters[i].second; j = findpar(j)) {
                if (par[j] == j) {
                    flag = true;
                    par[j] = j + 1;
                    j++;
                }
            }
            if (flag) res++;
        }
        cout << res << endl;
    }
}
```
