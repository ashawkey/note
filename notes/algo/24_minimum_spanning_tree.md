# 最小生成树 (Minimum Spanning Tree)

## 核心思想

连通无向图所有生成树中，边权和最小的一棵。两个经典算法都是贪心：

- **Kruskal**：按边权从小到大，用**并查集**维护连通性，能加入就加入（不形成环）。$O(E\log E)$，适合稀疏图。
- **Prim**：从一个点出发，每次选"连接已选集合与未选集合"的最小边，用**优先队列**维护。$O((V+E)\log V)$，适合稠密图。

正确性依据（切割性质）：任意割的最小横跨边必在某个 MST 中——Kruskal 每次加入的是"当前两个连通块之间的最小边"，Prim 每次加入的是"已选/未选集合之间的最小边"，都是切割性质的应用。

## 模板

### Kruskal（并查集）

```c++
#include <iostream>
#include <algorithm>
#include <vector>
using namespace std;

const int maxn = 10005;

struct edge {
    int u, v, w;
    bool operator< (const edge& b) const { return w < b.w; }
};

vector<edge> edges;
int parent[maxn];

int getRoot(int a) {
    if (parent[a] != a) parent[a] = getRoot(parent[a]);
    return parent[a];
}

// 返回最小生成树边权和；图不连通返回 -1
int kruskal(int N) {
    for (int i = 0; i < N; i++) parent[i] = i;
    sort(edges.begin(), edges.end());
    int ans = 0, cnt = 0;
    for (auto& e : edges) {
        int fu = getRoot(e.u), fv = getRoot(e.v);
        if (fu != fv) {
            parent[fv] = fu;
            ans += e.w;
            if (++cnt == N - 1) return ans;
        }
    }
    return -1; // not connected
}
```

### Prim（优先队列）

```c++
#include <iostream>
#include <queue>
#include <vector>
#include <cstring>
using namespace std;

const int maxn = 10005;
const int inf = 0x3f3f3f3f;

vector<pair<int, int>> G[maxn]; // (v, w)
int dist[maxn];
bool vis[maxn];

// 返回最小生成树边权和；图不连通返回 -1
int prim(int s, int N) {
    memset(vis, 0, sizeof(vis));
    memset(dist, 0x3f, sizeof(dist));
    dist[s] = 0;
    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> q;
    q.emplace(0, s);
    int ans = 0, cnt = 0;
    while (!q.empty()) {
        auto [d, u] = q.top(); q.pop();
        if (vis[u]) continue;
        vis[u] = true;
        ans += d;
        if (++cnt == N) return ans;
        for (auto [v, w] : G[u]) {
            if (!vis[v] && w < dist[v]) {
                dist[v] = w;
                q.emplace(w, v);
            }
        }
    }
    return -1; // not connected
}
```

> Prim 与 Dijkstra 模板几乎一样，区别只在：Dijkstra 用 `dist[v] = dist[u] + w`（累加），Prim 用 `dist[v] = w`（取边权最小）。

## 例题

### [1584 连接所有点的最小费用](https://leetcode-cn.com/problems/min-cost-to-connect-all-points/)

把每个点之间建"曼哈顿距离"边，跑 Kruskal：

```cpp
class Solution {
public:
    struct edge {
        int u, v, w;
        bool operator< (const edge& b) const { return w < b.w; }
    };

    int find(vector<int>& p, int x) {
        return p[x] == x ? x : p[x] = find(p, p[x]);
    }

    int minCostConnectPoints(vector<vector<int>>& points) {
        int n = points.size();
        vector<edge> es;
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++)
                es.push_back({i, j, abs(points[i][0]-points[j][0]) + abs(points[i][1]-points[j][1])});
        sort(es.begin(), es.end());
        vector<int> p(n);
        for (int i = 0; i < n; i++) p[i] = i;
        int ans = 0, cnt = 0;
        for (auto& e : es) {
            int fu = find(p, e.u), fv = find(p, e.v);
            if (fu != fv) {
                p[fv] = fu;
                ans += e.w;
                if (++cnt == n - 1) break;
            }
        }
        return ans;
    }
};
```

## 有向图最小树形图（朱刘算法）

求以某点为根、能到达所有点的**最小有向生成树**（arborescence）。算法核心：每轮给每个非根点选一条**最小入边**，若形成环则把环缩成一个点、环内非树边权值减去其终点的最小入边权，重复直到无环：

```c++
// 返回根为 s 的最小树形图权值；不可达返回 -1
double zhuliu(int s) {
    double res = 0;
    int v = V; // localize
    while (true) {
        // in[i] = 指向 i 的最小入边权，pre[i] 是其起点
        for (int i = 0; i < v; i++) in[i] = inf;
        for (int i = 0; i < edges.size(); i++) {
            edge& e = edges[i];
            if (e.f != e.t && e.w < in[e.t]) {
                pre[e.t] = e.f;
                in[e.t] = e.w;
            }
        }
        // check non-connectivity
        for (int i = 0; i < v; i++)
            if (s != i && in[i] == inf) return -1;
        // id[] is scc id
        memset(id, -1, sizeof(id));
        memset(vis, -1, sizeof(vis));
        in[s] = 0;
        int scc = 0;
        for (int i = 0; i < v; i++) {
            res += in[i];
            int v = i;
            // find cycle
            while (vis[v] != i && id[v] == -1 && v != s) {
                vis[v] = i;
                v = pre[v];
            }
            if (v != s && id[v] == -1) {  // assign cycle id
                for (int u = pre[v]; u != v; u = pre[u]) id[u] = scc;
                id[v] = scc++;
            }
        }
        if (scc == 0) break; // no cycle, MST built
        for (int i = 0; i < v; i++)
            if (id[i] == -1) id[i] = scc++;
        // shrink cycles
        for (int i = 0; i < edges.size(); i++) {
            edge& e = edges[i];
            int v = e.t;
            e.f = id[e.f];
            e.t = id[e.t];
            if (e.f != e.t) e.w -= in[v]; // 缩环后修正边权
        }
        v = scc;
        s = id[s];
    }
    return res;
}
```

（出处：POJ 地震之后——给定平面点集与有向距离边，求 0 号点的最小树形图。）

## 相关

- 并查集模板见 `14_union_find`。
- 次小生成树（严格/非严格）：先求 MST，再枚举非树边替换环上的最大边（可用 LCA 或树剖维护最大值）。
