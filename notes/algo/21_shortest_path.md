# 最短路 (Shortest Path)

## 算法总览

| 算法 | 适用 | 思想 | 时间复杂度 | 备注 |
| --- | --- | --- | --- | --- |
| **Dijkstra** | 不含负权边的有向/无向图，单源 | 贪心 | $O((E+V)\log V)$（堆） | 最常用 |
| **Bellman-Ford** | 含负权边的**有向**图，单源 | 动态规划 | $O(EV)$ | 可判负环 |
| **SPFA** | 同上 | 队列优化的 BF | 期望 $O(E)$，最坏 $O(EV)$ | 可判负环 |
| **Floyd** | 任意图，**全源** | 动态规划 | $O(V^3)$ | 支持负权，不能有负环 |

> 各算法的关系与选择（BFS 何时可以当最短路用）见 [22 BFS 与最短路算法的区别](bfs_vs_shortest_path/)。

**Dijkstra 为何不能处理负权边**：Dijkstra 是贪心算法，一旦从堆中弹出某个节点就认为其距离确定；负权边可能让"先绕远路再回来"更短，导致贪心失败：

```
a---3---b
|       |
4___c__-2
```

**Bellman-Ford 的时间复杂度分析**（Dijkstra 同理）：

$$
T(n) = O(E) \cdot dk_Q + O(V) \cdot em_Q
$$

- $dk_Q$（Decrease Key）：整个算法中**每条边最多触发一次**；
- $em_Q$（Extract Min）：**每个顶点取一次**下一个最短边。

|      | min heap            | brute            |
| ---- | ------------------- | ---------------- |
| dk   | logV (insert in heap) | 1 (no sorting) |
| em   | logV (pop heap)     | V^2 (one by one) |

最小堆 Dijkstra 为 $O((E+V)\log V) \approx O(E\log V)$；若考虑堆中保留旧边，操作最坏为 $O(\log E)$，即 $O((E+V)\log E)$。

## 模板

图存储（边表 + 邻接表）：

```c++
const int maxv = 30005;
const int maxe = 150005;
const int maxw = 0x7fffffff;
int N, M;

struct edge {
    int f, t, w;
    edge() {}
    edge(int u, int v, int w) :f(u), t(v), w(w) {}
};

vector<edge> edges;
vector<int> G[maxv];

// directed
void insert_edge(int a, int b, int c) {
    edges.push_back(edge(a, b, c));
    int s = edges.size() - 1;
    G[a].push_back(s);
}

int dist[maxv], vis[maxv];

bool cmp(int a, int b) {
    return dist[a] > dist[b];
}
```

### Dijkstra（堆优化）

```c++
void dijkstra(int s) {
    for (int i = 1; i <= N; i++) {
        dist[i] = maxw;
        vis[i] = 0;
    }
    dist[s] = 0;
    priority_queue<int, vector<int>, bool(*)(int, int)> que(cmp);
    que.push(s);
    while (!que.empty()) {
        int u = que.top(); que.pop();
        if (vis[u]) continue;
        vis[u] = 1;
        for (int i = 0; i < G[u].size(); i++) {
            int e = G[u][i];
            int v = edges[e].t;
            if (vis[v]) continue;
            if (dist[v] > dist[u] + edges[e].w) {
                dist[v] = dist[u] + edges[e].w;
                que.push(v);
            }
        }
    }
}
```

### Bellman-Ford（判负环）

```c++
bool bellmanford(int s) {
    for (int i = 1; i <= N; i++) dist[i] = maxw;
    dist[s] = 0;
    // run (N - 1) times
    for (int k = 1; k < N; k++) {
        // loop all edges
        for (int i = 0; i < edges.size(); i++) {
            int u = edges[i].f;
            int v = edges[i].t;
            if (dist[v] > dist[u] + edges[i].w)
                dist[v] = dist[u] + edges[i].w;
        }
    }
    // run 1 more time to detect negative loop.
    for (int i = 0; i < edges.size(); i++) {
        int u = edges[i].f;
        int v = edges[i].t;
        if (dist[v] > dist[u] + edges[i].w)
            return true; // if the dist can still be updated, there must be a negative loop.
    }
    return false;
}
```

> 最长路 + 正环判断同理：把 `>` 换成 `<`、`maxw` 换成 `0` 即可。且**判断正环不需要传播回源点**——BF 只额外传播一轮，源点可能没被更新到，但只要存在正环，多轮之后一定能传回源点。

### SPFA（判负环）

```c++
bool spfa(int s) {
    for (int i = 1; i <= N; i++) dist[i] = maxw, vis[i] = 0; // vis is used as cnt
    dist[s] = 0;
    queue<int> q; // stack is also OK.
    q.push(s);
    while (!q.empty()) {
        int p = q.front(); q.pop();
        for (int i = 0; i < G[p].size(); i++) {
            edge& e = edges[G[p][i]];
            if (dist[e.t] > dist[e.f] + e.w) {
                dist[e.t] = dist[e.f] + e.w;
                q.push(e.t);
                if (vis[e.t]++ >= N) return true; // if a point is enqueued for N times, there must be a negative loop.
            }
        }
    }
    return false;
}
```

### Floyd 与路径找回

Floyd 是 $O(V^3)$ 的全源最短路（DP）。要**输出具体路径**，需要记录 `parent[i][j]`（i→j 路径上 j 的前驱边）：

```c++
int dist[maxv][maxv];
int parent[maxv][maxv]; // 存边的编号（或前驱节点）

void floyd() {
    memset(dist, 0x3f, sizeof(dist));
    memset(parent, -1, sizeof(parent));
    for (int i = 1; i <= V; i++) {
        dist[i][i] = 0;
        for (int j = 0; j < G[i].size(); j++) {
            int e = G[i][j];
            int v = edges[e].t;
            dist[i][v] = edges[e].w;
            parent[i][v] = e;
        }
    }
    for (int u = 1; u <= V; u++) {
        for (int i = 1; i <= V; i++) {
            for (int j = 1; j <= V; j++) {
                if (dist[i][j] > dist[i][u] + dist[u][j]) {
                    dist[i][j] = dist[i][u] + dist[u][j];
                    parent[i][j] = parent[u][j];  // 更新前驱
                }
            }
        }
    }
}

// 从 v 沿 parent 一路回退到 u 即得路径（逆序）
void print_path(int u, int v) {
    cout << v;
    while (u != v) {
        int e = parent[u][v];
        v = edges[e].f;
        cout << "<-" << v;
    }
}
```

（出处：POJ うさぎと桜——无向图全源最短路 + 多组路径输出。）

## 例题

### [Currency Exchange](http://poj.org/problem?id=1860)（POJ 1860，最长路 + 正环）

源点为持有货币 V，其他点初始化为 0，按兑换规则算最长路；判断正环：

```c++
#include <iostream>
#include <cstring>
#include <algorithm>
#include <vector>
#include <queue>
#include <stack>
#include <set>
using namespace std;

const int maxv = 105;
const int maxe = 1005;

struct edge {
    int f, t;
    double w, c;
    edge() {}
    edge(int u, int v, double w, double c) :f(u), t(v), w(w), c(c) {}
};

vector<edge> edges;
vector<int> G[maxv];

void insert_edge(int a, int b, double w, double c) {
    edges.push_back(edge(a, b, w, c));
    int s = edges.size() - 1;
    G[a].push_back(s);
}

int N, M, S;
double V;

double dist[maxv];

bool bellmanford(int s, double v) {
    memset(dist, 0, sizeof(dist));
    dist[s] = v;
    for (int k = 1; k < N; k++) {
        bool flag = false;
        for (int u = 1; u <= N; u++) {
            for (int i = 0; i < G[u].size(); i++) {
                int e = G[u][i];
                int v = edges[e].t;
                if (dist[v] < (dist[u] - edges[e].c) * edges[e].w) {
                    dist[v] = (dist[u] - edges[e].c) * edges[e].w;
                    flag = true;
                }
            }
        }
        if (!flag) break; // optim
    }
    for (int u = 1; u <= N; u++) {
        for (int i = 0; i < G[u].size(); i++) {
            int e = G[u][i];
            int v = edges[e].t;
            if (dist[v] < (dist[u] - edges[e].c) * edges[e].w) return true;
        }
    }
    return false;
}

int main() {
    cin >> N >> M >> S >> V;
    int a, b;
    double c, d, e, f;
    for (int i = 0; i < M; i++) {
        cin >> a >> b >> c >> d >> e >> f;
        insert_edge(a, b, c, d);
        insert_edge(b, a, e, f);
    }
    cout << (bellmanford(S, V) ? "YES" : "NO") << endl;
}
```

### [Wormholes](http://poj.org/problem?id=3259)（POJ 3259，SPFA 判负环）

普通双向边 + 负权单向虫洞，SPFA 判断负环：

```c++
#include <iostream>
#include <cstdio>
#include <cstring>
#include <string>
#include <queue>
#include <stack>
#include <vector>
#include <algorithm>
#define P pair<int,int>

using namespace std;

const int maxn = 505;
const int inf = 0x3f3f3f3f;
int N, M, W;

struct edge {
    int f, t, w;
    edge() {}
    edge(int f, int t, int w) :f(f), t(t), w(w) {}
};

vector<edge> G[maxn];

int dist[maxn], times[maxn];

bool spfa(int s) {
    fill(dist, dist + N + 1, inf);
    fill(times, times + N + 1, 0);
    dist[s] = 0;
    queue<int> q; // stack is also OK.
    q.push(s);
    while (!q.empty()) {
        int p = q.front(); q.pop();
        for (int i = 0; i < G[p].size(); i++) {
            edge& e = G[p][i];
            if (dist[e.t] > dist[e.f] + e.w) {
                dist[e.t] = dist[e.f] + e.w;
                q.push(e.t);
                if (times[e.t]++ >= N) return true;
            }
        }
    }
    return false;
}

int main() {
    int T, x, y, w;
    scanf("%d", &T);
    while (T--) {
        scanf("%d%d%d", &N, &M, &W);
        for (int i = 1; i <= N; i++) G[i].clear();
        for (int i = 0; i < M; i++) {
            scanf("%d%d%d", &x, &y, &w);
            G[x].push_back(edge(x, y, w));
            G[y].push_back(edge(y, x, w));
        }
        for (int i = 0; i < W; i++) {
            scanf("%d%d%d", &x, &y, &w);
            G[x].push_back(edge(x, y, -w));
        }
        cout << (spfa(1) ? "YES" : "NO") << endl;
    }
}
```
