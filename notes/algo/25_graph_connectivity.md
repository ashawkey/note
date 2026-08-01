# 图的连通性 (Graph Connectivity)

## 有向图：强连通分量（SCC）

- **强连通**：$v_i$ 与 $v_j$ 之间存在 $v_i \rightarrow v_j$ 与 $v_j \rightarrow v_i$ 两条路径。
- **强连通分量**：极大强连通子图。
- **缩点**：把每个 SCC 缩成一个点，任意有向图变为 DAG——这是连通性问题最常用的转化。

### Tarjan 模板

- `dfn[u]`：u 的 DFS 序；
- `low[u]`：从 u 出发能访问到的最早 dfn；
- `dfn[u] == low[u]`：u 是一个 SCC 的根（DFS 序中第一个被访问的点）。

```
Tarjan(int u):
    low[u] = dfn[u] = ++index;
    stack.push(u);
    for each (u, v) in E:
        if not visited v:
            Tarjan(v);
            low[u] = min(low[u], low[v]);
        else if v in stack:
            low[u] = min(low[u], dfn[v]);
    if dfn[u] == low[u]:
        repeat: v = stack.pop(); dye v; until u == v
```

实现（含染色与出栈标记）：

```c++
const int maxn = 10005;
vector<int> G[maxn];
int dfn[maxn], low[maxn], color[maxn], vis[maxn];  // vis: 0 未访问, 1 在栈中, 2 已出栈
int N, M;
int ncolor = 0, idx = 0;
stack<int> stk;

void tarjan(int u) {
    dfn[u] = low[u] = ++idx;
    vis[u] = 1;
    stk.push(u);
    for (int v : G[u]) {
        if (!vis[v]) {
            tarjan(v);
            low[u] = min(low[u], low[v]);
        }
        else if (vis[v] == 1) {
            low[u] = min(low[u], dfn[v]);
        }
    }
    if (dfn[u] == low[u]) {
        int v;
        do {
            v = stk.top(); stk.pop();
            color[v] = ncolor;
            vis[v] = 2;
        } while (u != v);
        ncolor++;
    }
}
```

（Kosaraju 也能求 SCC 但比 Tarjan 慢，一般用 Tarjan。）

### 例题

#### Popular Cows（POJ 2186）

缩点成 DAG 后，**唯一出度为零的 SCC** 中的点就是能被所有点访问到的点：

```c++
int solve() {
    // 统计每个 SCC 的出度（跨 SCC 的边）
    for (int i = 1; i <= N; i++) {
        for (int j : G[i])
            if (color[j] != color[i]) degree[color[i]]++;
    }
    int end = -1;
    for (int i = 0; i < ncolor; i++) {
        if (degree[i] == 0) {
            if (end == -1) end = i;
            else return 0; // 不止一个出度 0 的 SCC
        }
    }
    int ans = 0;
    for (int i = 1; i <= N; i++)
        if (color[i] == end) ans++;
    return ans;
}
```

#### Network of Schools（POJ 1236）

- 至少选几个点才能遍历全图？**入度为 0 的 SCC 个数**。
- 至少加几条边才能强连通？`max(入度为 0 的 SCC 数, 出度为 0 的 SCC 数)`。

#### Going from u to v or v to u?（POJ 2762）

半连通（任意两点至少一方可达另一方）。缩点后，**半连通 DAG 一定是一条链**——检查每个 SCC 的出度/入度都不超过 1：

```c++
bool solve() {
    for (int i = 1; i <= N; i++) {
        for (int j : G[i]) {
            if (color[j] != color[i]) {
                outd[color[i]]++;
                ind[color[j]]++;
            }
        }
    }
    for (int i = 0; i < ncolor; i++) {
        if (outd[i] > 1 || ind[i] > 1) return false;
    }
    return true;
}
```

## 无向图：割点与桥

删除该点/边后图不再连通。也是 Tarjan：

- **割点**：`dfn[u] <= low[v]`（v 的子树无法回到 u 之前）→ u 是割点；
- **桥**：`dfn[u] < low[v]` → 边 (u, v) 是桥；
- **根节点的特判**：根只有一个孩子子树时不是割点（`nrs >= 2` 才是）；
- **重边**：桥的判断需要额外检测重边（数一下 u→v 的边数，只有 1 条才算桥）。

```c++
const int maxv = 100;
const int maxe = 10000;

vector<int> G[maxv];
int dfn[maxv], low[maxv], vis[maxv], parent[maxv];
int idx, N, M, nrs;
int iscut[maxv], isbrd[maxe];

void tarjan(int u) {
    dfn[u] = low[u] = ++idx;
    vis[u] = 1;
    for (int v : G[u]) {
        if (!vis[v]) {
            parent[v] = u;
            if (u == 1) nrs++;
            tarjan(v);
            low[u] = min(low[u], low[v]);
            /* WHY cut vertex: dfn[u] <= low[v] means subtree v can't point back
               to vertices earlier than u, so removing u isolates subtree v. */
            if (dfn[u] <= low[v]) iscut[u] = 1;
            /* WHY bridge: if dfn[u] == low[v], edge is in a loop, deleting it
               can't separate the graph. */
            if (dfn[u] < low[v]) {
                // detect multiple edges.
                int cnt = 0;
                for (int j : G[u]) if (j == v) cnt++;
                if (cnt == 1) isbrd[G[u][i]] = 1;
            }
        }
        // avoid undirected reverse edge
        else if (parent[u] != v) {
            low[u] = min(low[u], dfn[v]);  // must be dfn[v]
        }
    }
    // special for root !!!
    if (u == 1 && nrs < 2) iscut[u] = 0;
}
```

### Caocao's Bridge（HDU 4738）

求边权最小的桥；图不连通输出 0，无桥输出 -1，权为 0 的桥至少要 1 个人去炸：

```c++
vector<pair<int, int>> brd;
void bridge() {
    brd.clear();
    for (int i = 1; i <= N; i++) {
        int v = parent[i];
        if (dfn[v] < low[i]) {
            int cnt = 0;
            for (int j : G[v]) if (j == i) cnt++; // 重边检测
            if (cnt == 1) brd.push_back(make_pair(v, i));
        }
    }
}
// main 中：tarjan(1) 后检查是否所有点都访问过（不连通 → 0）；
// 无桥 → -1；否则取 brd 中最小的边权，若为 0 则输出 1。
```

## 双连通分量

### 点双连通（BCC，块）

不包含割点的极大连通子图。**原图的割点可以属于多个点双连通分量**，其他点与边只属于一个。用**边栈**：DFS 时把边入栈，`dfn[u] <= low[v]` 时弹出一个块：

```c++
stack<int> stk;
void tarjan(int u) {
    dfn[u] = low[u] = ++idx;
    vis[u] = 1;
    for (int i = 0; i < G[u].size(); i++) {
        int v = edges[G[u][i]].t;
        if (!vis[v]) {
            stk.push(G[u][i]); // push new edge
            parent[v] = u;
            tarjan(v);
            low[u] = min(low[u], low[v]);
            if (dfn[u] <= low[v]) {  // u 是割点 → 弹出一个 BCC 块
                // 弹出直到边 (u, v) 为止的所有边，即一个块
            }
        }
        else if (parent[u] != v) {
            low[u] = min(low[u], dfn[v]);
            if (dfn[u] > dfn[v]) stk.push(G[u][i]); // 回边（连向祖先）
        }
    }
}
```

### 边双连通（不含桥的极大连通子图）

去掉所有桥即可。**low 数组本身就是边双连通的一个染色**，缩点甚至不用显式找桥。

#### Road Construction（POJ 3352）

缩点后是一棵树，把它变成边双连通需要添加的边数为 **⌈叶子数 / 2⌉ = ⌊(叶子数+1)/2⌋**：

```c++
int solve() {
    for (int u = 1; u <= N; u++) {
        for (int i = 0; i < G[u].size(); i++) {
            int v = edges[G[u][i]].t;
            if (low[u] != low[v]) deg[low[u]]++; // 跨边双连通块的边
        }
    }
    int cnt = 0;
    for (int c = 1; c <= N; c++)
        if (deg[c] == 1) cnt++;  // 叶子数
    return (cnt + 1) / 2;
}
```

#### SPF（UVA 101）

割点删掉后分裂出的连通块数 = 该割点各子树中**不同 low 值的个数**：

```c++
for (int u = 1; u <= N; u++) {
    if (!iscut[u]) continue;
    set<int> s;
    for (int v : G[u]) s.insert(low[v]);
    // s.size() > 1 时即分裂块数
}
```

## 相关

- 拓扑排序与 DAG 性质见 `23_topological_sort`。
