# 拓扑排序 (Topological Sort)

## 核心思想

对有向无环图（DAG）求一个线性顺序，使每条边 `u→v` 中 u 都在 v 之前。满足限制条件的线性顺序**不唯一**。

**Kahn 算法**：反复删除入度为 0 的点及其出边。

```cpp
// 判环 + 拓扑序（Kahn）
vector<int> ind(N, 0);
for (auto& e : edges) ind[e[1]]++;
queue<int> q;
for (int i = 0; i < N; i++) if (ind[i] == 0) q.push(i);
vector<int> order;
while (!q.empty()) {
    int u = q.front(); q.pop();
    order.push_back(u);
    for (int v : G[u]) {
        if (--ind[v] == 0) q.push(v);
    }
}
// order.size() < N 说明有环（剩下的节点都在环上）
```

**DFS 法**：按 DFS 完成时间逆序输出（后序的逆）。

## DAG 的性质

- **入度为零的点**：不与其他点连通；扩散信息时，至少要从**所有入度为零的点**开始才能扩散到全图。
- **出度为零的点**：如果只有一个，则该点可以被其他所有点抵达。
- **添加多少条边才能使 DAG 强连通**：`max(入度为零的点数, 出度为零的点数)`。

## 基环树（pseudo-tree）

**N 个节点、N 条边的连通图**（不连通则为基环树森林）。基环树 = 环 + 环上的树枝：

- **基环内向树**：每个点只有一条出边；
- **基环外向树**：每个点只有一条入边。

基本处理思路：**先用拓扑排序剥离树枝，剩下的就是环**；再用反图计算挂在环上的最长链。

## 例题

### [851 喧闹与富有](https://leetcode-cn.com/problems/loud-and-rich/)

`answer[x]` = 所有比 x 有钱或同样有钱的人中安静值最小者。图按"更有钱 → 更穷"建边，拓扑序传播答案（或记忆化 DFS，避免每个点单独搜超时）。

**DFS + 记忆化**：

```cpp
class Solution {
public:
    vector<int> loudAndRich(vector<vector<int>> &richer, vector<int> &quiet) {
        int n = quiet.size();
        vector<vector<int>> g(n);
        for (auto &r : richer) {
            g[r[1]].emplace_back(r[0]); // 比 x 有钱的人
        }

        vector<int> ans(n, -1);
        function<void(int)> dfs = [&](int x) {
            if (ans[x] != -1) return;
            ans[x] = x;
            for (int y : g[x]) {
                dfs(y);
                if (quiet[ans[y]] < quiet[ans[x]]) {
                    ans[x] = ans[y];
                }
            }
        };
        for (int i = 0; i < n; ++i) {
            dfs(i);
        }
        return ans;
    }
};
```

**拓扑排序**：

```cpp
class Solution {
public:
    vector<int> loudAndRich(vector<vector<int>> &richer, vector<int> &quiet) {
        int n = quiet.size();
        vector<vector<int>> g(n);
        vector<int> inDeg(n);
        for (auto &r : richer) {
            g[r[0]].emplace_back(r[1]);
            ++inDeg[r[1]];
        }

        vector<int> ans(n);
        iota(ans.begin(), ans.end(), 0);
        queue<int> q;
        for (int i = 0; i < n; ++i) {
            if (inDeg[i] == 0) q.emplace(i);
        }
        while (!q.empty()) {
            int x = q.front();
            q.pop();
            for (int y : g[x]) {
                if (quiet[ans[x]] < quiet[ans[y]]) {
                    ans[y] = ans[x]; // 传播 x 的答案给 y
                }
                if (--inDeg[y] == 0) q.emplace(y);
            }
        }
        return ans;
    }
};
```

### [2127 参与会议的最多员工数](https://leetcode-cn.com/problems/maximum-employees-to-be-invited-to-a-meeting/)

每个人喜欢一个人（每个点出度为 1）→ **基环内向树森林**。圆桌安排分两类：

- **三人及以上环**：一张桌子只能有一个环，且不能挂支链 → 取最大环长；
- **二人环**：可以放任意多个，且允许挂链 → 所有"二人环 + 两侧最长链"之和。

先 DFS 找最大环，再拓扑排序剥离树枝、统计支链：

```cpp
class Solution {
public:
    int maximumInvitations(vector<int>& p) {
        int n = p.size();
        // detect largest loop
        int size3 = 0;
        vector<int> vis(n, -1), steps(n, 0);
        function<void(int, int, int)> dfs = [&](int x, int id, int step) {
            steps[x] = step;
            int y = p[x];
            if (vis[y] == -1) {
                vis[y] = id;
                dfs(y, id, step + 1);
            } else if (vis[y] == id) {
                // found loop, update max size
                size3 = max(size3, step - steps[y] + 1);
            }
        };
        for (int i = 0; i < n; i++) if (vis[i] == -1) {
            vis[i] = i;
            dfs(i, i, 1);
        }
        // toposort
        vector<int> ind(n), subchain(n, 1); // in degree, max subchain length
        queue<int> q;
        for (int i = 0; i < n; i++) ind[p[i]] += 1;
        for (int i = 0; i < n; i++) if (ind[i] == 0) q.push(i);
        while (!q.empty()) {
            int u = q.front(); q.pop();
            subchain[p[u]] = max(subchain[p[u]], subchain[u] + 1);
            if (--ind[p[u]] == 0) q.push(p[u]);
        }
        int size2 = 0;
        for (int i = 0; i < n; i++)
            // assures it's a 2-loop, update subchain length
            if (p[p[i]] == i && p[i] > i) size2 += subchain[i] + subchain[p[i]];
        return max(size3, size2);
    }
};
```

官方题解的另一种思路：**先拓扑排序再对环分类**（用拓扑剩下的入度≠0 的点找环，通过反图求最长链）：

```cpp
class Solution {
public:
    int maximumInvitations(vector<int> &favorite) {
        int n = favorite.size();
        vector<vector<int>> g(n), rg(n); // rg 为图 g 的反图
        vector<int> deg(n);
        for (int v = 0; v < n; ++v) {
            int w = favorite[v];
            g[v].emplace_back(w);
            rg[w].emplace_back(v);
            ++deg[w];
        }

        // 拓扑排序，剪掉图 g 上的所有树枝
        queue<int> q;
        for (int i = 0; i < n; ++i) {
            if (deg[i] == 0) q.emplace(i);
        }
        while (!q.empty()) {
            int v = q.front();
            q.pop();
            for (int w : g[v]) {
                if (--deg[w] == 0) q.emplace(w);
            }
        }

        // 寻找图 g 上的基环
        vector<int> ring;
        vector<int> vis(n);
        function<void(int)> dfs = [&](int v) {
            vis[v] = true;
            ring.emplace_back(v);
            for (int w: g[v]) {
                if (!vis[w]) dfs(w);
            }
        };

        // 通过反图 rg 寻找树枝上最深的链
        int max_depth = 0;
        function<void(int, int, int)> rdfs = [&](int v, int fa, int depth) {
            max_depth = max(max_depth, depth);
            for (int w: rg[v]) {
                if (w != fa) rdfs(w, v, depth + 1);
            }
        };

        int max_ring_size = 0, sum_list_size = 0;
        for (int i = 0; i < n; ++i) {
            if (!vis[i] && deg[i]) { // 遍历基环上的点（拓扑排序后入度不为 0）
                ring.resize(0);
                dfs(i);
                int sz = ring.size();
                if (sz == 2) { // 基环大小为 2：累加两侧最长链
                    int v = ring[0], w = ring[1];
                    max_depth = 0;
                    rdfs(v, w, 1);
                    sum_list_size += max_depth;
                    max_depth = 0;
                    rdfs(w, v, 1);
                    sum_list_size += max_depth;
                } else {
                    max_ring_size = max(max_ring_size, sz); // 取所有基环的最大值
                }
            }
        }
        return max(max_ring_size, sum_list_size);
    }
};
```

## 相关

- 强连通分量缩点后得到 DAG，拓扑性质在连通性问题里大量复用（见 `25_graph_connectivity`）。
