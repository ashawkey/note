# 网络流 (Network Flow)

## 最大流问题

有向图，单源单汇，边权为容量，求源到汇的最大流量。

**残量网络（Residual Network）**：对每条从源到汇的路径，从当前图中减去它并添加其**反向路径**。反向边提供了**撤销**（把流量推回去/重新安排流量）的能力——这是所有增广路算法的核心。

| 算法 | 找增广路方式 | 复杂度 |
| --- | --- | --- |
| Ford-Fulkerson | DFS | $O(Ef)$，$f$ 为最大流 |
| Edmonds-Karp | BFS | $O(VE^2)$ |
| Dinic | 分层图（BFS 分层 + DFS 多路增广） | $O(V^2E)$，实际很快 |

## 模板：Dinic

（POJ 1273 Drainage Ditches，邻接矩阵版）

```c++
const int inf = 0x3f3f3f3f;
const int maxv = 205;
int V, E, s, t;

int G[maxv][maxv]; // adjacency matrix (capacity)
int vis[maxv], dep[maxv];

// BFS 分层，能否到 t
bool bfs() {
    deque<int> q;
    memset(dep, -1, sizeof(dep));
    dep[s] = 0;
    q.push_back(s);
    while (!q.empty()) {
        int u = q.front(); q.pop_front();
        for (int i = 1; i <= V; i++) {
            if (G[u][i] > 0 && dep[i] == -1) {
                dep[i] = dep[u] + 1;
                if (i == t) return true;
                else q.push_back(i);
            }
        }
    }
    return false;
}

long long dinic() {
    long long mxflow = 0;
    deque<int> q;
    while (bfs()) {
        q.push_back(s);
        memset(vis, 0, sizeof(vis));
        vis[s] = 1;
        while (!q.empty()) {
            int u = q.back();
            // u is target: 沿当前 DFS 栈找最窄边并增广
            if (u == t) {
                int mnflow = inf;
                int v; // mnflow_start_vertex
                for (int i = 1; i < q.size(); i++) {
                    int vs = q[i - 1];
                    int ve = q[i];
                    if (G[vs][ve] > 0 && mnflow > G[vs][ve]) {
                        mnflow = G[vs][ve];
                        v = vs;
                    }
                }
                for (int i = 1; i < q.size(); i++) {
                    int vs = q[i - 1];
                    int ve = q[i];
                    G[vs][ve] -= mnflow;
                    G[ve][vs] += mnflow;
                }
                mxflow += mnflow;
                // 退栈到最窄边起点，继续找下一条增广路
                while (!q.empty() && q.back() != v) {
                    vis[q.back()] = 0;
                    q.pop_back();
                }
            }
            // u is not target: 尝试向下一层扩展（每次只进一个点）
            else {
                bool found = false;
                for (int i = 1; i <= V; i++) {
                    if (G[u][i] > 0 && dep[i] == dep[u] + 1 && !vis[i]) {
                        vis[i] = 1;
                        q.push_back(i);
                        found = true;
                        break;
                    }
                }
                if (!found) q.pop_back();
            }
        }
    }
    return mxflow;
}
```

## 拆点

把"点的容量限制"转化为"边容量"：把一个点拆成**入点 + 出点**，中间用容量为该点限制的边连接。

**ACM Computer Factory（POJ 3436）**：每个机器拆成输入/产出两个点，用产量连接；总源点连所有"输入全 0 或 2"的机器输入，总汇点连所有"产出全 1"的机器输出；机器之间按规格兼容连无穷边。**回溯最大流的每条边**：保存旧容量数组，跑完 Dinic 后 `旧容量 - 新容量` 即实际流量。

## 二分答案 + 最大流判定

**Optimal Milking（POJ 2116）**：奶牛到挤奶机的分配问题。Floyd 求最短路后**二分最大距离**，建图（源→奶牛 容量1，奶牛→可达挤奶机 容量1，挤奶机→汇 容量 M）跑最大流，等于奶牛数即可行。

## 最小割

**最大流 = 最小割**（min-cut max-flow theorem）。最小割的容量等于最大流，且最小割中"从源可达、汇不可达"的边划分了两个集合。常见转化：把"删边代价最小"问题建模成最小割。

## 二部图最大匹配

超源点用容量 1 的边连所有左部点，左部点连可匹配的右部点（容量 1），右部点连超汇点（容量 1），最大流即最大匹配（**The Perfect Stall**，POJ 1274）。更简单的专用算法是**匈牙利算法**（增广路 + 可撤销匹配，$O(VE)$）。

## 有流量下界的最大流

把每条有下界的边拆成"**必要边 + 不必要边**"：必要边（下界流量）先强制流掉，再在残量网络上求可行流。**Budget（POJ 2396）**是构造带下界网络流的经典题（行列和上下界 → 判断是否存在可行流并输出）。

## 最小费用最大流

每条边有单位流量费用，求所有最大流中**费用最小**的：把 EK 的 BFS 换成 **SPFA 求最短路**（边权为费用，注意反向边费用取负），增广时累加 `flow × 最短路费用`。**Farm Tour（POJ 2135）**：无向图两点间两条不相交路径 = 每条边容量 1、费用为长度，源到汇流量 2 的最小费用流。

## 相关

- 二分答案的判定思想见 `01_binary_search`；最短路（SPFA）见 `21_shortest_path`。
