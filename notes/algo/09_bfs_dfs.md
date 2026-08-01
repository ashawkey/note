# 搜索算法 BFS / DFS

## 核心思想

把问题建模为**状态空间**上的搜索：每个状态是图中的一个节点，转移是边。两种基础遍历方式：

- **Uninformed (Blind) Search**：对要搜索的节点一无所知；
- **Informed (Heuristic) Search**：对要搜索的节点有一些启发式知识，可用来加速搜索。

**vis 状态的设计是搜索的关键**：BFS 的 vis 必须是"全局的"（抵达一个状态后，之后通过其他路径再抵达它都是次优的）；DFS 的 vis 常常需要撤销（`vis[i]=1; dfs(i); vis[i]=0`），用于搜索允许路径重叠的问题。

## BFS

```python
def bfs(s0):
    q = [s0]
    vis = set()

    while not q.empty():
        s = q[0]; q = q[1:]
        if is_goal(s):
            return s
        vis.add(s)
        for n in s.neighbors(): # direct neighbors (distance = 1)
            if not n in vis:
                q.append(n)
                
    return None
```

- **空间**：$O(|V|)$（队列需要存某一深度的所有节点）。
- **时间**：$O(|V|)$（每条边/节点入队一次）。
- **适用**：无权图的最短路（首次抵达即最短）、分层遍历、按层扩展的问题。

**BFS 的分层技巧**：需要按层处理时，**必须把入队前的队列大小固定下来**，`for (; i < q.size(); )` 是错的（`q.size()` 会随入队增长）：

```cpp
// 分层打印二叉树（见例题）
int s = q.size();
for (int i = 0; i < s; i++) { ... }
```

## DFS

```python
vis = set()
def dfs(s):
    if is_goal(s): return s
    for n in s.neighbors(): # direct neighbors (distance = 1)
        if not n in vis:
            vis.add(n)
            res = dfs(n)
            if res is not None:
                return res
    return None
```

- **空间**：图为 $O(|V|)$（vis 表）；**树为 $O(D)$**（深度，无需 vis）。
- **时间**：$O(|V|)$。
- **适用**：搜索"全部路径 / 路径是否存在"（允许路径重叠时只能用 DFS），配合剪枝。

## BFS 与 DFS 的选择

| 任务 | 选择 |
| --- | --- |
| 搜索**满足条件的节点** | 都可以，$O(\|V\|)$ |
| 搜索**最优路径**（无权图） | **BFS** 时间效率高，但要存队列；DFS 空间效率高但时间长 |
| 搜索**全部路径 / 路径是否存在**（路径可重叠） | **只能 DFS**（vis 撤销使复杂度变为指数级，BFS 的全局 vis 无法表达"路径重叠"），通常需要剪枝 |

## 例题

### [剑指 Offer 12 矩阵中的路径](https://leetcode-cn.com/problems/ju-zhen-zhong-de-lu-jing-lcof/)

单词搜索路径允许重叠 → 只能用 DFS + vis 撤销，且必须剪枝（找到即返回），否则 TLE。最坏复杂度 $O(3^L \cdot MN)$（每步三个方向、$MN$ 个起点）：

```cpp
class Solution {
public:
    int dx[4] = {0, 0, 1, -1};
    int dy[4] = {1, -1, 0, 0};
    bool exist(vector<vector<char>>& board, string word) {
        int N = board.size(), M = board[0].size(), L = word.size();
        
        vector<vector<int>> vis(N, vector<int>(M, 0));
        bool flag = false;

        function<void(int, int, int)> dfs = [&](int i, int j, int l) {
            if (l == L) {
                flag = true;
                return;
            }
            for (int d = 0; d < 4; d++) {
                int ni = i + dx[d], nj = j + dy[d];
                if (ni >= 0 && nj >= 0 && ni < N && nj < M && vis[ni][nj] == 0 && board[ni][nj] == word[l]) {
                    vis[ni][nj] = 1;
                    dfs(ni, nj, l+1);
                    if (flag) return; // necessary pruning! else TLE.
                    vis[ni][nj] = 0;
                }
            }
        };

        for (int i = 0; i < N; i++) {
            for (int j = 0; j < M; j++) {
                if (board[i][j] == word[0]) {
                    vis = vector<vector<int>>(N, vector<int>(M, 0));
                    vis[i][j] = 1;
                    dfs(i, j, 1);
                    if (flag) return true;
                }
            }
        }
        return false;
    }
};
```

### [剑指 Offer 32 分层打印二叉树](https://leetcode-cn.com/problems/cong-shang-dao-xia-da-yin-er-cha-shu-ii-lcof/)

不用显式记录节点层数/两个队列轮换的小技巧——**先固定当前队列大小**：

```cpp
class Solution {
public:
    vector<vector<int>> levelOrder(TreeNode* root) {
        vector<vector<int>> ans;
        vector<int> tmp;
        if (root == NULL) return ans;
        queue<TreeNode*> q;
        q.push(root);
        while (!q.empty()) {
            // must fix the current queue size ! `for (;i<q.size();)` is wrong.
            int s = q.size();
            for (int i = 0; i < s; i++) {
                TreeNode* p = q.front(); q.pop();
                tmp.push_back(p->val);
                if (p->left) q.push(p->left);
                if (p->right) q.push(p->right);
            }
            ans.push_back(tmp);
            tmp.clear();
        }
        return ans;        
    }
};
```

### [847 访问所有节点的最短路径](https://leetcode-cn.com/problems/shortest-path-visiting-all-nodes/)

无权图从任意点出发、可重复经过、访问所有节点的最短路——NP 问题，最优 $O(n^2 2^n)$。BFS 可行，但**状态不只包含当前节点，还包含已访问集合**，用状态压缩记录：

```cpp
class Solution {
public:
    int shortestPathLength(vector<vector<int>>& graph) {
        int n = graph.size(); // <= 12
        int S = pow(2, n) - 1;
        // state compression + bfs
        queue<tuple<int, int, int>> q;
        vector<vector<int>> v(n, vector<int>(S+1, 0)); // faster than set<tuple<int,int>>
        for (int i = 0; i < n; i++) {
            q.emplace(i, 1 << i, 0); // emplace is faster than push
            v[i][1 << i] = 1; // always set to vis right after push
        }
        while (!q.empty()) {
            auto [p, s, d] = q.front(); q.pop();
            if (s == S) return d;
            for (int next_p: graph[p]) {
                int next_s = s | (1 << next_p);
                if (!v[next_p][next_s]) {
                    q.emplace(next_p, next_s, d+1);
                    v[next_p][next_s] = 1;
                }
            }
        }
        return -1;
    }
};
```

## 进阶变体

### IDS（迭代加深搜索）

DFS 的缺陷是树的深度可能无限。IDS 限制每次 DFS 的最大深度，逐次放宽。完全多叉树下，新节点总远多于重复搜索的节点。

### Best-FS / A*

用两个函数刻画节点信息：

- $g(s)$：从 $s_0$ 到 $s$ 的代价（如已走步数）；
- $h(s)$：从 $s$ 到终点的**启发式估计**。

按 $x(s)$ 排序扩展邻居：

- $x(s) = g(s)$：退化为 BFS；
- $x(s) = h(s)$：Greedy Search；
- $x(s) = g(s) + h(s)$：Heuristic Search（A*）。

**A\***：若 $h(s)$ 是 **admissible** 的（$h(s) \le C(s, s_g)$，即不高估），则 A* 完备且最优。难处在于好的 $h(s)$ 不好找。

**IDA\***：用 IDS 替换 A* 的 BFS 骨架，以降低空间复杂度。

### alpha-beta 剪枝

对双方、零和、确定性的游戏（井字棋、五子棋、围棋），搜索树是 MIN/MAX 层交替结构。双方绝对理性时可用 AB 剪枝：

- alpha：我方（MAX）能获得的最大收益下界；
- beta：对方（MIN）能给我的最大收益上界。

```python
def minimax(node, depth, a=-inf, b=+inf):
    if node is a terminal node or depth == 0:
        return the heuristic value of node
    if node is min node:
        foreach child of node:
            b = min(b, minimax(child, depth-1, a, b))
            if b <= a:
                  return b
        return b
    else:
        foreach child of node:
            a = max(a, minimax(child, depth-1, a, b))
            if b <= a:
                   return a
        return a
```

- 对方（MIN 层）发现能给我们一个**比我们当前最大收益更小**的值时，剪枝；
- 我方（MAX 层）发现能获得一个**比我们当前最小收益更大**的值时，剪枝。
