## [参与会议的最多员工数](https://leetcode-cn.com/problems/maximum-employees-to-be-invited-to-a-meeting/)

一个公司准备组织一场会议，邀请名单上有 n 位员工。公司准备了一张 圆形 的桌子，可以坐下 任意数目 的员工。

员工编号为 0 到 n - 1 。每位员工都有一位 喜欢 的员工，每位员工 当且仅当 他被安排在喜欢员工的旁边，他才会参加会议。每位员工喜欢的员工 不会 是他自己。

给你一个下标从 0 开始的整数数组 favorite ，其中 favorite[i] 表示第 i 位员工喜欢的员工。请你返回参加会议的 最多员工数目 。

### 思路

根据题意，每个人都有一个出度，所以必然成环。

成环分两种情况：

* 二人成环：这种环可以在一个桌子上有任意多个，并且允许有支链连接到环上（链状）。

  ```
  ... --> 1 <--> 2 <-- ...
            table
  ... --> 3 <--> 4 <-- ...
  ```

* 三及以上人成环：这种环在一个桌子上只能有一个，且不允许任何环外支链。

  ```
  1 --> 2 --> 3
  ^   table   │
  └-----------┘
  ```

所以问题关键是：先检测所有环，根据环长分类讨论。三人以上环只需要统计环长，取最大值即可，可以通过DFS解决。二人成环则还需要统计连接到环上的支链长度，通过DFS不好解决（因为需要提前知道环结构），但可以通过**拓扑排序**解决，统计所有二人环加上最长支链的总和。最后，取这两个的极大值即可。

> 反思：首先就没能正确分类讨论，WA了几次才注意到三人环是不兼容的（只能取一个最大三人环，或是取全部二人环）。其次没有想到拓扑排序，为了能够正确的计算最长支链，时间复杂度超过了O(n)，被TLE制裁。然后企图用并查集，但结果无法保证正确。
>
> 不要想着能够把问题全都塞到一个算法里，如果能够放弃用DFS统计二人环，再写一个拓扑排序，就能过了...

别人的优秀解答：

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



### 拓展

**基环树/环套树(pseudo-tree)**：**具有N个节点，N条边的连通图。**若不连通，则成为**基环树森林(pseudo-forest)**。

基环树通常包含两个部分，即**环与环上的树枝**，故又称环套树。

基环内向树：每个点只有一条出边。

基环外向树：每个点只有一条入边。

此题属于**基环内向树森林**。

基本处理思路为**通过拓扑排序分离树枝与环**。

本题的另一种思路即先拓扑排序，再对环长分类讨论，长度3及以上的环只需要统计最大环长，长度2的环则可以通过**反图**计算最长树枝。

```cpp
class Solution {
public:
    int maximumInvitations(vector<int> &favorite) {
        int n = favorite.size();
        vector<vector<int>> g(n), rg(n); // rg 为图 g 的反图
        vector<int> deg(n); // 图 g 上每个节点的入度
        for (int v = 0; v < n; ++v) {
            int w = favorite[v];
            g[v].emplace_back(w);
            rg[w].emplace_back(v);
            ++deg[w];
        }

        // 拓扑排序，剪掉图 g 上的所有树枝
        queue<int> q;
        for (int i = 0; i < n; ++i) {
            if (deg[i] == 0) {
                q.emplace(i);
            }
        }
        while (!q.empty()) {
            int v = q.front();
            q.pop();
            for (int w : g[v]) {
                if (--deg[w] == 0) {
                    q.emplace(w);
                }
            }
        }

        // 寻找图 g 上的基环
        vector<int> ring;
        vector<int> vis(n);
        function<void(int)> dfs = [&](int v) {
            vis[v] = true;
            ring.emplace_back(v);
            for (int w: g[v]) {
                if (!vis[w]) {
                    dfs(w);
                }
            }
        };

        // 通过反图 rg 寻找树枝上最深的链
        int max_depth = 0;
        function<void(int, int, int)> rdfs = [&](int v, int fa, int depth) {
            max_depth = max(max_depth, depth);
            for (int w: rg[v]) {
                if (w != fa) {
                    rdfs(w, v, depth + 1);
                }
            }
        };

        int max_ring_size = 0, sum_list_size = 0;
        for (int i = 0; i < n; ++i) {
            if (!vis[i] && deg[i]) { // 遍历基环上的点（拓扑排序后入度不为 0）
                ring.resize(0);
                dfs(i);
                int sz = ring.size();
                if (sz == 2) { // 基环大小为 2
                    int v = ring[0], w = ring[1];
                    max_depth = 0;
                    rdfs(v, w, 1);
                    sum_list_size += max_depth; // 累加 v 这一侧的最长链的长度
                    max_depth = 0;
                    rdfs(w, v, 1);
                    sum_list_size += max_depth; // 累加 w 这一侧的最长链的长度
                } else {
                    max_ring_size = max(max_ring_size, sz); // 取所有基环的最大值
                }
            }
        }
        return max(max_ring_size, sum_list_size);
    }
};
```



