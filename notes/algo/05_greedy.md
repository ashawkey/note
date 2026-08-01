# 贪心 (Greedy)

## 核心思想

每一步做**当下看起来最好的选择**（局部最优），并证明该选择不会影响后续步骤的最优性，从而局部最优拼出全局最优。

- 贪心**没有状态**：每一步只做一次决策，不回头、不记忆，这是与动态规划的本质区别。
- 贪心通常需要先**排序**（按某个键排），再线性扫描。
- 常见套路：区间调度类（按结束时间/起点排序）、每次取当前最值（配合堆）、差值/统计类。

## 适用条件与证明

贪心只有在前两条都成立时才正确：

1. **最优子结构**：全局最优解包含子问题的最优解（与 DP 相同）；
2. **贪心选择性质**：贪心选的"当下最优"一定出现在某个全局最优解里。

常用证明方法：

- **交换论证（Exchange Argument）**：假设某个最优解不含贪心选择的元素，把它与贪心元素交换，证明不劣——最常见的证明方式。
- **归纳**：证明贪心第 $k$ 步之后，贪心前缀仍然可扩展成全局最优。

> 经典区间贪心三件套（均可用交换论证证明）：
> - **区间调度**（选最多不相交区间）：按**结束时间**升序，能选就选；
> - **区间选点**（最少点覆盖所有区间）：按**结束时间**升序，每次把点放在当前区间右端点；
> - **区间覆盖**（最少区间覆盖一段目标）：按**起点**排序，每次选能覆盖当前最左未覆盖点的、右端点最远的区间。

## 例题

### [517 超级洗衣机](https://leetcode-cn.com/problems/super-washing-machines/)

> n 台洗衣机排成一排，每台有一定量衣服；每步可让任意 m 台洗衣机各送 1 件衣服给相邻洗衣机。求使所有洗衣机衣服数相等的最少步数。

贪心魔法：答案 = max(单个洗衣机超出均值的最大量, 某前缀累计溢出量的绝对值最大值)。

```cpp
class Solution {
public:
    int findMinMoves(vector<int> &machines) {
        int tot = accumulate(machines.begin(), machines.end(), 0);
        int n = machines.size();
        if (tot % n) {
            return -1;
        }
        int avg = tot / n;
        int ans = 0, sum = 0;
        for (int num: machines) {
            num -= avg;
            sum += num;
            ans = max(ans, max(abs(sum), num));
        }
        return ans;
    }
};
```

### [42 接雨水](https://leetcode-cn.com/problems/trapping-rain-water/)

> 给定 n 个非负整数表示每根柱子高度，计算下雨后能接多少雨水。

**三遍循环的优雅贪心**：每个位置能接的水 = min(左边最高, 右边最高) - 自身高度。

```cpp
class Solution {
public:
    int trap(vector<int>& height) {
        int size = height.size();
        vector<int> left_max(size), right_max(size);
        left_max[0] = 0;
        right_max[size - 1] = 0;
        int max = height[0];
        for (int i = 1; i < size; ++i){
            left_max[i] = max;
            max = std::max(max, height[i]);
        }
        max = height[size - 1];
        for (int i = size - 2; i >= 0; --i){
            right_max[i] = max;
            max = std::max(max, height[i]);
        }
        int res = 0, con;
        // final loop
        for (int i = 0; i < size; ++i){
            con = std::min(left_max[i], right_max[i]);
            res += (con - height[i] < 0) ? 0 : con - height[i];
        }
        return res;
    }
};
```

另一种实现：用栈做"动态离散化"，思路绕且容易写错，仅作记录：

```cpp
class Solution {
public:
    int trap(vector<int>& height) {
        stack<pair<int, int>> s; // id, height
        map<int, int> d;
        int ans = 0;
        for (int i = 0; i < height.size(); i++) {
            int H = height[i];
            d[H]++;
            if (!s.empty()) {
                int last = 0;
                for (auto it = d.begin(); it != d.end(); it++) {
                    int h = it->first;
                    if (h == 0) continue;
                    if (h > H) break;
                    while (!s.empty() && s.top().second < h) {
                        // must remove useless discrete values, else TLE
                        if (--d[s.top().second] <= 0) d.erase(s.top().second);
                        s.pop();
                    }
                    if (!s.empty()) {
                        ans += (i - s.top().first - 1) * (h - last);
                    }
                    last = h;
                }
            }
            if (H > 0) s.push({i, H});
        }
        return ans;
    }
};
```

### [407 接雨水 II（2D）](https://leetcode-cn.com/problems/trapping-rain-water-ii/)

仍然是贪心，但 1D 的"左右最高点"在 2D 中变成了"该位置到任意边界**所有路径**的最高点中的最小值"——这正是最短路问题，用**优先队列 BFS（Dijkstra 思路）**解。可见 Dijkstra 不仅用于最短路（见 `21_shortest_path`）。

```cpp
typedef pair<int,int> pii;

class Solution {
public:
    int trapRainWater(vector<vector<int>>& heightMap) {  
        if (heightMap.size() <= 2 || heightMap[0].size() <= 2) {
            return 0;
        }  
        int m = heightMap.size();
        int n = heightMap[0].size();
        priority_queue<pii, vector<pii>, greater<pii>> pq;
        vector<vector<bool>> visit(m, vector<bool>(n, false));
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (i == 0 || i == m - 1 || j == 0 || j == n - 1) {
                    pq.push({heightMap[i][j], i * n + j});
                    visit[i][j] = true;
                }
            }
        }

        int res = 0;
        int dirs[] = {-1, 0, 1, 0, -1};
        while (!pq.empty()) {
            pii curr = pq.top();
            pq.pop();            
            for (int k = 0; k < 4; ++k) {
                int nx = curr.second / n + dirs[k];
                int ny = curr.second % n + dirs[k + 1];
                if( nx >= 0 && nx < m && ny >= 0 && ny < n && !visit[nx][ny]) {
                    if (heightMap[nx][ny] < curr.first) {
                        res += curr.first - heightMap[nx][ny]; 
                    }
                    visit[nx][ny] = true;
                    pq.push({max(heightMap[nx][ny], curr.first), nx * n + ny});
                }
            }
        }
        
        return res;
    }
};
```

### [475 供暖器](https://leetcode-cn.com/problems/heaters/)

二分答案可行（见 `01_binary_search`），但更直接的是贪心：对每个房子找最近的供暖器，答案取所有最近距离的最大值。$O(n\log n)$（排序）。

```cpp
class Solution {
public:
    int findRadius(vector<int>& houses, vector<int>& heaters) {
        sort(houses.begin(),houses.end());
        sort(heaters.begin(),heaters.end());
        int ans = 0;
        int h = 0;
        // O(n)
        for (int i=0;i<houses.size();i++){
            while (h + 1 < heaters.size() && 
                   abs(houses[i] - heaters[h]) >= abs(houses[i] - heaters[h + 1])) h++;
            ans = max(ans,abs(houses[i]-heaters[h]));
        }
        return ans;
    }
};
```

### [334 递增的三元子序列](https://leetcode-cn.com/problems/increasing-triplet-subsequence/)

只记录第一小、第二小，一旦出现更大的值就说明存在三元组——与 LIS 的贪心（`lower_bound` 替换）同思路，但只维护两个值。

```cpp
class Solution {
public:
    bool increasingTriplet(vector<int>& nums) {
        int len = nums.size();
        if (len < 3) return false;
        int small = INT_MAX, mid = INT_MAX;
        for (auto num : nums) {
            if (num <= small) small = num;
            else if (num <= mid) mid = num;
            else if (num > mid) return true;
        }
        return false;    
    }
};
```

（用完整 LIS 的解法见 `07_dynamic_programming`。）
