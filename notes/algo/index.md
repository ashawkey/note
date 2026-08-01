# Algorithm

> 按**典型算法**组织的刷题/竞赛笔记：一个文件一个算法族 = 核心思想 + 模板代码 + 例题与题解。

## 基础技术

- [01 二分法 (Binary Search)](binary_search/) — 模板、答案二分、浮点二分、STL
- [02 双指针 (Two Pointers)](two_pointers/) — 相向/同向/快慢、三数之和、链表相交
- [03 滑动窗口 (Sliding Window)](sliding_window/) — 定长/变长、可修改 k 次
- [04 前缀和/差分 (Prefix Sum)](prefix_sum/) — 1D/2D、稀疏差分、树上前缀和
- [05 贪心 (Greedy)](greedy/) — 区间贪心、接雨水、证明方法
- [06 背包 (Knapsack)](knapsack/) — 0/1、完全、二维、计数
- [07 动态规划 (Dynamic Programming)](dynamic_programming/) — 线性/区间/状压/数位/计数 DP
- [08 回溯 (Backtracking)](backtracking/) — 选择/撤销、剪枝、数独/N皇后
- [09 BFS / DFS](bfs_dfs/) — 遍历模板、分层、状态压缩、A* 与剪枝
- [10 位运算 (Bit Manipulation)](bit_manipulation/) — 技巧、状态压缩、只出现一次系列

## 数据结构

- [11 栈/队列（单调栈/单调队列）](stack_queue/) — 单调栈/单调队列、括号类问题
- [12 堆 (Heap)](heap/) — 双堆中位数、多路归并、topK
- [13 哈希 (Hash)](hash/) — 前缀和+哈希、计数、LRU 类
- [14 并查集 (Union-Find)](union_find/) — 路径压缩、带权、食物链
- [15 Trie（前缀树）](trie/) — 前缀查询、01-Trie
- [16 线段树 (Segment Tree)](segment_tree/) — 区间修改/查询、动态开点、归并树、扫描线
- [17 树状数组 (Binary Indexed Tree)](binary_indexed_tree/) — 单点改/区间查、二维 BIT
- [18 树 (Tree)](tree/) — 遍历、重构、BST、Huffman、直径

## 图论

- [21 最短路 (Shortest Path)](shortest_path/) — Dijkstra / Bellman-Ford / SPFA / Floyd（含路径找回）
- [22 BFS 与最短路算法的区别](bfs_vs_shortest_path/) — 澄清型笔记
- [23 拓扑排序 (Topological Sort)](topological_sort/) — Kahn、判环、基环树
- [24 最小生成树 (MST)](minimum_spanning_tree/) — Kruskal / Prim、朱刘算法
- [25 图的连通性 (Graph Connectivity)](graph_connectivity/) — Tarjan、SCC、桥、双连通
- [26 网络流 (Network Flow)](network_flow/) — Dinic、拆点、最小割、二分图匹配

## 字符串

- [31 字符串 (String)](string/) — KMP、循环匹配、字符串哈希、Manacher
- [32 AC 自动机 (Trie Automaton)](trie_automaton/) — fail 指针、AC 自动机 + DP
- [33 后缀数组 (Suffix Array)](suffix_array/) — 倍增法、height/LCP、ST 表

## 数学

- [41 数论 (Math)](math/) — 快速幂、组合数、约瑟夫、凸优化
- [42 计算几何 (Geometry)](geometry/) — 叉积、线段相交、凸包

## 理论 & 杂项

- [51 排序 (Sorting)](sorting/) — 八大排序对比、第 k 小、基数排序
- [52 复杂度分析 (Complexity)](complexity/) — 主定理、逆序对、O(1) 空间技巧
- [53 NP 理论 (NP)](np/) — NPC 证明、归约实例、近似算法
- [54 常用技巧 (Common Sense)](common_sense/) — 0x3f、快速 IO、摩尔投票、模板

## 学习路线（建议顺序）

1. **基础技术**：01 二分 → 02 双指针 → 03 滑动窗口 → 04 前缀和 → 05 贪心 → 06 背包 → 07 DP → 08 回溯 → 09 搜索
2. **数据结构**：11 → 13 → 12 → 15 → 14 → 17 → 16 → 18
3. **图论**：21 → 22（BFS vs 最短路）→ 23 → 24 → 25 → 26
4. **字符串 / 数学 / 理论**：按需查阅
