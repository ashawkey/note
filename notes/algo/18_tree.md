# 树 (Tree / Binary Tree)

## 二叉树基础

### 基本性质

- 每个节点有 5 种形态：NULL、只有根、根+左、根+右、根+左右。
- N 个节点的二叉树有 **N − 1** 条边。
- 第 $i$ 层（根为第 0 层）最多 $2^i$ 个节点；高度为 K 的二叉树最多 $2^K - 1$ 个节点。
- **高度 = 层数，深度 = 最长路径长度**（仅根节点的树高度 1、深度 0）。
- 任意二叉树 $N_0 = N_2 + 1$（叶子数 = 度为 2 的节点数 + 1）。证明：$N = N_0+N_1+N_2 = E+1$ 且 $E = N_1 + 2N_2$。
- N 个节点的二叉树**空指针数 = N + 1**（推广：N 个节点的 K 叉树空指针 $= N(K-1)+1$）。
- N 个节点能构成 $\frac{1}{n+1}C_{2n}^n$（**Catalan 数**）种不同二叉树：$f(n) = \sum_{i=0}^{n-1} f(i)f(n-1-i)$。

### 完全二叉树（Complete BT）

- 只有**最后两层**的节点度可以小于 2，最后一层节点**从左到右连续**。
- 叶子只存在于最后两层；内部路径长度和在相同节点数的二叉树中最短；可以连续存在顺序表中。
- N 个节点的完全二叉树高度 $\lceil \log_2(N+1) \rceil$；第 n 层首元素为 $2^n - 1$（从 0 编号）。
- 从上到下、从左到右编号（数组存储，根为 0，最后一个为 n−1）：
  - $i$ 的父节点 $\lfloor (i-1)/2 \rfloor$；
  - 左孩子 $2i+1$、右孩子 $2i+2$（超过 n−1 则不存在）；
  - $i$ 为偶数时左兄弟 $i-1$，$i$ 为奇数时右兄弟 $i+1$。

### 扩充二叉树

把空指针替换成空树叶（外部节点）。性质：扩充二叉树是满二叉树；外部节点数 = 内部节点数 + 1；**外部路径长度 E、内部路径长度 I、内部节点数 n：$E = I + 2n$**（归纳可证）。

## 遍历

### 递归与重构

- 先序（根左右）、中序（左根右）、后序（左右根）。
- **重构定理**：先序或后序 **+ 中序**可以唯一确定一棵二叉树；**先序 + 后序不行**（无法区分左右子树）。

### 非递归（手动栈）

中序的非递归是最常用模板（BST 相关题大量使用，见例题）：

```c++
// infix
void infix(node* root) {
    stack<node*> stk;
    node* p = root;
    while (!stk.empty() || p) {
        if (p) {
            stk.push(p);
            p = p->left;
        }
        else {
            p = stk.top(), stk.pop();
            visit(p);
            p = p->right;
        }
    }
}

// prefix: 访问后先压右孩子再压左孩子
// postfix: 需要额外的 tag 标记左右（枚举 Left/Right，左子树归来改 Right 再进右子树）
```

用 `enum tag {Left, Right}` 的"通用 DFS"可以一个函数同时实现三种序。

### BFS（层序）

```c++
void bfs(node* root) {
    queue<node*> que;
    que.push(root);
    while (!que.empty()) {
        node* cur = que.front(); que.pop();
        visit(cur);
        if (cur->l != NULL) que.push(cur->l);
        if (cur->r != NULL) que.push(cur->r);
    }
}
```

分层打印等按层处理的技巧见 `09_bfs_dfs`。

## 二叉搜索树 BST

每个节点 K：左子树所有节点 < K，右子树所有节点 > K。值唯一，**中序遍历有序**。最佳情况下插入/删除/查找 $O(\log n)$。

```c++
struct node {
    node *L, *R;
    int data;
};

node* build(node* root, int d) {
    root = new node();
    root->data = d;
    return root;
}

void insert(node* root, int d) {
    if (root->data == d) return;
    else if (root->data < d) {
        if (root->R == NULL) {
            root->R = build(root->R, d);
            return;
        }
        else insert(root->R, d);
    }
    else {
        if (root->L == NULL) {
            root->L = build(root->L, d);
            return;
        }
        else insert(root->L, d);
    }
}

// 非递归删除：左子树为空直接挂右孩子；
// 否则找左子树最大值（或右子树最小值）替换被删节点
void del(node* root, node* p) {
    node* fp = parent(root, p);
    if (p->L == NULL) {
        if (fp->L == p) fp->L = p->R;
        else fp->R = p->R;
    }
    else {
        node* lmax = p->L;
        node* lpar = p;
        while (lmax->R != NULL) {
            lpar = lmax;
            lmax = lmax->R;
        }
        lpar->R = NULL;
        lmax->L = p->L;
        lmax->R = p->R;
        if (fp->L == p) fp->L = lmax;
        else fp->R = lmax;
    }
    delete p;
}
```

**定理**：随机构造 n 个不同节点的 BST 平均深度 $O(\log n)$、期望内部路径总和 $O(n\log n)$（由 Harmonic Series $\sum 1/i \in O(\log n)$ 推导）。

## Huffman 树（最优二叉树）

- 定义：带权路径长度（**外部路径长度**，含叶子权重）最小的二叉树，一定是**满二叉树**。
- 应用：频率不等的字符用 Huffman 编码得到最优不等长前缀编码；**任何字符编码都不是另一个编码的前缀**（反编码唯一）。
- K 叉 Huffman：先补**虚叶子（权重 0）**使 $(N-1) \bmod (K-1) = 0$，再反复取最小 K 个合并：

```c++
const int K = 2;

struct node {
    node* children[K];
    int w;
    node(int w) :w(w) {
        for (int i = 0; i < K; i++) children[i] = NULL;
    }
    node(int w, node** _children) :w(w) {
        for (int i = 0; i < K; i++) children[i] = _children[i];
    }
};

node* build(vector<int>& weights) {
    vector<node*> tree;
    int N = weights.size();
    // add virtual nodes !!!
    for (int i = 0; i < K - 1 - (N - 1) % (K - 1); i++) weights.push_back(0);
    priority_queue<node*> q;
    for (int w : weights) {
        tree.push_back(new node(w));
        q.push(tree.back());
    }
    while (true) {
        node* vs[K];
        int sum = 0;
        for (int i = 0; i < K; i++) {
            vs[i] = q.top();
            sum += vs[i]->w;
            q.pop();
        }
        tree.push_back(new node(sum, vs));
        if (q.empty()) return tree.back();
        q.push(tree.back());
    }
}
```

（堆的操作与实现见 `12_heap`。）

## 例题

### [230 二叉搜索树中第 K 小的元素](https://leetcode-cn.com/problems/kth-smallest-element-in-a-bst/)

中序遍历即有序。三种写法：

```cpp
// 写法一：递归返回子树大小
class Solution {
public:
    int ans;
    int infix(TreeNode* n, int K) {
        if (n == nullptr) return 0;
        int l = infix(n->left, K);
        if (ans != -1) return -1;
        if (l == K - 1) {
            ans = n->val;
            return -1;
        }
        int r = infix(n->right, K - l - 1);
        if (ans != -1) return -1;
        return l + r + 1;
    }
    int kthSmallest(TreeNode* root, int k) {
        ans = -1;
        infix(root, k);
        return ans;
    }
};
```

```cpp
// 写法二：递减计数
class Solution {
public:
    int ans, K;
    void infix(TreeNode* n) {
        if (n == nullptr) return;
        infix(n->left);
        if (ans != -1) return;
        if (--K == 0) {
            ans = n->val;
            return;
        }
        infix(n->right);
        if (ans != -1) return;
    }
    int kthSmallest(TreeNode* root, int k) {
        K = k;
        ans = -1;
        infix(root);
        return ans;
    }
};
```

```cpp
// 写法三：迭代（手动栈）
class Solution {
public:
    int kthSmallest(TreeNode* root, int k) {
        stack<TreeNode*> s;
        while (root != nullptr || !s.empty()) {
            while (root != nullptr) {
                s.push(root);
                root = root->left;
            }
            root = s.top(); s.pop();
            if (--k == 0) return root->val;
            root = root->right;
        }
        return -1;
    }
};
```

### [310 最小高度树](https://leetcode-cn.com/problems/minimum-height-trees/)

最小高度树的根在**树直径的中点**上。求直径技巧：**先以任意点为根 BFS 找最远点 x，再以 x 为根 BFS 找最远点 y，x–y 即直径**；BFS 时记录 parent，最后沿路径找回中点：

```cpp
class Solution {
public:
    vector<int> findMinHeightTrees(int n, vector<vector<int>>& edges) {
        // convert to G
        vector<vector<int>> G(n, vector<int>());
        for (auto& e: edges) {
            G[e[0]].push_back(e[1]);
            G[e[1]].push_back(e[0]);
        }
        // bfs
        vector<int> p(n); // parent
        auto bfs = [&] (int r) {
            for (int i = 0; i < n; i++) p[i] = -1; // -1 means not visited
            queue<pair<int, int>> q;
            q.emplace(r, 0);
            p[r] = -2; // just a number to differ from -1 and non-neg node id.
            int mxd = 0;
            int mxi = r;
            while (!q.empty()) {
                auto [x, d] = q.front(); q.pop();
                if (d > mxd) {
                    mxd = d;
                    mxi = x;
                }
                for (int y: G[x]) {
                    if (p[y] == -1) {
                        p[y] = x;
                        q.emplace(y, d + 1);
                    }
                }
            }
            return mxi;
        };
        // find longest path
        int x = bfs(0);
        int y = bfs(x);
        // retrieve middle node
        vector<int> path;
        while (y != -2) {
            path.push_back(y);
            y = p[y];
        }
        int l = path.size();
        if (l % 2 == 0) return {path[l/2 - 1], path[l/2]};
        else return {path[l/2]};
    }
};
```

> 注：`p[r] = -2` 用作"根节点无父节点"的哨兵，以区分 -1（未访问）。

## 相关

- 树形 DP 见 `07_dynamic_programming`；树的层序遍历技巧见 `09_bfs_dfs`。
