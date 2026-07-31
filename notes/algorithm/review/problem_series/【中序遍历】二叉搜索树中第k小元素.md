## [二叉搜索树中第k小元素](https://leetcode-cn.com/problems/kth-smallest-element-in-a-bst/)

> 给定一个二叉搜索树的根节点 `root` ，和一个整数 `k` ，请你设计一个算法查找其中第 `k` 个最小元素（从 1 开始计数）。

中序遍历的几种写法：

```cpp
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

可以用递减代替计数：

```cpp
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

迭代写法：

```cpp
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
        
    }
};
```

