# Trie（前缀树 / 字典树）

## 核心思想

输入一个词典（多个模式串），用于检索任意字符串（的**前缀**）是否出现在词典中，实现 $O(|S|)$ 时间的插入与检索。

- 每个节点代表一个前缀；从根到某节点的路径拼出一个前缀。
- 通常用字符集大小的孩子数组（如 26）；空间换时间。
- 与哈希的对比：哈希能做等值查询，Trie 擅长**前缀查询**、求共同前缀、按前缀遍历。
- 变体：01-Trie（把数按二进制位建成 Trie，用于异或最值）。

## 模板

指针版：

```cpp
class Trie {
public:
    struct node {
        bool exist = false;
        node* children[26];
    };
    node* root;

    Trie() {
        root = new node();
    }

    void insert(string word) {
        node *cur = root;
        for (char c : word) {
            int idx = c - 'a';
            if (cur->children[idx] == nullptr) cur->children[idx] = new node();
            cur = cur->children[idx];
        }
        cur->exist = true;
    }

    bool search(string word) {
        node *cur = root;
        for (char c : word) {
            int idx = c - 'a';
            if (cur->children[idx] == nullptr) return false;
            cur = cur->children[idx];
        }
        return cur->exist;
    }

    bool startsWith(string prefix) {
        node *cur = root;
        for (char c : prefix) {
            int idx = c - 'a';
            if (cur->children[idx] == nullptr) return false;
            cur = cur->children[idx];
        }
        return true;
    }
};
```

简洁数组版（竞赛常用，`nex[p][c]` 存子节点编号）：

```cpp
struct trie {
    int nex[100000][26], cnt;
    bool exist[100000];  // 该结点结尾的字符串是否存在

    void insert(char *s, int l) {  // 插入字符串
        int p = 0;
        for (int i = 0; i < l; i++) {
            int c = s[i] - 'a';
            if (!nex[p][c]) nex[p][c] = ++cnt;  // 如果没有，就添加结点
            p = nex[p][c];
        }
        exist[p] = 1;
    }
    bool find(char *s, int l) {  // 查找字符串
        int p = 0;
        for (int i = 0; i < l; i++) {
            int c = s[i] - 'a';
            if (!nex[p][c]) return 0;
            p = nex[p][c];
        }
        return exist[p];
    }
};
```

## 进阶：Trie 图（AC 自动机）

在 Trie 上加上 **fail 指针**（类比 KMP 的 next），实现"任意字符串中是否包含词典中任意单词"的 $O(L)$ 匹配——这就是 AC 自动机，见 `32_trie_automaton`。
