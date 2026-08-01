# AC 自动机（Trie 图）(Aho-Corasick Automaton)

## 核心思想

**多模式匹配**：KMP 的拓展（一个母串匹配多个模式串）。

- **Trie 树**：指针处于某个节点，代表"具有该前缀的所有子串都匹配了"。
- **Trie 图 / AC 自动机**：在 Trie 之上添加**前缀指针（fail / pre）**使其成为图（DFA），类似 KMP 的 next 数组。建图复杂度 $O(\sum |P_i|)$。
- **匹配**：指针沿母串移动，**失配时沿前缀指针回溯**；经过**危险节点**（终止节点或前缀指针指向终止节点的节点）即说明匹配到一个模式串。匹配复杂度 $O(|S|)$。

与 Trie 的区别：不要求是前缀，单词可以出现在字符串任意位置。与 KMP 的区别：可以同时对多个模式串匹配。

## 模板

```cpp
const int letters = 26;

struct node {
    bool bad;          // 危险节点：终止节点 or fail 指向终止节点
    node* pre;         // prefix pointer (fail)
    node* child[letters];
    node() {
        memset(child, 0, sizeof(child));
        pre = NULL;
        bad = false;
    }
};

void insert(node* rt, string s) {
    int len = s.length();
    for (int i = 0; i < len; i++) {
        int idx = s[i] - 'a';
        if (rt->child[idx] == NULL) rt->child[idx] = new node();
        rt = rt->child[idx];
    }
    rt->bad = true;
}

// BFS 建 fail 指针
void buildDFA(node* rt) {
    node* rrt = new node();       // 虚拟根：让根的所有孩子 fail 指向根
    rt->pre = rrt;
    for (int i = 0; i < letters; i++) rrt->child[i] = rt;
    queue<node*> q;
    q.push(rt);
    while (!q.empty()) {
        node* p = q.front(); q.pop();
        for (int i = 0; i < letters; i++) {
            node* pc = p->child[i];
            if (pc) {
                node* ppre = p->pre;
                while (ppre->child[i] == NULL) ppre = ppre->pre; // 沿 fail 找有该孩子的祖先
                pc->pre = ppre->child[i];
                if (pc->pre->bad) pc->bad = true;  // 危险节点传染
                q.push(pc);
            }
        }
    }
}

// 母串匹配：返回是否包含任一模式串
bool match(node* rt, string s) {
    int len = s.length();
    node* p = rt;
    for (int i = 0; i < len; i++) {
        int idx = s[i] - 'a';
        while (p != rt && p->child[idx] == NULL) p = p->pre;
        if (p->child[idx]) {
            p = p->child[idx];
            if (p->bad) return true;
        }
        else continue;
    }
    return false;
}
```

> 模板关键：**buildDFA 的 while 找 fail 链**（`while (ppre->child[i] == NULL) ppre = ppre->pre`）与 **match 的失配回溯**（`while (p != rt && p->child[idx] == NULL) p = p->pre`）是同一套逻辑。

## 例题

### 原型题（HDU 2222 风格）

建图 + 匹配即可，见模板。

### Inevitable Virus（POJ 3691 变体，判环）

问是否存在**无穷长**且不包含任何模式串的字符串 → AC 自动机预处理（跳过坏节点，缺失转移补到 fail 链）后**判有向图是否有环**：

```cpp
// 预处理：把缺失的 child 补到 fail 链上（跳过坏节点），得到"安全转移图"
void preprocess(node* rt) {
    for (int i = 0; i < letters; i++) {
        if (rt->child[i]) {
            if (rt->child[i]->bad) continue;
            rt->c.push_back(rt->child[i]);
            preprocess(rt->child[i]);
        }
        else {
            node* tmp = rt;
            while (tmp->pre && !tmp->pre->bad && tmp->child[i] == NULL) tmp = tmp->pre;
            rt->c.push_back(tmp->child[i]);
        }
    }
}

// 简单有向图判环
set<node*> vis;
bool isloop(node* rt) {
    vis.insert(rt);
    for (node* p : rt->c) {
        if (vis.count(p)) return true;
        else if (isloop(p)) return true;
    }
    vis.erase(rt);
    return false;
}
```

### Computer Virus on Planet Pandora（HDU 3695）

统计母串（含压缩格式 `[num]` 展开）中出现的模式串个数，**正反各跑一遍**；用 `used` 标记保证每个模式串只计一次：

```cpp
int count(node* rt, string& s) {
    int cnt = 0;
    int len = s.length();
    node* p = rt;
    for (int i = 0; i < len; i++) {
        int idx = s[i] - 'A';
        while (p != rt && p->child[idx] == NULL) p = p->pre;
        if (p->child[idx]) {
            p = p->child[idx];
            node* pp = p;
            if (pp->bad) {
                // 沿 fail 链向上把所有未被计数的终止节点计数（used 去重）
                while (pp != rt && !pp->used) {
                    pp->used = true;
                    if (pp->reallybad) cnt++;
                    pp = pp->pre;
                }
            }
        }
    }
    return cnt;
}
// 主流程：ans += count(root, t); reverse(t); ans += count(root, t);
```

### DNA Repair（HDU 2457，AC 自动机 + DP）

修改最少字符使母串不含任何模式串。**DP 状态 = (母串位置 i, 自动机节点 j)**，转移枚举 4 个字符，走到坏节点则跳过：

```cpp
// tr[] 数组版自动机；bad 标记危险节点
void solve(int cnt) {
    for (int i = 0; i < maxl; i++)
        for (int j = 0; j < maxl; j++) dp[i][j] = inf;
    dp[0][1] = 0; // len i, final node j
    int len = s.length();
    for (int i = 1; i <= len; i++) {  // use s[i-1], i is length
        for (int j = 1; j < cnt; j++) {
            if (dp[i - 1][j] == inf) continue;
            for (int k = 0; k < maxc; k++) {
                int tmp = j;
                while (tr[tmp].cs[k] == -1) tmp = tr[tmp].pre;
                if (tr[tr[tmp].cs[k]].bad) continue; // bad node
                // DP: min of self or move from parent (+1 if char changed)
                dp[i][tr[tmp].cs[k]] = min(dp[i][tr[tmp].cs[k]], dp[i - 1][j] + (dict[s[i - 1]] != k));
            }
        }
    }
}
```

## 相关

- Trie 基础见 `15_trie`；KMP 见 `31_string`。
- 待补充经典题：**Censored!**（带屏蔽词的字符串计数，AC 自动机 + DP/矩阵快速幂）。
