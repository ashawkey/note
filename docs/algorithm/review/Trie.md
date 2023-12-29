# Trie树

前缀树/字典树。输入一个词典（多个模式字符串），用于检索任意字符串（的前缀）是否出现在词典中。

实现$O(|S|)$时间的插入字符串、检索字符串（前缀）功能。

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
        for (char c: word) {
            int idx = c - 'a';
            if (cur->children[idx] == nullptr) cur->children[idx] = new node();
            cur = cur->children[idx];
        }
        cur->exist = true;
    }
    
    bool search(string word) {
        node *cur = root;
        for (char c: word) {
            int idx = c - 'a';
            if (cur->children[idx] == nullptr) return false;
            cur = cur->children[idx];
        }
        if (cur->exist) return true;
        else return false;
    }
    
    bool startsWith(string prefix) {
        node *cur = root;
        for (char c: prefix) {
            int idx = c - 'a';
            if (cur->children[idx] == nullptr) return false;
            cur = cur->children[idx];
        }
        return true;
    }
};


// concise ver.
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


# Trie图

AC自动机（Trie树上的KMP）。输入一个词典，用于检查任意字符串是否包含词典中的单词。

* 与Trie树的区别：不一定是前缀，单词可以出现在字符串任意位置。
* 与KMP的区别：可以同时对多个模式串（词典）进行匹配。

假设词典包含N个模式串$\{P_i\}$，要检索的字符串长度L。

* 对每个模式串单独进行KMP匹配：$O(\sum_i^N|P_i|+NL)$。
* 先建立Trie树，再对字符串枚举起点：$O(\sum_i^N|P_i| +L^2)$。

Trie图可以实现$O(\sum_i^N|P_i| +L)$的复杂度。


```cpp

const int letters = 26;

struct node {
	bool bad;
	node* pre;  // prefix pointer
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

void buildDFA(node* rt) {
	node* rrt = new node();
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
				while (ppre->child[i] == NULL) ppre = ppre->pre;
				pc->pre = ppre->child[i];
				if (pc->pre->bad) pc->bad = true;
				q.push(pc);
			}
		}
	}
}

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

