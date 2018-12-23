# 字典树

* 多模式匹配

  KMP的拓展。

* Trie树/字典树

  指针处于某个节点，代表具有该前缀的所有子串都匹配了。

* Trie图/DFA/AC自动机

  在Trie树之上添加**前缀指针**使其成为图。类似于Next数组。

  建图时间复杂度 $O(\sum len(s))$

* 匹配方法

  * 危险节点：终止节点&前缀指针指向终止节点的节点

  * 根据母串移动指针，失配时沿着前缀指针回溯。

    经过危险节点时意味着有一个子串已经匹配了，需要对其标记。

    时间复杂度 $O(len(S))$, S为母串。

* 例题

  * prototype

    ```c++
    #include <iostream>
    #include <queue>
    #include <string>
    #include <cstring>
    
    using namespace std;
    
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
    
    int N, M;
    string s;
    
    int main() {
    	cin >> N;
    	node* root = new node();
    	for (int i = 0; i < N; i++) {
    		cin >> s;
    		insert(root, s);
    	}
    	buildDFA(root);
    	cin >> M;
    	for (int i = 0; i < M; i++) {
    		cin >> s;
    		cout << (match(root, s) ? "YES" : "NO") << endl;
    	}
    }
    ```

  * Inevitable Virus

    还是预处理再判断环好写啊，waste 2h.

    ```c++
    #include <iostream>
    #include <stack>
    #include <set>
    #include <algorithm>
    #include <queue>
    #include <string>
    #include <cstring>
    
    using namespace std;
    
    const int letters = 2;
    const char stt = '0';
    
    struct node {
    	bool bad;
    	node* pre;  // prefix pointer
    	node* child[letters];
    	vector<node*> c;
    	node() {
    		memset(child, 0, sizeof(child));
    		pre = NULL;
    		bad = false;
    	}
    	bool full() { return child[0] != NULL && child[1] != NULL; }
    };
    
    void insert(node* rt, string s) {
    	int len = s.length();
    	for (int i = 0; i < len; i++) {
    		int idx = s[i] - stt;
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
    
    // **** the pre pointer!
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
    
    // simple cycle detection for a directed graph
    set<node*> vis;
    bool isloop(node* rt) {
    	vis.insert(rt);
    	int len = rt->c.size();
    	for (int i = 0; i < len; i++) {
    		node* p = rt->c[i];
    		if (vis.count(p)) return true;
    		else if (isloop(p)) return true;
    	}
    	vis.erase(rt);
    	return false;
    }
    
    int N;
    string s;
    
    int main() {
    	cin.sync_with_stdio(false);
    	cin >> N;
    	node* root = new node();
    	for (int i = 0; i < N; i++) {
    		cin >> s;
    		insert(root, s);
    	}
    	buildDFA(root);
    	preprocess(root);
    	vis.clear();
    	bool flag = isloop(root);
    	cout << (flag ? "TAK" : "NIE") << endl;
    }
    ```

  * Computer Virus on Planet Pandora

    ```c++
    #include <iostream>
    #include <algorithm>
    #include <queue>
    #include <string>
    #include <cstring>
    
    using namespace std;
    
    const int letters = 26;
    const char stt = 'A';
    
    struct node {
    	bool bad, reallybad, used;
    	node* pre;  // prefix pointer
    	node* child[letters];
    	node() {
    		memset(child, 0, sizeof(child));
    		pre = NULL;
    		bad = false;
    		reallybad = false;
    		used = false;
    	}
    };
    
    void insert(node* rt, string s) {
    	int len = s.length();
    	for (int i = 0; i < len; i++) {
    		int idx = s[i] - stt;
    		if (rt->child[idx] == NULL) rt->child[idx] = new node();
    		rt = rt->child[idx];
    	}
    	rt->bad = true;
    	rt->reallybad = true;
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
    
    int count(node* rt, string& s) {
    	int cnt = 0;
    	int len = s.length();
    	node* p = rt;
    	for (int i = 0; i < len; i++) {
    		int idx = s[i] - stt;
    		while (p != rt && p->child[idx] == NULL) p = p->pre;
    		if (p->child[idx]) {
    			p = p->child[idx];
    			node* pp = p;
    			if (pp->bad) {
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
    
    void del(node* rt) {
    	for (int i = 0; i < letters; i++)
    		if (rt->child[i]) del(rt->child[i]);
    	delete rt;
    }
    
    int T, N, M;
    string s, t;
    
    void decompress(string& s, string& t) {
    	int len = s.length();
    	t.clear();
    	for (int i = 0; i < len; i++) {
    		if (isalpha(s[i])) t += s[i];
    		else if (s[i] == '[') {
    			int num = 0;
    			for (int j = i + 1; j < len; j++) {
    				if (isdigit(s[j])) {
    					num *= 10;
    					num += s[j] - '0';
    				}
    				else if (s[j] == ']') {
    					i = j;
    					break;
    				}
    			}
    			char c = s[i - 1];
    			for (int i = 0; i < num; i++) {
    				t += c;
    			}
    		}
    	}
    }
    
    
    int main() {
    	cin.sync_with_stdio(false); // !!!
    	cin >> T;
    	while (T--) {
    		cin >> N;
    		node* root = new node();
    		for (int i = 0; i < N; i++) {
    			cin >> s;
    			insert(root, s);
    		}
    		buildDFA(root);
    		cin >> s;
    		decompress(s, t);
    		int ans = 0;
    		ans += count(root, t);
    		reverse(t.begin(), t.end());
    		ans += count(root, t);
    		cout << ans << endl;
    		del(root);
    	}
    }
    ```

  * DNA repair

    问题十分经典。动归思想！

    ```c++
    #include <iostream>
    #include <stack>
    #include <set>
    #include <algorithm>
    #include <queue>
    #include <string>
    #include <map>
    #include <cstring>
    
    using namespace std;
    
    const int maxc = 4;
    const int maxl = 1005;
    map<char, int> dict{ {'A',0},{'T',1},{'G',2},{'C', 3} };
    int dp[maxl][maxl];
    const int inf = 0x3fffffff;
    string s;
    int N, res, cnt;
    
    // array proto, very trivial...
    struct node {
    	bool bad;
    	int cs[maxc];
    	int pre;
    	node() {
    		memset(cs, -1, sizeof(cs));
    		pre = -1;
    		bad = false;
    	}
    } tr[maxl];
    
    void insert(int rt, string& s) {
    	int len = s.length();
    	int cur = rt;
    	for (int i = 0; i < len; i++) {
    		if (tr[cur].cs[dict[s[i]]] == -1)
    			tr[cur].cs[dict[s[i]]] = cnt++;
    		cur = tr[cur].cs[dict[s[i]]];
    	}
    	tr[cur].bad = true;
    }
    
    void build(int rt) {
    	tr[rt].pre = 0;
    	for (int i = 0; i < maxc; i++) tr[0].cs[i] = rt;
    	queue<int> q;
    	q.push(rt);
    	while (!q.empty()) {
    		int p = q.front(); q.pop();
    		for (int i = 0; i < maxc; i++) {
    			if (tr[p].cs[i] != -1) {
    				int c = tr[p].cs[i];
    				int ppre = tr[p].pre;
    				while (tr[ppre].cs[i] == -1) ppre = tr[ppre].pre;
    				tr[c].pre = tr[ppre].cs[i];
    				if (tr[tr[c].pre].bad) tr[c].bad = true;  // don't forget
    				q.push(tr[p].cs[i]);
    			}
    		}
    	}
    }
    
    void solve(int cnt) {
    	for (int i = 0; i < maxl; i++) {
    		for (int j = 0; j < maxl; j++) {
    			dp[i][j] = inf;
    		}
    	}
    	dp[0][1] = 0; // len i, final node j (id starts from 1)
    	int len = s.length();
    	for (int i = 1; i <= len; i++) {  // use s[i-1], i is length.
    		for (int j = 1; j < cnt; j++) {
    			//cout << "solve" << i << " " << j << endl;
    			if (dp[i - 1][j] == inf) continue;  // not necessary, deletion  makes 75ms -> 390ms.
    			for (int k = 0; k < maxc; k++) {
    				int tmp = j;
    				while (tr[tmp].cs[k] == -1) tmp = tr[tmp].pre;
    				if (tr[tr[tmp].cs[k]].bad) continue; // bad node
                    // DP: min of self or move from parent (+1 is changed char)
    				dp[i][tr[tmp].cs[k]] = min(dp[i][tr[tmp].cs[k]], dp[i - 1][j] + (dict[s[i - 1]] != k));
    				//cout << "dp " << i << "," << tr[tmp].cs[k] << "  = " << dp[i][tr[tmp].cs[k]] << endl;
    			}
    		}
    	}
    
    }
    
    int main() {
    	int cas = 0;
    	while (cin >> N) {
    		if (N == 0) break;
            // clear
    		for (int i = 0; i < cnt; i++) tr[i] = node();
    		cnt = 1;
    		int root = cnt++;
            // init
    		for (int i = 0; i < N; i++) {
    			cin >> s;
    			insert(root, s);
    		}
    		build(root);
            
    		cin >> s;
    		solve(cnt);
    
    		int len = s.size();
    		res = inf;
    		for (int i = 1; i < cnt; i++) {
    			if (!tr[i].bad) res = min(res, dp[len][i]);
    		}
    		cout << "Case " << ++cas << ": " << (res == inf ? -1 : res) << endl;
    	}
    }
    ```

  * Censored！