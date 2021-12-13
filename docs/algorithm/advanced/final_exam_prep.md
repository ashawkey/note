### DJ set

##### food chain

```c++
#include <iostream>
using namespace std;

const int maxn = 50005;
const int inf = 0x3f3f3f3f;

int N, K;
int p[maxn], w[maxn];

void init(int N) {
	for (int i = 0; i <= N; i++) {
		p[i] = i;
		w[i] = 0;
	}
}

int par(int x) {
	if (p[x] == x) return x;
	int fx = p[x];
	p[x] = par(fx);
	w[x] = (w[x] + w[fx]) % 3;
	return p[x];
}

void merge(int x, int y, int d) {
	int fx = par(x);
	int fy = par(y);
	p[fx] = fy;
	w[fx] = (d - 1 + w[y] - w[x] + 3) % 3;
}

int ans, d, x, y;

int main() {
	cin >> N >> K;
	init(N);
	ans = 0;
	for (int i = 0; i < K; i++) {
		cin >> d >> x >> y;
		if (x > N || y > N) {
			ans++;
			continue;
		}
		if (par(x) != par(y)) merge(x, y, d);
		else {
			if (d == 1) {
				if (w[x] == w[y]) continue;
				else ans++;
			}
			else {
				if ((w[x] - w[y] + 3) % 3 == 1) continue;
				else ans++;
			}
		}
	}
	cout << ans << endl;
};
```

##### cube stacking

```c++
#define  _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <cstdio>
#include <algorithm>
#include <cstring>
#include <string>
#include <vector>
#include <stack>
#include <queue>
#include <bitset>
using namespace std;

const int maxn = 30005;

int N, M;
int p[maxn], sum[maxn], under[maxn];

void init(int N) {
	for (int i = 1; i <= N; i++) {
		p[i] = i;
		sum[i] = 1;
		under[i] = 0;
	}
}

int par(int x) {
	if (p[x] == x) return x;
	int f = p[x];
	p[x] = par(f);
	under[x] += under[f];
	return p[x];
}

void merge(int x, int y) {
	int fx = par(x);
	int fy = par(y);
	p[fy] = fx;
	under[fy] = sum[fx];
	sum[fx] += sum[fy];
}

char c;
int x, y;

int main() {
	init(maxn);
	cin >> M;
	for (int i = 0; i < M; i++) {
		cin >> c;
		if (c == 'C') {
			cin >> x;
			par(x);
			cout << under[x] << endl;
		}
		else {
			cin >> x >> y;
			merge(y, x);
		}
	}
}
```

##### the suspects

```c++
#include <iostream>
using namespace std;

const int maxn = 30005;
const int inf = 0x3f3f3f3f;

int N, M;
int p[maxn];

void init(int N) {
	for (int i = 0; i < N; i++) {
		p[i] = i;
	}
}

int par(int x) {
	if (p[x] == x) return x;
	int fx = p[x];
	p[x] = par(fx);
	return p[x];
}

void merge(int x, int y) {
	int fx = par(x);
	int fy = par(y);
	p[fx] = fy;
}

int ans, x, y, z;

int main() {
	while (cin >> N >> M && N) {
		init(N);
		for(int i = 0; i < M; i++) {
			cin >> x >> y;
			for (int i = 1; i < x; i++) {
				cin >> z;
				if (par(y) != par(z)) merge(y, z);
			}
		}
		ans = 0;
		int tmp = par(0);
		for (int i = 0; i < N; i++) {
			if (par(i) == tmp) ans++;
		}
		cout << ans << endl;
	}
}
```



### BIT

##### apple tree

```c++
#include <iostream>
#include <algorithm>
#include <cstring>
#include <vector>
using namespace std;

const int maxn = 100005;
int N;

int arr[maxn], bit[maxn], in[maxn], out[maxn];
vector<int> G[maxn];

int lowbit(int x) { return x & (-x); }

void add(int i, int x) {
	arr[i] += x;
	for (i; i <= N; i+=lowbit(i)) bit[i] += x;
}

int query(int i) {
	int ans = 0;
	for (i; i > 0; i -= lowbit(i)) ans += bit[i];
	return ans;
}

int t;
void dfs(int x) {
	in[x] = t++;
	for (int i = 0; i < G[x].size(); i++) dfs(G[x][i]);
	out[x] = t;
}

int M, x, y;
char c;

int main() {
	cin >> N;
	memset(arr, 0, sizeof(arr));
	memset(bit, 0, sizeof(bit));
	for (int i = 1; i <= N; i++) add(i, 1);
	for (int i = 1; i < N; i++) {
		cin >> x >> y;
		G[x].push_back(y);
	}
	t = 0;
	dfs(1);
	cin >> M;
	for (int i = 0; i < M; i++) {
		cin >> c >> x;
		if (c == 'Q') cout << query(out[x]) - query(in[x]) << endl;
		else {
			int p = in[x] + 1;
			add(p, (arr[p] ? -1 : 1));
		}
	}
}
```

##### LIS`(NlgN BS+DP)`

```c++
#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <cstdio>
#include <algorithm>
#include <cstring>
using namespace std;

const int maxn = 300005;
const int inf = 0x7fffffff;
int N;
int arr[maxn], dp[maxn];

int main() {
	scanf("%d", &N);
	for (int i = 0; i < N; i++) scanf("%d", arr + i);
	for (int i = 0; i < N; i++) dp[i] = inf;
	int ans = 0;
	for (int i = 0; i < N; i++) {
		int idx = lower_bound(dp, dp + N, arr[i]) - dp;
		dp[idx] = arr[i];
		if (idx == ans) ans++;
	}
	cout << ans << endl;
}
```



### Segment Tree

##### Balanced Lineup 

静态求区间最值。

卡`cin`就算了，没想到这个题`cout`都卡。

```c++
#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <cstdio>
#include <algorithm>
#include <cstring>
#define P pair<int,int>
#define lc rt*2+1
#define rc rt*2+2
using namespace std;

const int maxn = 50005;
const int inf = 0x7fffffff;
int h[maxn];
int N, Q;

struct node {
	int l, r;
	int mx, mn;
	int m() { return (l + r) / 2; }
} tr[maxn*4];

void build(int rt, int l, int r) {
	tr[rt].l = l;
	tr[rt].r = r;
	if (l == r) {
		tr[rt].mx = tr[rt].mn = h[l];
		return;
	}
	int m = (l + r) / 2;
	build(lc, l, m);
	build(rc, m + 1, r);
	tr[rt].mx = max(tr[lc].mx, tr[rc].mx);
	tr[rt].mn = min(tr[lc].mn, tr[rc].mn);
}

int mx, mn;
void query(int rt, int l, int r) {
	if (l == tr[rt].l && r == tr[rt].r) {
		mx = max(mx, tr[rt].mx);
		mn = min(mn, tr[rt].mn);
		return;
	}
	int m = tr[rt].m();
	if (r <= m) query(lc, l, r);
	else if (l > m) query(rc, l, r);
	else {
		query(lc, l, m);
		query(rc, m + 1, r);
	}
}

int x, y;

int main() {
	scanf("%d%d", &N, &Q);
	for (int i = 0; i < N; i++) scanf("%d", h + i);
	build(0, 0, N - 1);
	for (int i = 0; i < Q; i++) {
		scanf("%d%d", &x, &y);
		mn = inf, mx = 0;
		query(0, x - 1, y - 1);
		printf("%d\n", mx - mn);
	}
}
```



##### integers

```c++
#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <iomanip>
#include <cstring>
#include <algorithm>
#include <string>
#include <queue>
#include <cmath>
#include <vector>
#include <stack>
using namespace std;

#define ll long long

const int maxn = 100005;
ll arr[maxn];
int N, Q;

struct node {
	int l, r;
	ll sum, inc;
	int mid() { return (l + r) / 2; }
} tr[maxn<<2];

#define lc 2*rt+1
#define rc 2*rt+2

void pushup(int rt) {
	tr[rt].sum = tr[lc].sum + tr[rc].sum;
}

void pushdown(int rt) {
	if (tr[rt].inc) {
		tr[rt].sum += (tr[rt].r - tr[rt].l + 1)*tr[rt].inc;
		tr[lc].inc += tr[rt].inc;
		tr[rc].inc += tr[rt].inc;
		tr[rt].inc = 0;
	}
}

void build(int rt, int l, int r) {
	tr[rt].l = l;
	tr[rt].r = r;
	tr[rt].inc = 0;
	if (l == r) {
		tr[rt].sum = arr[l];
		return;
	}
	int m = (l + r) / 2;
	build(lc, l, m);
	build(rc, m + 1, r);
	pushup(rt);
}


void add(int rt, int l, int r, int v) {
	if (tr[rt].l == l && tr[rt].r == r) {
		tr[rt].inc += v;
		return;
	}
	tr[rt].sum += (r - l + 1)*v;
	int m = tr[rt].mid();
	if (l > m) add(rc, l, r, v);
	else if (r <= m) add(lc, l, r, v);
	else {
		add(lc, l, m, v);
		add(rc, m + 1, r, v);
	}
}


ll query(int rt, int l, int r) {
	if (tr[rt].l == l && tr[rt].r == r) return tr[rt].sum + tr[rt].inc*(r - l + 1);
	pushdown(rt);
	int m = tr[rt].mid();
	if (l > m) return query(rc, l, r);
	else if (r <= m) return query(lc, l, r);
	else return query(lc, l, m) + query(rc, m + 1, r);
}

char c;
int a, b, d;

int main() {
	cin >> N >> Q;
	for (int i = 0; i < N; i++) cin >> arr[i];
	build(0, 0, N - 1);
	for (int i = 0; i < Q; i++) {
		cin >> c;
		if (c == 'Q') {
			cin >> a >> b;
			cout << query(0, a-1, b-1) << endl;
		}
		else {
			cin >> a >> b >> d;
			add(0, a-1, b-1, d);
		}
	}
}
```



##### kth-Number

```c++
#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <map>
#include <iomanip>
#include <cstring>
#include <algorithm>
#include <string>
#include <queue>
#include <cmath>
#include <vector>
#include <stack>
using namespace std;

const int maxn = 100005;
int N;
int arr[maxn];

#define lc 2*rt+1
#define rc 2*rt+2

struct node {
	int l, r;
	vector<int> v;
	int m() { return (l + r) / 2; }
} tr[maxn<<2];

void pushup(int rt) {
	vector<int>& a = tr[lc].v;
	vector<int>& b = tr[rc].v;
	vector<int>& c = tr[rt].v;
	c.resize(a.size() + b.size()); // necessary
	merge(a.begin(), a.end(), b.begin(), b.end(), c.begin());
}

void build(int rt, int l, int r) {
	tr[rt].l = l;
	tr[rt].r = r;
	if (l == r) {
		tr[rt].v.push_back(arr[l]);
		return;
	}
	int m = (l + r) / 2;
	build(lc, l, m);
	build(rc, m + 1, r);
	pushup(rt);
}

int query(int rt, int l, int r, int v) {
	if (tr[rt].l == l && tr[rt].r == r) {
		return upper_bound(tr[rt].v.begin(), tr[rt].v.end(), v) - tr[rt].v.begin();
	}
	int m = tr[rt].m();
	if (l > m) return query(rc, l, r, v);
	else if (r <= m) return query(lc, l, r, v);
	else return query(lc, l, m, v) + query(rc, m + 1, r, v);
}

void solve(int l, int r, int k) {
	int left = -1e9 - 1, right = 1e9 + 1;
	while (left < right) {
		int mid = left + (right - left) / 2;
		if (query(0, l, r, mid) < k) left = mid + 1;
		else right = mid;
	}
	cout << left << endl;
}

int Q, a, b, c;

int main() {
	cin.sync_with_stdio(false);
	cin >> N >> Q;
	for (int i = 0; i < N; i++) cin >> arr[i];
	build(0, 0, N - 1);
	for (int i = 0; i < Q; i++) {
		cin >> a >> b >> c;
		solve(a - 1, b - 1, c);
	}
}
```



##### Mayor's Poster

```c++
#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <map>
#include <iomanip>
#include <cstring>
#include <algorithm>
#include <string>
#include <queue>
#include <cmath>
#include <vector>
#include <stack>
using namespace std;

const int maxn = 10005;
int p[maxn][2];
int N;

struct node {
	int l, r;
	bool occ;
	int mid() { return (l + r) / 2; }
} tr[maxn*8];

#define lc 2*rt+1
#define rc 2*rt+2

void pushup(int rt) {
	tr[rt].occ = tr[lc].occ & tr[rc].occ;
}

void pushdown(int rt) {
	if (tr[rt].occ) {
		tr[lc].occ = 1;
		tr[rc].occ = 1;
	}
}

void build(int rt, int l, int r) {
	tr[rt].l = l;
	tr[rt].r = r;
	tr[rt].occ = 0;
	if (l == r) return;
	int m = tr[rt].mid();
	build(lc, l, m);
	build(rc, m + 1, r);
}

bool add(int rt, int l, int r) {
	if (tr[rt].l == l && tr[rt].r == r) {
		if (tr[rt].occ == 0) {
			tr[rt].occ = 1;
			return true;
		}
		else return false;
	}
	pushdown(rt);
	int m = tr[rt].mid();
	bool flag;
	if (l > m) flag = add(rc, l, r);
	else if (r <= m) flag = add(lc, l, r);
	else flag = add(lc, l, m) | add(rc, m + 1, r);
	pushup(rt);
	return flag;
}

int main() {
	int cas;
	cin >> cas;
	while (cas--) {
		cin >> N;
		vector<int> xs;
		map<int, int> m;
		for (int i = 0; i < N; i++) {
			cin >> p[i][0] >> p[i][1];
			xs.push_back(p[i][0]);
			xs.push_back(p[i][1]);
		}
		sort(xs.begin(), xs.end());
		int uN = unique(xs.begin(), xs.end()) - xs.begin();
		for (int i = 0; i < uN; i++) m[xs[i]] = i;
		build(0, 0, uN - 1);
		int ans = 0;
		for (int i = N - 1; i >= 0; i--) {
			if (add(0, m[p[i][0]], m[p[i][1]])) 
				ans++;
		}
		cout << ans << endl;
	}
}
```





### Trie

##### Proto

```c++
#include <iostream>
#include <cstring>
#include <string>
#include <queue>
#include <vector>
#include <algorithm>
#define ll long long

using namespace std;

const int maxc = 26;

struct node {
	node* cs[maxc];
	node* fail;
	bool bad;
	node() {
		fail = 0;
		memset(cs, 0, sizeof(cs));
		bad = false;
	}
};

void insert(node* rt, string s) {
	for (int i = 0; i < s.size(); i++) {
		int idx = s[i] - 'a';
		if (!rt->cs[idx]) rt->cs[idx] = new node();
		rt = rt->cs[idx];
	}
	rt->bad = true;
}

void build(node* rt) {
	node* rrt = new node();
	rt->fail = rrt;
	for (int i = 0; i < maxc; i++) rrt->cs[i] = rt;
	queue<node*> q;
	q.push(rt);
	while (!q.empty()) {
		node* p = q.front(); q.pop();
		for (int i = 0; i < maxc; i++) {
			node* pc = p->cs[i];
			if (pc) {
				node* pre = p->fail; // father's fail
				while (!pre->cs[i]) pre = pre->fail;
				pc->fail = pre->cs[i];
				if (pc->fail->bad) pc->bad = true;
				q.push(pc);
			}
		}
	}
}

bool match(node* rt, string s) {
	node* p = rt;
	for (int i = 0; i < s.size(); i++) {
		int idx = s[i] - 'a';
		while (p != rt && !p->cs[idx]) p = p->fail; // back tracing first
		if (p->cs[idx]) {
			p = p->cs[idx]; // move
			if (p->bad) return true;
		}
		else continue;  // fail anyway
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
	build(root); // don't forget
	cin >> M;
	for (int i = 0; i < M; i++) {
		cin >> s;
		if (match(root, s)) cout << "YES" << endl;
		else cout << "NO" << endl;
	}
}
```



### Directed Tarjan

##### popular cows

```c++
#include <iostream>
#include <cstring>
#include <string>
#include <queue>
#include <stack>
#include <vector>
#include <algorithm>
#define P pair<int,int>

using namespace std;

// 4:53

const int maxn = 10005;
int N, M;

struct edge {
	int f, t;
	edge() {}
	edge(int f, int t) :f(f), t(t) {}
};

vector<edge> G[maxn];

int dfn[maxn], low[maxn], vis[maxn], color[maxn], outd[maxn];
int idx, ncol;
stack<int> stk;

void tarjan(int s) {
	dfn[s] = low[s] = idx++;
	vis[s] = 1;
	stk.push(s);
	for (int i = 0; i < G[s].size(); i++) {
		int t = G[s][i].t;
		if (!vis[t]) {
			tarjan(t);
			low[s] = min(low[s], low[t]);
		}
		else if (vis[t] == 1) {
			low[s] = min(low[s], dfn[t]);
		}
	}
	if (low[s] == dfn[s]) {
		int t;
		do {
			t = stk.top(); stk.pop();
			color[t] = ncol;
			vis[t] = 2;
		} while (t != s);
		ncol++;
	}
}

int solve() {
	memset(vis, 0, sizeof(vis));
	memset(outd, 0, sizeof(outd));
	memset(color, -1, sizeof(color));
    // check un-connectivity
	for (int i = 1; i <= N; i++) { 
		if (!vis[i]) tarjan(i);
	}
	for (int i = 1; i <= N; i++) {
		for (int j = 0; j < G[i].size(); j++) {
			edge& e = G[i][j];
			if (color[e.t] != color[e.f]) {
				outd[color[e.f]] ++;
			}
		}
	}
	int res = -1;
	for (int i = 0; i < ncol; i++) {
		if (outd[i] == 0) {
			if (res == -1) res = i;
			else return 0;
		}
	}
	int cnt = 0;
	for (int i = 1; i <= N; i++) {
		if (color[i] == res) cnt++;
	}
	return cnt;
}

int x, y;

int main() {
	cin >> N >> M;
	for (int i = 0; i < M; i++) {
		cin >> x >> y;
		G[x].push_back(edge(x, y));
	}
	cout << solve() << endl;
}

// 5:08
```



### Shortest Path

###### candies

```c++
#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <cstdio>
#include <cstring>
#include <string>
#include <queue>
#include <stack>
#include <vector>
#include <algorithm>
#define P pair<int,int>

using namespace std;

const int maxn = 30100;
const int inf = 0x3f3f3f3f;
int N, M;

struct edge {
	int f, t, w;
	edge() {}
	edge(int f, int t, int w) :f(f), t(t), w(w) {}
};

vector<edge> G[maxn];

int dist[maxn];

void dijstra(int s) {
	fill(dist, dist + N + 1, inf);
	dist[s] = 0;
	priority_queue<P, vector<P>, greater<P>> q;
	q.push(make_pair(dist[s], s));
	while (!q.empty()) {
		P p = q.top(); q.pop();
		int v = p.second;
		if (dist[v] != p.first) continue; // vis
		for (int i = 0; i < G[v].size(); i++) {
			edge& e = G[v][i];
			if (dist[e.t] > dist[e.f] + e.w) {
				dist[e.t] = dist[e.f] + e.w;
				q.push(make_pair(dist[e.t], e.t));
			}
		}
	}
}

/*
int dijstra(int s, int t) {
	fill(dist, dist + N + 1, inf);
	dist[s] = 0;
	priority_queue<P, vector<P>, greater<P>> q;
	q.push(make_pair(dist[s], s));
	while (!q.empty()) {
		P p = q.top(); q.pop();
		int v = p.second;
		if (dist[v] < p.first) continue;
		if (v == t) return dist[v]; // return after check !!!
		for (int i = 0; i < G[v].size(); i++) {
			edge& e = G[v][i];
			if (dist[e.t] > dist[e.f] + e.w) {
				dist[e.t] = dist[e.f] + e.w;
				q.push(make_pair(dist[e.t], e.t));
			}
		}
	}
	return -1;
}
*/

int x, y, w;

int main() {
	scanf("%d%d", &N, &M);
	for (int i = 0; i < M; i++) {
		scanf("%d%d%d", &x, &y, &w);
		G[x].push_back(edge(x, y, w));
	}
	dijstra(1);
	printf("%d\n", dist[N]);
}
```



##### currency exchange

```c++
#include <iostream>
#include <cstring>
#include <string>
#include <queue>
#include <stack>
#include <vector>
#include <algorithm>
#define P pair<int,int>

using namespace std;

const int maxn = 105;
int N, M;

struct edge {
	int f, t;
	double r, c;
	edge() {}
	edge(int f, int t, double r, double c) :f(f), t(t), r(r), c(c) {}
};

vector<edge> G[maxn];

int s;
double v;
double dist[maxn];

bool bf() {
	memset(dist, 0, sizeof(dist));
	dist[s] = v;
	for (int k = 1; k < N; k++) {
		bool flag = true;
		for (int i = 1; i <= N; i++) {
			for (int j = 0; j < G[i].size(); j++) {
				edge& e = G[i][j];
				if (dist[e.t] < (dist[e.f] - e.c) * e.r) {
					dist[e.t] = (dist[e.f] - e.c) * e.r;
					flag = false;
				}
			}
		}
		if (flag) return false;
	}
	for (int i = 1; i <= N; i++) {
		for (int j = 0; j < G[i].size(); j++) {
			edge& e = G[i][j];
			if (dist[e.t] < (dist[e.f] - e.c)*e.r) return true;
		}
	}
	return false;
}

int x, y;
double a, b, c, d;

int main() {
	cin >> N >> M >> s >> v;
	for (int i = 0; i < M; i++) {
		cin >> x >> y >> a >> b >> c >> d;
		G[x].push_back(edge(x, y, a, b));
		G[y].push_back(edge(y, x, c, d));
	}
	cout << (bf() ? "YES" : "NO") << endl;
}
```



##### Warmholes

```c++
#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <cstdio>
#include <cstring>
#include <string>
#include <queue>
#include <stack>
#include <vector>
#include <algorithm>
#define P pair<int,int>

using namespace std;

const int maxn = 505;
const int inf = 0x3f3f3f3f;
int N, M, W;

struct edge {
	int f, t, w;
	edge() {}
	edge(int f, int t, int w) :f(f), t(t), w(w) {}
};

vector<edge> G[maxn];

int dist[maxn], times[maxn];

bool spfa(int s) {
	fill(dist, dist + N + 1, inf);
	fill(times, times + N + 1, 0);
	dist[s] = 0;
	//queue<int> q;
	stack<int> q;
	q.push(s);
	while (!q.empty()) {
		//int p = q.front(); q.pop();
		int p = q.top(); q.pop();
		for (int i = 0; i < G[p].size(); i++) {
			edge& e = G[p][i];
			if (dist[e.t] > dist[e.f] + e.w) {
				dist[e.t] = dist[e.f] + e.w;
				q.push(e.t);
				if (times[e.t]++ >= N) return true;
			}
		}
	}
	return false;
}

int T, x, y, w;

int main() {
	scanf("%d", &T);
	while (T--) {
		scanf("%d%d%d", &N, &M, &W);
		for (int i = 1; i <= N; i++) G[i].clear();
		for (int i = 0; i < M; i++) {
			scanf("%d%d%d", &x, &y, &w);
			G[x].push_back(edge(x, y, w));
			G[y].push_back(edge(y, x, w));
		}
		for (int i = 0; i < W; i++) {
			scanf("%d%d%d", &x, &y, &w);
			G[x].push_back(edge(x, y, -w));
		}
		cout << (spfa(1) ? "YES" : "NO") << endl;
	}
}
```



##### cow contest

```c++
#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <cstdio>
#include <cstring>
#include <string>
#include <queue>
#include <stack>
#include <vector>
#include <algorithm>
#define P pair<int,int>

using namespace std;

const int maxn = 105;
const int inf = 0x3f3f3f3f;
int N, M, W;

struct edge {
	int f, t;
	edge() {}
	edge(int f, int t) :f(f), t(t) {}
};

vector<edge> G[maxn];

int dist[maxn][maxn];

void floyd() {
	memset(dist, 0x3f, sizeof(dist));
	for (int i = 1; i <= N; i++) {
		dist[i][i] = 0;
		for (int j = 0; j < G[i].size(); j++) {
			edge& e = G[i][j];
			dist[e.f][e.t] = 1;
		}
	}
	for (int k = 1; k <= N; k++) {
		for (int i = 1; i <= N; i++) {
			for (int j = 1; j <= N; j++) {
				if (dist[i][j] > dist[i][k] + dist[k][j]) {
					dist[i][j] = dist[i][k] + dist[k][j];
				}
			}
		}
	}
}


int T, x, y;

int main() {
	while (~scanf("%d%d", &N, &M)) {
		for (int i = 1; i <= N; i++) G[i].clear();
		for (int i = 0; i < M; i++) {
			scanf("%d%d", &x, &y);
			G[x].push_back(edge(x, y));
		}
		floyd();
		int ans = 0;
		for (int i = 1; i <= N; i++) {
			int cnt = 0;
			for (int j = 1; j <= N; j++) {
				if (j == i) continue;
				if (dist[i][j] != inf) cnt++;
				if (dist[j][i] != inf) cnt++;
			}
			if (cnt == N - 1) ans++;
		}
		cout << ans << endl;
	}
}
```





----

### Undirected Tarjan

···



### Maximum Flow

##### Alice's Chance

build the graph...

```c++
#include <iostream>    
#include <cstring>
#include <queue>   
#include <vector>  
#include <algorithm>   
#include <deque>   
#define inf 999999999

using namespace std;

const int maxn = 400;

int G[maxn][maxn];
bool vis[maxn];
int Layer[maxn];   //Layer[i]是节点i的层号  
int M = 371;

bool CountLayer(int src, int dst) { //分层
	int layer = 0;
	deque<int>q;
	memset(Layer, 0xff, sizeof(Layer)); //都初始化成-1
	Layer[src] = 0;
	q.push_back(src);
	while (!q.empty()) {
		int v = q.front();
		q.pop_front();
		for (int j = 1; j <= M; j++) {
			if (G[v][j] > 0 && Layer[j] == -1) {
				//Layer[j] == -1 说明j还没有访问过
				Layer[j] = Layer[v] + 1;
				if (j == dst) return true;  //分层到汇点即可
				else q.push_back(j);
			}
		}
	}
	return false;
}
int Dinic(int src, int dst) {
	int i;
	int s;
	int nMaxFlow = 0;
	deque<int> q; //DFS用的栈
	while (CountLayer(src, dst)) { //只要能分层
		q.push_back(src);	//源点入栈	
		memset(vis, 0, sizeof(vis));
		vis[src] = 1;
		while (!q.empty()) {
			int nd = q.back();
			if (nd == dst) { // nd是汇点
				//在栈中找容量最小边
				int nMinC = inf;
				int nMinC_vs; //容量最小边的起点
				for (i = 1; i < q.size(); i++) {
					int vs = q[i - 1];
					int ve = q[i];
					if (G[vs][ve] > 0) {
						if (nMinC > G[vs][ve]) {
							nMinC = G[vs][ve];
							nMinC_vs = vs;
						}
					}
				}
				//增广，改图
				nMaxFlow += nMinC;
				for (i = 1; i < q.size(); i++) {
					int vs = q[i - 1];
					int ve = q[i];
					G[vs][ve] -= nMinC; //修改边容量 
					G[ve][vs] += nMinC; //添加反向边
				}
				//退栈到 nMinC_vs成为栈顶，以便继续dfs
				while (!q.empty() && q.back() != nMinC_vs) {
					vis[q.back()] = 0;
					q.pop_back();
				}

			}
			else { //nd不是汇点
				for (i = 1; i <= M; i++) {
					if (G[nd][i] > 0 && Layer[i] == Layer[nd] + 1 &&
						!vis[i]) {
						//只往下一层的没有走过的节点走
						vis[i] = 1;
						q.push_back(i);
						break;
					}
				}
				if (i > M)  //找不到下一个点
					q.pop_back(); //回溯
			}
		}
	}
	return nMaxFlow;
}

const int maxk = 25;
int movies[maxk][10];

int T, K;
int src = 0, dst = 371;

void build() {
	for (int i = 1; i <= 350; i++) G[src][i] = 1;
	for (int k = 0; k < K; k++) {
		int tmp = 351 + k;
		for (int j = 1; j <= movies[k][8] * 7; j++) {
			if (movies[k][(j - 1) % 7]) {
				G[j][tmp] = 1;
			}
		}
		G[tmp][dst] = movies[k][7];
	}
}

int main() {
	cin >> T;
	while (T--) {
		cin >> K;
		int tot = 0;
		for (int i = 0; i < K; i++) {
			for (int j = 0; j < 9; j++) {
				cin >> movies[i][j];
			}
			tot += movies[i][7];
		}
		memset(G, 0, sizeof(G));
		build();
		int ans = Dinic(src, dst);
		if (ans == tot) cout << "Yes" << endl;
		else cout << "No" << endl;
	}
}
```

##### Asteroids

bipartite

```c++
#include <iostream>    
#include <cstring>
#include <queue>   
#include <vector>  
#include <algorithm>   
#include <deque>   
#define inf 999999999

using namespace std;

const int maxn = 2000;

int G[maxn][maxn];
bool vis[maxn];
int Layer[maxn];   //Layer[i]是节点i的层号  
int M;

bool CountLayer(int src, int dst) { //分层
	int layer = 0;
	deque<int>q;
	memset(Layer, 0xff, sizeof(Layer)); //都初始化成-1
	Layer[src] = 0;
	q.push_back(src);
	while (!q.empty()) {
		int v = q.front();
		q.pop_front();
		for (int j = 1; j <= M; j++) {
			if (G[v][j] > 0 && Layer[j] == -1) {
				//Layer[j] == -1 说明j还没有访问过
				Layer[j] = Layer[v] + 1;
				if (j == dst) return true;  //分层到汇点即可
				else q.push_back(j);
			}
		}
	}
	return false;
}
int Dinic(int src, int dst) {
	int i;
	int s;
	int nMaxFlow = 0;
	deque<int> q; //DFS用的栈
	while (CountLayer(src, dst)) { //只要能分层
		q.push_back(src);	//源点入栈	
		memset(vis, 0, sizeof(vis));
		vis[src] = 1;
		while (!q.empty()) {
			int nd = q.back();
			if (nd == dst) { // nd是汇点
				//在栈中找容量最小边
				int nMinC = inf;
				int nMinC_vs; //容量最小边的起点
				for (i = 1; i < q.size(); i++) {
					int vs = q[i - 1];
					int ve = q[i];
					if (G[vs][ve] > 0) {
						if (nMinC > G[vs][ve]) {
							nMinC = G[vs][ve];
							nMinC_vs = vs;
						}
					}
				}
				//增广，改图
				nMaxFlow += nMinC;
				for (i = 1; i < q.size(); i++) {
					int vs = q[i - 1];
					int ve = q[i];
					G[vs][ve] -= nMinC; //修改边容量 
					G[ve][vs] += nMinC; //添加反向边
				}
				//退栈到 nMinC_vs成为栈顶，以便继续dfs
				while (!q.empty() && q.back() != nMinC_vs) {
					vis[q.back()] = 0;
					q.pop_back();
				}

			}
			else { //nd不是汇点
				for (i = 1; i <= M; i++) {
					if (G[nd][i] > 0 && Layer[i] == Layer[nd] + 1 &&
						!vis[i]) {
						//只往下一层的没有走过的节点走
						vis[i] = 1;
						q.push_back(i);
						break;
					}
				}
				if (i > M)  //找不到下一个点
					q.pop_back(); //回溯
			}
		}
	}
	return nMaxFlow;
}

int src, dst;
int N, K, x, y;

int main() {
	cin >> N >> K;
	M = 2 * N + 1;
	memset(G, 0, sizeof(G));
	src = 0;
	dst = 2 * N + 1;
	for (int i = 1; i <= N; i++) G[src][i] = 1;
	for (int i = N + 1; i <= 2 * N; i++) G[i][dst] = 1;
	for (int i = 0; i < K; i++) {
		cin >> x >> y;
		G[x][y+N] = inf;
	}
	cout << Dinic(src, dst) << endl;
}
```





### Geometry

##### TOYS

```c++
#include <iostream>    
#include <cstring>
#include <string>
#include <queue>   
#include <vector>  
#include <algorithm>   
#include <deque>   

using namespace std;

double PI = acos(-1);
double INF = 1e20;
double EPS = 1e-3;

bool IsZero(double x) {
	return -EPS < x && x < EPS;
}
bool FLarger(double a, double b) {
	return a - b > EPS;
}
bool FLess(double a, double b) {
	return b - a > EPS;
}

struct CVector {
	double x, y;
	CVector() {}
	CVector(double x, double y) :x(x), y(y) {}
};

typedef CVector CPoint;

struct CLine {
	CPoint a, b;
	CLine() {}
	CLine(CPoint a, CPoint b) :a(a), b(b) {}
};

CVector operator +(CVector p, CVector q) {
	return CVector(p.x + q.x, p.y + q.y);
}

CVector operator -(CVector p, CVector q) {
	return CVector(p.x - q.x, p.y - q.y);
}

CVector operator *(double k, CVector p) {
	return CVector(k * p.x, k * p.y);
}

CVector operator /(CVector p, double k) {
	return CVector(p.x / k, p.y / k);
}

double operator *(CVector p, CVector q) {
	return p.x * q.x + p.y * q.y;
}

double length(CVector p) {
	return sqrt(p * p);
}

CVector unit(CVector p) {
	return 1 / length(p) * p;
}

double project(CVector p, CVector n) {
	return p * unit(n); //点积
}

double operator ^(CVector p, CVector q) {
	return p.x * q.y - q.x * p.y;
}

double area(CVector p, CVector q) {
	return (p ^ q) / 2;
}

double dist(CPoint p, CPoint q) {
	return length(p - q);
}

double dist(CPoint p, CLine l) {
	return fabs((p - l.a) ^ (l.b - l.a))
		/ length(l.b - l.a);
}

CPoint rotate(CPoint b, CPoint a,
	double alpha) {//返回点C坐标
	CVector p = b - a;
	return CPoint(a.x + (p.x * cos(alpha)
		- p.y * sin(alpha)),
		a.y + (p.x * sin(alpha)
			+ p.y * cos(alpha)));
}

int sideOfLine(CPoint p, CPoint a, CPoint b)
{ //判断p在直线 a->b的哪一侧
	double result = (b - a) ^ (p - a);
	if (IsZero(result))
		return 0; //p 在 a->b上
	else if (result > 0)
		return 1; //p在a->b左侧
	else
		return -1; //p在a->b右侧
}

int n, m, x1, Y1, x2, y2;

const int maxn = 5005;

CPoint polys[maxn][4];
int ans[maxn];

int u, l, ou, ol;
int x, y;

bool PointInPoly(int poly, CPoint p) {
	for (int i = 0; i < 4; i++) {
		CLine l(polys[poly][i], polys[poly][(i + 1) % 4]);
		//if (sideOfLine(p, l.a, l.b) == 0) return true;
		CVector a = polys[poly][i] - p;
		CVector b = polys[poly][(i + 1) % 4] - p;
		if (FLess(a^b, 0)) return false;
	}
	return true;
}

int main() {
	int cas = 0;
	while (cin >> n && n) {
		if (cas != 0) cout << endl;
		cas++;
		cin >> m >> x1 >> Y1 >> x2 >> y2;
		memset(ans, 0, sizeof(ans));
		ou = x1, ol = x1;
		for (int i = 0; i < n; i++) {
			cin >> u >> l;
			polys[i][0] = CPoint(ou, Y1);
			polys[i][1] = CPoint(ol, y2);
			polys[i][2] = CPoint(l, y2);
			polys[i][3] = CPoint(u, Y1);
			ou = u;
			ol = l;
		}
		polys[n][0] = CPoint(ou, Y1);
		polys[n][1] = CPoint(ou, y2);
		polys[n][2] = CPoint(x2, y2);
		polys[n][3] = CPoint(x2, Y1);
		for (int i = 0; i < m; i++) {
			cin >> x >> y;
			CPoint p(x, y);
			int l = 0, r = n + 1;
			while (l < r) {
				int m = (l + r) / 2;
				if (sideOfLine(p, polys[m][2], polys[m][3]) == -1) l = m + 1;
				else r = m;
			}
			ans[l]++;
		}
		for (int i = 0; i < n+1; i++) {
			cout << i << ": " << ans[i] << endl;
		}
	}
}
```

##### Space ant

极角排序。

```c++
#include <iostream>    
#include <cstring>
#include <string>
#include <queue>   
#include <vector>  
#include <algorithm>   
#include <deque>   

using namespace std;

double PI = acos(-1);
double INF = 1e20;
double EPS = 1e-3;

int Sign(double x) { // 判断 x 是大于0,等于0还是小于0 
	return fabs(x) < EPS ? 0 : x > 0 ? 1 : -1;
}

struct Point {
	double x, y;
	int id;
	Point(double xx = 0, double yy = 0, int id = 0) :x(xx), y(yy), id(id) { }
	Point operator-(const Point & p) const {
		return Point(x - p.x, y - p.y);
	}
	bool operator <(const Point & p) const {
		if (y < p.y)
			return true;
		else if (y > p.y)
			return false;
		else
			return x < p.x;
	}
};

typedef Point Vector;

double Cross(const Vector & v1, const Vector & v2)
{//叉积
	return v1.x * v2.y - v2.x * v1.y;
}

double Distance(const Point & p1, const Point & p2)
{
	return sqrt((p1.x - p2.x)*(p1.x - p2.x) + (p1.y - p2.y)*(p1.y - p2.y));
}

struct Comp { //用来定义极角排序规则的函数对象
	Point p0; //以p0为原点进行极角排序,极角相同的，离p0近算小
	Comp(const Point & p) :p0(p.x, p.y) { }
	bool operator ()(const Point & p1, const Point & p2) const {
		int s = Sign(Cross(p1 - p0, p2 - p0));
		if (s > 0)
			return true;
		else if (s < 0)
			return false;
		else {
			if (Distance(p0, p1) < Distance(p0, p2))
				return true;
			else
				return false;
		}
	}
};

const int maxn = 55;
Point ps[maxn];

int T, N;
int x, y, z;

void solve() {
	vector<int> ans;
	sort(ps, ps + N);
	ans.push_back(ps[0].id);
	for (int p = 1; p < N; p++) {
		sort(ps + p, ps + N, Comp(ps[p - 1]));
		ans.push_back(ps[p].id);
	}
	cout << ans.size();
	for (int i = 0; i < ans.size(); i++) cout << " " << ans[i];
	cout << endl;
}

int main() {
	cin >> T;
	while (T--) {
		cin >> N;
		for (int i = 0; i < N; i++) {
			cin >> z >> x >> y;
			ps[i] = Point(x, y, z);
		}
		solve();
	}
}
```

##### Line of Sight

区间合并很有趣。

```c++
#include <iostream>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <vector>
using namespace std;

const double EPS=1e-8;
const int MAXN=200;

int dblcmp(double x){
    return fabs(x)<EPS?0:(x>0?1:-1);
}

struct Point;
typedef Point Vector;

struct Point{
    double x,y;
    Point(){}
    Point(double _x,double _y):x(_x),y(_y){}
    Vector operator -(Point p){
        return Vector(x-p.x,y-p.y);
    }
    double operator ^(Vector v){ //叉积
        return x*v.y-y*v.x;
    }
};

struct Region{ //遮挡区域
    double lx,rx;
    Region(){}
    Region(double _lx,double _rx):lx(_lx),rx(_rx){}
};

struct Line{
    Point p1,p2;
    Line(){}
    Line(Point _p1,Point _p2):p1(_p1),p2(_p2){}
    bool input(){
        double x1,x2,y;
        cin>>x1>>x2>>y;
        if(x1==0&&x2==0&&y==0) return false;
        p1=Point(x1,y); p2=Point(x2,y);
        return true;
    }
    double intersectionPointX(Line l){ //定点分比法求交点(x坐标)
        double s1=(l.p1-p1)^(p2-p1);
        double s2=(l.p2-p1)^(p2-p1);
        return (s2*l.p1.x-s1*l.p2.x)/(s2-s1);
    }
    double length(){
        return p2.x-p1.x;
    }
};


Line house,property;
vector<Line> obstacles;
vector<Region> regions;

bool compare(const Region& r1,const Region& r2){
    return r1.lx<r2.lx;
}

void solve(){
    int m=obstacles.size();
    if(!m){
        cout<<fixed<<setprecision(2)<<property.length()<<endl;
        return ;
    }
    regions.clear();
    for(int i=0;i<m;++i){ //求每个障碍线的遮挡区域
        Point left=obstacles[i].p1;
        Point right=obstacles[i].p2;
        double x1=property.intersectionPointX(Line(left,house.p2)); //左端点
        double x2=property.intersectionPointX(Line(right,house.p1)); //右端点
        regions.push_back(Region(max(property.p1.x,x1),min(property.p2.x,x2)));
    }
    sort(regions.begin(),regions.end(),compare); //排序，方便后面合并相连的遮挡区域
    vector<Region> r;
    Region region=Region(regions[0].lx,regions[0].rx);
    for(int i=1;i<m;++i){ //合并相连的遮挡区域
        if(region.rx>regions[i].lx-EPS)
            region.rx=max(region.rx,regions[i].rx);
        else{
            if(region.rx>region.lx) //一开始没有判断这个，wa了好久。。
                r.push_back(region);
            region=Region(regions[i].lx,regions[i].rx);
        }
    }
    if(region.rx>region.lx)
        r.push_back(region);
    m=r.size();
    double ans=r[0].lx-property.p1.x;
    for(int i=0;i<m-1;++i){
        ans=max(ans,r[i+1].lx-r[i].rx);
    }
    ans=max(ans,property.p2.x-r[m-1].rx);
    if(ans<EPS) cout<<"No View"<<endl;
    else cout<<fixed<<setprecision(2)<<ans<<endl;
}

int main(){
    while(house.input()){
        property.input();
        int n;
        cin>>n;
        Line obstacle;
        obstacles.clear();
        for(int i=0;i<n;++i){
            obstacle.input();
            if(obstacle.p1.y>house.p1.y-EPS||obstacle.p1.y<property.p1.y+EPS)
                continue; //排除不在房屋和观光线之间的
            obstacles.push_back(obstacle);
        }
        solve();
    }
    return 0;
}
```

##### The Fortified Forest

凸包+状压。

```c++
#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <iomanip>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <vector>
#include <stack>
using namespace std;

const int maxn = 16;
int N;

struct tree {
	int x, y, v, l;
	tree() {}
	tree(int x, int y, int v, int l) :x(x), y(y), v(v), l(l) {}
} f[maxn];

#define EPS 1e-3
#define inf 0x3f3f3f3f

int Sign(double x) { // 判断 x 是大于0,等于0还是小于0 
	return fabs(x) < EPS ? 0 : x > 0 ? 1 : -1;
}

struct Point {
	double x, y;
	Point(double xx = 0, double yy = 0) :x(xx), y(yy) { }
	Point operator-(const Point & p) const {
		return Point(x - p.x, y - p.y);
	}
	bool operator <(const Point & p) const {
		if (y < p.y)
			return true;
		else if (y > p.y)
			return false;
		else
			return x < p.x;
	}
};

typedef Point Vector;

double Cross(const Vector & v1, const Vector & v2)
{//叉积
	return v1.x * v2.y - v2.x * v1.y;
}

double Distance(const Point & p1, const Point & p2)
{
	return sqrt((p1.x - p2.x)*(p1.x - p2.x) + (p1.y - p2.y)*(p1.y - p2.y));
}

struct Comp { //用来定义极角排序规则的函数对象
	Point p0; //以p0为原点进行极角排序,极角相同的，离p0近算小
	Comp(const Point & p) :p0(p.x, p.y) { }
	bool operator ()(const Point & p1, const Point & p2) const {
		int s = Sign(Cross(p1 - p0, p2 - p0));
		if (s > 0)
			return true;
		else if (s < 0)
			return false;
		else {
			if (Distance(p0, p1) < Distance(p0, p2))
				return true;
			else
				return false;
		}
	}
};

// return wood needed
double Graham(vector<Point> & points, vector<Point> & stack) {
	//points是点集合
	if (points.size() == 1) return 0;
	if (points.size() == 2) return Distance(points[0], points[1]) * 2;
	stack.clear();
	//先按坐标排序，最左下的放到points[0] 
	sort(points.begin(), points.end());
	//以points[0] 为原点进行极角排序 
	sort(points.begin() + 1, points.end(), Comp(points[0]));
	stack.push_back(points[0]);
	stack.push_back(points[1]);
	stack.push_back(points[2]);
	for (int i = 3; i < points.size(); ++i) {
		while (true) {
			Point p2 = *(stack.end() - 1);
			Point p1 = *(stack.end() - 2);
			if (Sign(Cross(p2 - p1, points[i] - p2) <= 0))
				//p2->points[i]没有向左转，就让p2出栈 
				stack.pop_back();
			else
				break;
		}
		stack.push_back(points[i]);
	}
	double res = 0;
	for (int i = 0; i < stack.size(); i++) {
		res += Distance(stack[i], stack[(i + 1) % stack.size()]);
	}
	return res;
}

int x, y, v, l;
int main() {
	int cas = 0;
	while (cin >> N && N) {
		if (cas) cout << endl;
		cas++;

		for (int i = 0; i < N; i++) {
			cin >> x >> y >> v >> l;
			f[i] = tree(x, y, v, l);
		}

		int mnv = inf;
		double exw = 0;
		vector<int> mncuts;

		int mx_st = (1 << N) - 1;
		for (int st = 1; st < mx_st; st++) {
			vector<Point> poly;
			vector<int> cuts;
			int tv = 0, tl = 0;
			for (int i = 0; i < N; i++) {
				if (st&(1 << i)) {
					tv += f[i].v;
					tl += f[i].l;
					cuts.push_back(i);
				}
				else {
					poly.push_back(Point(f[i].x, f[i].y));
				}
			}
			vector<Point> hull;
			double s = Graham(poly, hull);
			if (tl > s && tv < mnv) {
				mnv = tv;
				exw = tl - s;
				mncuts = cuts;
			}
		}
		cout << "Forest " << cas << endl;
		cout << "Cut these trees:";
		for (int i = 0; i < mncuts.size(); i++) cout << " " << mncuts[i] + 1;
		cout << endl;
		cout << "Extra wood: " << fixed << setprecision(2) << exw << endl;
	}
}
```







### Suffix Array

##### Substrings

多字串最长公共子串背板。

```c++
#define _CRT_SECURE_NO_WARNINGS
#include <cstdio>
#include <cstring>
#include <string>
#include <iostream>
#include <algorithm>
#include <string>

using namespace std;

const int maxn = 10005;
int N;
int wa[maxn], wb[maxn], wc[maxn], wd[maxn];
int sa[maxn];

// n: strlen(s), m: 128(ASCII unique chars)
void buildSA(int* s, int n, int* sa, int m = 255) {
	int i, j, p, *pm = wa, *k2sa = wb, *t;
	// k1 radix sort
	for (i = 0; i < m; i++) wd[i] = 0;
	for (i = 0; i < n; i++) wd[pm[i] = s[i]]++;
	for (i = 1; i < m; i++) wd[i] += wd[i - 1];
	for (i = n - 1; i >= 0; i--) sa[--wd[pm[i]]] = i;
	// loops j->2j
	for (j = p = 1; p < n; j <<= 1, m = p) {
		// generate k2sa
		for (p = 0, i = n - j; i < n; i++) k2sa[p++] = i; // null k2
		for (i = 0; i < n; i++) if (sa[i] >= j) k2sa[p++] = sa[i] - j;
		// k2 radix sort
		for (i = 0; i < m; i++) wd[i] = 0;
		for (i = 0; i < n; i++) wd[wc[i] = pm[k2sa[i]]]++;
		for (i = 1; i < m; i++) wd[i] += wd[i - 1];
		for (i = n - 1; i >= 0; i--) sa[--wd[wc[i]]] = k2sa[i];
		// update pm
		for (t = pm, pm = k2sa, k2sa = t, pm[sa[0]] = 0, p = i = 1; i < n; i++) {
			int a = sa[i - 1], b = sa[i];
			if (k2sa[a] == k2sa[b] && k2sa[a + j] == k2sa[b + j])
				pm[sa[i]] = p - 1;
			else pm[sa[i]] = p++;
		}
	}
}

int Rank[maxn], height[maxn];
void buildHeight(int* str, int n) {
	int i, j, k;
	for (i = 0; i < n; i++) Rank[sa[i]] = i;
	for (i = k = 0; i < n; height[Rank[i++]] = k)
		for (k ? k-- : 0, j = sa[Rank[i] - 1];
			str[i + k] == str[j + k];
			k++);
}


int s[maxn], id[maxn];
int vis[105];

int T, M;
string a;

bool check(int mid) {
	memset(vis, 0, sizeof(vis));
	int cnt = 0;
	for (int i = 1; i <= N; i++) {
		if (height[i] >= mid) {
			for (int j = 1; j <= M; j++) {
				if (id[sa[i]] == j) {
					if (!vis[j]) cnt++;
					vis[j] = 1;
				}
				if (id[sa[i-1]] == j) {
					if (!vis[j]) cnt++;
					vis[j] = 1;
				}
			}
		}
		else {
			if (cnt >= M) return true;
			cnt = 0;
			memset(vis, 0, sizeof(vis));
		}
	}
	if (cnt >= M) return true;
	return false;
}

int main() {
	cin >> T;
	while (T--) {
		int l = 0, offset = 1;
		cin >> M;
		for (int i = 1; i <= M; i++) {
			cin >> a;
			for (int j = 0; j < a.size(); j++) id[l] = i, s[l++] = a[j];
			s[l++] = '#' + offset++;
			for (int j = a.size() - 1; j >= 0; j--) id[l] = i, s[l++] = a[j];
			s[l++] = '#' + offset++;
		}
		N = l;
		buildSA(s, N, sa);
		buildHeight(s, N);
		int left = 0, right = N;
		while (left < right) {
			int mid = (left + right) / 2;
			if (check(mid)) left = mid + 1;
			else right = mid;
		}
		cout << left - 1 << endl;
	}
}
```





