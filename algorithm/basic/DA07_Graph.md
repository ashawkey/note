# Graph

### Definition

$$
G = (V, E)
$$

We only considers **simple graphs** that have **No self-Connection, No parallel edge.**

* Complete graph: contains all possible edges, $E = \frac {V(V-1)}{2}$

* neighbors: vertices connected by an edge.
* degree: in and out. leaf's degree is 0.
* subgraph: $V' \in V, E' \in E$, and all vertices connected by $E'$ are in $V'$.
* path: $V_p \rightarrow ... \rightarrow V_q$

* simple path: 除了起点终点，其他顶点无重复。
* Simple cycle：起点终点相同的简单路径。

* DAG：Directed Acyclic Graph. 有向无环图。
* 有根图：有向图中，从顶点V可以抵达其他所有顶点，则称之为根。
* 连通图：无向图任意两个顶点连通，则称为连通图。
* 强连通图：有向图任意两个顶点强联通，则成为强连通图。
* 连通分量：**无向图的最大连通子图**。
* 强连通分量：有向图的最大连通子图。
* 网络：带权连通图。
* 自由树：无回路的连通无向图，且$E=V-1$.

### Storage

* 邻接矩阵 Adjacency Matrix

  空间代价$O(|V|^2)$. （可以稀疏矩阵优化）

  * 无向图：对称。
  * 有向图：不一定对称。

* 邻接表

  最常使用的结构，空间代价小，但失去了邻接矩阵的性质。

  * 无向图空间代价$O(|V|+2|E|)$.

  * 有向图空间代价$O(|V|+|E|)$，只记录入边表或出边表。

* 十字链表

  * 顶点表，边链表

### Traversal

* 问题：非连通，回路。

  使用VIS标记来避免回路。

* DFS

  时间复杂度与存储结构的空间复杂度同数量级。

* BFS

  时间复杂度与存储结构的空间复杂度同数量级。

* 拓扑排序

  A **Linear order** that can finish the task as well as satisfying all pre request. 

  **Usually Not Unique.**

  Can only be performed on a **DAG**. （有拓扑排序等价于是DAG）


### Shortest Path

* Dijkstra

  Single source, Directed or Undirected, **Non-negative edge weight**, Greedy.

  $O((V+E)logE)$ or $O(V^2 + E)$

* Floyd

  All pairs, Dynamic programming.

  $O(V^3)$


### Minimum cost Spanning Tree (MST)

MST may be not unique, but the min cost is the same.

* Prim

   $O((V+E)logV)$.

  The structure is very similar to Dijkstra, but here we find the shortest edge to current tree, instead of the shortest point to source point.

  Need to assign the Root.

* Kruskal

  $O(ElogE)$

  Use disjoint set to merge points from shortest edge.

### Toy Graph

```c++
#include <iostream>
#include <cstring>
#include <set>
#include <queue>
#include <vector>

using namespace std;

const int maxv = 100;
const int maxw = 0x3f3f3f3f; // if use floyd, 2*maxw must not overflow.
int V;

struct node {
	int ind, outd;
	int data;
	node() {
		ind = 0;
		outd = 0;
		data = 0;
	}
} nodes[maxv];

struct edge {
	int f, t, w;
	edge() {}
	edge(int f, int t, int w) :f(f), t(t), w(w) {}
    bool operator < (const edge& b) const { return w > b.w; }
};

vector<edge> G[maxv];
int vis[maxv];

void insert_edge(int a, int b, int w) {
	G[a].push_back(edge(a, b, w));
	//G[b].push_back(edge(b, a, w));
	nodes[a].outd++;
	nodes[b].ind++;
}

void visit(int n) {
	cout << "visit:" << n << endl;
}

// clear vis first
void dfs(int rt) {
	if (vis[rt]) return;
	vis[rt] = 1;
	int len = G[rt].size();
	for (int i = 0; i < len; i++) {
		dfs(G[rt][i].t);
	}
	visit(rt);
}

void bfs(int rt) {
	memset(vis, 0, sizeof(vis));
	queue<int> q;
	q.push(rt);
	while (!q.empty()) {
		int p = q.front(); q.pop();
		if (vis[p]) return;
		vis[p] = 1;
		visit(p);
		int len = G[p].size();
		for (int i = 0; i < len; i++) {
			q.push(G[p][i].t);
		}
	}
}

// output one of the toposorts.
// 有向图判断有无环的方法之一。
void toposort_bfs() {
	memset(vis, 0, sizeof(vis));
	queue<int> q;
	for (int i = 0; i < V; i++)
		if (nodes[i].ind == 0)
			q.push(i);
	while (!q.empty()) {
		int v = q.front(); q.pop();
		visit(v);
		vis[v] = 1;
		int len = G[v].size();
		for (int i = 0; i < len; i++) {
			nodes[G[v][i].t].ind--;
			if (nodes[G[v][i].t].ind == 0) q.push(G[v][i].t);
		}
		for (int i = 0; i < maxv; i++) {
			if (!vis[i]) {
				cout << "Loop Detected" << endl;
				break;
			}
		}
	}
}

// can't detect loop! 
vector<int> res; // reverse-ordered
void _toposort_dfs(int n) {
	vis[n] = 1;
	for (int i = 0; i < G[n].size(); i++) {
		int m = G[n][i].t;
		if (!vis[m])
			_toposort_dfs(m);
	}
	res.push_back(n);
}

void toposort_dfs() {
	memset(vis, 0, sizeof(vis));
	for (int i = 0; i < V; i++)
		if (!vis[i])
			_toposort_dfs(i);
	for (int i = res.size() - 1; i >= 0; i--)
		visit(res[i]);
}


int dist[maxv], parent[maxv];

bool cmp(int a, int b) {
	return dist[a] > dist[b];  // min heap should use greater<>
}

// O((V+E)logE)，单源最短路，求起点s到任意其他点t的最短距离dist[t]
/*
实现方式：最小堆+不删除旧值。
其他方式：
	扫描dist来寻找下一个最小元素。 O(V^2 + E)
	最小堆+删除旧元素：不方便用std优先队列实现。O(VlogV)？
不连接的点距离为maxw。可以用来检查图的连接。
*/
void dijkstra(int s) {
	for (int i = 0; i < V; i++) {
		vis[i] = 0;
		parent[i] = -1;
		dist[i] = maxw;
	}
	dist[s] = 0;
	priority_queue<int, vector<int>, bool(*)(int,int)> que(cmp);  // note the use of function cmp
	que.push(s);
	while (!que.empty()) {
		int u = que.top(); que.pop();
		if (vis[u]) continue; // prevent old element rescanning, necessary.
		vis[u] = 1;
		for (int i = 0; i < G[u].size(); i++) {
			int v = G[u][i].t;
			if (vis[v]) continue;  // this can be deleted. Simple pruning.
			if (dist[v] > dist[u] + G[u][i].w) {
				dist[v] = dist[u] + G[u][i].w;
				parent[v] = u;
				que.push(v);
			}
		}
	}
}

// Simplified Version !!!
int d[maxn];
#define P pair<int,int>

void dijkstra(int s) {
	memset(d, 0x3f, sizeof(d));
	priority_queue<P, vector<P>, greater<P>> q; // use pair<int,int> 
	q.push(P(0, s));
	d[s] = 0;
	while (!q.empty()) {
		P p = q.top(); q.pop();
		int u = p.second;
		if (d[u] < p.first) continue; // outd records
		for (int i = 0; i < G[u].size(); i++) {
			edge& e = G[u][i];
			int v = e.t;
			if (d[v] > d[u] + e.w) {
				d[v] = d[u] + e.w;
				q.push(P(d[v], v));
			}
		}
	}
}

// a STL version
vector<vector<pair<int, int>>> G(n+1);
for (auto& p: times) {
    G[p[0]].push_back({p[1], p[2]}); // p = {f, t, w}
}
vector<int> d(n+1, 0x7fffffff);
priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> q;
q.push({0, k});
d[k] = 0;
while (!q.empty()) {
    auto p = q.top(); q.pop();
    int u = p.second;
    //cout << "dij " << u << " " << d[u] << endl;
    if (d[u] < p.first) continue;
    for (auto& [v, w]: G[u]) {
        if (d[v] > d[u] + w) {
            d[v] = d[u] + w;
            q.push({d[v], v});
        }
    }
}

// floyd
int dist2[maxv][maxv], parent2[maxv][maxv];
void floyd() {
    memset(dist2, maxw, sizeof(dist2));
    memset(parent2, -1, sizeof(parent2));
    // init the graph's edge
	for (int u = 0; u < V; u++) {
        dist2[u][u] = 0;
		for (int i = 0; i < G[u].size(); i++) {
			int v = G[u][i].t;
			dist2[u][v] = G[u][i].w;
			parent2[u][v] = u;
		}
	}
    // O(n^3), the order is i->u->j
	for (int u = 0; u < V; u++) {
		for (int i = 0; i < V; i++) {
			for (int j = 0; j < V; j++) {
				if (dist2[i][j] != maxw && dist2[i][j] > dist2[i][u] + dist2[u][j]) {
					dist2[i][j] = dist2[i][u] + dist2[u][j];
					parent2[i][j] = parent2[u][j];
				}
			}
		}
	}
}

vector<edge> MST;
// only for connected graph, so check vis later for whether it's connected.
// O((V+E)logE), dense graph.
void prim(int s) {
    MST.clear();
    memset(vis, 0, sizeof(vis));
    vis[s] = 1;
	priority_queue<edge> que;  
    for(int i=0; i<G[s].size(); i++) que.push(G[s][i]);
	while (!que.empty()) {
		edge e = que.top(); que.pop();
        if (vis[e.t]) continue; // prevent out of date element
        vis[e.t] = 1;
        MST.push_back(e);
        for(int i=0; i<G[e.t].size(); i++){
            edge ee = G[e.t][i];
            if(!vis[ee.t]) que.push(ee);
        }
	}
}

// disjoint set for kruskal
int par[maxn];
void init(int n) {
	for (int i = 0; i <= n; i++) par[i] = i;
}
int getpar(int x) {
	if (par[x] == x) return x;
	return par[x] = getpar(par[x]);
}
void merge(int x, int y) {
	par[getpar(x)] = getpar(y);
}
bool query(int a, int b){
    return getpar(a) == getpar(b);
}


// kruskal MST, O(ElogE), sparse graph.
void kruskal(){
    MST.clear();
    priority_queue<edge> que;
    // push all edges
    for(int i=0; i<V; i++)
        for(int j=0; j<G[i].size(); j++)
            que.push(G[i][j]);
    int n = V;
    init(n);
    // merge from shortest edge until only one class left.
    while(n>1){
        edge e = que.top(); que.pop();
        if(!query(e.f, e.t)){
            merge(e.f, e.t);
            MST.push_back(e);
            n--;
        }
    }
}


int main() {
	V = 4;
	insert_edge(0, 2, 1);
	insert_edge(0, 1, 3);
	insert_edge(0, 3, 1);
	insert_edge(1, 3, 6);
	insert_edge(2, 3, 2);
    kruskal();
    for(int i=0; i<V-1; i++){
        cout<<MST[i].f<<" "<<MST[i].t<<" "<<MST[i].w<<endl;
    }
}
```



### Examples

* ウサギと桜

  弗洛伊德+路径找回。

```c++
#include <iostream>
#include <sstream>
#include <string>
#include <cstring>
#include <algorithm>
#include <vector>
#include <queue>
#include <map>

using namespace std;

const int maxv = 30;
int V, E;

struct edge {
	int f, t, w;
	edge() {}
	edge(int f, int t, int w) :f(f), t(t), w(w) {}
	bool operator < (const edge& b)const {
		return w > b.w;
	}
};

vector<edge> edges;
vector<int> G[maxv];

void insert_edge(int f, int t, int w) {
	edges.push_back(edge(f,t,w));
	int s = edges.size() - 1;
	G[f].push_back(s);
}

int dist[maxv][maxv];
int parent[maxv][maxv];

void floyd() {
	memset(dist, 0x3f, sizeof(dist));
	memset(parent, -1, sizeof(parent));
	for (int i = 1; i <= V; i++) {
		dist[i][i] = 0; // self-connection
		for (int j = 0; j < G[i].size(); j++) {
			int e = G[i][j];
			int v = edges[e].t;
			dist[i][v] = edges[e].w;
			parent[i][v] = e;
		}
	}
	for (int u = 1; u <= V; u++) {
		for (int i = 1; i <= V; i++) {
			for (int j = 1; j <= V; j++) {
				if (dist[i][j] > dist[i][u] + dist[u][j]) {
					dist[i][j] = dist[i][u] + dist[u][j];
					parent[i][j] = parent[u][j];
				}
			}
		}
	}
}

map<string, int> m;
map<int, string> mm;

int main() {
	cin >> V;
	string s, ss;
	int w;
	for (int i = 1; i <= V; i++) {
		cin >> s;
		m[s] = i;
		mm[i] = s;
	}
	cin >> E;
	for (int i = 0; i < E; i++) {
		cin >> s >> ss >> w;
		insert_edge(m[s], m[ss], w);
		insert_edge(m[ss], m[s], w);
	}
	floyd();
	int R;
	vector<int> path;
	cin >> R;
	for (int i = 0; i < R; i++) {
		cin >> s >> ss;
		cout << s;
		int u = m[s], v = m[ss];
		path.clear();
		while (u != v) {
			//cout << mm[u] << " to " << mm[v] << endl;
			int e = parent[u][v];
			path.push_back(e);
			v = edges[e].f;
		}
		for (int i = path.size() - 1; i >= 0; i--) {
			edge e = edges[path[i]];
			cout << "->(" << e.w << ")->" << mm[e.t];
		}
		cout << endl;
	}
}
```

* 地震之后

  有向图最小树形图，**朱刘算法。**

  ```c++
  #include <iostream>
  #include <sstream>
  #include <string>
  #include <cstring>
  #include <algorithm>
  #include <vector>
  #include <queue>
  #include <map>
  
  using namespace std;
  
  const int maxv = 105;
  const int inf = 0x3f3f3f3f;
  int V, E;
  
  struct node {
  	double x, y;
  } nodes[maxv];
  
  double dist(const node& a, const node& b) {
  	return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2));
  }
  
  struct edge {
  	int f, t;
  	double w;
  	edge() {}
  	edge(int f, int t, double w) :f(f), t(t), w(w) {}
  	bool operator < (const edge& b)const {
  		return w > b.w;
  	}
  };
  
  vector<edge> edges;
  vector<int> G[maxv];
  
  void init() {
  	for (int i = 0; i < V; i++) G[i].clear();
  	edges.clear();
  }
  
  void insert_edge(int f, int t, double w) {
  	edges.push_back(edge(f,t,w));
  	int s = edges.size() - 1;
  	G[f].push_back(s);
  }
  
  int x, y;
  double in[maxv];
  int vis[maxv], pre[maxv], id[maxv];
  
  // v starts from 0
  double zhuliu(int s) {
  	double res = 0;
  	int v = V; // localize
  	while (true) {
  		// in[] is the longest in edge's weight
  		for (int i = 0; i < v; i++) in[i] = inf;
  		// calc in[]
  		for (int i = 0; i < edges.size(); i++) {
  			edge& e = edges[i];
  			if (e.f != e.t && e.w < in[e.t]) {
  				pre[e.t] = e.f;
  				in[e.t] = e.w;
  			}
  		}
  		// check non-connectivity
  		for (int i = 0; i < v; i++)
  			if (s != i && in[i] == inf)
  				return -1;
  		// id[] is scc id
  		memset(id, -1, sizeof(id));
  		memset(vis, -1, sizeof(vis));
  		// set root
  		in[s] = 0;
  		// count scc
  		int scc = 0;
  		// 
  		for (int i = 0; i < v; i++) {
  			res += in[i];
  			int v = i;
  			// find scc 
  			while (vis[v] != i && id[v] == -1 && v != s) {
  				vis[v] = i;
  				v = pre[v];
  			}
  			// assign scc id
  			if (v != s && id[v] == -1) {
  				for (int u = pre[v]; u != v; u = pre[u]) id[u] = scc;
  				id[v] = scc++;
  			}
  		}
  		// no scc, MST built
  		if (scc == 0) break;
  		// non-connected single points are also scc
  		for (int i = 0; i < v; i++) 
  			if (id[i] == -1)
  				id[i] = scc++;
  		// shrink scc
  		for (int i = 0; i < edges.size(); i++) {
  			edge& e = edges[i];
  			int v = e.t;
  			e.f = id[e.f];
  			e.t = id[e.t];
  			if (e.f != e.t) e.w -= in[v];
  		}
  		v = scc;
  		s = id[s];
  	}
  	return res;
  }
  
  int main() {
  	while (cin >> V >> E) {
  		init();
  		for (int i = 0; i < V; i++) {
  			cin >> x >> y;
  			nodes[i].x = x;
  			nodes[i].y = y;
  		}
  		for (int i = 0; i < E; i++) {
  			cin >> x >> y;
  			x--; y--;
  			insert_edge(x, y, dist(nodes[x], nodes[y]));
  		}
  		double ans = zhuliu(0);
  		if (ans == -1) printf("NO\n");
  		else printf("%.2f\n", ans);
  	}
  }
  ```
