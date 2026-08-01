# 计算几何 (Computational Geometry)

## 核心技巧

**三点共线判断用乘法代替除法**，避免分母为零/精度问题：

```cpp
// WRONG: (a[0]-b[0]) / (a[1]-b[1]) != (a[0]-c[0]) / (a[1]-c[1]);
// OK:    叉积为零即共线
(a[0] - b[0]) * (a[1] - c[1]) != (a[0] - c[0]) * (a[1] - b[1]);
```

## 基础模板

点/向量运算、线段相交、点与多边形关系、凸包（Graham 扫描）：

```c++
const double eps = 1e-6;
int epssgn(double x) {
    if (fabs(x) < eps) return 0;
    else return x < 0 ? -1 : 1;
}

struct Point {
    double x, y;
    Point(double a = 0, double b = 0) { x = a; y = b; }
    bool operator < (const Point& b) const {
        if (x == b.x) return y < b.y;
        return x < b.x;
    }
};
typedef Point Vector;

Vector operator + (Point a, Point b) { return Vector(a.x + b.x, a.y + b.y); }
Vector operator - (Point a, Point b) { return Vector(a.x - b.x, a.y - b.y); }
Vector operator * (Point a, double k) { return Vector(a.x * k, a.y * k); }
Vector operator / (Point a, double k) { return Vector(a.x / k, a.y / k); }
double operator * (Vector a, Vector b) { return a.x * b.x + a.y * b.y; }   // 点积
double operator ^ (Vector a, Vector b) { return a.x * b.y - b.x * a.y; }   // 叉积
double len(Vector a) { return sqrt(a.x * a.x + a.y * a.y); }

struct Line {
    Point a, b;
    Line() {}
    Line(Point a, Point b) :a(a), b(b) {}
};

int SideOfLine(Point p, Point a, Point b) {  // 点与直线的关系：-1/0/1
    double res = (b - a) ^ (p - a);
    return epssgn(res);
}

bool PointInSeg(Point p, Line l) {  // 点是否在线段上
    double tmp = (l.a - p) ^ (l.a - l.b);
    if (epssgn(tmp) != 0) return false;
    return epssgn(min(l.a.x, l.b.x) - p.x) <= 0 &&
           epssgn(p.x - max(l.a.x, l.b.x)) <= 0 &&
           epssgn(min(l.a.y, l.b.y) - p.y) <= 0 &&
           epssgn(p.y - max(l.a.y, l.b.y)) <= 0;
}

Point LineCrossLine(Line l, Line m) {  // 两直线交点
    Point a = l.b - l.a, b = m.b - m.a, s = m.a - l.a;
    return l.a + a * ((b ^ s) / (b ^ a));
}
```

**凸包 Graham 扫描**（先按 x/y 排序，再按极角排序，维护单调栈）：

```c++
// polar angle order comparator
struct comp {
    Point p0;
    comp(const Point& p) :p0(p) {}
    bool operator ()(const Point& p1, const Point& p2) const {
        int s = epssgn((p1 - p0) ^ (p2 - p0));
        if (s == 0) return dist(p0, p1) < dist(p0, p2);
        else return s > 0;
    }
};

int graham(vector<Point>& ps, vector<Point>& stk) {
    if (ps.size() < 3) return 0;
    stk.clear();
    sort(ps.begin(), ps.end());
    sort(ps.begin() + 1, ps.end(), comp(ps[0]));
    stk.push_back(ps[0]);
    stk.push_back(ps[1]);
    stk.push_back(ps[2]);
    for (int i = 3; i < ps.size(); i++) {
        while (true) {
            Point p2 = *(stk.end() - 1);
            Point p1 = *(stk.end() - 2);
            if (epssgn((p2 - p1) ^ (ps[i] - p2)) <= 0) stk.pop_back();
            else break;
        }
        stk.push_back(ps[i]);
    }
    return stk.size();
}
```

## 例题

### [POJ 1228 Grandpa's Estate](http://poj.org/problem?id=1228)（凸包唯一确定）

给定若干点（全部位于某凸包边上），判断凸包是否**唯一确定**——即凸包每条边上有至少 3 个给定点（否则该边可以"松动"）。做法：求凸包后检查每条边是否还有第三个给定点（共线点）。

### [POJ 3525 Most Distant Point from the Sea](http://poj.org/problem?id=3525)（半平面交 + 二分）

求多边形内离边界最远的点（最大内接圆半径）。**二分距离 d，把每条边向内平移 d，若半平面交非空则可行**——"收缩后仍非空" ⇔ 存在到海边距离 ≥ d 的点。

### [POJ 1039 Pipe](http://poj.org/problem?id=1039)（枚举直线 + 线段相交）

折线管道中光线能到达的最远 x 坐标。枚举一条**经过两个顶点（上/下壁）**的直线，用 `LineCrossSeg` 检查它能穿过多少段管道，并计算最远交点。
