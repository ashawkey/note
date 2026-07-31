# ACM 风格 I/O

## scanf/printf

Not recommended to use...

```cpp
#include <stdio.h>

int main() {
    printf("int %d, float %f, double %lf, string %s, char %c, endl \n", 42, 3.141, 3.141, "string", 'c');

    // float precision
    printf("float %.2f \n", 3.141); // 3.14
}
```

```cpp
#include <stdio.h>

int main() {
    int x;
    scanf("%d", &x);

    int x, y;
    scanf("%d%d", &x, &y);

    float f;
    scanf("%f", &f);

    char c;
    scanf("%c", &c); // this will scan spaces too!
    scanf(" %c", &c); // this will ignore all leading spaces...
}
```

## iostream

加速：

```cpp
#include <iostream>

using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(0);
}
```

`cin >> x` 以空格或换行分隔，类型自动转换。

```cpp
#include <iostream>

using namespace std;

const static int N = 1000;
int xs[N];

int main() {
    // input until EOF
    int x;
    int i = 0;
    while (cin >> x) {
        xs[i++] = x;
    }
}
```

```cpp
#include <iostream>
#include <string>

using namespace std;

int main() {
    string s;
    cin >> s; // only get one word
}
```

读取**一整行（可能含空格）**到 string：

```cpp
#include <iostream>
#include <string>
#include <sstream>

using namespace std;

int main() {
    string s;
    getline(cin, s); // NOT cin.getline(c_str);

    // if s is a single int
    int x = stoi(s);

    // if s is a list of int (unknown length), to extract them:
    istringstream iss(s);
    int x;
    while (iss >> x) {
        cout << x << " ";
    }
}
```

输入共 N+1 行，第一行是 N，之后 N 行是长度未知的整数：

```cpp
int N;
string s;
getline(cin, s); // cannot cin >> N; cin won't consume the endline, and will cause the next getline to get an empty string!
N = stoi(s);
for (int i = 0; i < N; i++) {
    // get a line of ints
    getline(cin, s);
    istringstream iss(s);
    int x;
    while (iss >> x) {
        // ...
    }
}
```

`cout` 精度：

```cpp
#include <iostream>
#include <iomanip>

using namespace std;

int main() {
    double f = 3.1415926;
    // default, in total 5 effective numbers.
    cout << setprecision(5) << f << endl; // 3.1416

    // fixed-point float notation, always 5 numbers after floating-point.
    cout << fixed << setprecision(5) << f << endl; // 3.14159
}
```
