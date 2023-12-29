### ACM style I/O


### scanf/printf

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


### iostream

Acceleration:

```cpp
#include <iostream>

using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(0);
}
```


`cin >> x` are separated by spaces or line breaks. types are auto-casted.

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


If you want to get **a whole input line (may contain spaces)** to string:

```cpp
#include <iostream>
#include <string>

using namespace std;

int main() {
	string s;
	getline(cin, s); // NOT cin.getline(c_str);
}
```


`cout` precision:

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


