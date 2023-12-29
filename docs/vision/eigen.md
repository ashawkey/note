# Eigen 

A matrix/vector arithmetic library for c/c++.


### Install

download the source at [homepage](https://eigen.tuxfamily.org/index.php?title=Main_Page).

to use it, include it in compiling : `g++ -I /path/to/eigen source.cpp`

or simply symbol link it to `\usr\local\include`.


> Troubleshooting: `Eigen::all, Eigen::seq` is not a member of Eigen.
>
> These operations belong to `dev` branch of Eigen, as you can see from the documentation:
>
> * dev: https://eigen.tuxfamily.org/dox-devel/group__TutorialSlicingIndexing.html
> * stable: https://eigen.tuxfamily.org/dox/ (there is no page about slicing)
>
> To use the `dev` branch, download the source at https://eigen.tuxfamily.org/dox/


### Examples

simple matrix

```c++
#include <iostream>
#include <Eigen/Dense>
 
using Eigen::MatrixXd;
 
int main()
{
  MatrixXd m(2,2);
  m(0,0) = 3;
  m(1,0) = 2.5;
  m(0,1) = -1;
  m(1,1) = m(1,0) + m(0,1);
  std::cout << m << std::endl;
}
```

matrix vector multiplication

```c++
#include <iostream>
#include <Eigen/Dense>
 
using namespace Eigen;
using namespace std;
 
int main()
{
  Matrix3d m = Matrix3d::Random();
  m = (m + Matrix3d::Constant(1.2)) * 50; 
  cout << "m =" << endl << m << endl; // [3, 3]
  Vector3d v(1,2,3); // [3, 1]
  
  cout << "m * v =" << endl << m * v << endl; // [3, 1]
}
```


### Matrix

all matrices and vectors are instances of the `Matrix` class:

```c++
Matrix<// the first 3 are necessary 
       typename Scalar,
       int RowsAtCompileTime,
       int ColsAtCompileTime,
       // defaulf 0 means ColMajor, use `RowMajor` to change to row-major storage order.
       int Options = 0, 
       // can be used to avoid dynamic allocating.
       int MaxRowsAtCompileTime = RowsAtCompileTime, 
       int MaxColsAtCompileTime = ColsAtCompileTime
       >
```

e.g., `typedef Matrix<float, 4, 4> Matrix4f;`

Initial coefficients are uninitialized.

To init with values: `Vector2d a(5.0, 6.0);`. (support max to `Vector4d`)

The default storage order is **Column-major**.

Built-in typedefs for `N in [2,3,4,X]` and `t in [i, f, d, cf, cd]`:

* `MatrixNt` == `Matrix<t, N, N>`
* `VectorNt` == `Matrix<t, N, 1>`
* `RowVectorNt` == `Matrix<t, 1, N>`


#### dynamic matrix

Eigen also supports **dynamic** matrix size: `typedef Matrix<double, Dynamic, Dynamic> MatrixXd;`

Initial size is 0-by-0. 

To init with a size: `MatrixXd a(4,4);`

When to use dynamic matrix: **use fixed sizes for very small sizes (<= 16) where you can, and use dynamic sizes for larger sizes or where you have to.**

```cpp
// example for [N, 3] dynamic matrix
using Points = Matrix<float, Dynamic, 3, RowMajor>;
using Triangles = Matrix<uint32_t, Dynamic, 3, RowMajor>;
```


#### Access coefficients

Coefficient accessors: `m(i, j)`

Use comma-initialization:

```c++
Matrix3f m;
m << 1, 2, 3,
     4, 5, 6,
     7, 8, 9;
```


#### Size

Size and resize: `rows(), cols(), size()`

```c++
int main()
{
  MatrixXd m(2,5);
  m.resize(4,3);
  std::cout << m.rows() << "x" << m.cols() << std::endl; // 4x3
  std::cout << m.size() << std::endl; // 12
}
```

Assignment will cause resizing too:

```c++
MatrixXf a(2,2);
MatrixXf b(3,3);
a = b; // a is now 3x3
```


### Arithmetic

#### Add/Sub:

```c++
#include <iostream>
#include <Eigen/Dense>
 
using namespace Eigen;
 
int main()
{
  Matrix2d a;
  a << 1, 2,
       3, 4;
  MatrixXd b(2,2);
  b << 2, 3,
       1, 4;
  
  // element-wise add with another matrix
  std::cout << "a + b =\n" << a + b << std::endl;
  std::cout << "a - b =\n" << a - b << std::endl;
    
  std::cout << "Doing a += b;" << std::endl;
  a += b;
  std::cout << "Now a =\n" << a << std::endl;
    
  // Matrix do not support adding/subtracting scalar !!!
  //a += 1; // CompileError, use Array for element-wise add.
    
  Vector3d v(1,2,3);
  Vector3d w(1,0,0);
  std::cout << "-v + w - v =\n" << -v + w - v << std::endl;
}
```

#### Scalar Mul/Div:

```c++
int main()
{
  Matrix2d a;
  a << 1, 2,
       3, 4;
  Vector3d v(1,2,3);
    
  std::cout << "a * 2.5 =\n" << a * 2.5 << std::endl;
  std::cout << "0.1 * v =\n" << 0.1 * v << std::endl;
    
  std::cout << "Doing v *= 2;" << std::endl;
  v *= 2;
  std::cout << "Now v =\n" << v << std::endl;
}
```

#### Transpose/Conjugate:

```c++
MatrixXcf a = MatrixXcf::Random(2,2);
cout << "Here is the matrix a\n" << a << endl;
cout << "Here is the matrix a^T\n" << a.transpose() << endl;
cout << "Here is the conjugate of a\n" << a.conjugate() << endl;
cout << "Here is the matrix a^*\n" << a.adjoint() << endl;
```

Note: `a.transpose()` will not copy the matrix, and just creates a proxy.

`b = a.transpose()` copy `a.transpose()` to `b`, and is safe.

However, `a = a.tranpose()` is WRONG and never use it. (e.g., it will cause`[[12][34]] --> [[12][24]]`)

Instead, if must, use `a = a.transposeInPlace()`.

#### Matrix Mul:

```c++
int main()
{
  Matrix2d mat;
  mat << 1, 2,
         3, 4;
  Vector2d u(-1,1), v(2,0);
  std::cout << "Here is mat*mat:\n" << mat*mat << std::endl;
  std::cout << "Here is mat*u:\n" << mat*u << std::endl;
  std::cout << "Here is u^T*mat:\n" << u.transpose()*mat << std::endl;
  std::cout << "Here is u^T*v:\n" << u.transpose()*v << std::endl;
  std::cout << "Here is u*v^T:\n" << u*v.transpose() << std::endl;
  std::cout << "Let's multiply mat by itself" << std::endl;
  mat = mat*mat;
  std::cout << "Now mat is mat:\n" << mat << std::endl;
}
```

Note: `m = m*m` is safe as a special case. It equals: `tmp = m*m; m = tmp;` 

We can avoid this default copy if we are sure there is no aliasing problem by: `a.noalias() += b * c;`

#### Dot/Cross product of vectors

```c++
int main()
{
  Vector3d v(1,2,3);
  Vector3d w(0,1,2);
 
  cout << "Dot product: " << v.dot(w) << endl; // 8

  double dp = v.transpose()*w; // automatic conversion of the inner product to a scalar
  cout << "Dot product via a matrix product: " << dp << endl;
    
  cout << "Cross product:\n" << v.cross(w) << endl; // [1, -2, 1]
}
```

#### norm and normalization of vectors

```c++
int main() {
    Vector3f v(1,2,3);
    v.normalize(); // inplace
    auto v2 = v.normalized();
    
    float n = v.norm();
    float n2 = v.squaredNorm(); // n^2
}
```


#### Basic reductions

```c++
int main()
{
  Eigen::Matrix2d mat;
  mat << 1, 2,
         3, 4;
  cout << "Here is mat.sum():       " << mat.sum()       << endl;
  cout << "Here is mat.prod():      " << mat.prod()      << endl;
  cout << "Here is mat.mean():      " << mat.mean()      << endl;
  cout << "Here is mat.minCoeff():  " << mat.minCoeff()  << endl; // max element
  cout << "Here is mat.maxCoeff():  " << mat.maxCoeff()  << endl;
  cout << "Here is mat.trace():     " << mat.trace()     << endl;
    
  // to return index of min/max element:
  Matrix3f m = Matrix3f::Random();
  std::ptrdiff_t i, j;
  float minOfM = m.minCoeff(&i,&j);
  cout << "Here is the matrix m:\n" << m << endl;
  cout << "Its minimum coefficient (" << minOfM 
       << ") is at position (" << i << "," << j << ")\n\n";
 
  RowVector4i v = RowVector4i::Random();
  int maxOfV = v.maxCoeff(&i);
  cout << "Here is the vector v: " << v << endl;
  cout << "Its maximum coefficient (" << maxOfV 
       << ") is at position " << i << endl;
}
```

#### Element-wise Mat Mul:

Since `*` is overload by Mat Mul, we have to use `cwiseProduct` for coefficient-wise (element-wise) product,

```c++
Matrix3f a;
Matrix3f b;
a.cwiseProduct(b);
```


### Array

`Array` provides general-purposed arrays, while `Matrix` provides linear-algebra purposed arrays.

The interface of `Array` are almost the same with `Matrix`, but all the operations are by default **element-wise**!

Also, `Array` supports adding scalars, e.g., `a + 4`, while `Matrix` doesn't support.

```c++
int main()
{
  // element-wise mul
  ArrayXXf a(2,2);
  ArrayXXf b(2,2);
  a << 1,2,
       3,4;
  b << 5,6,
       7,8;
  cout << "a * b = " << endl << a * b << endl;
  
  // element-wise operations
  ArrayXf a = ArrayXf::Random(5);
  a += 2;
  cout << "a =" << endl 
       << a << endl;
  cout << "a.abs() =" << endl 
       << a.abs() << endl;
  cout << "a.abs().sqrt() =" << endl 
       << a.abs().sqrt() << endl;
  cout << "a.min(a.abs().sqrt()) =" << endl 
       << a.min(a.abs().sqrt()) << endl;
}
```

Since array expressed both `Vector` and `Matrix`, the name convention is slightly different:

* for vector: `ArrayNt` == `Array<type, N, 1>`
* for matrix: `ArrayNNt` == `Array<type, N, N>`

#### Conversion between Matrix and Array

use `.array()` and `.matrix()`. 

**Conversions will not copy!** just proxies!

```c++
int main()
{
  MatrixXf m(2,2);
  MatrixXf n(2,2);
  MatrixXf result(2,2);
 
  m << 1,2,
       3,4;
  n << 5,6,
       7,8;
 
  result = m * n;
  cout << "-- Matrix m*n: --" << endl << result << endl << endl;
    
  result = m.array() * n.array();
  cout << "-- Array m*n: --" << endl << result << endl << endl;
    
  result = m.cwiseProduct(n); // same as m.array() * n.array()
  cout << "-- With cwiseProduct: --" << endl << result << endl << endl;
  
  result = (m.array() + 4).matrix() * m;
  cout << "-- Combination 1: --" << endl << result << endl << endl;
    
  result = (m.array() * n.array()).matrix() * m;
  cout << "-- Combination 2: --" << endl << result << endl << endl;
}
```

```c++
int main() 
{
  Matrix2f m;
  m << 0,0,0,0;
  cout << m << endl; // 0,0,0,0
  m.array() += 3;
  cout << m << endl; // 3,3,3,3
}
```


### Block Operations

use `.block(i,j,p,q)` for dynamic-size block or `.block<p,q>(i,j)` for a fixed-size block, starting at `(i,j)` with size `(p,q)`.

This block can be used for both l-value and r-value. 

```c++
int main()
{
  Array22f m;
  m << 1,2,
       3,4;
  Array44f a = Array44f::Constant(0.6);
  cout << "Here is the array a:" << endl << a << endl << endl;
  a.block<2,2>(1,1) = m;
  cout << "Here is now a with m copied into its central 2x2 block:" << endl << a << endl << endl;
  a.block(0,0,2,3) = a.block(2,1,2,3);
  cout << "Here is now a with bottom-right 2x3 block copied into top-left 2x3 block:" << endl << a << endl << endl;
}
```

#### special case: only one row or col

Use `.row()` and `.col()` for optimized performance.

```c++
int main()
{
  Eigen::MatrixXf m(3,3);
  m << 1,2,3,
       4,5,6,
       7,8,9;
  cout << "Here is the matrix m:" << endl << m << endl;
  cout << "2nd Row: " << m.row(1) << endl;
  m.col(2) += 3 * m.col(0);
  cout << "After adding 3 times the first column into the third column, the matrix m is:\n";
  cout << m << endl;
}
```

#### special case: corner-centered blocks

Some short names. For example:

```c++
m.topLeftCorner(p, q) == m.block(0, 0, p, q)
m.topLeftCorner<p, q>() == m.block<p, q>(0, 0)
```

Also have `bottemLeftCorner, topRightCorner, bottomRightCorner, topRows, bottomRows, leftCols, rightCols `.

#### special case: block for vectors

```c++
v.head(n) == v.head<n>()
v.tail(n) == v.tail<n>()
v.segment(i, n) == v.segment<n>(i)
```


### Advanced Initializations

#### joined comma 

```c++
RowVectorXd vec1(3);
vec1 << 1, 2, 3;
 
RowVectorXd vec2(4);
vec2 << 1, 4, 9, 16;
 
RowVectorXd joined(7);
joined << vec1, vec2;
std::cout << "joined = " << joined << std::endl;
```


### Special matrices

```c++
// zero matrices
MatrixXd::Zero(3, 3);

// identity matrices
MatrixXd::Identity(5, 5); 

// constant matrices
MatrixXd::Constant(3, 4, 1.2);

// linspace
ArrayXXf table(10, 4);
table.col(0) = ArrayXf::LinSpaced(10, 0, 90);
table.col(1) = M_PI / 180 * table.col(0);
table.col(2) = table.col(1).sin();
table.col(3) = table.col(1).cos();
std::cout << "  Degrees   Radians      Sine    Cosine\n";
std::cout << table << std::endl;

// identity matrices
const int size = 6;
MatrixXd mat1(size, size);
mat1.topLeftCorner(size/2, size/2)     = MatrixXd::Zero(size/2, size/2);
mat1.topRightCorner(size/2, size/2)    = MatrixXd::Identity(size/2, size/2);
mat1.bottomLeftCorner(size/2, size/2)  = MatrixXd::Identity(size/2, size/2);
mat1.bottomRightCorner(size/2, size/2) = MatrixXd::Zero(size/2, size/2);
std::cout << mat1 << std::endl << std::endl;

// in-place version
MatrixXd mat2(size, size);
mat2.topLeftCorner(size/2, size/2).setZero();
mat2.topRightCorner(size/2, size/2).setIdentity();
mat2.bottomLeftCorner(size/2, size/2).setIdentity();
mat2.bottomRightCorner(size/2, size/2).setZero();
std::cout << mat2 << std::endl << std::endl;
 
// comma version
MatrixXd mat3(size, size);
mat3 << MatrixXd::Zero(size/2, size/2), MatrixXd::Identity(size/2, size/2),
        MatrixXd::Identity(size/2, size/2), MatrixXd::Zero(size/2, size/2);
std::cout << mat3 << std::endl;

// as a temporary objects
MatrixXf mat = MatrixXf::Random(2, 3);
mat = (MatrixXf(2,2) << 0, 1, 1, 0).finished() * mat; // .finished() is a must!
```


### Reductions

#### norm

```c++
int main()
{
  VectorXf v(2);
  MatrixXf m(2,2), n(2,2);
  v << -1,
       2;
  m << 1,-2,
       -3,4;
 
  cout << "v.squaredNorm() = " << v.squaredNorm() << endl;
  cout << "v.norm() = " << v.norm() << endl;
  cout << "v.lpNorm<1>() = " << v.lpNorm<1>() << endl;
  cout << "v.lpNorm<Infinity>() = " << v.lpNorm<Infinity>() << endl;

  cout << "m.squaredNorm() = " << m.squaredNorm() << endl;
  cout << "m.norm() = " << m.norm() << endl;
  cout << "m.lpNorm<1>() = " << m.lpNorm<1>() << endl;
  cout << "m.lpNorm<Infinity>() = " << m.lpNorm<Infinity>() << endl;
}
```

#### boolean

```c++
int main()
{
  ArrayXXf a(2,2);
  a << 1,2,
       3,4;
 
  cout << "(a > 2).all()   = " << (a > 2).all() << endl;
  cout << "(a > 2).any()   = " << (a > 2).any() << endl;
  cout << "(a > 2).count() = " << (a > 2).count() << endl;
}
```

#### visitors

to retrieve location of min/max reduction.

```c++
int main()
{
  Eigen::MatrixXf m(2,2);
  
  m << 1, 2,
       3, 4;
 
  //get location of maximum
  MatrixXf::Index maxRow, maxCol;
  float max = m.maxCoeff(&maxRow, &maxCol);
 
  //get location of minimum
  MatrixXf::Index minRow, minCol;
  float min = m.minCoeff(&minRow, &minCol);
 
  cout << "Max: " << max <<  ", at: " <<
     maxRow << "," << maxCol << endl;
  cout << "Min: " << min << ", at: " <<
     minRow << "," << minCol << endl;
}
```

#### partial reduction

since we at most consider 2-dimensional arrays...

```c++
int main()
{
  Eigen::MatrixXf mat(2,4);
  mat << 1, 2, 6, 9,
         3, 1, 7, 2;
  
  std::cout << "Column's maximum: " << std::endl
   << mat.colwise().maxCoeff() << std::endl; // a row vector
    
  std::cout << "Row's maximum: " << std::endl
   << mat.rowwise().maxCoeff() << std::endl; // a col vector   
}
```

Also, `rowwise()` and `colwise()` support other reductions:

```c++
int main()
{
  MatrixXf mat(2,4);
  mat << 1, 2, 6, 9,
         3, 1, 7, 2;
  
  MatrixXf::Index   maxIndex;
  float maxNorm = mat.colwise().sum().maxCoeff(&maxIndex);
  
  std::cout << "Maximum col-wise sum at position " << maxIndex << std::endl;
 
  std::cout << "The corresponding vector is: " << std::endl;
  std::cout << mat.col( maxIndex ) << std::endl;
  std::cout << "And its sum is is: " << maxNorm << std::endl;
}
```

#### Broadcast

Element-wise broadcasts are achieved by `Array`

Col/Row-wise broadcasts are achieved by `colwise() / rowwise()`.

```c++
int main()
{
  Eigen::MatrixXf mat(2,4);
  mat << 1, 2, 6, 9,
         3, 1, 7, 2;
    
  Eigen::VectorXf v(2);         
  v << 0,
       1;
       
  //add v to each column of m
  mat.colwise() += v;
  std::cout << mat << std::endl;

  Eigen::VectorXf v2(4);       
  v2 << 0,1,2,3;
       
  //add v2 to each row of m
  mat.rowwise() += v2.transpose();
  std::cout << mat << std::endl;
}
```


### Map: from raw data to eigen

```cpp
float* data;
Map<const Vector3f> pos(data, 3); // (pointer, size)
```


### Ref: generic type without template

```cpp
void fn(const Ref<const MatrixXf>& a) {
    // a can be MatrixXf or block of MatrixXf
    // a is read only
}

void fn(Ref<MatrixXf> a) {
    // a is writable
}
```

