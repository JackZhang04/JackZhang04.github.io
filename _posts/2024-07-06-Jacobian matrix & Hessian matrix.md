---
title: Jacobian Matrix & Hessian Matrix
date: 2024-07-06 23:10:00 +0800
categories: [Mathematics]
tags: [Linear Algebra, Calculus]     # TAG names should always be lowercase
math: true
---

前几天在看深度学习经典之作（花书）的时候突然发觉自己的数学基础过于薄弱，于是打算恶补机器学习相关数学理论，进而熟练地打开csDIY网站，找到了吴恩达讲授的cs229这门课程。该说不愧是吴恩达，课程中对于机器学习算法的数学推导和解释深入浅出、赏心悦目，但是碍于自己薄弱的数学基础，当听到Hessian矩阵时还是感到一头雾水（小声bb：线代高数老师也不讲这个呀orz），遂上网查找相关资料。经过一番查询，也算是有了一点自己的体悟和收获，于是就有了这篇博客。
（主要参考了wikipedia上的解释并且掺杂了一些个人想法，或有浅薄之处，期望批评指正）

## 雅可比矩阵（Jacobian matrix）

*如果有一函数 $f: \mathbb{R}^{n} \to \mathbb{R}^{m}$ （即从 $x\in \mathbb{R}^n$ 映射到 $f(x) \in \mathbb{R}^m$ ），在点 $x$ 处可微，则其微分或导数的矩阵形式即为该函数的雅可比矩阵*。

$$
\boldsymbol{J}=
\begin{bmatrix}
\frac{\partial \boldsymbol{f}}{\partial x_1} & \cdots & \frac{\partial \boldsymbol{f}}{\partial x_n}
\end{bmatrix}
=\begin{bmatrix}
\frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_n}\\
\vdots & \ddots & \vdots \\
\frac{\partial f_m}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_n}
\end{bmatrix}
$$

换句话说，雅可比矩阵就是单变量实数函数的微分向**多变量向量函数微分**的推广。

雅可比矩阵的分量：

$$
\boldsymbol{J}_{ij} = \frac{\partial f_i}{\partial x_i}
$$

### 雅可比行列式

如果 $f$ 函数映射的维数不变，即 $m = n$ ，此时该函数的雅可比矩阵为 $n \times n$方阵。于是我们可以取该矩阵的行列式，即**雅可比行列式**。

雅可比行列式绝对值的大小代表对应的雅可比矩阵张成的空间大小，所以我们根据雅可比行列式的绝对值可以判定经过函数 $f$ 向量 $x$ 是增大还是缩小了体积（如果行列式的值大于1，则说明体积增大，以此类推体积保持不变和减小的情况）。

这也说明了为什么雅可比行列式会用在换元积分法当中：换元总是伴随着空间体积的变换，我们需要在换元过程中考虑到这种体积变换给我们的观测系带来的影响，所以总要乘以该空间放缩的倍数，即雅可比行列式绝对值的大小。

## 黑塞矩阵（Hessian matrix）

**黑塞矩阵**（Hessian matrix）又译作**海森矩阵**、**海塞矩阵**等，是一个由多变量实数函数的二阶导数组成的方阵。与雅可比矩阵类似，黑塞矩阵可以理解为单变量实数函数的二阶导向**多变量实数函数**的二阶导数的推广。

假设有一个实数函数 $f(x_1, x_2, ..., x_n)$ ，如果 $f$ 的所有二阶偏导数都存在且在定义域内连续，则 $f$ 的黑塞矩阵为：

$$
\mathbf{H} =
\begin{bmatrix}
\frac{\partial ^2 f}{\partial x_1^2} & \frac{\partial ^2 f}{\partial x_1 \partial x_2} & \cdots & \frac{\partial ^2 f}{\partial x_1 \partial x_n} \\
\frac{\partial ^2 f}{\partial x_2 \partial x_1} & \frac{\partial ^2 f}{\partial x_2^2} & \cdots & \frac{\partial ^2 f}{\partial x_2 \partial x_n} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial ^2 f}{\partial x_n \partial x_1} & \frac{\partial ^2 f}{\partial x_n \partial x_2} & \cdots & \frac{\partial ^2 f}{\partial x_n^2}
\end{bmatrix}
$$

黑塞矩阵的分量：

$$
\mathbf{H}_{ij} = \frac{\partial ^2 f}{\partial x_i \partial x_j}
$$

显然黑塞矩阵是一个 $n \times n$ 方阵，故黑塞矩阵一定存在行列式。其行列式称为**黑塞行列式**。

事实上，黑塞矩阵与雅可比矩阵有如下关系：

$$
\mathbf{H}(f) = \mathbf{J}(\nabla f)^{\mathbf{T}}
$$

即 $f$ 的黑塞矩阵等于其梯度的雅可比矩阵。这在导数层面是非常好理解的，因为函数的二阶导总是等于其导数的一阶导。

## 从泰勒展开理解雅可比矩阵和黑塞矩阵

由高等数学可知，假设存在单变量函数 $f(x)$ ，令 $x-x_0 = \Delta x$，在 $x_0$ 处进行泰勒展开：

$$
f(x) = f(x_0) + f^\prime(x_0)\Delta x + \frac{f^{\prime\prime}(x_0)}{2!}\Delta x^2 + ...
$$

泰勒展开可以拓展到多元函数，做法如下。假设我们有n元函数$F(x_1, x_2, ..., x_n)$ ，我们期望将其在 $x_0 = (x_{10}, x_{20}, ..., x_{n0})$ 处进行泰勒展开，则我们可以将上式中的 $f^\prime(x_0)$ 和 $f^{\prime\prime}(x_0)$ 分别替换为该n元函数在该点处的雅可比矩阵 $\mathbf{J}(x_0)$ 和黑塞矩阵 $\mathbf{G}(x_0)$ ，将实数 $\Delta x$ 替换为向量 $\Delta \boldsymbol{x}$ ，进而我们得到下式：

$$
F(x) = F(x_0) + \mathbf{J}(x_0)\Delta \boldsymbol{x} + \frac{1}{2!}\Delta\boldsymbol{x}^\mathbf{T}\mathbf{G}(x_0)\Delta \boldsymbol{x} + ...
$$

其中，多元实数函数 $F$ 在 $x_0$ 点处的雅可比矩阵亦是该函数在该点处的梯度 $\nabla F(x_0)$（二者等价）。

## 应用

正如前文所说，雅可比矩阵和黑塞矩阵分别是一元实数函数的一阶、二阶导数在更高维度上的拓展。这种高维特性与现代多维数据优化的需求相匹配，被广泛运用于优化设计、机器学习等领域中；这两种重要矩阵也高频地出现在不同经典算法中，例如[梯度下降](https://zh.wikipedia.org/wiki/%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D%E6%B3%95)、[牛顿法](https://zh.wikipedia.org/wiki/%E7%89%9B%E9%A1%BF%E6%B3%95)等。
