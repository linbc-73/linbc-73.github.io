# Paper-Notes-01 DimeNet 基于方向信息传递的分子图模型
##### 文章标题: Directional Message Passing for Molecular Graphs
##### Arxiv链接: https://arxiv.org/abs/2003.03123

-
### 1. 概述

ICLR2020 的一篇文章。
图神经网络GNN在预测分子性质方面取得了优秀的表现，但是在该模型出现以前，人们构建分子图时只利用了原子间的距离信息，没有考虑原子之间的空间方向信息spatial direction（比如角度信息）。实际上，在一些分子的经验势函数中，原子之间的角度信息却起着关键作用。因此，作者基于 MPNN 框架设计了 DimeNet 网络结构，在信息传递过程中加入了角度信息。（看完后自己总结一遍！）

**文章贡献：**
1） 


### 2. 模型设计
** 相关学习：等变性equivariance与不变性invariance，根据置信度传播belief propagation **
** 2.0 科学规律 **
1) 所有的分子预测模型首先需要尊重物理世界的基本规律，比如对称性symmetry和不变性invariance。其中最重要的有：平移和旋转不变性、同类一致性、各向同性、排列不变性。
如果模型没有考虑到这些因素，则很有可能引入重复的权重、增加模型复杂度进而造成训练时间过长。
2) 根据分子动力学模拟Molecular Dynamics simulation，视体系中的每个原子为遵守牛顿第二定律，根据分子的势能函数：
$$ E = E_{bounds}+E_{angle}+E_{torsion}+E_{non-bonded} $$
可以得到作用在每个原子上的力：
$$ F_{i} (X,z) = -\frac{\partial }{\partial x_{i} } f_{\theta }  (X,z) $$
注意如果要使用势函数，要求神经网路中的函数满足连续二阶可微。
** 2.1 输入与输出 **
网络的输入是各种分子，它们都可以由原子序数 z={z1,...,zN} 和原子位置 X={x1,...,xN} 独立地确定。该模型主要设计以解决回归任务，因此输出target可以视为实数。总结来说，网络可以表示为： $$  f_{\theta } : \left \{ X,z \right \} \longrightarrow \mathcal{R}  $$
不同于一些模型，作者没有引入例如化学键等辅助信息，因为牵扯到了人工设计hand-engineered的知识、于模型而言是non-essential不关键的。
** 2.2 损失函数 **
结合之前2.0提到的MD simulation，可以设计损失函数：
（公式LMD）
** 2.3 方向信息传递 **
在揭示方向信息传递如何运作之前，先看一下传统的GNN是如何将分子转化为图graph的。首先，每个原子都有一个单独的编码atom embedding： $ h_{i} $，原子和原子之间的边（不一定是化学键）也有独立的编码edge embedding：$ e_{ij} $，在神经网络的第l层layer中用来更新编码的消息传递函数构造如下：
$$ h_{i}^{(l+1)} = f_{update}(h_{i}^{(l)}, \sum_{j\in N_{i}} f_{int}(h_{j}^{(l)},e_{ij}^{(l)}))  $$
GNN一般不会使用完整的距离矩阵，这样会造成时间复杂度过高，而是引入截断距离cut-off distance，将时间复杂度从O(N\*N)降低到O(C\*N)。这也造成了一定的缺陷，比如不能分辨苯环和两个C3环。
一种解决这个问题的方法是使用两个原子之间的方向代替距离，而方向是随坐标系所改变的，因此需要设计一种在特定转换下具有等变性的网络。一个例子是G-CNN，群等变卷积神经网络Group Equivariant Convolutional Networks，然而这种方法不足在于只能实现离散的等变性，不能实现连续的等变性。（啥意思？）
因此，这篇文章转换另一种思路。首先，这里不再称为边的距离编码，而是使用包含了距离和角度的方向信息编码message embedding：$ m_{ji} $（代表了从原子j到原子i的信息传递），根据置信度传播算法belief propagation，边的消息传递函数构造如下：
$$ m_{ji}^{(l+1)} = f_{update}(m_{ji}^{(l)}, \sum_{k\in N_{i}\setminus \left \{ i \right \} } f_{int}(m_{kj}^{(l)},e_{RBF}^{(ji)},a_{SBF}^{(kj,ji)})) $$
![9](https://user-images.githubusercontent.com/83945633/183275193-9b26785e-5372-48fe-a451-0b0b0ad9d6ad.png)
