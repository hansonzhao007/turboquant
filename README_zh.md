# TurboQuant

[English](README.md) | **中文**

TurboQuant 是一个面向近似点积搜索的非对称量化方案。它将基础向量离线压缩为
4-bit 表示，在查询时将 query 向量动态量化为 int8，并通过解析缩放项重建
浮点分数。

![TurboQuant 流程图](turboquant_flow_diagram.svg)

## 仓库内容

本仓库包含：

- [`turboquant.py`](turboquant.py)：TurboQuant 的参考实现
- [`turboquant_codebook.py`](turboquant_codebook.py)：码本生成脚本与 4 维完整示例
- [`turboquant_test.py`](turboquant_test.py) 与
  [`turboquant_codebook_test.py`](turboquant_codebook_test.py)：回归与精度测试
- [`turboquant_explainer.html`](https://hansonzhao007.github.io/turboquant/turboquant_explainer.html)：可交互的流程可视化页面

实现本身比较紧凑，因此这份 README 更强调算法主流程、仓库结构，以及如何快速
体验交互式说明页面。

## 算法概览

TurboQuant 是一种**非对称**量化：

- 基础向量 `U` 离线量化一次，并以 4-bit 派生的 int8 代表值存储
- 查询向量 `V` 在打分时动态旋转并量化为 int8
- 最终点积分数按如下方式重建：

`Score = P * Scale_U * Scale_V`

其中 `P = sum(U_int8 * V_int8)`。

对于 2 的幂维度，旋转使用未归一化的 Fast Walsh-Hadamard Transform（FWHT）。
在通用实现中，[`HadamardRotation`](turboquant.py) 还支持非 2 的幂维度：先将
`d` 分解为 `r * 2^k`，再对 `2^k` 子空间应用 FWHT，并用一个正交矩阵混合剩余
的 `r` 个子空间。旋转前还可以选择性地施加随机符号向量 `D`。

## 打分流程

1. **记录原始范数**  
   在量化前保存 `Norm_U` 和 `Norm_V`，后续用来恢复最终分数的量级。

2. **旋转到更适合量化的空间**  
   对两个向量应用结构化旋转。对于 README 中的 4 维示例，这一步就是
   `H4 * U` 和 `H4 * V`。

3. **仅标准化基础向量**  
   将 `U_rot` 变换为单位方差形式：

   `U_std = (U_rot / ||U_rot||) * sqrt(d)`

   这样它的分布就与 TurboQuant 基础侧使用的标准正态码本相匹配。

4. **非对称量化**

   - **基础向量路径**：根据 `abs(U_std[i])` 落入的 bucket，再结合符号映射到
     4-bit 码本，得到存储用的 `U_int8`
   - **查询向量路径**：对 `V_rot` 做线性缩放，使 `max(abs(V_rot))` 对应到 `127`，
     然后四舍五入得到 `V_int8`

5. **计算整数点积**

   `P = sum(U_int8 * V_int8)`

   这是吞吐最关键的一步，也是这种表示适合大规模相似性搜索的核心原因。

6. **重建分数量级**

   - `Scale_U` 使用去偏后的质心能量表
     `SQUARED_CENTROIDS_WITH_DEBIAS`
   - `Scale_V` 将 int8 查询恢复到原始查询向量的量级

7. **返回近似浮点分数**

   `Score = P * Scale_U * Scale_V`

## 交互式说明页面

仓库中自带一个独立的交互页面：

- 本地文件：[`turboquant_explainer.html`](turboquant_explainer.html)

直接在浏览器中打开即可查看完整流程，编辑输入向量，并实时观察所有中间状态。

### 如何从 README 首页引出这个页面

比较实际的 README 首页集成方式是：

1. 在 README 中保留静态流程图
2. 在显眼位置放一个交互页面链接
3. 如果希望 GitHub 仓库首页也能直接打开可交互版本，就把
   `turboquant_explainer.html` 发布到 GitHub Pages，再将 README 链接指向该地址

GitHub README 可以渲染图片和相对链接，但不会在 README 内联执行这个 HTML 页面
里的 JavaScript。所以 README 可以展示预览图并提供跳转入口，但不能直接在仓库首页
里把这个交互页面运行起来。

如果你为这个仓库启用了 GitHub Pages，按当前 `origin` remote 推断，页面通常会发布到：

https://hansonzhao007.github.io/turboquant/turboquant_explainer.html

## 快速开始

安装依赖：

```bash
pip install -r requirements.txt
```

运行测试：

```bash
python -m unittest turboquant_test.py turboquant_codebook_test.py
```

运行性能测试脚本：

```bash
python profile_tq.py -d 1024 -N 10000 -Q 100
```

## 最小使用示例

```python
import numpy as np
from turboquant import TurboQuant

U = np.array([
    [2.0, 4.0, -2.0, 0.0],
    [1.5, 3.0, -1.0, 2.0],
], dtype=np.float64)

V = np.array([7.0, -1.0, 0.0, 1.0], dtype=np.float64)

tq = TurboQuant(d=4, use_signs=False)
tq.add_base_embeddings(U)
scores = tq.score(V)
print(scores)
```

## 文件说明

- [`turboquant.py`](turboquant.py)：`HadamardRotation` 与 `TurboQuant` 的参考实现
- [`turboquant_codebook.py`](turboquant_codebook.py)：码本推导与详细的 4 维示例
- [`turboquant_explainer.html`](https://hansonzhao007.github.io/turboquant/turboquant_explainer.html)：量化流程的交互式可视化
- [`fast_hadamard.cc`](fast_hadamard.cc)：FWHT 与批量点积的可选 C++ 加速实现
- [`profile_tq.py`](profile_tq.py)：快速性能测试入口

## 说明

- README 中的 4 维手工示例使用 `use_signs=False`，这样更容易逐步验证数学过程。
- 库本身支持通过 `HadamardRotation(..., use_signs=True)` 启用随机符号预处理。
- 更高维度的数据更能体现 TurboQuant 的真实效果；如果精确点积本身非常接近 0，
  小型玩具示例的相对误差可能会显得很大。

## 参考资料

- 实现细节：[`turboquant.py`](turboquant.py)、
  [`turboquant_codebook.py`](turboquant_codebook.py)
- 验证脚本：[`turboquant_test.py`](turboquant_test.py)、
  [`turboquant_codebook_test.py`](turboquant_codebook_test.py)
- GitHub README / Pages 相关文档：
  [About READMEs](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-readmes)、
  [Basic writing and formatting syntax](https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax)、
  [What is GitHub Pages?](https://docs.github.com/articles/mime-types-on-github-pages)
