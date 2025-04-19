---
# try also 'default' to start simple
theme: seriph
# random image from a curated Unsplash collection by Anthony
# like them? see https://unsplash.com/collections/94734566/slidev
background: https://source.unsplash.com/collection/94734566/1920x1080
# apply any windi css classes to the current slide
class: 'text-center'
# https://sli.dev/custom/highlighters.html
highlighter: shiki
# show line numbers in code blocks
lineNumbers: true 
# persist drawings in exports and build
drawings:
  persist: false
# page transition
transition: slide-left
# use UnoCSS
css: unocss
---

<style>
.col-right {
  padding-left: 20px;
}
</style>

# 如何做 PPT

bifnudozhao@tencent.com

---
layout: two-cols
---

# 概要

- 如何做 PPT
  - 基本工具
  - 形式
    - 基本
    - 代码
    - 数学公式
    - 画图

---
---

# 如何做 PPT - 基本工具

<img src="/slidev.png" class="h-90" />

Slidev官网：https://sli.dev/

---
layout: two-cols
---

# 如何做 PPT - Slidev 特点

- 适用场景
  - 用于紧急的 PPT 制作
  - 用于一些平实的技术或者知识性主题演讲

- 优点
  - 直接写 markdown 就可以了
  - 样式有主题可以选择，字号对齐等问题不会存在
  - 如果某些地方的样式希望自己调整，自己写样式就好
  - 各种展示形式都非常方便

- 扩展
  - 可以自己写主题，比如写个腾讯 PPT 主题
  - 本质是前端项目，想改什么都可以

::right::

<div class="h-100" style="display: flex; align-items: center; justify-content: center;">
```markdown
---
layout: two-cols
---
- 适用场景
  - 用于紧急的 PPT 制作
  - 用于一些平实的技术或者知识性主题演讲

- 优点
  - 直接写 markdown 就可以了
  - 样式有主题可以选择，字号对齐等问题不会存在
  - 如果某些地方的样式希望自己调整，自己写样式就好
  - 各种展示形式都非常方便

- 扩展
  - 可以自己写主题，比如写个腾讯 PPT 主题
  - 本质是前端项目，想改什么都可以
```
</div>

---
---

# 如何做 PPT - 形式

- PPT 是一个表达思想或者传授知识的载体
- 以讲为主，展示为辅，和文章是截然不同的载体
- 所以建议用多种形式表达
  - 文字：表达主要观点，言简意赅
  - 代码：一行代码胜千言
  - 公式：一条公式胜千言
  - 图示：一图胜千言

---
---

# 如何做 PPT - 代码

在座各位都是开发，PPT 里使用代码的场景应该挺高。工整漂亮的代码是非常重要的，使用 Slidev，代码块也是直接支持的，用 markdown 语法就可以了。

```markdown
const ppt = Slidev()
ppt.isProfessional = true
```

上面的 markdown 代码会直接渲染成下面的代码块。

```js
const ppt = Slidev()
ppt.isProfessional = true
```

- 书写方便快捷
- Slidev 还有更多高级功能
  - 支持代码高亮某行
  - 支持代码分段高亮
  - 支持各种代码颜色主题

---
layout: two-cols
---

# 如何做 PPT - 公式

在讲述一些复杂机制的时候，可能需要用数学公式。数学公式可以使用 Latex 或者 Katex。Katex 是 Latex 的一个子集，Slidev 支持的是 Katex。

数学公式实际上没有想象中困难，下面简单介绍下。

- 普通公式如果只使用字母或者数字，可以直接打。

```tex
a + b = c, 1 + 1 = 2
```

$$
a + b = c, 1 + 1 = 2
$$

- Tex 会对字符进行渲染（可以看到上面的数字以及字母都是 serif 以及倾斜），如果想用正常文字，可用“text”命令字。

```tex
\text{This is a text.}
```

$$
\text{This is a text.}
$$

::right::

- 特殊符号用反斜杠“\”加“命令字”，下面是一些常用特殊符号。
  
```tex
\sum, \int, \prod, \sin, \cos
```

$$
\sum, \int, \prod, \sin, \cos
$$

- 上标使用 `^`，下标使用 `_`，用于命令字也生效。

```tex
1 + \cdots + n
= \sum^n_{i = 0}\frac{n(n + 1)}{2}
```

$$
1 + 2 + \cdots + n = \sum^n_{i = 0}\frac{n(n + 1)}{2}
$$

---
layout: two-cols
---

# 如何做 PPT - 公式

前面提到的都是一些单行公式，有时候我们希望以公式组的形式对齐。Katex 里有一个环境的概念，比如在矩阵环境里编辑，那就是对矩阵的排版，在公式组环境里编辑，那就是对一组公式的排版。

- 矩阵环境里编辑一个矩阵，其中“&”可以理解为制表符，是用来对齐的。“\\”用于换行。

```tex
M = 
\begin{vmatrix}
1 & 2 & 3 & 4 \\
2 & 3 & 4 & 5 \\
3 & 4 & 5 & 6
\end{vmatrix}
```

$$
M = 
\begin{vmatrix}
1 & 2 & 3 & 4 \\
2 & 3 & 4 & 5 \\
3 & 4 & 5 & 6
\end{vmatrix}
$$

::right::

- 对齐环境比较通用，可以用于对齐你想对齐的地方。其中“&”在等号前，表明以等号对齐。

```tex
\begin{aligned}
x &= a + b \\
y &= c + d + e + f \\
\end{aligned}
```

$$
\begin{aligned}
x &= a + b \\
y &= c + d + e + f \\
\end{aligned}
$$

- 条件式环境。

```tex
f(x) =
\begin{cases}
  x + 2, \text{if} x > 0 \\
  2x, \text{if} x \leq 0 \\
\end{cases}
```

$$
f(x) = 
\begin{cases}
x + 2, &\text{ if } x > 0 \\
2x, &\text{ if } x \leq 0 \\
\end{cases}
$$

---
---

# 如何做 PPT - 图示

一图胜千言，很多时候画一个很好的图示可以节省很多解释的时间。学会画图，学会画一些清晰美观的图非常重要。

画图功能有几类：

- DSL 画图，比如 markdown, graphviz
  - 缺点：DSL 比较繁琐，需要熟悉 DSL 并且写代码
  - 图片自动渲染很难单独修改
- 复杂的可视化编辑器，比如 PS
  - 大部分技术和知识分享不需要太绚丽的图
  - 学习成本高以及花费大量时间
- 简易可视化编辑器，比如 draw.io，Omnigraffle
  - 操作简单
  - 学习成本低

从节省时间以及方便程度来看，draw.io 是比较适合大家的。

---
layout: two-cols-header
---

# 如何做 PPT - 图示 tips

背景最好和 PPT 背景一致并且是纯色，最好把 draw.io 的背景网格去掉。

<img src="/graph-grid-setting.png" class="h-30" />

下面是效果对比，如果有网格，会和背景有点格格不入。没有背景可以更好融合上下文，看着更为干净。

::left::
<img src="/graph-with-grid.png" class="h-60" />

::right::
<img src="/graph-without-grid.png" class="h-60" />

---
layout: two-cols-header
---

# 如何做 PPT - 图示 tips

图示里的各单元大小最好尽量一致，以及使用 draw.io 本身的对齐能力尽可能对齐。

- 对单元内容稍作评估之后，先拖出一个单元，然后直接复制多个，这样能保证每个一样大
- 在有了一堆单元之后，使用 draw.io 本身的参考线进行对齐
- 整体调整的时候考虑让整个图变得整齐稠密，并且尽量让对称或者类似的地方间距一致

<p style="margin-top: 20px" />

::left::
<img src="/alignment-bad.png" class="h-60" />

::right::
<img src="/alignment-good.png" class="h-60" />

---
layout: two-cols
---

# 如何做 PPT - 图示 tips

图示的颜色恰当，和谐，可以让图示更清晰，甚至增色不少。

- 没有太好的明确想法的时候，最好不要超过三个颜色
  - 划定区域的背景色
  - 字本身的颜色
  - 强调的颜色
- 三个颜色最好是同一个色系，或者刚好是对比强烈的颜色
  - 同色系：白色，灰色，黑色
  - 对比强烈：红色，蓝色

::right::

<div class="h-100" style="display: flex; justify-content: center; align-items: center;">
  <img src="/color-not-good.png" class="w-100" />
</div>

---
layout: two-cols
---

# 如何做 PPT - 图示 tips

图示的颜色恰当，和谐，可以让图示更清晰，甚至增色不少。

- 没有太好的明确想法的时候，最好不要超过三个颜色
  - 划定区域的背景色
  - 字本身的颜色
  - 强调的颜色
- 三个颜色最好是同一个色系，或者刚好是对比强烈的颜色
  - 同色系：白色，灰色，黑色
  - 对比强烈：红色，蓝色

::right::

<div class="h-100" style="display: flex; justify-content: center; align-items: center;">
  <img src="/color-better.png" class="w-100" />
</div>

---
layout: center
---

<h1>虽然内容才是最重要的，但是表现形式不容忽视</h1>


---
---

<div style="display: flex; width: 100%; height: 100%; justify-content: center; align-items: center">
  <h1 style="font-size: 3em;">Q & A</h1>
</div>