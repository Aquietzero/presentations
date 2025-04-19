---
# try also 'default' to start simple
theme: neversink 
# random image from a curated Unsplash collection by Anthony
# like them? see https://unsplash.com/collections/94734566/slidev
background: https://source.unsplash.com/collection/94734566/1920x1080
color: blue-light
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
strong {
  color: #3b82f6;
}

.slidev-layout li {
  margin-bottom: 0.5em;
}
</style>

# 逻辑引擎与推理

bifnudozhao@tencent.com

---
layout: side-title
color: blue-light
---

::title::

# 概要

::content::

- 背景
  - 基于知识库的代理
  - 例子
- 介词逻辑
  - 语法
  - 语义
  - 一个简单的知识库
  - 一个简单的推理过程
- 基于介词逻辑的代理

---
layout: top-title
color: blue-light
---

::title::

# 背景 - `基于知识库的代理`

::content::

基于知识库的代理的核心在于**知识库（knowledge base）**。一个知识库就是一组**句子**的集合，而每个句子，都必须符合知识库所规定的语法。当一个句子是直接给出而不是推导出来的话，则被称为**公理（axiom）**。

知识库里面有两个操作**TELL**以及**ASK**，两者都可以在知识库里产生新句子，其中蕴含着**逻辑推导**的过程。

```ts
const KnowledgeBase;
const timer;

const KnowledgeBaseAgent = (percept: Percept): Action => {
  TELL(KnowledgeBase, makePerceptSentence(percept, timer))
  const action = ASK(KnowledgeBase, makeActionQuery(timer))
  TELL(KnowledgeBase, makeActionSentence(action, timer))
  timer.tick()
  return action
}
```


---
layout: top-title-two-cols
color: blue-light
---

::title::

# 背景 - `例子`

::left::

## 怪物世界

一个怪物世界是由一些格子组成，每个格子会有不同的属性

- **start**: 起点
- **pit**: 深渊
- **breeze**: 在深渊四周会有微风
- **monster**: 怪兽
- **stench**: 怪兽四周会有气味
- **gold**: 金块

::right::

<img src="/wumpus-world.png" class="w-120" />

---
layout: top-title-two-cols
color: blue-light
---

::title::

# 背景 - `例子`

::left::

表现衡量

- 获得金块得 +1000 分
- 掉进深渊或者被怪物吃掉得 -1000 分
- 每走一步得 -1 分
- 使用了弓箭的话得 -10 分

效应器（动作）

- **Forward**: 前进一格
- **TurnLeft** | **TurnRight**: 左转，或者右转 90 度
- **Grab**: 拿起金块
- **Shoot**: 用箭射击
- **Climb**: 爬出世界

::right::

<img src="/wumpus-world.png" class="w-120" />

---
layout: top-title-two-cols
color: blue-light
---

::title::

# 背景 - `例子`

::left::

传感器

- **Stench**: 在怪物周围的格子里，会感受到臭味
- **Breeze**: 在深渊周围的格子里，会感受到微风
- **Glitter**: 在有金块的格子里，会感受到光滑
- **Bump**: 碰到墙了会感知到撞击
- **Scream**: 怪物被杀死了，会感受到尖叫

在某一个格子里，代理能收到的信号，是传感器传递的信息序列，比如

```ts
const percept = [
  Stench, // Stench
  Breeze, // Breeze
  None,   // Glitter
  None,   // Bump
  None,   // Scream
]
```

::right::

<img src="/wumpus-world.png" class="w-120" />


---
layout: top-title-two-cols
color: blue-light
---

::title::

# 介词逻辑 - `语法`

::left::

介词逻辑的**语法**定义了合法的句子，**原子句子（atomic sentences）** 由一个单独的 **介词符号（prepositional symbol）** 组成。每个符号都能和一个真值所关联，而介词逻辑里，有两个具有固定意思的符号：**True** 以及 **False**。

复杂的句子可以由简单的句子组成，括号以及一些连接符，统一称为**逻辑连接符（logical connectives）**

::right::

- $\neg$ (not): 否定一个句子。
- $\land$ (and): 一个句子如果用 $\land$ 连接，则这个句子是一个 **合取**。
- $\lor$ (or): 一个句子如果用 $\lor$ 连接，则这个句子是一个 **析取**。
- $\Rarr$ (implies): 推出，可以理解为 **规则** 或者 **if-then** 语句。
- $\lrArr$ (if and only if): 句子 $A \iff B$ 是 **双向的**。

---
layout: top-title
color: blue-light
---

::title::

# 介词逻辑 - `语法`

::content::

**BNF（Backus-Naur Form）语法**

$$
\begin{align*}
\text{Sentence} &\to \text{AtomicSentence} | \text{ComplexSentence} \\
\text{AtomicSentence} &\to True | False | P | Q | R | \cdots \\
\text{ComplexSentence} &\to (\text{Sentence}) \\
&\ \ \ | \ \   \neg \text{Sentence} \\
&\ \ \ | \ \    \text{Sentence} \land \text{Sentence} \\
&\ \ \ | \ \  \text{Sentence} \lor \text{Sentence} \\
&\ \ \ | \ \  \text{Sentence} \implies \text{Sentence} \\
&\ \ \ | \ \  \text{Sentence} \iff \text{Sentence} \\
\text{Operator Precedence} &\ \ : \ \ \neg, \land, \lor, \implies, \iff
\end{align*}
$$


---
layout: top-title
color: green-light
---

::title::

# 介词逻辑 - `语义`

::content::

在介词逻辑里，**语义**实际上就是定义一个句子的真值，判断句子真值的方式如下

原子句子

- **True** 永远为真，**False** 永远为假
- 每个介词符号的真值，都必须明确指定

复合句子

- $\neg P$ 为真当且仅当 $P$ 在 $m$ 里为假。
- $P \land Q$ 为真当且仅当 $P$ 和 $Q$ 在 $m$ 里为真。
- $P \lor Q$ 为真当且仅当 $P$ 或者 $Q$ 在 $m$ 里为真。
- $P \implies Q$ 为真当且仅当在 $m$ 里 $P$ 为真并且 $Q$ 为假。
- $P \iff Q$ 为真当且仅当 $P$ 和 $Q$ 在 $m$ 里都为真或者都为假。

---
layout: top-title-two-cols
color: green-light
---

::title::

# 介词逻辑 - `知识库`

::left::

对于怪物世界来说，我们可以针对每一个位置 $[x, y]$ 定义如下所示的符号。

- $P_{x, y}$ 为真如果在 $[x, y]$ 有一个深渊。
- $W_{x, y}$ 为真如果在 $[x, y]$ 有一直怪兽。
- $B_{x, y}$ 为真如果在 $[x, y]$ 有微风。
- $S_{x, y}$ 为真如果在 $[x, y]$ 有气味。
- $L_{x, y}$ 为真如果代理在 $[x, y]$。

::right::

<img src="/wumpus-world.png" class="w-120" />

---
layout: top-title-two-cols
color: green-light
---

::title::

# 介词逻辑 - `知识库`

::left::

一些例子

- 在 $[1, 1]$ 没有深渊。
    
    $$
    R_1: \neg P_{1, 1}
    $$

- 一个格子有微风当且仅当它附近有深渊。
    
    $$
    \begin{align*}
    R_2 &: B_{1, 1} \iff (P_{1, 2} \lor P_{2, 1}) \\
    R_3 &: B_{2, 1} \iff (P_{1, 1} \lor P_{2, 2} \lor P_{3, 1})  \\
    \end{align*}
    $$

::right::

<img src="/wumpus-world.png" class="w-120" />


---
layout: top-title-two-cols
color: blue-light
---

::title::

# 基于介词逻辑的代理 - `世界状态`

::left::

一个知识库由两部分组成，一部分是世界的公理，另一部分是由这些公理推导出来的定理。对于怪物世界来说，我们对每个格子都知道一些基本事实：

- 深渊四周的格子是有风的
- 怪物四周的格子是有气味的

用介词逻辑表示如下

$$
\begin{align*}
B_{1, 1} &\iff (P_{1, 2} \lor P_{2, 1}) \\
S_{1, 1} &\iff (W_{1, 2} \lor W_{2, 1}) \\
&\ \ \ \ \ \vdots
\end{align*}
$$

::right::

<img src="/wumpus-world.png" class="w-120" />

---
layout: top-title-two-cols
color: blue-light
---

::title::

# 基于介词逻辑的代理 - `世界状态`

::left::

我们还知道整个世界只有一只怪兽，那么我们可以先表述整个世界至少有一只怪兽，用介词逻辑表示如下

$$
W_{1, 1} \lor W_{1, 2} \lor \cdots \lor W_{4, 3} \lor W_{4, 4}
$$

然后再描述最多只有一只怪兽，用介词逻辑表示如下

$$
\begin{align*}
\neg W_{1, 1} &\lor \neg W_{1, 2} \\
\neg W_{1, 1} &\lor \neg W_{1, 3} \\
&\ \vdots \\
\neg W_{4, 3} &\lor \neg W_{4, 4} \\
\end{align*}
$$

我们还知道代理一开始在左下角，所以这个位置信息也是一个公理

$$
L_{1, 1}
$$

::right::

<img src="/wumpus-world.png" class="w-120" />

对于整个怪物世界来说，刚才提到的句子集合，可以作为这个世界的公理集。

---
layout: top-title-two-cols
color: blue-light
---

::title::

# 基于介词逻辑的代理 - `世界状态`

::left::

因为代理在世界探险的过程中，有很多状态会发生变化，而状态毫无疑问会跟时间挂钩，所以对于时间的描述，非常重要。我们可以在介词逻辑里引入时间的概念，只需要通过给符号增加时间标记即可

考虑在 **t=0** 时刻，代理在左下角，那么有公理

$$
L_{1, 1}^0
$$

如果在 **t** 时刻，代理感受到微风以及气味，我们可以表示为

$$
L^t_{x, y} \implies (Breeze^t \iff B_{x, y}) \\
L^t_{x, y} \implies (Stench^t \iff S_{x, y})
$$

我们甚至可以基于这样的语法，写出行动的语句

$$
L^0_{1, 1} \land FacingEast^0 \land Forward^0 \implies (L^1_{2, 1} \land \neg L^1_{1, 1})
$$

::right::

<img src="/wumpus-world.png" class="w-120" />

---
layout: top-title
color: blue-light
---

::title::

# 基于介词逻辑的代理 - `世界状态`

::content::

我们可以基于带时间状态的句子，写出一些**效应公理（effect axiom）**

<img src="/effect-axioms.png" class="w-170" />

---
layout: top-title
color: blue-light
---

::title::

# 基于介词逻辑的代理 - `流`

::content::

我们把这类带有时间信息的公理称为**流（fluent）**，对于一个流 $F$，我们都可以写出一条公理来表达 $F^{t+1}$ 的真值，下面是一个状态变化公理的形式

$$
F^{t + 1} \iff ActionCausesF^t \lor (F^t \land \neg ActionCausesNotF^t)
$$

对于代理的位置，可以用下面的公理来了解状态变化公理的形式

<img src="/wumpus-world-2.png" class="w-170" />

---
layout: top-title
color: blue-light
---

::title::

# 基于介词逻辑的代理 - `流`

::content::

而我们最关心的一个问题是，哪里可以走哪里不可以走，走哪里比较安全。由此我们可以定义一个 **OK 流** 来表述

$$
OK^t_{x, y} \iff \neg P_{x, y} \land \neg(W_{x, y} \land WumpusAlive^t)
$$

所以代理可以移动到任何满足 $\text{ASK}(KB, OK^t_{x, y}) = true$ 的格子。

---
layout: cover
color: blue-light
---

# Q & A
