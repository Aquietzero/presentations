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

# 语言学习以及学习方法杂谈

bifnudozhao@tencent.com

---
layout: side-title
color: blue-light
---

::title::

# 概要

::content::

- 语言学习
  - 语言结构
  - 语音
  - 符号
  - 词汇
  - 句法
- 学习方法
  - 简单化
  - 结构化
  - 抽象化


---
layout: top-title
color: blue-light
---

::title::

# 语言学习 - `语言结构`

::content::

回忆我们学习编程语言的过程（简化版）：

## 学习基础语法


```ts
// 比如变量定义，简单计算
const a = 1
const b = 2
const c = a + b
```

## 学习标准库的使用

```ts
// 比如用 `findIndex` 方法来查找元素下标
const arr = ['bad', 'bad', 'good', 'bad']
const index = arr.findIndex(item => item === 'good')
```

```ts
// 比如用 `concat` 来拼接
const hello = 'hello'
const world = 'world'
const greet = hello.concat(world)
```

---
layout: top-title-two-cols
color: blue-light
---

::title::

# 语言学习 - `语言结构`

::left::

## 基础词汇

**以 typescript/javascript 为例子**

关键字

- `var`, `const`, `let`：定义变量
- `function`：定义函数
- `for`, `while`：循环
- `break`：跳出代码块
- `if`, `else`：条件判断
- `switch`, `case`：多条件分支

::right::

运算符

- `=`：用于变量赋值
- `>`, `<`, `>=`, `<=`, `===`, `!==`：变量比较
- `=>`：函数定义
- `{`, `}`：代码块


---
layout: top-title-two-cols
color: blue-light
---

::title::

# 语言学习 - `语言结构`

::left::

## 基础语法

- 变量定义语法
  
```ts
const 变量名 = 初始值
```

- 条件分支语法

```ts
if (条件) { 语句 }
```

- `for` 循环语法

```ts
for (循环初始赋值;循环边界判断;循环递增赋值) { 语句 }
```

- `while` 循环语法

```ts
while (循环边界判断) { 语句 }
```

::right::

## 基础库使用

- 字符串基础方法：`length`, `match`, `replace`, `toLowerCase` 等
- 数组基础方法：`find`, `concat`, `map`, `join` 等
- 文件处理基础方法：`readFile`, `writeFile` 等
- 网络请求：`fetch` 等

---
layout: quote
color: pink-light
---

“如果要学习一种新的语言该怎么学？”

---
layout: top-title
color: green-light
---

::title::

# 语言学习 - `语言结构`

::content::

既然语言的结构可以划分为

- 基础词汇
- 基础语法
- 基础库使用

那在学习一门新语言的时候

- 分步学习语言结构
- 与已经熟悉的语言进行类比于联想

---
layout: top-title
color: green-light
---

::title::

# 语言学习 - `语言结构`

::content::

|   |   | ts/js | c++ | python | go |
|---|---|-------|-----|--------|----|
|基础词汇|关键字| `var`, `let`, `const` | `int`, `bool` | `def`, `__init__` | `func`, `:=` |
|      |语句分割| 分号/换行 | 分号 | 换行 | 分号以及换行 | 
|      |缩进规则| 无 | 无 | 对齐 | 对齐 | 
|基础语法|条件语句| `if (cond) {}` | `if (cond) {}` | `if cond:` | `if cond {}` | 


---
layout: top-title
color: blue-light
---

::title::

# 技术实现 - `概念`

::content::

通过上面的分析，我们先定义一组概念

- **测试用例**: 输入是数据，输出是 UI 界面，`{ input: params, output: image }`
- **测试用例正确性**: 测试用例输出的结果和正确的 UI 是否一致
- **测试用例覆盖率**: 分支覆盖率
- **有效用例**: 可以使覆盖率增加的用例

- **类型定义**: 一个数据结构的定义，包含其字段以及类型
- **元数据**: 一个用于描述数据结构的数据结构
- **数据生成器**: 给定一个类型描述，输出对应类型的数据

---
layout: top-title-two-cols
color: blue-light
---

::title::

# 技术实现 - `自动生成测试用例`

::left::

## 数据类型定义

在商品卡片例子中，商品卡片组件的输入数据是商品数据，我们需要随机生成商品数据，就需要一个商品数据生成器，可以根据类型元数据来构造**数据生成器**。

```ts
interface Product {
  title: string,         // 商品标题
  currentPrice: number,  // 商品价格
  originalPrice: number, // 商品原价
  tags: string[],        // 商品标签
  image: string[],       // 商品图片
  viewNumber: number,    // 商品热度
  saleNumber: number,    // 商品销量
}
```

::right::

## 数据元数据

有了这种类型定义，我们可以将其转化为**元数据**。

```ts
const ProductSchema = {
  name: 'Product',
  fields: [
    { name: 'title', type: 'string' },
    { name: 'currentPrice', type: 'number' },
    { name: 'originalPrice', type: 'number' },
    { name: 'tags', type: 'array[string]' },
    { name: 'image', type: 'array[image]' },
    { name: 'viewNumber', type: 'number' },
    { name: 'saleNumber', type: 'number' },
  ]
}
```

---
layout: top-title-two-cols
color: blue-light
---

::title::

# 技术实现 - `自动生成测试用例`

::left::

## 数据生成器

- 数据生成器接受一个数据元数据 `model`，输出对应数据元数据的数据

```ts {*}{lines: false}
type Generator = (schema) => data
```

- 随机数据的每个字段，是由对应类型的随机生成器生成
- 生成的数据是一个符合类型的随机数据
- 生成数据的随机性，可以通过每种类型的生成器进行控制
- 复合类型可嵌套

::right::

```ts
export const productGenerator = (schema: ProductSchema) => {
  const randomData = {}

  _.each(schema.fields, field => {
    let value
    switch (field.type) {
      case Types.Number:
        value = randomeNumber()
        break
      case Types.String:
        value = randomString()
        break
      case Types.Boolean:
        value = randomBoolean()
        break
      // 各种类型调用对应的随机生成器
    }
    randomData[field.name] = value
  })

  return randomData
}
```

---
layout: top-title-two-cols
color: blue-light
---

::title::

# 技术实现 - `自动生成测试用例`

::left::

## 随机数据

通过数据生成器，便可以得到符合类型的随机数据，比如一个随机的商品数据如下所示

```ts
const randomProduct = {
  title: 'asjdflkjadlk',
  currentPrice: 278,
  originalPrice: 823,
  tags: ['ad', '3ukadjsfl'],
  image: ['http://asdf.com/adksjfkj'],
  viewNumber: 23748,
  saleNumber: 9283,
}
```

有了随机数据作为渲染函数的输入，我们就能得到对应这个随机数据的界面

```vue
<Product :product="randomProduct" />
```

::right::

## 小结

由此，我们得到了这样的一条链路（依然以商品卡片为例）

1. 被测函数的输入数据类型定义 `Product`
2. 从数据类型推导出其元数据 `ProductSchema`
3. 从元数据得到数据生成器 `generateProduct`
4. 使用生成器生成随机数据 `randomProduct`
5. 使用随机数据渲染组件 `<Product />`
6. 组件渲染结果为一张图

由此我们得到了一个**用例**

---
layout: quote
color: pink-light
---

“上面生成的都是随机用例，我们怎样得到有用的用例？”

---
layout: top-title-two-cols
color: blue-light
---

::title::

# 技术实现 - `自动生成测试用例`

::left::

## 有效用例

有效的用例是能使覆盖率增加的用例

- 执行用例前的覆盖率为 `coverageBefore`
- 执行用例 `case`
- 执行用例后的覆盖率为 `coverageAfter`

如果 `coverageAfter > coverateBefore`，那么 `case` 就是有效的

<Admonition title="注意" color='pink-light' width="300px">
因为这个用例，让函数执行了新的逻辑，所以才能导致覆盖率增加
</Admonition>

::right::

<img src="/覆盖率说明.png" class="h-110" />

---
layout: top-title-two-cols
color: blue-light
---

::title::

# 技术实现 - `自动生成测试用例`

::left::

```ts
const collectTestCases = (fn, generator, opts) => {
  const testCases = []
  let coverageBefore = 0

  while (true) {
    // 使用生成器生成随机数据
    const randomData = generator()
    // 使用随机数据调用被测函数
    const UI = fn(randomData)
    // 观测本次调用后的累积分支覆盖率
    const coverageAfter = instrument.coverage()
    // 分支覆盖率有所增加，则判断为有效用例
    if (coverageAfter > coverageBefore)
      testCases.push({ input: randomData, output: UI })
      
    coverageBefore = coverageAfter

    // 覆盖率满足阈值，就能退出循环
    if (coverage >= opts.coverageThreshold) break
  }

  return testCases
}
```

::right::

- 循环次数是可控的，可以设定一个覆盖率目标
- 整体流程的每个组件，都存在灵活配置的空间
  - 随机数据生成器可以控制随机的算法
  - 覆盖率定义与计算可以使用不同的方式
  - 测试用例的输出类型多样，可以是数据，也可以是图片

---
layout: top-title-two-cols
color: violet-light
---

::title::

# 技术实现 - `自动化测试`

::left::

## 收集正例

在进行测试的时候，我们实际上是用测试结果来与正例进行对比，所以我们需要一个正例库

由于通过随机生成的方式我们可以得到满足一定覆盖率的测试用例集，而这些用例的输出实际上是图片，所以通过人工的方式建立正例库成本并不大

正例库可能是如下形式的

```ts
const correctCases = [
  { input: params_1, output: UI_1 },
  { input: params_2, output: UI_2 },
  { input: params_3, output: UI_3 },
  // ...
  { input: params_N, output: UI_N },
]
```

::right::

## UI 比对获得测试通过率

有了正例库，我们便可以执行正例库里的用例

```ts
const runCases = (correctCases, fn) => {
  let correctCount = 0
  // 检查每个正例
  _.each(correctCases, case => {
    // 使用正例里的数据作为输入
    const UI = fn(case.input)
    // 如果被测函数的输出与标准输出一致，则通过
    if (isCorrect(UI, case.output)) {
      correctCount += 1
    }
  })

  // 计算正确比率
  return correctCount / correctCases.length
}
```

---
layout: top-title
color: violet-light
---

::title::

# 技术实现 - `自动化测试`

::content::

前面使用到的 `isCorrect` 也是一个可以灵活配置的组件，其目标在于判断两个被测函数的输出是否一致。在当前场景下被测函数的输出是图片，所以对比图片是否一致有很多办法。

- 精确匹配：像素级匹配，由于输入是固定的，如果代码改动正确，理论上输出也应该一致
- 模糊匹配：可以使用 AI 方法，使用正例进行训练，得到一个有容错并相对智能的图片匹配器

<img src="/匹配方式说明.png" class="h-70" />

---
layout: top-title
color: green-light
---

::title::

# 技术实现 - `可持续迭代闭环`

::content::

最开始提到的一些痛点消失了

| 之前    | 现在  |
|--------|-------|
| 测试用例建立成本很大 | 测试用例自动生成 <br /> ==需要手动调整元数据== |
| 测试用例结果需要人工检查 | 测试用例结果可以自动比对 <br /> ==需要人工先确定正例== |


---
layout: quote
color: pink-light
---

“那在迭代的过程中，如何使用以及更新这些用例呢？”

---
layout: top-title-two-cols
color: green-light
columns: is-4
---

::title::

# 技术实现 - `可持续迭代闭环`

::left::

## 需求与代码改动

对于多状态组件，我们在面对代码的时候实际上是很抽象的
  - 不清楚某一行代码是否会引起某些界面的变动
  - 不清楚它是否会引入问题

有了一个组件的所有用例以及批量渲染UI的方式，我们可以实现下面的工作流。

::right::

<img src="/需求与代码改动.png" class="h-90" />

---
layout: top-title-two-cols
color: green-light
---

::title::

# 技术实现 - `可持续迭代闭环`

::left::

## 用例与界面的变动

- 代码改动前存在一批有效用例
- 代码改动后再次执行，获得一批新的用例
- 执行 diff 算法比对不同的用例
- 人工判别哪些用例需要更新
- 用例库更新

::right::

<img src="/用例变更.png" class="h-110" />

---
layout: top-title-two-cols
color: green-light
---

::title::

# 技术实现 - `可持续迭代闭环`

::left::

## 测试粒度控制

在前面提到的例子中，测试粒度是组件，实际上这个方法是可以在不同粒度上进行的，因为本质是函数的测试。

<Admonition title="注意" color='pink-light' width="300px">
对于组件来说，输入是组件函数的入参类型
对于页面来说，输入是页面函数的入参类型
</Admonition>

::right::

## 页面维度

如果在页面维度实施这个方案，那么

- 输入：能控制一个页面渲染结果的因素
  - url：地址，以及地址上的参数
  - localStorage：浏览器本地存储
  - cookie
  - 接口返回
- 输出：渲染结果截图

---
layout: cover
color: blue-light
---

# Q & A
