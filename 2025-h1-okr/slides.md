---
theme: seriph
class: 'text-center'
highlighter: shiki
lineNumbers: true
drawings:
  persist: false
transition: slide-left
css: unocss
---

<style>
h2, h3, h4, h5, h6 {
  margin-top: 20px !important;
}

.slidev-layout {
  font-family: sans-serif;
}

.slidev-layout.cover {
  color: black !important;
  background: white !important;
  background-image: unset !important;
}
.slidev-layout.cover p {
  color: black !important;
}

.slidev-layout h1 {
  color: #005cc5 !important;
  font-weight: bold;
  border-bottom: solid 2px #005cc5;
  width: fit-content;
}
.slidev-layout h2 {
  font-size: 1.6rem;
  margin-bottom: 10px;
}
.slidev-layout .my-auto h1 {
  color: #005cc5 !important;
  border-bottom: none;
  width: auto;
}
.slidev-layout h1 + p {
  opacity: 1 !important;
  padding-top: 20px;
}
.col-right {
  padding-left: 25px;
  display: flex;
  justify-content: center;
  flex-direction: column;
}
</style>

# 2025 H1 OKR

矩阵端技术中心/矩阵前端研发二组

---

# 中心 OKR

### O3：「创新」探索AI技术助力研发提效，通过技术创新创造用户价值

- **KR1**: 探索Design-to-Code在前端社区&增长活动页等场景的应用。
- **KR2**: 完善前端自动化黑盒测试能力，在实际业务场景中应用且产生收益。 

---

# 概览

### 基于 Figma + LLM 的 D2C 工作流（简称 D2C 工作流）
- 基于 Figma + LLM 的设计稿转代码
- 多状态组件生成
- 自动化多轮优化

### 基于 RAG + LLM 的 codebase 理解（简称 RAG+LLM）
- 基于 RAG + LLM 的自动化黑盒测试
- 智能元数据生成
- 多轮对话优化

---

# D2C 工作流 - 背景

### 能力

- LLM 与 MLLM 的飞速发展，让 D2C 这个课题的可行性变得更高了
- 现有工具已经能具备肉眼可见不错的效果

### 不足

- 司内工具不稳定，不开源，效果一般，sloth
- cline + figma MCP 可以实现编辑器流程整合，但依然是封闭的流程
- 无法支持多状态组件
- 无法使用本地仓库代码

---
layout: two-cols
---

# D2C 工作流 - 目标

## 定量目标

- 建立 D2C 工具链
  - Figma 插件：选择组件，获取结构及样式
  - 生成服务：Prompt 构造及 LLM 调用
  - VSCode 插件：代码应用及多轮优化
  - 组件渲染器：可视化效果对比

- 建立 D2C 工作流
  - 设计规范，Figma 使用规范
  - 组件代码生成流程
  - 多轮对话微调

::right::

## 定性目标

- 产出高质量可用代码
- 节省人力及代码量
- 支持多状态组件生成
- 实现自动化多轮优化

---
layout: two-cols
---

# D2C 工作流 - 核心架构

- **figma plugin**：在 figma app 内部，获取设计稿的 dom 以及 style，通过请求的方式传到 D2C server 作进一步处理。
- **D2C server**：一个承接 D2C 主要工作流的服务器，用于接收 figma plugin 传出来的数据，构造合适的 prompt，然后与 LLM server 进行交互，获得代码。
- **LLM server**：大语言模型的服务器，用于将 prompt 转化为组件代码。
- **vscode plugin**：在 vscode（或者其他代码编辑器）内部，用于接收代码，并且将代码应用到对应的项目目录里。
- **renderer**：对生成代码进行可视化，是一个带有可执行环境的小型前端 sandbox 环境。


::right::

<img class="w-120" src="/d2c整体架构.png" />

---

# D2C 工作流 - 核心流程 - 多状态 D2C 流程

多状态 D2C 流程将在 figma 插件中允许多选，并且允许添加每种状态的标注，状态+标注构造为一个 figma 元数据，交给 D2C server 进行处理。

<img class="w-150" src="/d2c多状态.png" />

---

# D2C 工作流 - 核心流程 - 自动化多轮优化流程

可以进一步通过建立 AI 闭环来进行多轮优化。这个闭环在下图的蓝色高亮部分表示。

<img class="w-190" src="/d2c多轮优化.png" />

---
layout: two-cols
---

# D2C 工作流 - 业务应用

### 组件代码生成

组件代码生成是本方案最基本的目标。希望能达到设计稿的还原度 80% 左右的效果。针对各种独立组件。

这些都是看着比较简单的组件，实现也不需要很多时间，但是能直接高质量生成是可以节省不少时间的。

::right::

<img class="w-80" src="/d2c应用1.png" />

---
layout: two-cols
---

# D2C 工作流 - 业务应用

### 静态页面代码生成

对于组件数量不多，相对静态的页面能直接生成。

::right::

<img class="w-100" src="/d2c应用2.png" />

---
layout: two-cols
---

# D2C 工作流 - 业务应用

### 多状态组件代码生成

目前市面上大多数工具不支持多状态组件的代码生成，但日常开发生产中绝大部分组件都是多状态的，本方案的另一个目标是覆盖这类场景。

::right::

<img class="w-100" src="/d2c应用3.png" />

---

# D2C 工作流 - 里程碑

<div class="timeline-container my-10">
  <div class="timeline-line"></div>
  <div class="timeline-items">
    <div class="timeline-item">
      <div class="timeline-content top">
        <h3>2025-03-31</h3>
        <p>工程建设，流程搭建<br>基础流程跑通<br>基础 Prompt 设计</p>
      </div>
      <div class="timeline-dot"></div>
    </div>
    <div class="timeline-item">
      <div class="timeline-content bottom">
        <h3>2025-04-30</h3>
        <p>提升代码质量<br>组件拆分支持<br>VSCode 插件实现</p>
      </div>
      <div class="timeline-dot"></div>
    </div>
    <div class="timeline-item">
      <div class="timeline-content top">
        <h3>2025-05-31</h3>
        <p>多状态组件生成<br>节点批注支持<br>多状态预览</p>
      </div>
      <div class="timeline-dot"></div>
    </div>
    <div class="timeline-item">
      <div class="timeline-content bottom">
        <h3>2025-06-30</h3>
        <p>AI 闭环优化<br>结果评判标准<br>Prompt 优化</p>
      </div>
      <div class="timeline-dot"></div>
    </div>
  </div>
</div>

<style>
.timeline-container {
  position: relative;
  height: 250px;  /* 改动：减小整体高度使内容更紧凑 */
  display: flex;
  align-items: center;
  padding: 0 40px;
}

.timeline-line {
  position: absolute;
  width: 100%;
  height: 2px;
  background: #93c5fd;
  top: 50%;
  transform: translateY(-50%);
}

.timeline-items {
  position: relative;
  display: flex;
  justify-content: space-between;
  width: 100%;
}

.timeline-item {
  position: relative;
  flex: 1;
  display: flex;
  justify-content: center;
}

.timeline-content {
  position: absolute;
  width: 200px;
  text-align: center;
  left: 50%;
  transform: translateX(-50%);
}

.timeline-content.top {
  bottom: 15px;  /* 改动：更靠近节点 */
}

.timeline-content.bottom {
  top: 15px;  /* 改动：更靠近节点 */
}

.timeline-content h3 {
  color: #3b82f6;
  font-size: 1.2rem;  /* 改动：增大标题字号 */
  font-weight: bold;
  margin-bottom: 0.2rem;  /* 改动：减小标题和内容间距 */
}

.timeline-content p {
  font-size: 1rem;  /* 改动：增大内容字号 */
  line-height: 1.3;  /* 改动：适当减小行高使文字更紧凑 */
}

.timeline-dot {
  width: 16px;
  height: 16px;
  background: #3b82f6;
  border-radius: 50%;
  position: absolute;
  top: 50%;
  transform: translateY(-50%);
}
</style>

---
layout: two-cols
---

# RAG + LLM - 目标

## 定量目标

- 建立基于仓库理解的 RAG + LLM 工具流
  - LLM 代码分析理解
  - RAG 构建
  - 多轮对话优化

- 整合元数据生成流程
  - 组件元数据生成
  - 生成时机设计
  - 迭代流程优化

::right::

## 定性目标

- 减轻元数据生成压力
- 实现自动化黑盒测试
- 扩展到其他项目应用

---
layout: two-cols
---

# RAG + LLM - 核心架构

- **代码仓库解释**: 对仓库代码进行解释，生成文档。
- **RAG 构建**: 对文档进行分片，向量化，持久化，方便后续查阅。
- **LLM 问答**: 向 LLM 提问的时候寻找关联片段，结合为一个统一的上下文结构，然后交给 LLM 进行回答。
- **多轮优化**: 对于特定场景，比如自动化黑盒测试里的元数据生成，支持多轮优化，提升元数据生成质量。

::right::

<img class="w-120" src="/ragllm整体架构.png" />

---
layout: two-cols
---

# RAG + LLM - 仓库解释

对于现成代码仓库来说，要求代码仓库有一个对应的文档库实在非常严苛，毕竟没有人会在写代码的时候，同时写上一堆解释代码的文档，而 coder 模型，刚好可以胜任这个任务。我们可以先通过 coder 模型对代码仓库进行解释。

```python
def explain_codebase(repo_path):
  for root, dirs, files in os.walk(repo_path):
    # 过滤无关目录
    if any(folder in dirs for f in filter_folders):
        continue

    for file in files:
      file_path = os.path.join(root, file)
      # 使用 coder LLM 解释文件
      explain_file(file_path)
```

::right::

<img class="w-120" src="/ragllm仓库解释.png" />

---

# RAG + LLM - 构建 RAG

RAG 的关键是将文档嵌入（embedding），然后持久化到向量数据库中。

<img class="w-200" src="/ragllm构建rag.png" />

其中 embedding 可选各种 embedding 模型，vector database 也可以使用各种开源库。下面的选择是跑通流程所使用的最基础的选择。

- embedding 模型使用 `all-MiniLM-L6-v2`
- vector database 使用 faiss

由于对解释文档进行索引是一个固定的行为，所以解释并且索引结束之后，可以将结果持久化，所以这个流程需要加入持久化模块。

---

# RAG + LLM - 回答问题

由于解释文档已经 embedding，所以可以根据向量的相似性进行寻找，一般向量数据库也提供了这样的能力。

<img class="w-140" src="/ragllm回答问题.png" />

需要注意的是，embedding 只是为了将内容转化为向量来求相似性，但 LLM 的输入是文本，所以在找出相似性之后，需要将 embedding 转化回文本。因为构建 RAG 使用的 embedding 与 LLM 的 embedding 不同，所以不能直接透传。

---

# RAG + LLM - 应用方向

### 自动化黑盒测试

本方案的目标本来就是为自动化黑盒测试生成元数据，所以这个流程可以被应用在自动化黑盒测试里。为组件生成元数据，让自动化黑盒测试的元数据及人工微调流程更为便捷。

<img class="w-180" src="/ragllm应用1.png" />

### 智能问答

考虑到长期使用闭源 cursor 或者 trae 会有安全风险问题，自己掌握这种技术以及流程可以让我们在 AI Editor 使用上有更多的选择，不至于在后面无法使用这些工具手足无措。

<img class="w-110" src="/ragllm应用2.png" />

---

# RAG + LLM - 应用方向

### 项目分析理解与组件可视化
对于现有项目来说，组件数量规模庞大，对于不熟悉（甚至熟悉）项目的同学来说，会有以下问题：

- 需要预览组件的时候非常困难，需要运行项目，寻找组件存在的页面，构造页面数据，渲染组件。
- 需要预览组件的某一个状态非常困难，需要搞清楚逻辑之后 mock 对应状态的数据，才能渲染出组件的对应状态。
- 迭代的时候需要清楚了解组件的所有状态，此时也会遇到上面的问题。

<img class="w-180" src="/ragllm应用3.png" />

通过本流程，可以让 AI 直接构造组件不同状态下的 mock 数据，可以根据需要选择持久化这些 mock 数据，从而解决上面提到的问题。

---

# RAG + LLM 里程碑

<div class="timeline-container my-10">
  <div class="timeline-line"></div>
  <div class="timeline-items">
    <div class="timeline-item">
      <div class="timeline-content top">
        <h3>2025-03-31</h3>
        <div>工程建设搭建</div>
        <div>RAG + LLM 流程</div>
        <div>基础 Prompt 设计</div>
      </div>
      <div class="timeline-dot"></div>
    </div>
    <div class="timeline-item">
      <div class="timeline-content bottom">
        <h3>2025-04-30</h3>
        <div>Mock 数据质量提升</div>
        <div>元数据生成优化</div>
        <div>持久化支持</div>
      </div>
      <div class="timeline-dot"></div>
    </div>
    <div class="timeline-item">
      <div class="timeline-content top">
        <h3>2025-05-31</h3>
        <div>黑盒测试整合</div>
        <div>生成器整合</div>
        <div>代码变更流程</div>
      </div>
      <div class="timeline-dot"></div>
    </div>
    <div class="timeline-item">
      <div class="timeline-content bottom">
        <h3>2025-06-30</h3>
        <div>AI 闭环优化</div>
        <div>代码覆盖率统计</div>
        <div>优化效果评估</div>
      </div>
      <div class="timeline-dot"></div>
    </div>
  </div>
</div>

<style>
.timeline-container {
  position: relative;
  height: 250px;  /* 改动：减小整体高度使内容更紧凑 */
  display: flex;
  align-items: center;
  padding: 0 40px;
}

.timeline-line {
  position: absolute;
  width: 100%;
  height: 2px;
  background: #93c5fd;
  top: 50%;
  transform: translateY(-50%);
}

.timeline-items {
  position: relative;
  display: flex;
  justify-content: space-between;
  width: 100%;
}

.timeline-item {
  position: relative;
  flex: 1;
  display: flex;
  justify-content: center;
}

.timeline-content {
  position: absolute;
  width: 200px;
  text-align: center;
  left: 50%;
  transform: translateX(-50%);
}

.timeline-content.top {
  bottom: 15px;  /* 改动：更靠近节点 */
}

.timeline-content.bottom {
  top: 15px;  /* 改动：更靠近节点 */
}

.timeline-content h3 {
  color: #3b82f6;
  font-size: 1.2rem;  /* 改动：增大标题字号 */
  font-weight: bold;
  margin-bottom: 0.2rem;  /* 改动：减小标题和内容间距 */
}

.timeline-content p {
  font-size: 1rem;  /* 改动：增大内容字号 */
  line-height: 1.3;  /* 改动：适当减小行高使文字更紧凑 */
}

.timeline-dot {
  width: 16px;
  height: 16px;
  background: #3b82f6;
  border-radius: 50%;
  position: absolute;
  top: 50%;
  transform: translateY(-50%);
}
</style>

---

# 总结

## 技术创新

- 结合 LLM 的智能化工具
- 自动化流程优化
- 多轮对话改进

## 业务价值

- 提升开发效率
- 降低维护成本
- 提高代码质量
