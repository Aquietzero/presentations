基于 figma + LLM 的 D2C 工作流
背景
●https://doc.weixin.qq.com/doc/w3_AaIAYwY5AOcXtnmbtqkRBSsxtoRSv?scode=AJEAIQdfAAoptpcW53AaIAYwY5AOc 前期整体设计，包括行业背景调研等
●https://doc.weixin.qq.com/doc/w3_AaIAYwY5AOcu0Mcbn9eQSSxEXvq3L?scode=AJEAIQdfAAoDN59tKaAaIAYwY5AOc&folder_share_code=AJEAIQdfAAo39u13iY 使用不同形式的 prompt 以及不同模型进行探究
目标
定量目标
●建立 D2C 工具链
○figma 插件：选择组件，获取结构以及样式
○生成服务：承接 figma 插件的调用，prompt 构造以及大语言模型的调用
○vscode 插件：将代码应用于项目代码，提供多轮对话优化接口
○组件渲染器：可视化生成代码的效果，与设计图进行比对
○使用信息统计
●建立 D2C 工作流
○设计规范，figma 使用规范
○选择与生成组件代码流程
○多轮对话微调
○代码应用与组件预览
●文档与专利沉淀
定性目标
●能通过 D2C 工作流产出较为高质量的可用代码
●通过这个工作流能节省一定的人力以及代码量
方案设计
概述
从https://doc.weixin.qq.com/doc/w3_AaIAYwY5AOcu0Mcbn9eQSSxEXvq3L?scode=AJEAIQdfAAoDN59tKaAaIAYwY5AOc&folder_share_code=AJEAIQdfAAo39u13iY 的效果中可以看到纯结构的 prompt 效果还算可以，有可以继续调优的可能性，考虑到自己训练多模态大模型不现实，以及市面上开源多模态模型并不多的前提下，可行性最高的做法依然是基于 figma 数据进行 prompt 构建，然后交由 coder LLM 进行代码生成。
结合市面上已有的工具进行分析，在整个 D2C 工作流里还有不少优化的空间，包括但不限于
●多状态组件的生成
●模块化生成代码
●标注 + 多轮对话提升生成质量
●代码预览以及应用
基础工具实现（目标中提到的工具集以及工作流）以及上面的优化点共同组成了本方案设计目标。
整体架构
下面是整体架构图，简单表述各组件之间的交互以及通讯内容。

●figma plugin：在 figma app 内部，获取设计稿的 dom 以及 style，通过请求的方式传到 D2C server 作进一步处理。
●D2C server：一个承接 D2C 主要工作流的服务器，用于接收 figma plugin 传出来的数据，构造合适的 prompt，然后与 LLM server 进行交互，获得代码。
●LLM server：大语言模型的服务器，用于将 prompt 转化为组件代码。
●vscode plugin：在 vscode（或者其他代码编辑器）内部，用于接收代码，并且将代码应用到对应的项目目录里。
●renderer：对生成代码进行可视化，是一个带有可执行环境的小型前端 sandbox 环境。
模块划分方式
基于上面的整体架构，实际模块划分还可以有其他可能。

这种组织方式可以让暴露在外的组件只有两个，分别是 figma plugin 以及 vscode plugin，不过把 D2C 流程，渲染器，coder 等集中在一个 vscode 插件里的坏处是不利于后续扩展，很多功能必须限制在了 vscode 窗口内部。
考虑到各组件的独立性，这个决策也不需要在最初的时候进行，可以独立实现组件，最后再结合实际使用来进行模块职责的调整。
安装与部署方式
基于整体架构，安装与部署方式也有多种可能性。

最极限的方案是全部组件都是本地的，好处是全链路可控，每个组件的服务选择都可以自由切换，不过缺点也很明显，本地部署大模型成本高，能部署的大模型也不可能是最优秀的大模型，另外执行速度也很慢，严重影响其他任务的执行。
退一小步的方案是使用线上的大模型服务，这个线上可以选择大模型服务提供商，或者是内网自行部署。选择提供商的好处是选择多以及速度快，坏处是成本与隐私问题问题。内网部署应该是一个比较合理的选择。
最后一个方案是将 D2C server 也放在 remote，好处是它可以作为一个公共服务提供给所有使用者，更进一步的好处是可以统一收集使用者的记录，进而优化整个流程。
建议最后的实现采用 本地插件 - 远程 D2C 服务 - 内网大模型 的方式。
组件设计
经过上面讨论，不是一般性，可以将整体架构里的每个组件独立实现，最后再根据需要组合成模块。下面分别给出每个组件的具体职责以及设计。
figma plugin
figma 插件的职责是获取 dom 结构以及 style 样式，然后将数据通过接口调用的方式传到 D2C server。
下面是 figma 插件具体需要的能力：
●【基础功能】支持选择节点
●【基础功能】支持多选节点
●【基础功能】显示节点 dom 结构以及样式
●【基础功能】调用 D2C server 接口发送节点结构及样式数据
●【进阶功能】支持人为划分组件
●【进阶功能】支持对划分的组件进行标注
●【进阶功能】发送数据的时候包含结构化的组件划分与标注
因为 figma 插件相对封闭，所以功能应该尽量简单，对数据不做过多处理，直接传出来交给 D2C server 处理，下面是 figma 插件架构图。

figma sdk 提供能力监听当前选择的节点。
window.onmessage = event => {
  const message = event.data.pluginMessage;
  if (message.type === 'node-info') {
    currentNodeData = message.data;
  }
}

如果要支持多选，需要维护一个本地 store 记录已经选择的节点，并且有移除已选节点的能力。
D2C server
D2C server 的职责是承接整个 D2C 流程的数据传递。下面是 D2C server 必须具备的分模块职责。
●figma 数据流（/figma）
○接收 figma plugin 的数据
●LLM 服务（/llm）
○figma 数据预处理（包括无效节点去除，DOM 结构组织，多状态结构组织等）
○根据需要构建 prompt（设计固定的 prompt 模板）
○对接 LLM 接口（如果是对接远程服务，需要有 api key 管理）
○对 LLM 调用的返回结果进行预处理（通过 prompt 限制输出格式，获取组件代码或者执行命令）
○切换模型的能力
○正常对话能力
●vscode 数据流（/vscode）
○生成的代码最终需要在 vscode 使用，可以考虑使用 ws 服务推送结果
○代码 diff 服务（这个可以通过大模型实现，也可以使用现有的 diff 库实现，主要用在多轮对话中）
○命令执行服务
以上职责主要针对 D2C 工作流的一个单向流程。如果 D2C server 部署在远端，那么还可以提供用户系统，项目系统等能力，用于记录用户信息，方便用户管理项目。不过这些功能并不阻塞 D2C 基础流程，也不是整个项目目标里的核心功能，可以在二期再考虑。
考虑到 LLM 服务一般处理时间较长，D2C server 可能还需要提供任务机制，调用 LLM server 的时候，以任务的方式进行提交，任务结束之后再主动通知。任务机制也可以用于 vscode 执行命令这个场景。
LLM server
测试环境或者开发环境，可以使用 ollama 本地部署的方式实现。生产环境可以考虑在 venus 平台私有化部署模型。
vscode plugin · renderer
vscode plugin 建立与本地 D2C server 的 ws 通讯，接收 D2C server 的推送。另外的代码相关功能采用接口调用的方式。vscode plugin 的职责如下：
●监听并接受命令或者代码推送
●使用内置 renderer 将代码渲染为组件
●应用代码
●基于代码的多轮对话优化
核心流程
多状态 D2C 流程
多状态 D2C 流程将在 figma 插件中允许多选，并且允许添加每种状态的标注，状态+标注构造为一个 figma 元数据，交给 D2C server 进行处理。
在 D2C server 内部先对数据进行清洗与一轮结构整理，将处理后的数据嵌入实现定义好的 prompt 模板中，交给 LLM server 生成回答。LLM 生成的回答会被 prompt 限定为组件代码或者命令。这个结果通过 D2C server 内部的 websocket server 推送到 vscode 插件。
vscode 插件接收代码之后，使用内置的 renderer 对代码进行渲染，如果用户觉得没问题，既可以应用代码到项目中。

自动化多轮优化流程
可以进一步通过建立 AI 闭环来进行多轮优化。这个闭环在下图的蓝色高亮部分表示。

要实现多轮闭环优化，需要在主要组件中实现下面的能力。
●figma plugin
○整合信息的时候，必须带上节点对应的截图。用于后面做图片比对
●D2C server
○图片比对能力可以使用现有算法，也可以使用图片匹配模型
○需要设计用于改进代码的 prompt
○渲染器需要内置到 D2C server 并且有截图能力
○建立 AI 闭环的应用逻辑
业务应用
下面列举一些 D2C 工作流的业务应用。
组件代码生成
组件代码生成是本方案最基本的目标。希望能达到设计稿的还原度 80% 左右的效果。针对各种独立组件。



上面这些都是看着比较简单的组件，实现也不需要很多时间，但是能直接高质量生成是可以节省不少时间的。
静态类页面生成代码
对于组件数量不多，相对静态的页面能直接生成。

多状态组件生成代码
目前市面上大多数工具不支持多状态组件的代码生成，但日常开发生产中绝大部分组件都是多状态的，本方案的另一个目标是覆盖这类场景。

里程碑
下面按月度定里程碑，一方面因为方案中有不少需要探索的环节，时间不方便确定；另一方面因为人力问题，每周不一定有明显的进展。
2025-03-31
这阶段主要工作为工程建设，流程搭建，实现以下事项。
基础流程跑通，包括 figma 插件，D2C server，LLM server 对接，renderer
基础 prompt 设计，可以通过 prompt 限制 LLM 输出为可执行代码
2025-04-30
这阶段主要工作是提升生成的代码质量，提高代码可用性。
更细致的 prompt 设计，支持组件拆分，文件拆分
prompt 可以将 LLM 输出限制为分文件的代码内容，以及包含命令字，比如”新建“，”修改“等
vscode 插件实现
应用代码
2025-05-31
这阶段主要工作是针对多状态组件的生成进行探索与实现
figma 组件支持多选节点，支持节点批注，整体批注
针对多状态结构数据的 prompt 设计
渲染器预览支持多状态
2025-06-30
这阶段主要工作是针对 AI 闭环多轮优化结果的探索与实现
实现 D2C server 内部的多轮应用逻辑
单轮结果评判的标准制定与实现
改进 prompt 的设计

