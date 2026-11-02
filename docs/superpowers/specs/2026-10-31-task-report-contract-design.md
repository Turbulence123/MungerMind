# 任务与报告契约（A 路线，最小切片）· 设计说明

- 日期：2026-10-31
- 依据：`MungerMind_v0.2.md` 2.3、3.2、4.3、9.1、10.3、12.1
- 范围：把「一次性问答」变成「任务内多轮」；给已有范式一个统一输出契约；产出 2.3 的八段报告。不碰专家、思维模型内容、调度器、工具网关、隐私模式、前端。

## 意图

ReverseReAct 已能独立跑通，但仍是孤岛：没有任务对象，没有模式间契约，没有报告结构。之后接专家、调度器、权限时，这三样都会被临时发明一次再返工。这一刀只立三个契约：

- **Task**：状态、取消、任务内上下文归程序层，模式只读；
- **ModeResult**：模式产出什么，编排层只认这个；
- **DecisionReport**：给用户看什么，固定八段，缺的明说。

## 目标

1. 同一 Task 内可多轮追问，第二轮能看到第一轮的结论。
2. Task 持有取消与任务内上下文，模式不持有这些状态。
3. 取消后不再发起任何模型调用。
4. ReverseReAct 成为第一个合规模式，结构化结果完整保留。
5. `--report` 写出八段报告（json/md），缺的段落标「未生成」。
6. 四组离线测试覆盖上面五条，不依赖网络与 API Key。

## 非目标

- 不做预算控制（模型调用次数、token、时间、成本）；成本需求明确后再加。
- 不做持久化与进程重启恢复；上报文件只是快照。
- 不做暂停；暂停需要前端与恢复语义，随第 3 章一起做。
- 不给 simple / react / reflection / plan_solve 做结构化输出，只保留现有 Agent。
- 不定义 EvidenceItem / ExpertOpinion 的字段；证据结构与证据层一起设计。
- 不建模式注册表；注册表属于第 6 章调度器。

## 备选与取舍

- **A 契约优先（选中）**：Task + ModeResult + Report。后面所有模块都读这三样，先立不返工；代价是这一版没有新能力。
- **B 内容优先**：先做思维模型库与专家。演示效果好，但专家输出格式和报告字段只能先拍脑袋。
- **C 证据优先**：先做工具网关。能解决「复核=盖章」，但权限与取消依赖 Task，规格也把它排在阶段 2。

## 关键决策（我的判断，非用户指定）

| 决策 | 取值 | 理由 |
|---|---|---|
| 状态集 | created / running / waiting_for_user / completed / cancelled / failed | 只有这六个在 A 里被真正使用；paused 与 cleanup_pending 各自等前端和隐私层 |
| 取消语义 | Task 记状态；引擎已有的 `cancel()` 负责「不再发起调用」 | 10.3 要求的是尽力取消与真实状态记录，不引入新机制 |
| 报告内容 | 八段纯文本字段，未生成留空，输出时统一标注 | 2.3 规定的是内容；结构化数据留在 ModeResult.details，等证据/专家层再升级 |
| 多轮上下文 | 只带上一轮问答，截断 600 字 | 最小可用；滚动窗口是 9.5 的话题，交给记忆层 |
| 交互模式 | 无参数启动时循环追问，空行 / exit / EOF 结束任务；带问题参数仍是一次性运行 | 保持现有 CLI 行为与测试兼容 |

## 模块与职责

| 文件 | 内容 |
|---|---|
| `core/task.py` | TaskState、Turn、Task（状态机、取消、上下文） |
| `core/report.py` | DecisionReport（八段）、NOT_GENERATED、render_markdown |
| `core/exceptions.py` | 增加 TaskStateError |
| `reasoning/mode.py` | FindingRef、ModeResult、ReasoningMode 协议 |
| `reasoning/adapters.py` | ReverseReActMode（run / run_stream / to_mode_result） |
| `reasoning/report.py` | build_report(task, mode_result, question, turn) |
| `agents/reverse_react.py` | 只加一个 `context` 参数，循环逻辑不动 |
| `__main__.py` | 任务循环、`--report` |

附带：README 里「只是骨架」「12 学科 built-in」「roadmap 缺 ReverseReAct」三处失实一并改掉。

## 测试（只测核心）

1. 状态机：非法转移被拒，终态不可再动（含取消）。
2. 适配：ReverseReActResult → ModeResult 的答案 / 发现 / 行动 / 停止原因。
3. 报告：八段齐全、缺段标「未生成」、JSON 可序列化。
4. 多轮与落盘：第二轮提示含第一轮结论；`--report` 真的写出文件。

不测：Markdown 排版、枚举解析、上下文截断边界、帮助文本、日志格式。

## 验收

- 同一 Task 连问两轮，第二轮的分析提示里出现第一轮答案。
- `--report r.json` 产出含八个段的文件，缺的段落为「未生成」。
- 取消后 Task 状态为 cancelled；引擎不再发起调用（既有测试覆盖）。
- 原有 39 个测试不改断言、全部通过。

## 留给后续

- 预算：Task 已是程序层状态容器，调用计数加在这里不会返工。
- 调度器：ModeResult 已够做路由比较；模式注册表在 6.1 落地。
- 专家层：ExpertOpinion 填报告第 4 段。
- 证据层：EvidenceItem 替换 evidence_refs 字符串，`externally_verified` 变成计算结果。
- 前端：Task 的六个状态可直接映射状态视图与确认流程。
