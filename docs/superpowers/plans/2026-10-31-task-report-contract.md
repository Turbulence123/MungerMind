# 任务与报告契约 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把一次性问答升级为任务内多轮，并用统一的模式输出契约产出八段决策报告。

**Architecture:** Task 持有状态、取消与上一轮上下文；ReverseReAct 通过适配器实现 `ReasoningMode`，把既有结构化结果映射为 `ModeResult`；`build_report` 把 `ModeResult` 装配成 `DecisionReport`；CLI 负责任务循环与落盘。

**Tech Stack:** Python 3.13、Pydantic 2、unittest（离线，脚本化假模型）

**Spec:** `docs/superpowers/specs/2026-10-31-task-report-contract-design.md`

## Global Constraints

- Python >= 3.13，不新增依赖。
- 测试离线运行，不访问网络、不需要 API Key，沿用脚本化假模型。
- 不改 ReverseReAct 的循环、校验与停止规则；只加一个 `context` 参数。
- 原有 39 个测试的断言一条不改。
- 不做预算、暂停、持久化、专家、调度器、工具网关。
- 中文 docstring 与注释，风格与现有模块一致。

## Review Focus

- 空问题或纯空白：CLI 在创建 Task 前退出，不应崩溃。
- 第二轮上下文：只带上一轮且截断，避免提示无限增长。
- Ctrl+C 发生在流式输出中：任务置取消，不再发起新调用。
- 报告路径不可写：报错退出，不静默吞掉报告。
- 模型输出缺段：报告不得伪造证据，缺段标「未生成」。

---

### Task 1: Task 状态机与上下文

**Files:**
- Create: `mungermind/core/task.py`
- Modify: `mungermind/core/exceptions.py`, `mungermind/core/__init__.py`
- Test: `tests/test_task_report.py`（`TaskStateTest`）

**Interfaces:**
- Produces: `TaskState`（六个成员）、`Turn(question, answer)`、`Task(question, *, task_id=None)`，方法 `start() / wait_for_user() / finish() / cancel() / fail(reason) / add_turn(q, a) / context_text(max_chars=600)`，属性 `state / turns / question / id / is_terminal`；异常 `TaskStateError`。

- [ ] **Step 1: 写失败测试**

```python
def test_task_rejects_illegal_transitions(self) -> None:
    task = Task("要不要换工作？")
    with self.assertRaises(TaskStateError):
        task.finish()
    task.start(); task.wait_for_user(); task.start(); task.cancel()
    self.assertEqual(task.state, TaskState.CANCELLED)
    with self.assertRaises(TaskStateError):
        task.finish()
```

- [ ] **Step 2: 运行，确认失败**

Run: `.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_task_report.py" -k TaskStateTest -v`
Expected: FAIL（`ModuleNotFoundError: mungermind.core.task`）

- [ ] **Step 3: 实现**

`_TRANSITIONS` 只允许 `created → running|cancelled`，`running → waiting_for_user|completed|cancelled|failed`，`waiting_for_user → running|completed|cancelled|failed`；非法转移抛 `TaskStateError`。`context_text` 只取最后一轮，超过 `max_chars` 截断并加 `…`。

- [ ] **Step 4: 运行，确认通过**

Run: 同上。Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add mungermind/core/task.py mungermind/core/exceptions.py mungermind/core/__init__.py tests/test_task_report.py
git commit -m "feat(task): add task state machine and scoped context"
```

---

### Task 2: 模式契约与 ReverseReAct 适配

**Files:**
- Create: `mungermind/reasoning/mode.py`, `mungermind/reasoning/adapters.py`
- Modify: `mungermind/agents/reverse_react.py`, `mungermind/reasoning/__init__.py`
- Test: `tests/test_task_report.py`（`ModeResultTest`）

**Interfaces:**
- Produces: `FindingRef(statement, status="unverified", source="")`；`ModeResult(mode, answer, stop_reason, findings, evidence_refs, unresolved_issues, next_actions, externally_verified, details)`；`ReasoningMode.run(task, question) -> ModeResult`、`ReasoningMode.run_stream(task, question, *, on_progress=None) -> Iterator[str]`；`ReverseReActMode(agent)` 带 `last_result`；`to_mode_result(ReverseReActResult) -> ModeResult`。
- Consumes: Task 1 的 `Task.context_text()`。

- [ ] **Step 1: 写失败测试**

断言 `mode.run()` 返回的 `answer`、`stop_reason`、两条 `supported` 发现、`next_actions == ["估算成本"]`、`details["cycles"]`。

- [ ] **Step 2: 运行，确认失败**

Run: `.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_task_report.py" -k ModeResultTest -v`
Expected: FAIL（`ModuleNotFoundError: mungermind.reasoning.adapters`）

- [ ] **Step 3: 实现**

`ReverseReActAgent.run / run_with_result / run_stream / _execute / _run_cycles / _run_cycle` 增加 `context: str = ""`，分析提示词增加 `{task_context}` 段；适配器把每轮 findings、最后一轮 actions/unresolved、证据引用与 `details.cycles` 映射过去。

- [ ] **Step 4: 运行，确认通过**

Run: 同上。Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add mungermind/reasoning mungermind/agents/reverse_react.py tests/test_task_report.py
git commit -m "feat(reasoning): add mode contract and reverse-react adapter"
```

---

### Task 3: 八段决策报告

**Files:**
- Create: `mungermind/core/report.py`, `mungermind/reasoning/report.py`
- Test: `tests/test_task_report.py`（`ReportTest`）

**Interfaces:**
- Produces: `NOT_GENERATED`、`SECTIONS`（八段的字段名与中文标题）、`DecisionReport(task_id, question, turn, mode, problem, criteria_and_gaps, options, experts_and_models, evidence, counter_arguments, recommendation, next_steps, stop_reason, externally_verified)`，`to_dict()` 用占位符填充空段，`render_markdown()`；`build_report(task, mode_result, question, *, turn=1) -> DecisionReport`。

- [ ] **Step 1: 写失败测试**

断言八个段名都在 `to_dict()` 中、`options == NOT_GENERATED`、`recommendation` 等于答案、`evidence` 含未核实主张、Markdown 含「未生成」。

- [ ] **Step 2: 运行，确认失败**

Run: `.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_task_report.py" -k ReportTest -v`
Expected: FAIL（`ModuleNotFoundError: mungermind.core.report`）

- [ ] **Step 3: 实现**

`build_report` 映射：第 1 段写问题并注明目标/约束未单独整理；第 5 段列未核实主张与外部验证状态；第 6 段列非 supported 发现；第 7 段用模式答案；第 8 段用行动与未决事项；第 2/3/4 段留空。

- [ ] **Step 4: 运行，确认通过**

Run: 同上。Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add mungermind/core/report.py mungermind/reasoning/report.py tests/test_task_report.py
git commit -m "feat(report): add eight-section decision report"
```

---

### Task 4: CLI 任务循环与 `--report`

**Files:**
- Modify: `mungermind/__main__.py`
- Test: `tests/test_task_report.py`（`CliMultiTurnTest`）

**Interfaces:**
- Consumes: Task、ReverseReActMode、build_report。
- Produces: `main(argv, *, llm_factory=LLM)` 的交互循环；`--report PATH`（`.json` 写 JSON，其余写 Markdown）；带问题参数仍单次运行。

- [ ] **Step 1: 写失败测试**

两问的 stdin + 10 条脚本化响应，断言第二次分析提示含「上一轮问题」与第一轮答案、报告文件可解析且 `recommendation` 为第二轮答案。

- [ ] **Step 2: 运行，确认失败**

Run: `.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_task_report.py" -k CliMultiTurnTest -v`
Expected: FAIL（`TypeError: main() got an unexpected keyword...` 或报告文件缺失）

- [ ] **Step 3: 实现**

保持既有行为：带问题参数单次运行、stdout 只有答案、进度与提示走 stderr、Ctrl+C 退出码 0 且打印「已取消」。无参数时循环追问，空行/exit/EOF 结束任务；每轮结束 `wait_for_user`，退出时 `finish`。

- [ ] **Step 4: 运行，确认通过**

Run: `.\.venv\Scripts\python.exe -m unittest discover -s tests -v`
Expected: 43 个测试全部 PASS（原 39 + 新 4）

- [ ] **Step 5: 提交**

```bash
git add mungermind/__main__.py tests/test_task_report.py
git commit -m "feat(cli): run tasks across turns and write reports"
```

---

### Task 5: README 同步

**Files:**
- Modify: `README.md`, `README.zh-CN.md`

- [ ] **Step 1: 改掉三处失实**

「package ships as a skeleton」→ 说明 ReverseReAct 与任务/报告契约已可用；「Twelve built-in libraries」→「planned / in progress」；roadmap 勾上 ReverseReAct 与任务报告契约。

- [ ] **Step 2: 全量测试**

Run: `.\.venv\Scripts\python.exe -m unittest discover -s tests`
Expected: OK

- [ ] **Step 3: 提交**

```bash
git add README.md README.zh-CN.md
git commit -m "docs: sync README with the usable CLI"
```
