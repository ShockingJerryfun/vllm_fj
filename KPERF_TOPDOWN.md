# vLLM 0.26.0 默认V2 Decode八阶段与端到端PMU采集

本分支只对真实decode路径中的八个串行函数做CPU PMU打点。探针在函数外层
调用 `kperf_begin()`，并在 `finally` 中调用 `kperf_finish()`；原函数主体放在
对应的 `*_inner()` 中，八个统计区间不互相嵌套。

## 阶段边界

| 顺序 | 阶段名 | 外层函数 | 主要内容 |
| --- | --- | --- | --- |
| 1 | `add_requests` | `GPUModelRunner.add_requests()` | 把scheduler的新请求加入GPU侧持久请求状态 |
| 2 | `prepare_inputs` | `GPUModelRunner.prepare_inputs()` | 准备本轮token、position和模型输入buffer |
| 3 | `prepare_attn_runner` | `GPUModelRunner.prepare_attn()` | runner侧attention输入与metadata准备 |
| 4 | `prepare_attn_model_state` | `DefaultModelState.prepare_attn()` | model state侧attention metadata准备 |
| 5 | `run_fullgraph` | `CUDAGraphWrapper.run_fullgraph()` | replay完整CUDA Graph |
| 6 | `sample` | `GPUModelRunner.sample()` | logits处理与GPU采样 |
| 7 | `async_output_init` | `AsyncOutput.__init__()` | 创建异步输出并发起结果回传 |
| 8 | `postprocess_sampled` | `GPUModelRunner.postprocess_sampled()` | 更新请求状态并整理采样结果 |

920B和950额外采集 `execute_model_to_sample_tokens`，起点是Worker
`execute_model()` 进入，终点是 `sample_tokens()` 返回。它覆盖八阶段及区间之间
未单独打点的同一EngineCore/Worker执行线程代码，也包括两次Worker
调用之间的调度代码，但不包含异步输出线程随后等待D2H完成的部分。
由于当前计数器状态不支持嵌套，端到端通过 `KPERF_TARGET` 独立运行
time和13个PMU事件组；内部八阶段探针在这些轮次中为no-op。

本次采集配置以 `run_fullgraph` 为八阶段调用对齐锚点，只适用于走完整Graph的
decode轮次；eager和piecewise路径不属于这套八阶段统计口径。所有原始调用仍写入
明细页，第一轮prefill和未对齐调用不会进入汇总值。

这些数据是当前Python/C++执行线程的CPU PMU计数，反映主机调度、CUDA提交和
同步等待等CPU行为，不代表GPU内部cycles或GPU kernel效率。热点函数是独立的
`perf record` 采集，脚本优先把容器内完成符号解析的报告写入工作簿。

每次完整采集先单独运行一个不打开PMU的 `time` 轮次，再逐组运行PMU轮次。
time轮次同时记录 `perf_counter_ns()` 墙钟时间和 `thread_time_ns()` 当前线程
CPU时间；PMU轮次只做事件组reset、enable、disable和read，不再混入墙钟计时。
汇总中的 `time(us)` 取time轮次的墙钟平均值，`CPU利用率` 按同一阶段有效
decode样本的 `sum(thread CPU time) / sum(wall time)` 计算。它表示被打点线程在
该墙钟区间内的CPU占用比例，不是进程全部线程或整机利用率；脚本不裁剪或归一化
实测结果。

## 920B与950 Topdown L2/L3口径

920B的Topdown宽度为6，950（HIP12）为8。两者共用以下Backend事件：
`0x7001` execution stall、`0x7005` any-load stall、`0x7006` store stall、
`0x7007` L1 miss stall、`0x7008` L2 miss stall、`0x7009` L3 miss stall。
汇总比率均先对同一轮的分子和分母分别求和，再做除法。

- `Retire = retired / (width * cycles)`，`BadSpec = (spec-retired) / (width*cycles)`。
- `Branch Mispredicts = BadSpec * branch_mispredict / (branch_mispredict + ROB flush)`，
  `Machine Clears = BadSpec - Branch Mispredicts`。
- `Core Bound = execution stall / cycles - Memory Bound`。
- `Memory Bound = (any-load stall + store stall) / cycles`。
- `L1/L2/L3 Bound` 依次为 `(0x7005-0x7007)`、`(0x7007-0x7008)`、
  `(0x7008-0x7009)` 除以cycles；`Mem Bound=0x7009/cycles`，
  `Store Bound=0x7006/cycles`。
- 920B的 `Nuke Flush=0x200f/0x2010`；950为 `0x203f/0x2040`。
- 950额外计算 `Resource Bound=0x7000/cycles`；920B没有对齐事件，显示
  `未采集`。

因Core Bound与Memory Bound位于不同PMU组，跨组差值是各自重复轮次
归一化后的比率相减；脚本不对跨轮差值做裁剪或强制归一化。

## 脚本结构

- `scripts/run_topdown.sh`：宿主机统一入口。
- `scripts/config.env`：唯一的运行参数和芯片事件组配置。
- `scripts/run_one.sh`：容器内启动服务、发送请求并采集单个事件组或热点。
- `scripts/parse_run.py`：保留全部原始记录并标记prefill、decode和未对齐调用。
- `scripts/build_xlsx.py`：公共Excel生成器。
- `scripts/<芯片>/run.sh`：芯片采集流程。
- `scripts/<芯片>/report_config.json`：芯片事件列、公式和工作表配置。

目前支持 `920b`、`950` 和 `hygon_c86_7490`。公共生成逻辑复用，芯片事件和
公式仍由各自目录维护。

## 使用方法

容器内第一次使用时安装唯一的报表依赖：

```bash
/opt/vllm/bin/python3 -m pip install \
  -r /home/fj/vllm_topdown/scripts/requirements-report.txt
```

先在 `scripts/config.env` 中修改芯片、容器、仓库、模型和采集参数，
然后在宿主机执行：

```bash
bash /home/fj/vllm_topdown/scripts/run_topdown.sh
```

默认输入7000、输出100、模型简写 `qwen3`、版本简写 `0.26`。最终文件按
“芯片_vLLM版本_模型简写_输入输出”命名，例如：

```text
920b_vllm0.26_qwen3_7k100.xlsx
```

可在 `scripts/config.env` 中修改 `MODEL_SHORT` 和
`VLLM_VERSION_SHORT` 以改变文件名中的模型与版本。

## Excel内容

- 第一个sheet固定为 `汇总`，只汇总对齐的decode调用。920B、950和Hygon使用
  完全相同的指标行及顺序；920B和950的端到端列放在八阶段之前。
  芯片没有采集的等价事件显示
  `未采集`，已确认事件不响应的指标显示 `未支持`。
- `cycles` 下一行固定为 `cycle占比`；`time(us)` 和
  `CPU利用率` 来自独立time轮次。
- `IPC` 和 `Retire` 分别输出，不合并为一行。
- 920B按6-wide文档、950按HIP12的8-wide文档输出Retire、Frontend、
  BadSpec和Backend分层。Backend的Memory Bound继续拆分为L1/L2/L3/Mem/Store；
  950另输出Resource Bound，920B该行显示 `未采集`。
- 阶段比率按 `SUM(分子)/SUM(分母)` 计算，不对每行比率再取平均。
- 第二个sheet固定为 `热点函数`，使用容器内 `perf report` 解析后的报告。
- 其余明细sheet保留全部原始记录，包括prefill、decode和未对齐调用；明细中的
  `时间(us)` 也来自独立time轮次，`time_enabled/time_running` 只描述PMU调度。
- `prepare_attn` 明细包含runner与model state两个区段；`output` 明细包含
  `async_output_init` 与 `postprocess_sampled` 两个区段。
- 920B和950每个事件组另有一个 `execute_to_sample` 明细sheet；端到端列的
  `cycle占比` 为 `不适用`，不会加入八阶段cycles总和。
- 普通sheet冻结首行和首列；热点正文保持左对齐。
- 汇总sheet底部直接附带time轮 `benchmark.log` 中的请求吞吐、token吞吐和
  TTFT/TPOT/ITL等实际输出项。
- 920B和950均为13组PMU，默认各生成93个sheet；Hygon七组Core PMU加一组
  独立L3 Uncore采集，默认生成50个sheet。

每次运行结果位于 `results/<芯片>/<RUN_ID>/`，包含原始日志、解析CSV、
`summary.csv`、`collection_quality.csv`、热点文件、最终Excel和
`commands.txt`。`commands.txt` 记录当次实际展开后的vLLM服务、
benchmark、`perf record`、`perf report`、解析和报表命令。只有最终
Excel存在且非空时，宿主机入口才报告成功。
