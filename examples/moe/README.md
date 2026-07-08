# DeepSeek 算子卸载赛题说明

## 1. 赛题背景

大模型推理系统正在从单纯追求峰值算力，逐步转向对端到端延迟、尾延迟和系统资源协同效率的综合优化。以 DeepSeek、Mixtral 等模型中广泛使用的 MoE（Mixture of Experts）结构为例，FFN 层通常占据推理计算和权重访存的主要开销。MoE 通过路由器为每个 token 选择少量专家参与计算，降低了理论计算量，但也带来了专家权重规模大、访存模式不规则、尾延迟对系统调度和缓存策略敏感等新的系统问题。

在真实部署中，模型权重往往远大于单机 CPU cache 或加速器片上存储容量。对于 MoE FFN 这类权重密集型算子，如果能够将部分计算靠近数据所在位置执行，就有机会减少主机侧数据搬移和资源争用。因此，本赛题以 SPDK（Storage Performance Development Kit，一套用户态高性能存储开发框架）NVMe-oF target 为框架，将问题抽象为“DeepSeek 算子卸载”：MoE FFN 被全部卸载到存储侧执行，主机侧通过 NVMe vendor-specific 命令提交输入向量，并通过标准 READ 取回输出向量。

本赛题关注系统软件和算子实现的协同优化。初赛中的 MoE 经过了简化，只保留单层 MoE FFN 计算，不包含完整大模型中的多层 Transformer、Attention、KV Cache、采样等推理流程。参赛队伍需要在给定框架中优化存储侧 MoE FFN 执行路径，使 benchmark 观测到的高百分位端到端延迟尽可能低。

如果你此前没有接触过 SPDK，可以把它理解为一套用户态高性能存储开发框架。传统存储 I/O 往往需要经过内核协议栈和块层，而 SPDK 尽量在用户态通过轮询、无锁队列、大页内存等机制减少系统调用和中断开销，常用于构建低延迟 NVMe SSD、NVMe-oF target 和自定义块设备服务。本赛题不会要求参赛队伍完整掌握 SPDK 的所有模块，但需要理解它在这里扮演的角色：把主机发来的 NVMe 请求送到 target 侧实现，并把 target 侧计算结果返回给主机。

## 2. 赛题描述

仓库中的 `examples/moe` 提供了本赛题的基本框架：

### 2.1 术语说明

本赛题中几个常见术语的含义如下：

- SPDK：Storage Performance Development Kit，用户态高性能存储框架。本赛题使用它搭建 target、NVMe-oF 协议栈和自定义 bdev。
- NVMe-oF：NVMe over Fabrics，可以理解为通过网络协议访问 NVMe 设备。本赛题使用本地 TCP transport。
  initiator 和 target 运行在同一台机器上，通过 `127.0.0.1:4420` 通信。
- target：被访问的一端，即运行 `moe_tgt` 的存储侧服务。MoE FFN 计算发生在 target 侧。
- initiator：发起访问的一端，即运行 `moe_bench` 的评测程序。它负责生成输入、发送请求并统计延迟。
- bdev：SPDK 中的块设备抽象。发布版参考实现使用名为 `moe_bdev_0` 的 bdev 承接请求，但它只是内部实现方式，不是正式测评要求保留的外部接口。
- vendor opcode：NVMe 允许厂商自定义的命令编号。本赛题用 `0xC1` 表示“提交一个 MoE 输入向量并在 target 侧计算”。
- READ：标准 NVMe 读命令。本赛题中 vendor opcode 完成计算后，`moe_bench` 再用 READ 取回输出向量。

### 2.2 框架组成

- `examples/moe/tgt/moe_tgt`：存储侧 target 进程，启动 SPDK event 框架并加载 `examples/moe/moe_tgt.json`。
- `module/bdev/moe`：自定义 bdev 模块，创建 `moe_bdev_0`，接收 NVMe vendor opcode `0xC1`，在 target 侧执行 MoE FFN。
- `lib/moe_ffn` 与 `include/moe_ffn`：MoE FFN 参考实现和配置。demo 配置为 `d_model=2048`、`d_ff=7168`、`num_experts=8`、`top_k=2`。
  正式测评中的 `d_model` 和 `d_ff` 与 demo 完全一致。
- `examples/moe/bench/moe_bench`：随赛题发布的参考 benchmark initiator。每个样本会生成随机输入。
  计时区间覆盖 vendor opcode 计算请求和 READ 取回输出的完整本地 TCP NVMe-oF 往返路径。

### 2.3 参考实现与运行权限

上述 target 侧实现只是参考框架，不是必须沿用的架构。以下实现都可以重构、替换，甚至推倒重来：

- `module/bdev/moe`
- `lib/moe_ffn`
- `include/moe_ffn`

最终要求是 target 能与官方 initiator 按约定接口对接，并通过正确性和稳定性验证。

demo 实现刻意采用逻辑直白、便于阅读和调试的写法，并未追求性能。它包含大量可优化空间，效率极其低下。

参赛队伍在重构以下路径后，如果看到非常夸张的性能提升，属于正常现象：

- 计算路径。
- 内存管理路径。
- I/O 路径。

demo 为了降低本地运行门槛，放弃了一些需要特权权限或额外系统配置的 SPDK/NVMe 特性，因此在常见环境下不依赖 root 权限也能运行。

正式提交允许使用 root/sudo 启动程序。参赛队伍可以自行选用需要特权权限的特性，例如：

- 设备解绑。
- SPDK 用户态 NVMe 驱动。
- hugepage 配置。
- I/O 调度。
- CPU/NUMA 亲和性设置。

### 2.4 优化目标与测评方式

初赛评测对象是单层 MoE FFN 的一次前向计算。每个请求只包含一个输入向量，target 侧需要完成：

- 路由。
- Top-K 专家选择。
- 专家 FFN 计算。
- 加权求和。
- 返回一个输出向量。

题目名称中的 DeepSeek 算子卸载用于描述问题背景和优化方向，并不要求参赛队伍实现完整 DeepSeek 模型。

参赛目标是在保证以下前提不变的情况下，优化 `moe_tgt` 侧实现，使同一计时范围下观测到的尾延迟尽可能低：

- 结果正确。
- 接口兼容。
- 测评协议不变。

重点指标为 P99（99% 分位数）和 P99.99（99.99% 分位数）。参考 benchmark JSON 字段分别为：

- `latency_ns.percentiles.p99`
- `latency_ns.percentiles.p99_99`

随赛题发布的 `moe_bench` 是一个简单实现，仅供开发参考，用于帮助参赛队伍理解：

- 请求流程。
- 计时范围。
- 本地调试方式。

最终测评阶段会使用实现更完善的本地 TCP initiator 完成性能测评。它与发布版 `moe_bench` 保持一致的部分包括：

- 原理相同。
- 计时范围相同。
- initiator 和 target 仍运行在同一台机器上。
- 不引入跨机器网络延迟。

参赛队伍不应依赖发布版 `moe_bench` 的具体实现细节进行优化或规避。

发布版 `moe_bench` 采用单线程串行请求模型，同一时刻只有一个请求处于处理流程中。初赛阶段的最终性能测评也采用相同模型，不考察以下场景：

- 多 initiator。
- 多 qpair。
- 多请求并发下的吞吐扩展能力。

一次请求的基本时序如下：

```text
initiator                    NVMe-oF target / moe_tgt
    |                                  |
    |  vendor opcode 0xC1 + input      |
    |--------------------------------->|
    |                                  |  MoE router + Top-K
    |                                  |  expert FFN + weighted sum
    |                                  |  cache output for READ
    |  command completion              |
    |<---------------------------------|
    |                                  |
    |  READ output                     |
    |--------------------------------->|
    |                                  |
    |  output vector                   |
    |<---------------------------------|
```

### 2.5 正式规模与内存约束

Demo 参数与初赛参数对比如下：

| 项目 | Demo | 初赛正式测评 |
| --- | --- | --- |
| `d_model` | 2048 | 2048 |
| `d_ff` | 7168 | 7168 |
| 专家总数 | 8 | 256 |
| 每次激活专家数 | 2 | 8 |
| 权重常驻方式 | 全部专家可放入内存 | 约 8 个专家的权重容量可用于内存 buffer，其余位于 SPDK NVMe SSD |
| target DRAM 限制 | 无 cgroup 限制，实际运行低于 2GB | 2GB cgroup 限制 |
| SSD 使用 | 非必需 | 单块 NVMe SSD，可交由 SPDK 接管 |
| 测评并发 | 单线程串行 | 单线程串行 |

需要特别注意，仓库中的 demo 只用于说明接口和运行方式：

- demo 给出 8 个专家。
- demo 中全部专家权重均可放置在内存中。
- demo 每次推理激活 2 个专家。

正式测评中，`d_model=2048`、`d_ff=7168` 与 demo 完全一致，输入输出向量大小不变。变化的是：

- 专家总数扩大为 256。
- 每次推理激活 8 个专家。
- target 进程最多使用 2GB DRAM，由 cgroup 限制。

一个专家的权重指该专家对应的 `W_up`、`W_gate`、`W_down`。8 个专家的权重大约占用 1.3GB，主办方额外预留约 700MB DRAM。这里的“约 8 个专家”是容量估算，不是硬性要求。

参赛队伍可以按总体 2GB DRAM 约束自行权衡 buffer 数量。例如，管理结构和其他内存需求较大时，可以只保留 7 个专家的权重 buffer；如果通过优化节省了更多内存，也可以保留 9 个或更多专家的权重 buffer。最终要求是总内存使用满足 cgroup 限制。

约 700MB 余量可以按需用于：

- 权重缓存。
- 输入输出缓冲。
- 索引。
- 调度状态。
- 其他运行时数据结构。

以下内存也计入该内存预算，不在 2GB 限制之外额外提供：

- hugepage。
- SPDK 内存池。
- 其他预留内存。

参赛实现应面向“专家权重无法全部常驻内存”的场景设计，不应依赖 demo 中全量权重常驻内存的简化假设。

### 2.6 SSD 与权重组织

内存放不下的专家权重应放置在 SPDK 接管的 NVMe SSD 中。初赛环境开放单块 NVMe SSD。

参赛队伍可以将该设备从内核 NVMe 驱动解绑，并交由 SPDK 用户态驱动管理。

如何在 SPDK 内使用这块硬盘，属于参赛队伍的设计空间，例如：

- 权重文件或块布局。
- 读取粒度。
- 预取策略。
- 缓存替换。
- I/O 与计算流水。
- 队列深度。
- poller 调度。

推荐参赛队伍在接入 NVMe SSD 后进行一次全盘 TRIM。这样可以使 SSD 尽量处于稳定、干净的初始状态，并减少以下影响：

- 不同 SSD 历史写入状态对延迟的影响。
- 不同 SSD 状态对公平性的影响。

正式测评环境也会尽量采用统一的 SSD 初始化流程。

权重文件的初始形态仍然位于普通文件系统中，例如 demo 使用的 `/tmp/moe_weights`。

赛题组会提供权重生成程序。该程序生成的权重可以被发布版 `moe_tgt` 正确读取并处理。

参赛实现只需保持从文件系统读取权重这一层的格式和对齐逻辑不变。推荐流程是：

- target 初始化阶段先从文件系统读取权重。
- 再按自己的布局写入或组织到 SPDK 接管的 NVMe SSD 中。

不建议参赛队伍提前把权重手工写入裸盘，再将设备从内核解绑后交给 SPDK。权重进入 SPDK 接管的 NVMe SSD 后，内部形式可以自由设计，例如：

- 盘上块布局。
- 索引格式。
- 分片方式。
- 压缩或重排方式。

只要最终计算结果正确即可。这样可以保持评测流程清晰，也便于主办方统一下发和校验权重文件。

### 2.7 基础运行流程

基础运行流程如下：

```bash
make -C examples/moe

# 终端 1：启动 target。权重文件需位于 /tmp/moe_weights。
cd examples/moe
./tgt/moe_tgt moe_tgt.json

# 终端 2：运行 benchmark。
cd examples/moe
./bench/moe_bench --cpu=0 --warmup=3 --runtime=10 --json-output=./moe_bench_result.json
```

`moe_bench` 支持以下参数：

- `--seed`
- `--cpu`
- `--warmup`
- `--runtime`
- `--json-output`

正式评分时，主办方会在统一软硬件环境、统一权重文件、统一测评口径下运行本地 TCP 测评 initiator。

## 3. 相关要求

参赛队伍需要提交可编译、可运行的完整代码。target 侧改动范围原则上开放，包括但不限于：

- `examples/moe/moe_tgt.c`
- `module/bdev/moe`
- `lib/moe_ffn`
- `include/moe_ffn`
- target 侧构建配置
- 必要时对 SPDK 内部模块的侵入式修改，例如 `lib/nvmf` 中的 controller/request 处理路径。

发布版 bdev、MoE FFN library 和 target 组织方式均仅供参考。参赛队伍可以按自己的方案重构或替换。

随赛题发布的参考 benchmark `examples/moe/moe_bench.c` 不作为最终成绩依据。

除非赛题补充说明明确允许，参赛队伍不得通过修改以下内容来伪造、夸大或混淆本地自测结果：

- 计时逻辑。
- 统计逻辑。
- 连接逻辑。
- 输出字段。

提交说明、性能报告和交流材料中引用本地 benchmark 数据时，应基于未修改计时和统计口径的参考 benchmark。

提交版本必须保持以下接口兼容：

- NVMe-oF target 默认监听 `127.0.0.1:4420`，NQN 为 `nqn.2024-07.io.spdk:moe_target`。
- host-to-controller vendor opcode 为 `0xC1`。
- 输入和输出均为 `MOE_D_MODEL` 个 `float`。正式测评的 `MOE_D_MODEL` 与 demo 一致，输入和输出大小应与 `MOE_D_MODEL * sizeof(float)` 保持一致。
- 权重文件默认从 `/tmp/moe_weights` 加载，权重生成程序输出的文件命名和维度需与 `moe_config.h` 保持一致。

内部实现不要求必须保留参考 bdev 或参考 MoE library 的代码结构。官方只要求对外协议语义保持一致：

- initiator 能连接 target。
- initiator 能通过约定 vendor opcode 提交输入。
- initiator 能通过约定方式取回当前请求对应的正确输出。

参赛队伍需要在第 2.5 和第 2.6 所述规模、内存和 SSD 约束下完成正确计算，不得假设 256 个专家权重能够全部常驻 DRAM。

具体验证环境将由赛题组后续统一公布，包括：

- CPU 型号。
- 操作系统类型。
- Kernel 版本。
- NVMe SSD 规格。
- cgroup 版本。

若参赛实现依赖特定 CPU 特性或平台能力，建议及时与赛题组反馈沟通，并尽早申请验证环境进行调试。相关特性包括但不限于：

- AVX/AVX2/AVX-512。
- NUMA 拓扑。
- 特定 cache 行为。
- 特定 NVMe 指令或 SSD 特性。

优化实现必须保证计算正确性。最终测评会包含一套独立的准确性验证 initiator，使用独立权重、独立输入和参考结果进行验证。

准确性判定采用逐元素绝对误差：

- 对输出向量每个元素计算 `fabs(output[i] - expected[i])`。
- 取最大值 `max_absolute_error`。
- 要求 `max_absolute_error < MOE_MAX_ABS_ERROR`（当前宏值为 `1e-5f`）。

主办方可使用隐藏校验程序或参考实现，对多个随机 seed、多个输入样本和不同运行轮次进行结果比对。未通过准确性验证的提交，其性能成绩无效。

禁止通过以下方式获取非正常成绩：

- 修改、绕过或伪造 `moe_bench` 的延迟统计和 JSON 输出。
- 依赖发布版参考 benchmark 的实现细节，规避 vendor opcode、READ 和输出正确性的协议语义。
- 跳过实际 MoE FFN 计算、返回固定结果、缓存与当前输入不匹配的结果。
- 依赖评测 seed、输入分布或权重内容进行硬编码。
- 绕过 cgroup 内存限制，或通过外部常驻进程、共享内存、tmpfs 等方式规避 2GB DRAM 预算。
- 修改评测环境、系统时间、CPU 频率策略或主办方指定的运行脚本。
- 使用未在提交说明中声明、且评测环境不可复现的外部服务或预加载程序。

鼓励的优化方向包括但不限于：

- 减少 target 侧动态内存分配、拷贝和同步开销。
- 优化 Top-K、Softmax、SwiGLU FFN、矩阵向量乘等热点路径。
- 改善权重布局、cache locality、NUMA 亲和性和预取策略。
- 设计面向 256 专家规模的权重缓存、淘汰和按需加载策略，在有限专家权重 buffer 和 SPDK NVMe SSD 之间控制尾延迟。
- 用好 SPDK 接管的单块 NVMe SSD，例如优化盘上布局、异步读取、队列深度、I/O 与计算重叠和热点专家缓存策略。
- 利用 SIMD、线程并行、批内流水或 SPDK poller 模型降低尾延迟。
- 在不破坏接口的前提下优化 NVMe-oF target 配置和 bdev 请求处理路径。

## 4. 评分标准

最终成绩以官方最终测评环境中本地 TCP 测评 initiator 生成的结果为准。评测会先检查程序是否能正常完成以下流程：

- 构建。
- 启动。
- 连接。
- 预热。
- 正式运行。

评测还会通过独立准确性验证 initiator 校验输出正确性。准确性要求为逐元素最大绝对误差小于 `MOE_MAX_ABS_ERROR`（当前宏值为 `1e-5f`）。

正确性是性能评分的前置条件。未通过正确性校验的提交，其性能结果无效，不参与排名。

正式评分以如下配置下的运行结果为准：

- `d_model=2048`
- `d_ff=7168`
- 256 个专家。
- 每次激活 8 个专家。
- target 进程受 2GB DRAM cgroup 限制。

demo 中的 8 专家、每次激活 2 专家场景只作为本地开发、接口理解和功能验证样例，不作为最终性能排名依据。

性能分由 P99 和 P99.99 延迟决定，延迟单位为纳秒，数值越低越好。评分公式如下：

```text
score_latency = 0.6 * normalized(p99) + 0.4 * normalized(p99_99)
```

其中 `normalized(x)` 由主办方按照所有有效提交在对应指标上的相对排名计算。

两个指标含义如下：

- P99 体现稳定高负载下的大多数尾部请求表现。
- P99.99 体现极端尾延迟控制能力。

若两支队伍综合性能分接近，依次比较：

- `p99_99`
- `p99`
- 平均延迟
- 吞吐率 `rate_calls_per_sec`
- 多轮运行稳定性

总分构成如下：

- 性能得分：90%。依据官方最终测评 initiator 的 P99 与 P99.99 延迟计算。
- 工程质量：10%。包括代码可读性、改动范围合理性、构建方式清晰、优化说明完整、无明显不可复现实验依赖。

主办方可进行多轮评测并取中位数或最优稳定轮次，以降低系统抖动影响。

若提交在官方最终测评环境下出现稳定性问题，主办方可酌情要求队伍提交补丁进行修复。补丁要求如下：

- 单次大小不超过 10KB。
- 同时提供修复说明。

稳定性问题包括但不限于：

- 偶发崩溃。
- 偶发错误输出。
- 偶发连接失败。
- 内存超限。
- 无法稳定生成 JSON。

补丁只能用于修复稳定性或环境适配问题，不得引入新的性能优化逻辑或改变评测接口。补丁是否合规、是否接受，由赛题组判断。

若无法在规定时间内完成修复，或修复后仍不能稳定通过官方测评，则该提交成绩无效。

所有队伍应以官方公布的最终评测脚本、硬件配置和运行参数为准。

需要依赖硬件或系统特性的实现，应以赛题组公布的验证环境为适配目标，并尽早在该环境中完成验证。

赛题规则及测评结果的最终解释权归赛题组所有。
