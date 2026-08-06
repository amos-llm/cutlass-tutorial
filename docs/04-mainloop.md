## 第 4 章:深入 CollectiveMainloop(本教程核心价值)

如果本教程只选一章细读,选这一章。Mainloop 是 CUTLASS 3.x 写得最繁的部分——也是体现"5 层抽象是否值钱"的地方。

主文件:`include/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp`。

本章主线:**跟着 producer / consumer 各自的 K-loop,把里面每一步 CuTe 计算**(layout 摊分、smem Tensor 构造、rmem fragment 分配、`cute::gemm` 推到 WGMMA)**逐个拆开**。

读法:**从类头部分支开始 → 类型别名 → 8 个方法的角色 → `load_init` 怎么构造 CuTe 视图 → `load` 里 producer 怎么发 TMA → `mma` 里 consumer 怎么推到 WGMMA → 收尾**。

### 本章涉及 CUTLASS 源文件

- `include/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp` — 主文件(`CollectiveMma` 主体)
- `include/cutlass/gemm/collective/sm90_mma_tma_gmma_*.hpp` 变体族(SS/RS/FP8/blockwise/grouped/sparse)
- `include/cutlass/gemm/dispatch_policy.hpp:265` — `MainloopSm90TmaGmmaWarpSpecialized<Stages, ClusterShape, Schedule>` tag
- `include/cutlass/pipeline/sm90_pipeline.hpp:271` — `PipelineTmaAsync<Stages>`(被 builder 推到 MainloopPipeline)

### 4.0 本章导航

```text
§4.1  类头部分支(在文件前 1/3 段)             ← 文件结构入口
§4.2  类型别名总解                             ← 8 个 CollectiveMma 模板参数对应什么
§4.3  mainloop 的三阶段:setup → K-loop → teardown  ← 全局节奏
§4.4  8 个方法的契约总结                       ← 速查表
§4.5  Cluster 维度:CTA 间协作                  ← 物理实现(get_slice / multicast)
§4.6  SharedStorage union                      ← smem 复用
§4.7  本章没有合适的图                          ← callout
§4.8  章末:读完这一章你该做得到的事            ← 自检 checklist
```

读两次建议:

- **第一次**(顺着读):§4.1 → §4.7 一把过,跟着 producer/consumer 走完 K-loop。
- **第二次**(按需查):§4.4 速查表 + §4.8 checklist 当入门考卷,卡住回对应章节重读。

### 4.1 类头部分支(在文件前 1/3 段)

```cpp
namespace cutlass::gemm::collective {

template <
  class MainloopSm90TmaGmmaWarpSpecialized,   // dispatch policy
  class TileShape_,                            // (M, N, K)
  class ElementA_, class StrideA_,
  class ElementB_, class StrideB_,
  class TiledMma_,
  class GmemTiledCopyA_,                       // 通常 = SM90_TMA_LOAD
  class SmemLayoutAtomA_,
  class SmemCopyAtomA_,
  class TransformA_,
  class GmemTiledCopyB_,                       // 通常 = SM90_TMA_LOAD
  class SmemLayoutAtomB_,
  class SmemCopyAtomB_,
  class TransformB_
>
struct CollectiveMma<
    MainloopSm90TmaGmmaWarpSpecialized,
    TileShape_, ElementA_, StrideA_,
    ElementB_, StrideB_,
    TiledMma_,
    GmemTiledCopyA_, SmemLayoutAtomA_, SmemCopyAtomA_, TransformA_,
    GmemTiledCopyB_, SmemLayoutAtomB_, SmemCopyAtomB_, TransformB_
> {
  ...
};
```

**15 个模板参数**。看似吓人,分两组:

|组|内容|
|---|---|
|上下文(由 builder 推)|`MainloopSm90TmaGmmaWarpSpecialized`、`TileShape`、`ElementA/B`、`StrideA/B`、`TiledMma`|
|8 个原子(`A` 和 `B` 各 4 个)|`GmemTiledCopyA/B`(通常是 TMA)、`SmemLayoutAtomA/B`(smem 上 A/B 的 layout)、`SmemCopyAtomA/B`(smem 内部微调 copy 用的 sub-atom,通常空)、`TransformA/B`(swizzle / interleave 这类 layout 变换)|

前 7 个由 builder 推;后 8 个 A/B 原子是 "**smem A/B 各自的 4 件套**"。你写 mainloop 时不会手写这些;改 builder 输入 `ElementA * LayoutA * AlignmentA` 后,这些都会被推出来。

### 4.2 类型别名总解

```cpp
// 在 collective_mma 内部的类型,按"出现频率"排:

using DispatchPolicy = MainloopSm90TmaGmmaWarpSpecialized;  // tag Ch8
using TileShape      = TileShape_;
using ElementA       = ElementA_;
using StrideA        = StrideA_;
using ElementB       = ElementB_;
using StrideB        = StrideB_;
using TiledMma       = TiledMma_;
using ArchTag        = typename DispatchPolicy::ArchTag;
using MainloopPipeline = cutlass::PipelineTmaAsync<DispatchPolicy::Stages>;
//    ↑ 来自 include/cutlass/pipeline/sm90_pipeline.hpp;producer/consumer 屏障数组

using PipelineParams = typename MainloopPipeline::Params;
using PipelineState  = cutlass::PipelineState<DispatchPolicy::Stages>;

using SmemCopyAtomA = SmemCopyAtomA_;
using SmemCopyAtomB = SmemCopyAtomB_;
using TransformA    = TransformA_;
using TransformB    = TransformB_;

using SharedStorage  = ...;   // 共享 smem 区段
using TensorStorage  = typename SharedStorage::TensorStorage;   // smem 上 A/B 张量布局
using PipelineStorage = ...;  // pipeline 屏障的 smem 布局
```

**关键看 `MainloopPipeline` 的来源**——它来自 `include/cutlass/pipeline/sm90_pipeline.hpp` 的 `PipelineTmaAsync<Stages>`:

- `PipelineTmaAsync<N>` = "N 阶段 TMA producer/consumer 流水线"
- 内部有 `FullBarrier[N]` 和 `EmptyBarrier[N]`——N 对 barrier 信号,`producer_acquire / producer_commit / consumer_wait / consumer_release` 操作屏障
- 这是 CUTLASS 实现"流水线 producer/consumer 同步"的底层原语。Mainloop 和 epilogue 都用。

> **你手写 GEMM 的对照**:你写 raw `mbarrier.*` PTX(`mbarrier.try_wait.parity` / `mbarrier.arrive.expect_tx`)+ `__syncthreads()` + 手动维护的 stage 状态。`PipelineTmaAsync<Stages>` 就是把这套封装起来。**不要写 `__threadfence_block()`**:那是 Ampere 之前 cp.async 还没硬件 barrier 时的 workaround,Hopper mbarrier 自带 release/acquire 语义,再写就冗余。

### 4.3 mainloop 的三阶段:setup → K-loop → teardown

先把整个 mainloop 拆成三阶段,再逐个方法钻进去。

| 阶段 | 调什么 | 做什么 |
|---|---|---|
| **setup**(host)| `to_underlying_arguments` + `can_implement` | host 端把 runtime `Arguments` 烧成 `Params`,顺带把 TMA descriptor 编好 |
| **setup**(device, kernel 入口)| `prefetch_tma_descriptors` + `load_init` | 把 TMA desc 提到 L1;构造 CuTe gmem 视图 `(gA_mkl, gB_nkl)` |
| **K-loop**(device)| producer 调 `load`;consumer 调 `mma` | TMA 拉数据 + WGMMA 算,流水线并发 |
| **teardown**(device)| producer 调 `load_tail`;consumer 调 `mma_tail` | 等 K-loop 收尾,做 cluster 防早退 + 异步 mma fence |

把 K-loop 接力赛跑想象成:

```text
[ Producer 线程]
  load_init → while (k < K) { load → state++ } → load_tail

[ Consumer 线程]
  load_init → while (k < K) { consumer_wait(state); mma(state); state++; } → mma_tail
```

下面 4 节按 `load_init → load → mma → *_tail` 顺序,**逐行讲里面每个 CuTe 计算在做什么**。读完这 4 节,Ch4 真正讲的是什么就清楚了。


### 4.4 8 个方法的契约总结

producer 和 consumer 之间的契约**只通过两类东西传**:

1. **smem 上的数据**(通过 pipeline barrier 同步可见性)
2. **`PipelineState`**(推进 stage、parity 翻转)

所以 `mma` 和 `load` 各自带一个独立的 `PipelineState`(`smem_pipe_read` / `smem_pipe_write`),**初始位置**也不同:

- producer 起点:`make_producer_start_state<MainloopPipeline>()`(phase 让首次 `acquire` 直接成功)
- consumer 起点:默认构造(`smem_pipe_read` 默认从 stage 0 开始)

两个 state 在 K-loop 里各自 `++`,互相不依赖——通过 barrier 的 arrive/wait 间接同步。这是 **Ch0「5 层互不依赖」的体现**:mainloop 内部怎么编排 producer/consumer 都是 CollectiveMma 的事,kernel 层只调 5 个方法 + 维护两块 PipelineState。

8 个方法的简要总结(完整版本见每个子节):

| # | 方法 | 谁调 | 关键 CuTe 计算 |
|---|---|---|---|
| 1 | `to_underlying_arguments` | host 一次 | `make_tma_copy_A/B_sm90` 编 TMA descriptor |
| 2 | `can_implement` | host 一次 | `check_alignment` TMA 128-bit 对齐 |
| 3 | `prefetch_tma_descriptors` | device 单线程 | `prefetch_tma_descriptor` 提 desc 到 L1 |
| 4 | `load_init` | device 每 warpgroup | `get_tma_tensor` + `local_tile` 构 gmem 视图 |
| 5 | `load` | producer K-loop | `make_tensor` smem view + `get_slice` cluster 分摊 + `partition_S/D` TMA thread 摊分 + `copy` 发 TMA |
| 6 | `load_tail` | producer 一次 | `producer_tail` 防 cluster 早退踢 L2 |
| 7 | `mma` | consumer K-loop | `get_slice` warpgroup 切 TiledMma + `partition_A/B` smem 摊分 + `make_fragment_A/B` rmem 分配 + `warpgroup_fence/arrive/commit/wait` async mma 同步 + `cute::gemm` 推 WGMMA |
| 8 | `mma_tail` | consumer 一次 | `warpgroup_wait<0>` 等所有 mma retire |

### 4.5 Cluster 维度:CTA 间协作

`cute::cluster_size_v<ClusterShape>`、`cute::block_rank_in_cluster()` 出现在 `tile_to_shape` / `subtile` / `get_slice(cluster_local_block_id)` 等地方——具体用法是:

```cpp
// 让当前 CTA 看 DSMEM 里"邻居 CTA 的 smem"
auto neighbor_smem_A = cluster_collective_load(...);
```

> 你写 Hopper 时如果用过 cluster launch(`cudaLaunchKernelEx` + `clusterDim`),这里对应。CTA 间 DSMEM 互访省 smem 注册流量——尤其在大 BlockM × BlockN 的 tile 上。

`load` 里 `get_slice(cluster_local_block_id)`(Ch4.5.2)就是 cluster 协作在 TMA 加载时的物理实现:每个 CTA 拿 TMA desc 的不同切片,**整个 cluster 协作加载同一个 (BLK_M, BLK_N) tile**。`SM90_TMA_LOAD_MULTICAST` 是更进一步的优化——同一份数据 multicast 给 cluster 内多个 CTA,不用各自重新加载。

### 4.6 SharedStorage union

回到 Ch2.6。Mainloop 用完 smem 之后,这块交给 epilogue 复用——`union { MainloopTensorStorage; EpilogueTensorStorage; }`。

变体速览:同一 `sm90_mma_tma_gmma_*` 文件名下还有这些变体,都靠 dispatch policy tag 路由Ch8:

|文件|变体|
|---|---|
|`sm90_mma_tma_gmma_ss.hpp`|非 warp-specialized(MMA 在所有 warp 上,无 producer/consumer 切分)|
|`sm90_mma_tma_gmma_ss_warpspecialized.hpp`|**默认**:1 producer warp group + 1 consumer warp group|
|`sm90_mma_tma_gmma_rs_warpspecialized.hpp`|A 在 register(RS 而非 SS)|
|`sm90_mma_tma_gmma_rs_warpspecialized_mixed_input.hpp`|A/B 不同 dtype|
|`sm90_mma_tma_gmma_ss_warpspecialized_fp8.hpp`|FP8|
|`sm90_mma_tma_gmma_ss_warpspecialized_fp8_blockwise_scaling.hpp`|FP8 + block scale|
|`sm90_mma_array_tma_gmma_ss_warpspecialized.hpp`|Grouped / ptr-array(SS)|
|`sm90_mma_array_tma_gmma_ss_warpspecialized_fp8.hpp`|FP8 grouped / ptr-array|
|`sm90_mma_array_tma_gmma_rs_warpspecialized_mixed_input.hpp`|Mixed-input grouped / ptr-array|
|`sm90_sparse_mma_tma_gmma_ss_warpspecialized.hpp`|2:4 structured sparsity(SS)|
|`sm90_sparse_mma_tma_gmma_ss_warpspecialized_fp8.hpp`|2:4 structured sparsity + FP8|

> Pingpong / Cooperative 变体在 **kernel 层**(`kernel/sm90_gemm_tma_warpspecialized_pingpong.hpp` / `_cooperative.hpp`),不是 collective 层文件。dispatch policy tag 路由后,它们复用同一个 `sm90_mma_tma_gmma_ss_warpspecialized.hpp` collective mainloop,只换 kernel 编排逻辑。

#### kernel 编排层的 3 个变体:WarpSpec / Pingpong / Cooperative

注意: Ch4.11 上面那张表里的「变体」(SS/RS/FP8/sparse/grouped)改的是 **mainloop 内部**(数据从哪里来、什么 dtype);这里的 3 个变体改的是 **kernel 编排逻辑**(几个 warp group 一起算一个 CTA 的事)。后者是「**producer / consumer 怎么分工**」,跟 collective mainloop 解耦:

| 变体 | 编排 | 适合 | 文件 |
|---|---|---|---|
| `KernelTmaWarpSpecialized` | **1 producer warpgroup + 1 consumer warpgroup**。producer 走 TMA 加载,consumer 走 WGMMA。固定分工,简单 | 中等 tile(BM/BN ≤ 128)| `kernel/sm90_gemm_tma_warpspecialized.hpp` |
| `KernelTmaWarpSpecializedPingpong` | **1 producer + 2 consumer 交替**(软硬流水线重叠加倍)| 中大 tile(BM/BN ≈ 128) | `kernel/sm90_gemm_tma_warpspecialized_pingpong.hpp` |
| `KernelTmaWarpSpecializedCooperative` | **1 producer + N 个 consumer 协同**算一个超大 tile(每个 consumer 算 tile 的一部分,WGMMA 之间互不重叠)| **巨型 tile**(BM=256 / BN=256 这种) | `kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp` |

机制差别一句话:

- **WarpSpec**: 1 个 consumer warpgroup 负责整个 tile 的 WGMMA,K-loop 串行。最简单、stage 数最少、对 smem 压力最小。问题是单 consumer 算 128×128 的 tile 已经是它的带宽上限,再大就瓶颈。
- **Pingpong**: 2 个 consumer **交替**——一个跑 K-step 0、另一个跑 K-step 1,软硬流水线叠加。每个 consumer 算**整个** tile,所以 K-loop 总吞吐 = 2 × 单 consumer 带宽。代价:smem pipeline 要 **double buffer**(stage 数 × 2)、smem 压力大。适合中等 tile 想再压一压。
- **Cooperative**: N 个 consumer **协同**算一个 tile,每个 consumer 只算 tile 的 **1/N**(比如 4 个 consumer 每人算 128×32)。tile 整体很大时,这种切分让每个 consumer 的 smem footprint 变小,可以塞更多 stage;同时 K-loop 总吞吐 = N × 单 consumer 带宽。代价:consumer 之间需要协调(谁负责哪个子区域),dispatch policy 里要算清楚 fragment 切分。

**怎么选?** 经验启发式(具体见 Ch9 的 `media/docs/cpp/heuristics.md` + `python/cutlass_library/heuristics.py`):

- BM × BN ≤ 128 × 128:WarpSpec 够用,smem 压力最小。
- 128 × 128 ~ 256 × 128:Pingpong 收益明显。
- ≥ 256 × 256:Cooperative 才喂得满。

`KernelScheduleAuto` 就是按这条启发式选;用户写 `KernelTmaWarpSpecialized` / `_Pingpong` / `_Cooperative` 之一就是显式覆盖。**这些 tag 之间没有 C++ 继承关系**Ch8——是同辈空 struct,builder 用 `is_same_v` 硬枚举路由。

### 4.7 本章没有合适的图

Ch4 之前引用过两张流水线图(`software-pipeline.png` + `cutlass-threadblock-mma-pipelined.png`),但**两张都不对**:

- `cutlass-threadblock-mma-pipelined.png` 画的是 CUTLASS 2.x Ampere 时代的 `MmaPipelined` 类(`IteratorA / IteratorB` + `ReadableTileIterator` 概念),跟 Ch4 讲的 `CollectiveMma` + TMA + WGMMA + warp specialization 完全不是一个抽象层
- `software-pipeline.png` 是通用 software-pipeline 图,没画 Hopper 的关键特征——producer/consumer warpgroup 分离、`smem_pipe_write` vs `smem_pipe_read` 的 stage 切换、TMA + mbarrier

CUTLASS 自带图集(`media/images/`)里**没有 Hopper 专属的 mainloop 流水线示意图**。Ch4 §4.4-§4.7 已经把 producer / consumer 各自的 K-loop 拆成 CuTe 计算步骤讲清,**用文字+表格描述比强行塞错时代的图好**。

Ch4 把 mainloop 讲完了。下一章 Ch7 看 epilogue——几乎是镜像结构,但**写回**(而不是计算)+**EVT 融合算子**(而不是纯 mma)是主要差异。

### 4.8 章末:读完这一章你该做得到的事

- ✅ 在 `sm90_mma_tma_gmma_ss_warpspecialized.hpp` 看到 `load_init / load / mma / load_tail / mma_tail` 5 个函数,能口述每个函数做了什么(gmem 视图构造 / TMA 加载 + smem 摊分 / WGMMA 循环 / cluster 早退保护 / K-retire 等待)。
- ✅ 能解释 `ProducerWarpRole` / `ConsumerWarpRole` 的实际差异,以及为什么 mainloop 的常见入口有 `if (Role == Producer) { ... } else { ... }` 两次分支。
- ✅ 看到 `make_tma_tensor(make_shape(M, K, L))` 知道这是从 TMA desc 拿 gmem Tensor,`local_tile(...)` 之后的 `(gA_mkl, gB_nkl)` 是 **collective/kernel 之间的契约**。
- ✅ 看到 `get_slice(cluster_local_block_id)` 知道这是 cluster 内 CTA 分摊 TMA 切片的物理实现,以及 `SM90_TMA_LOAD_MULTICAST` 怎么把同一份数据 multicast 给 cluster 内多个 CTA。
- ✅ 看到 `partition_S / partition_D` 知道是"把 smem tile 切到本 TMA thread";看到 `make_fragment_A / B` 知道是 rmem 分配。
- ✅ 看到 `cute::gemm(tiled_mma, tCrA, tCrB, accum)` 知道它在走 5-case dispatch,根据 `accumulate_` 决定 scale-out,根据 atom 类型推到 PTX 字符串。
- ✅ 看到 `warpgroup_wait<0>()` 在 `mma_tail` 出现**而不是** `K_PIPE_MMAS`,知道这是"等所有 WGMMA retire,epilogue 才能安全读 accum"。
- ✅ 看到 `make_producer_start_state<MainloopPipeline>()` 知道这是为了"首轮 acquire 立刻成功"——必须显式构造,默认 state 不行。
- ✅ 区分 `PipelineTmaAsync` 的 `producer_commit` 在 TMA 路径是 **NoOp**,在 cp.async 路径才真 arrive barrier。
- ✅ 看 8 个 `CollectiveMma` 变体(SS/RS/FP8/blockwise/fp8-grp/rs-mixed/2:4-sparse/ds-sparse)能讲出每个变体改的是 **mainloop 内部**还是 **kernel 编排**。
- ✅ 知道 Ch4 重点不是把 CollectiveMma 整段代码抄下来——重点是 5 个函数每个在 mainloop 轨迹里不同位置上的"做的事"。**Ch7 (DispatchPolicy) 讲 tag 怎么路由到这些变体,Ch9 (Builder) 讲策略怎么选**。

---