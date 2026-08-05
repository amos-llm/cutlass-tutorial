# CUTLASS 3.x 教程:从手写 Hopper GEMM 到读懂工业级 GEMM 库

> **适用对象**:你已经手写过 Hopper GEMM 并达到 ~100% cuBLAS 性能,熟悉 WGMMA、cp.async / TMA、warp specialization、shared memory swizzle、thread block cluster。
>
> **本文不教**:GPU 入门 / GEMM 算法基础 / WGMMA 编码细节 / mma intrinsic 编程。
>
> **本文教**:CUTLASS 3.x 为什么要切片成 5 个 C++ 抽象,每个抽象在你已经熟悉的机制里**对应**什么,从哪里读起,怎么改默认值,以及为什么 Blackwell 上同样的 5 层抽象依然能复用。

---

## 目录

- [序章:为什么 CUTLASS 长成这样——5 件事、5 层类、5 个理由](#preface)
- [第 1 章:5 层架构图(整体鸟瞰)](#ch01-architecture)
- [第 2 章:跑通一个最小 GEMM——走读 `examples/48`](#ch02-minimal-gemm)
- [第 3 章:CuTe——CUTLASS 真正的"语言"](#ch03-cute)
- [第 4 章:深入 CollectiveMainloop(本教程核心价值)](#ch04-mainloop)
- [第 5 章:深入 CollectiveEpilogue + EVT](#ch05-epilogue)
- [第 6 章:Kernel orchestrator + TileScheduler](#ch06-kernel-orchestrator)
- [第 7 章:DispatchPolicy——tag-inheritance 模式](#ch07-dispatch-policy)
- [第 8 章:CollectiveBuilder——把"形状 + 类型"压成具体实现](#ch08-collective-builder)
- [第 9 章:`examples/49`——一个用户故事](#ch09-user-story)
- [第 10 章:调参世界观 + `cutlass_profiler`](#ch10-tuning)
- [第 11 章:Blackwell 桥接——同样的 5 层架构,换了一组原子](#ch11-blackwell)
- [附录 A:Hopper 原语 ↔ CUTLASS 封装文件(速查表)](#app-a-primitives)
- [附录 B:In-tree 散文定位("你想读 X 看 Y")](#app-b-treemap)
- [附录 C:你手写 GEMM 的 X 行 ↔ CUTLASS 哪里(对照表)](#app-c-mapping)
- [附录 D:再之后(Grouped / Sparse / Conv / SSD / PDL)](#app-d-future)

---

## 序章:为什么 CUTLASS 长成这样——5 件事、5 层类、5 个理由 {#preface}

你手写 Hopper GEMM 时,脑子里并行地想了这些事:

1. **问题切片**:M、N、K 三个维度各自切多大块(`BlockM × BlockN × BlockK`,你可能叫 tile size 或 threadblock shape)。
2. **CTA 调度**:`gridDim.x/y/z` 该怎么排,几个 CTA 跑同一个 tile、要不要做 L2-locality swizzle、要不要 persistent(每个 CTA 处理多 tile)。
3. **加载 A/B**:gmem → smem 走 TMA 还是 cp.async,描述符怎么准备、smem 怎么 swizzle 防 bank conflict、几条 stage 流水线。
4. **mma 计算**:A、B 在 smem 里怎么布局 → WGMMA,fragment 累积器怎么分给 warp / lane、流水线推进。
5. **写回 C/D**:做完的 accumulator 怎么搬到 gmem,要不要顺手做 bias / ReLU / swizzle。

CUTLASS 3.x 把这件事**正好**切成 5 个 C++ 类,每类一一对应一件事:

```text
┌─ 切层类(对应 GEMM 5 件事)
│
├─ 1. cutlass::gemm::device::GemmUniversalAdapter       ──  host 句柄:开工、收尾
├─ 2. cutlass::gemm::kernel::GemmUniversal<...>          ──  CTA 调度 + 主循环 orchestrator
├─ 3. cutlass::gemm::collective::CollectiveMma           ──  加载 A/B → mma(事 3 + 事 4)
├─ 4. cutlass::epilogue::collective::CollectiveEpilogue  ──  写回 C/D + 融合算子(事 5)
└─ 5. cutlass::gemm::kernel::*TileScheduler              ──  切 tile + 排队(事 1 + 事 2)
```

![CUTLASS 五层架构图](media/images/cutlass-layered-organization.png)

> **如果你只能记住这张图,就够了。** 后面每一章只是把这一张图"哪个方框里住着什么、代码长什么样、对应你手写的哪一段"展开。

### 5 个设计理由(为什么这样切、不那样切)

CUTLASS 3.x 设计文档 `media/docs/cpp/cutlass_3x_design.md` 里列了 5 条:

1. **可组合性**:5 层之间用模板参数互不依赖,可以单独替换某一层而不动其他层。比如你想改 epilogue 加 swizzle,只换第 4 个模板参数,不动 mainloop。
2. **可配置性**:每个具体实现都用"dispatch policy" tag(空 struct)标记,builder(第 8 章)用 tag 路由到对应 partial specialization——用户写 `Auto` 而不是写具体类名。
3. **关注点分离**:mma 编程模型(`CollectiveMma`)和 epilogue 融合(`CollectiveEpilogue`)是两个完全不同的领域专家的工作,CUTLASS 把它们解耦,各自演化。
4. **硬件可移植**:同一套 5 层框架可以在 Hopper、Blackwell、Volta、Ampere、Turing 上工作;具体某一层(比如 mma atom)从 `WGMMA` 换到 `UMMA` 不影响其他 4 层。
5. **默认正确**:`Auto*`(`StageCountAuto`、`KernelScheduleAuto`、`EpilogueScheduleAuto`)总是选一个合理的实现,你得手写错才能跑错——这是 3.x 比 2.x 显著进步的一处。

### 本教程的承诺

- **重写、自成一体**——不依赖读者已经读过 `media/docs/cpp/` 任何一篇文章。但附录 B 给"如果你想深挖,看哪里"的导航。
- **单文件**——便于全文搜、git 单 file diff、转 PDF。章末锚点 `<a name>` 可直接跳转。
- **类名不省略**——每个 `using`、`constexpr`、`typename`、`operator()` 前缀都不缩。
- **你不熟悉的 CUTLASS 概念 ↔ 你已经会的 Hopper 概念**——每个新类、每个调度 tag、每个 `SharedStorage` 段都有手写对照 callout。

---

## 第 1 章:5 层架构图(整体鸟瞰) {#ch01-architecture}

本章只立骨骼:每一层是谁、长在哪个文件、它做什么。具体细节在 Ch2–Ch11 展开。

### 1.1 第 1 层:Adapter(host 句柄)

文件:`include/cutlass/gemm/device/gemm_universal_adapter.h`。

**角色**:整个 GEMM 在 host 端的"开机关收尾"。你写:

```cpp
Gemm gemm;                                  // 实例化
auto arguments = args_from_options(options); // 组装 6 元组 Arguments
size_t ws = Gemm::get_workspace_size(arguments);
Gemm::device_memory::allocation<uint8_t> workspace(ws);
CUTLASS_CHECK(gemm.can_implement(arguments));   // 校验:类型/对齐/layout 合不合法
CUTLASS_CHECK(gemm.initialize(arguments, workspace.get())); // 编译所有模板参数,绑定参数
CUTLASS_CHECK(gemm.run());                       // 启动 kernel
```

5 个动作正好对应:**新建物理对象 → 算 workspace → 校验 → 实例化参数 → launch**。

Adapter 实质上是一个薄 stateful handle——内部持有一个 kernel `Params params_`。它本身没有任何 GEMM 知识,所有知识委托给内核。

> **你手写 GEMM 的对照**:你的 "host launch" 是一个 `cudaLaunchKernel(grid, block, args)`。Adapter 把这一行拆成"建句柄 → 校验 → 算 workspace → 绑参数 → launch"五步,每一步都暴露出来,所以你可以查可以插桩。

### 1.2 第 2 层:Kernel orchestrator(主循环 + 调度入口)

文件:`include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized.hpp`(对 Hopper SM90);还有 `sm90_gemm_tma_warpspecialized_pingpong.hpp`、`sm90_gemm_tma_warpspecialized_cooperative.hpp`、`sm90_gemm_tma.hpp`、`sm100_gemm_tma_warpspecialized.hpp` 等变体。

**角色**:kernel 体本身——CTA 之间谁跑哪个 tile、CTA 内 producer/consumer 各跑什么、pipeline 怎么推进、cluster barrier 怎么同步。模板签名:

```cpp
template <
  class ProblemShape_,        // M, N, K
  class CollectiveMainloop_,  // 第 3 层
  class CollectiveEpilogue_,  // 第 4 层
  class TileScheduler_        // 第 5 层(默认 void)
>
class GemmUniversal { ... };  // 这里包含 CUTLASS_DEVICE operator()(Params, char* smem_buf)
```

模板实例化时——这就是 Adapter 包的"东西":

```cpp
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int>,     // ProblemShape
    CollectiveMainloop,
    CollectiveEpilogue
>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
```

`TileScheduler_` 默认是 `void`,在 kernel 内部通过 `TileSchedulerSelector` 路由为 `PersistentTileSchedulerSm90`(SM90 Hopper 默认)。

> **你手写 GEMM 的对照**:你的 kernel 体本身——`__global__ void gemm(...) { ... 主循环 ... }`。CUTLASS kernel 把"我调度谁做什么"这件事做得更繁:生产者 warp group 调 `CollectiveMainloop::load`,消费者调 `mma`,收尾调 epilogue。这里**比手写更繁**,因为 CUTLASS 把 mainloop/epilogue/scheduler 都拆出去了(下文第 3–5 层)。

### 1.3 第 3 层:CollectiveMainloop(加载 A/B + mma)

主文件:`include/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp`。

**角色**:把 A/B 从 gmem 拉入 smem、smem 入 register、用 WGMMA / UMMA 做 mma、再把结果放进 accumulator。这一层封装最繁——这一章只在 Ch4 才展开。

调用接口标准签名(对任意架构、任意 mma 形态;成员对应 Ch4.3 的实际实现):

```cpp
struct CollectiveMma {
  // 一次性:把 host 端的指针 / stride 转成 device 端参数
  MainloopParams to_underlying_arguments(...);

  // 一次性:预取 TMA descriptor(单线程幂等)
  static void prefetch_tma_descriptors(MainloopParams const&);

  // 一次性:校验形状/对齐/类型是否合法
  static bool can_implement(...);

  // K-loop 起点:K-loop 进入前各做一次
  void load_init(...);

  // K-loop 中被反复调用(producer:load / consumer:mma)
  void load(...);
  void mma(...);

  // K-loop 末尾:把剩下的 mma 跑完 / 清空 producer pipeline
  void mma_tail(...);
  void load_tail(...);

  // SharedMemory / Pipeline 字段
  using TensorStorage  = ...;   // smem 上的 A/B 张量布局
  using PipelineStorage = ...;  // producer/consumer 屏障
  using SharedStorage   = ...;  // = union { TensorStorage; EpilogueStorage reserve; }
};
```

> ⚠ 这里**没有** `load_wait` / `mma_next`——早期 CUTLASS 草稿里有这两个 helper,后来为了减少函数拆分,等价的 `consumer_wait(state)` 与 `state++` 已经直接内联在 `mma(...)` 的函数体里。这一节只列当前实际存在的成员,Ch4.3 给出对应实现。

这一层决定:加载用 TMA 还是 cp.async、A 从 smem 还是直接 rmem、单 vs 双 warp-group、FP8 block-scale 等等。

> **你手写 GEMM 的对照**:你的 mainloop 函数——`__device__ void main_loop(cta, k_iter)` 自己写所有这些。CUTLASS 把这些都内置,你要选哪个变体,就换 dispatch policy(Ch7)。

### 1.4 第 4 层:CollectiveEpilogue(写回 + 融合)

主文件:`include/cutlass/epilogue/collective/sm90_epilogue_tma_warpspecialized.hpp`。

**角色**:`D = alpha * (A*B) + beta * C` 及更复杂的链——bias、ReLU、silu、top-K softmax、等等。EVT(Epilogue Visitor Tree,`include/cutlass/epilogue/fusion/`)是融合算子的"小 DSL"。

```cpp
struct CollectiveEpilogue {
  EpilogueParams to_underlying_arguments(...);
  static void prefetch_tma_descriptors(...);

  // CTA 内 epilogue warp group 进入
  void operator()(
      ProblemShape, // M, N, K + 分块偏移
      TileShape,    // 本 CTA 拿到的 M*N 子区域
      Coord /* m, n offset */,
      ...);
};
```

> **你手写 GEMM 的对照**:你的 epilogue 函数——`__device__ void epilogue(cta, k_done)` 自己写 bias / ReLU / store。CUTLASS 默认只给你标量路径,你想加 bias+relu 就用 EVT 把 `Sm90Compute<...>` 节点串起来。

### 1.5 第 5 层:TileScheduler(切 tile + 排队)

主文件:`include/cutlass/gemm/kernel/sm90_tile_scheduler.hpp`(默认 PersistentTileSchedulerSm90),变体:`sm90_tile_scheduler_stream_k.hpp`、`sm90_tile_scheduler_group.hpp`。

**角色**:把 `(M, N)` 切成若干 `(blockM, blockN)` tile,给每个 CTA 一个 work id 或一组 work id。

```cpp
struct PersistentTileSchedulerSm90 {
  // 在 kernel 入口被每个 CTA 调用,决定它跑哪个 tile
  WorkTileInfo get_current_work();

  // 在 epilogue 之后调用,推进到下一个 tile
  void advance();
};
```

Scheduler 也有自己的参数(由 host 端 `Arguments.scheduler` 填):

```cpp
arguments.scheduler.raster_order     = options.raster;       // AlongN / AlongM / Heuristic
arguments.scheduler.max_swizzle_size = options.swizzle;      // 1 / 2 / 4 / 8
```

> **你手写 GEMM 的对照**:你的 `grid_dim` 计算 + `blockIdx` 分配策略。CUTLASS 默认走"persistent + swizzle + raster"——见 Ch6 详细。

### 1.6 三句话侧栏:这些 `sm90_*_warpspecialized` 文件不是"5 个不同实现"

看到 `include/cutlass/gemm/collective/` 下有几十个 `sm90_*_warpspecialized*.hpp` 不要慌——它们是**同一个 partial specialization 的 6 种维度变体**:

|维度|取值|
|---|---|
|mma 形态|GMMA(SS:smem↔smem) / RS(A 在 register) / Array(多 CTA 协同)|
|调度|WarpSpecialized(1 个消费者 warp group) / Pingpong(2 个交替) / Cooperative(多 warp group)|
|类型|f16 / bf16 / tf32 / fp8 / fp8+block-scaling / mixed-input / sparse|
|输入|A/B 都是 gmem(A gmem→smem) / A register(A gmem→register→mma)|

第 7 章会讲——这种变体是怎么靠"tag-inheritance"模式**同构**地生成的,你改 dispatch policy tag 就换实现,不用"换文件"。

---

## 第 2 章:跑通一个最小 GEMM——走读 `examples/48` {#ch02-minimal-gemm}

本教程**最重要**的一章。所有 5 层如何拼起来、host 端 API 长什么样、`<...>` 里那一串 dispatch-policy tag 怎么设——全部以这一章为入口展开。

文件:`examples/48_hopper_warp_specialized_gemm/48_hopper_warp_specialized_gemm.cu`。**实质 user-facing 代码量很小,主体是 `Options` 解析 + reference GEMM + `main`**——这一章只读 user-facing 部分。

### 2.1 上手:用 CUTLASS 跑一个 TF32 Hopper GEMM 需要哪几步

```cpp
// 步骤 1:include 集合
#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"

// 步骤 2:using 别名集合(描述你的 GEMM)
using ElementA   = float;
using LayoutA    = cutlass::layout::RowMajor;
constexpr int AlignmentA = 128 / sizeof(float); // 4

using ElementB   = float;
using LayoutB    = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 4;

using ElementC   = float;
using LayoutC    = cutlass::layout::ColumnMajor;
constexpr int AlignmentC = 4;

using ElementAccumulator = float;
using ArchTag            = cutlass::arch::Sm90;
using OperatorClass       = cutlass::arch::OpClassTensorOp;
using TileShape           = Shape<_128, _128, _32>;   // BlockM × BlockN × BlockK
using ClusterShape        = Shape<_4, _2, _1>;        // block 集群大小

// 步骤 3:用 builder 拼 epilogue + mainloop
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutC, AlignmentC, ElementC, LayoutC, AlignmentC,
    cutlass::epilogue::collective::EpilogueScheduleAuto
>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccumulator,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;

// 步骤 4:GEMM 对象
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
```

读完之后,这个 4 步骤模式适用于一切 CUTLASS 3.x GEMM。换 tile size、换类型、换 schedule 都是改某些 using 而已。

### 2.2 拆解 using 别名

下面把上面那段"看着就头大"的 using **逐行**说明。

#### Element / Layout / Alignment(4 个一组,描述一份输入)

```cpp
using ElementA   = float;                  // 类型:FP32(经 TMA 加载到 smem 后变 TF32)
using LayoutA    = cutlass::layout::RowMajor;  // 行主序
constexpr int AlignmentA = 4;              // 4 个 fp32 = 128-bit,TMA 要求
```

> **Alignment 是什么?** 你的设备端指针有 128/256 bit 对齐要求时,要告诉 CUTLASS。`4 * sizeof(float) = 16` byte = 128 bit。RowMajor + N col-major 其实跟 stride 有关,TMA 能容忍非连续 stride 也支持 strided 模式。

#### ArchTag / OperatorClass

```cpp
using ArchTag      = cutlass::arch::Sm90;            // SM 9.0 = Hopper SM
using OperatorClass = cutlass::arch::OpClassTensorOp; // tensor core 路径(WGMMA)
```

> **你手写 GEMM 的对照**:等价于你的 `-arch=sm_90a -DUSE_WGMMA`。

#### TileShape / ClusterShape

```cpp
using TileShape    = Shape<_128, _128, _32>;  // 一个 CTA 拿 128×128 个 output tile,K 每片 32
using ClusterShape = Shape<_4, _2, _1>;       // 4×2×1 = 8 个 CTA 协作
```

`Shape<a, b, c>` 是 CuTe 的静态整数 tuple(Ch3 会深讲)。语义就是:

- **BlockM × BlockN**:一个 CTA 在 K-loop 结束后覆盖的 output tile 大小,常 128×128 或 256×128。
- **BlockK**:K 维 pipeline 的单 stage 大小,常 32 / 64。
- **ClusterShape**:配合 `cutlass/cluster_launch.hpp` 一起 launch,这 8 个 CTA 可以用 DSMEM(分布式 shared memory)互相看对方的 smem。

> **你手写 GEMM 的对照**:就是 `dim3 block_dim(...); dim3 grid_dim(...);`,加上 sm_90a 的 `cluster_dim = cdim(...)` 调用。Tile 越大,smem 越大、潜在 stage 数越少;Cluster 越大,CTA 间通信越频繁——调优第 10 章。

#### `*Auto*` 系列 tag(3 个)

```cpp
using StageCountType = cutlass::gemm::collective::StageCountAutoCarveout<...>; // 第 3 个参数
using KernelSchedule = cutlass::gemm::collective::KernelScheduleAuto;          // mainloop 调度
// (epilogue 也有) cutlass::epilogue::collective::EpilogueScheduleAuto;
```

这一行一次性出现了 3 个"auto"型的 tag,这是新手最容易懵的地方。它们不是"留给编译期决定"——它们**精确选了一个 dispatcher**,只是你不用手填。"auto"这一字背后是 builder 的 partial specialization 路由,Ch7 + Ch8 会讲清。

具体含义:

|Tag|默认挑什么|修改它意味着什么|
|---|---|---|
|`StageCountAutoCarveout<epi_bytes>`|smem 减去 epilogue 占用的字节后,塞下尽可能多的 mainloop stage|想强制 `StageCount<4>` 之类手动调|
|`KernelScheduleAuto`|`KernelTmaWarpSpecialized`(单消费者 warp group)|改 `KernelTmaWarpSpecializedPingpong`(双 warp group)或 `_Cooperative`|
|`EpilogueScheduleAuto`|`Sm90TmaWarpSpecialized` (TMA store)|改 `Sm90NoSmemWarpSpecialized`(纯 smem store)|

> **你手写 GEMM 的对照**:相当于你写"如果 smem 还够,就多塞几 stage;否则少塞"。CUTLASS 把它当 first-class 类型暴露是因为别的 GEMM 库都没有这种 step-by-step 的 auto 化。

### 2.3 CollectiveBuilder 模板参数逐个

```cpp
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag,           // 1.  架构:Sm90
    OperatorClass,     // 2.  Operator class:tensor core

    ElementA,          // 3.  A 元素类型
    LayoutA,           // 4.  A 内存布局
    AlignmentA,        // 5.  A 对齐(单位:元素)

    ElementB,          // 6.  B 元素类型
    LayoutB,           // 7.  B 内存布局
    AlignmentB,        // 8.  B 对齐

    ElementAccumulator,// 9.  累加器类型

    TileShape,         // 10. tile 形状
    ClusterShape,      // 11. cluster 形状

    StageCountType,    // 12. mainloop pipeline stage 数(auto / auto-carveout / 整数)
    KernelSchedule     // 13. mainloop schedule tag(auto / 具体 tag)
>::CollectiveOp;
```

> 这些参数**全部**会被 builder 用来在 `include/cutlass/gemm/collective/builders/sm90_gmma_builder.inl`(10+ partial specialization)**挑**出一个具体的 `CollectiveMma` 实现(详见 Ch8)。Builder 不是黑箱——它是 CUTLASS 3.x 帮你做"模板参数填空"的部分。

epilogue builder 同构但少一些参数(`EpilogueTile` 类型 + 三个 c-d 类型)。

### 2.4 `args_from_options` 六元组

host 端拿到的不是上面那些 using 别名,而是一个具体的 `Arguments` 对象(由 `make_arguments` 或自己组装)。`examples/48` 是这样组装的:

```cpp
typename Gemm::Arguments args_from_options(const Options& options) {
  int device_id = 0;
  cutlass::KernelHardwareInfo kernel_hw_info =
      cutlass::KernelHardwareInfo::make_kernel_hardware_info<Gemm::GemmKernel>(device_id);

  typename Gemm::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,   // 1. mode: 单 GEMM, 不切 K, 不用 grouped, ...

    {options.m, options.n, options.k},        // 2. ProblemShape = (M, N, K)

    {block_A.get(), stride_A,                 // 3. Mainloop args
     block_B.get(), stride_B},

    {{options.alpha, options.beta},           // 4. Epilogue args
     block_C.get(), stride_C,
     block_D.get(), stride_D},

    kernel_hw_info                            // 5. SM 信息(用于 persistent scheduling)

    // 6. 隐式构造的 scheduler params,后面 .scheduler.raster_order 改
  };

  arguments.scheduler.raster_order     = options.raster;   // 6.1
  arguments.scheduler.max_swizzle_size = options.swizzle;  // 6.2

  return arguments;
}
```

6 元组是一个常用表达:

1. **mode**:告诉你这是一个单 GEMM、还是 split-K、还是 grouped,等等。`examples/48` 走 `kGemm`。
2. **ProblemShape**:M / N / K 三个维度。也支持分维度动态(比如 `M=Int<dynamic>` 表示 host 端传大小)。
3. **Mainloop args**:A、B 指针和 stride。
4. **Epilogue args**:alpha / beta / C / D 指针和 stride(`alpha * A * B + beta * C`)。
5. **hw_info**:SM 数量、每 SM 最大 block 数。Builder 用这个判断 cluster 大小能不能容下、persistent 时要不要调整 grid。
6. **scheduler params**:`raster_order`(`AlongN` / `AlongM` / `Heuristic`)、`max_swizzle_size`(1 / 2 / 4 / 8)。

> **你手写 GEMM 的对照**:你的 host launch 准备 `cudaLaunchKernel(grid, block, A, B, C, D, alpha, beta, ...)`。CUTLASS 把这些参数**集中打包**到 `Arguments`,在 kernel 入口之前一次性 translate 成 `Params`(编译期已知的常数 + 运行时知道的指针),见下一节。

### 2.5 `run()` 三段式:`can_implement` / `initialize` / `run`

```cpp
int run(Options& options) {
  initialize(options);                            // 申请设备内存等

  Gemm gemm;                                       // 建 adapter
  auto arguments = args_from_options(options);     // 组装 6 元组
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  CUTLASS_CHECK(gemm.can_implement(arguments));    // 段 1: 校验
  CUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));  // 段 2: 翻译成 Params
  CUTLASS_CHECK(gemm.run());                       // 段 3: launch

  return 0;
}
```

为什么是 3 段?

|段|作用|你手写 GEMM 的对应|
|---|---|---|
|`can_implement`|检查:形状、对齐、layout、类型组合**是不是合法的**——有没有超出 smem、stage 数够不够、对齐够不够 16B|你的 `if (M % tileM != 0 && !can_handle_residue) ...` 之类的边界检查|
|`initialize`|`Arguments → Params`——把动态值固化成编译期已知 + 一次性的 kernel 参数(captured by value),并完成 TMA descriptor host 配置|你的 host 一次性把指针放到 kernel params 结构里|
|`run`|真正的 `cudaLaunchKernel`|`cudaLaunchKernel(...)`|

> **为什么拆 3 段?** 因为 `Arguments → Params` 的转换**很重**——可能触发编译期模板实例化,甚至要做"先按某种 shape 试一下 size、再调"——所以拆开让你可以**先 can 再 init**,避免意外触发编译。

### 2.6 callout:`SharedStorage = union { MainloopStorage; EpilogueStorage; }`

每一层(主 / 收尾)都有自己的 smem 需求,但 smem 是物理上一块。 CUTLASS 的惯例是:

```cpp
struct SharedStorage {
  union TensorStorage {
    MainloopTensorStorage mainloop;       // 加载 + mma 时用这个 smem
    EpilogueTensorStorage epilogue;       // 收尾时把同一段 smem 改用 epilogue
  } tensors;                                // union, 不同时存在

  struct PipelineStorage : cute::aligned_struct<16, _1> {
    MainloopPipelineStorage mainloop;      // 流水线屏障 + 状态
    EpiLoadPipelineStorage  epi_load;      // epilogue 加载屏障(取 C 用)
  } pipelines;
};
```

**union(不是 struct!)的语义**:mainloop 用完了之后,同一 smem 区被 epilogue 复用——因为 mainloop 结束后,我们不再需要它。这是"5 层共享 smem"的现实约束。**注意这是 kernel 整个生命周期里非并发(非 persistent)**。persistent kernel 的 SharedStorage 见 Ch6。

引用(`media/docs/cpp/programming_guidelines.md`)——讲 CUTLASS 的所有 kernel 的"Params / SharedStorage"惯例。这一段不重写。

### 2.7 callout:`Arguments → Params` 是 CUTLASS 全员惯例

```cpp
// kernel::GemmUniversal 类内:
struct Arguments { mode, problem_shape, mainloop_args, epilogue_args, hw_info, scheduler };
struct Params { mainloop::Params, epilogue::Params, scheduler::Params, ... };

// 在 initialize() 里:
Params params = to_underlying_arguments(arguments);
```

这是任意一个 kernel 都要做的:host 端的 runtime `Arguments` 在 `initialize()` 里转成 device 端捕获的 `Params`(指针、规模、layout 常量)。切分的好处:Params 一旦绑好,device 端**只读**——线程安全、零开销。

后面每一章我们都会看到 `to_underlying_arguments`,理解这一段就够了。

### 2.8 章末:读完这一章你该做得到的事

- ✅ 用 `examples/48` 在 5 分钟内跑通一个 GEMM,确认 cutlass build/install 没坏。
- ✅ 修改一个 Alignment 参数(如改成 8),确认 builder 还能 resolve(可能自动 fallback)。
- ✅ 解释 `StageCountAutoCarveout<sizeof(Epilogue::SharedStorage>>` 为何要"减去 epilogue smem"再选 mainloop stage 数——因为 mainloop 和 epilogue 共享同一块 smem。
- ✅ 解释 `KernelScheduleAuto` 不是"推迟到运行时",而是在编译期用 mainloop dispatcher 选了一个默认 tag。

---

## 第 3 章:CuTe——CUTLASS 真正的"语言" {#ch03-cute}

你可能认为 CuTe 是装饰用的工具库。其实不是——它是 CUTLASS **写 mainloop / epilogue 的语言**。Cutlass 文件里那一串 `cute::Shape` / `cute::Tile` / `cute::make_tensor` 不是装饰,是 mainloop 在描述"我这一 stage 的 smem 上 A 是什么样的"。

读 Ch4 之前,你需要 CuTe 的"语法熟悉"。这一章不讲 CuTe 怎么写 sgemm(那是 `media/docs/cpp/cute/0x_gemm_tutorial.md` 的事),只讲所有 CUTLASS mainloop **一定会用到**的 4 个核心抽象:`Shape` / `Stride` / `Layout` / `Tensor`,以及 6 个常用组合子:`make_shape` / `make_layout` / `make_tensor` / `composition` / `coalesce` / `tile_to_shape`。

读完之后,你应该能在 `sm90_mma_tma_gmma_ss_warpspecialized.hpp` 里把 `SmemLayoutAtomA` / `GmemTiledCopyA` / `TiledMma` 这些 `cute::` 类型认出来、能理解它们在 mainloop 里在做什么。

### 3.1 思想:布局代数

CuTe 把"内存布局"看成一个数学结构:`Layout: Coord → Index`。给定一个整数坐标 `(i, j, k)`,Layout 把它映射到一个一维 index `n`,n 是这个元素在内存中的字节偏移除以 sizeof(elem)。

```cpp
// "4 行 8 列,行主序,fp32"
using LayoutA = Layout<Shape<_4, _8>, Stride<_8, _1>>;

constexpr int idx(int i, int j) {
  // (i, j) → idx 展开(编译期常量)
  return i*8 + j;
}
```

> **你手写 GEMM 的对照**:你写 `A[i][j]` 在 row-major 下算字节偏移 `i * (cols * sizeof(E)) + j * sizeof(E)`。CuTe 的 Layout 就是把这个"展开"放到模板里——所有偏移都编译期算完。

### 3.2 4 个最常用类型

```cpp
#include <cute/layout.hpp>   // Shape / Stride / Layout
#include <cute/int_tuple.hpp> // Int<_N>
#include <cute/tensor.hpp>    // Tensor / make_tensor
#include <cute/pointer.hpp>   // make_gmem_ptr / make_smem_ptr / make_rmem_ptr
```

#### `Shape<a, b, c, ...>` 与 `_N`

```cpp
using Shape<_M, _N, _K>    = cute::tuple<Int<M>, Int<N>, Int<K>>;
using MyShape              = Shape<_128, _128, _32>;     // 编译期 128×128×32
using DynamicShape         = Shape<int, int, int>;       // 运行期 3-tuple
using MixedShape           = Shape<_128, _128, int>;     // 1+2 前两者编译期
```

`_N` 是 `Int<N>` 的 alias,语义是"一个值为 N 的类型"。`Shape` 是 `tuple<...>`,所以你只能 `get&lt;i&gt;` 访问,CUTLASS 大量这种"tuple of compile-time ints"模式,用模板元编程做下标计算。

> 你写 `TileShape = Shape<_128, _128, _32>`,CUDA 编译器会**完全**把这些数字 bake 进代码——没有运行时开销。

#### `Stride<a, b, c, ...>` 与 `_N`

```cpp
using RowMajor    = Stride<_N, _1>;     // (i, j) → i*N + j
using ColMajor    = Stride<_1, _M>;     // (i, j) → i + j*M
```

组合 `Shape<_M, _N> + Stride<_N, _1>` 就是 `(i,j) → i*N + j`——行主序。

#### `Layout`

```cpp
template <class Shape, class Stride = LayoutLeft::Apply<Shape>>
struct Layout : private cute::tuple<Shape, Stride> { ... };

using A_Layout = Layout<Shape<_4, _8>, Stride<_8, _1>>;
```

`LayoutLeft::Apply<Shape>` 是默认 stride:行主序。`LayoutRight::Apply<Shape>` 是列主序。直接写 row-major / col-major 不需要你填 Stride。

`Layout` 是一个轻量包装——>内部就是一个 pair(shape, stride) tuple,所有功能通过 free function 提供。

#### `Tensor = Pointer + Layout`

```cpp
#include <cute/tensor.hpp>

// 假设有指针
ElementA* A_ptr = ...;

auto A = make_tensor(make_gmem_ptr(A_ptr),
                     make_layout(make_shape(M, N),     // 运行时尺寸
                                 make_stride(N, 1)));   // RowMajor

// 把上面合成(常见):
auto A = make_tensor(A_ptr,                      // 原始指针(自动包为 gmem_ptr)
                     Layout<Shape<M, N>, Stride<N, 1>>{});
```

Tensor 的语义是"一段连续内存 + 怎么解读它"。同样的指针,不同的 Layout = 不同的视图:

```cpp
// 同一个 16 元素 buffer,横看作 4×4,纵看作 4×4,或看作 16
auto x_row = make_tensor(p, Layout<Shape<_4, _4>, Stride<_4, _1>>{}); // row-major
auto x_col = make_tensor(p, Layout<Shape<_4, _4>, Stride<_1, _4>>{}); // col-major
auto x_seq = make_tensor(p, Layout<Shape<_16>>{});                   // 1-D
```

> **你手写 GEMM 的对照**:你的 `cudaMemcpy2D(...)` 给定 pitch / ptr / width / height——Layout 做的就是同一个事,但**编译期全部固化**。

#### pointer 三件套

```cpp
ElementA* raw;
auto gptr = make_gmem_ptr(raw);   // 在 gmem 上的指针
auto sptr = make_smem_ptr(smem_addr);  // 在 smem 上的指针(__shared__ void* 转过来)
auto rptr = make_rmem_ptr(reg_addr);   // 在 register 上的指针
```

这三种 engine 唯一区别是 `Tensor` 自己做下标计算时,会按位置(memory space)做不同的"一致化假设"——比如 smem 上的指针在 swizzle 时会做 swizzle 修正。

### 3.3 6 个核心组合子

#### `make_shape` / `make_stride` / `make_layout`

```cpp
auto s = make_shape(_128, _128, _32);                   // = Shape<_128, _128, _32>
auto d = make_stride(_128, _1);                         // = Stride<_128, _1>
auto l = make_layout(make_shape(_128, _128),            // M × N tile
                     make_stride(_128, _1));            // row-major stride
// 等价 Layout<Shape<_128, _128>, Stride<_128, _1>>
```

#### `make_tensor`

上面 3.2 已经讲过。

#### `composition`:Layout 的"函数复合"

Layout 就是个映射函数;`composition(A, B)` = "先按 B 映射,再按 A 映射"——这相当于数学的 `f ∘ g`,后到先应用。

```cpp
auto L1 = make_layout(make_shape(_8, _8), make_stride(_8, _1));  // row-major 8x8
auto L2 = make_layout(make_shape(_2, _2), make_stride(_2, _8));  // block of 2x2

auto L = composition(L1, L2);
// L 把一个 2×2 tile 排进 8×8 大矩阵的某个 block 位置
// 这是 mainloop 里经常出现的"tile of tile"组合
```

> **你手写 GEMM 的对照**:你写 `threadIdx` + `warpIdx` 算 smem 偏移。`composition` 就是把"warp 内偏移"和"warp 在 CTA 内的偏移"复合。

#### `coalesce`:合并相邻同 shape 的 mode

```cpp
auto L = make_layout(make_shape(_2, _3, _4), make_stride(_6, _2, _1));
coalesce(L);  // → Shape<_6, _4>, Stride<_2, _1>(把前两个 mode 合并)
```

`coalesce` 等价于一个 layout simplification——把连续访问的 mode 合成一个 mode。这对编译器做循环优化非常友好。

#### `tile_to_shape`:把一个 tile 排到大矩阵里

```cpp
auto Tile = Layout<Shape<_8, _8>, Stride<_8, _1>>{};   // 8x8 tile
auto Placed = tile_to_shape(Tile,                     // tile 的内部 layout
                            make_shape(_4, _4),       // 在大矩阵里 4×4 个 tile
                            Step<_1, _2>{});          // 大矩阵 first-mode major
// → 在 32×32 的大矩阵里,沿第一维排 4 个 Tile,再沿第二维排 4 个 Tile
```

mainloop 里 `SmemLayoutAtomA = tile_to_shape(SwizzleAtom, make_shape(...block_M, ...block_K, Stages), ...)` 就是这个模式:smem 上 A 是 SwizzleAtom 的 layout,再 tile 到 "`BlockM × BlockK × Stages`"的 3D 上。Ch4 会具体讲到。

#### 其他常用(略讲,你会在 Ch4/Ch6 看到)

- `local_tile`:从"大矩阵 + tile 大小 + 起始坐标"切成子 tile。Mainloop 里获取"本 CTA 当前 stage 的 A":

  ```cpp
  auto A_smem_tile = local_tile(A_smem_layout, /*coord=*/make_coord(block_m_coord, k_coord, stage), /*proj=*/...);
  ```

- `slice`:把 `Tile (M_tile, N_tile, K_tile, ...)` 切成"由 thread / warp 看的子 view"。
- `make_tiled_mma` / `make_tiled_copy`:把一个 mma atom / copy atom 复制成 CTA / warp / thread 分工——见 3.5。

### 3.4 callout:`cute::gemm` 的 5-case dispatch(原始出处)

文件:`include/cute/algorithm/gemm.hpp` 的 5-case dispatch 表所在位置(原文):

```cpp
/** The gemm algorithm takes four (or three) tensors and computes

 *   D = A * B + C
 * It dispatches based on the number of modes each tensor has:
 *

 * 1. (V)        x (V)        => (V)        . The element-wise product of vectors. Dispatches to FMA or MMA.
 * 2. (M)        x (N)        => (M, N)     . The outer product of vectors.    Dispatches to [3] with new mode K=(1).
 * 3. (M, K)     x (N, K)     => (M, N)     . The product of matrices.         Dispatches to [5] with MMA vector-mode V.
 * 4. (V, M)     x (V, N)     => (V, M, N)  . The batched outer product of vectors.
 * 5. (V, M, K)  x (V, N, K)  => (V, M, N)  . The batched product of matrices.
 */
```

这是 `cute::gemm` 的语义总览——5 种 tensor 形态,5 种 dispatch 路径,具体哪条路径根据 dim / mode rank 选。最后都落到 `cute::gemm(...)` 的元编程调度。

**作用**:Ch4 里会出现 `cute::gemm(tiled_mma, A_frag_smem, B_frag_smem, acc)`。这一行的 dispatch 就是根据 `A / B / acc` 的 Layout rank(2D / 3D / V-mode)选具体走哪条 case。**真正的 mma 形态是 WGMMA 还是 cp.async.mma 还是 fp16 还是 fp8,都被吸进 `TiledMma` 内部**——`cute::gemm` 看到的是一致的 Layout 接口。

### 3.5 TiledMma / TiledCopy:从 atom 到 tile

```cpp
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/copy_atom.hpp>

// 单条 mma 指令的抽象
auto mma_atom = SM90_64x16x16_F32F16F16F32_SS;     // WGMMA 64x16x16 fp16→fp32, smem→smem
auto copy_atom_a = SM90_TMA_LOAD;                   // TMA load (gmem→smem)
auto copy_atom_b = SM90_TMA_LOAD;                   // 同上

// 把一个 mma atom 复制成"由线程分摊"的 TiledMma
auto TiledMma_ = make_tiled_mma(mma_atom,
                                Layout<Shape<_2, _2, _1>>{});  // thread 排布

// 同样,copy atom 也用 make_tiled_copy 复制
auto TiledCopyA = make_tiled_copy(copy_atom_a,
                                  TiledMma_);                  // copy 配 mma
```

`make_tiled_mma` 的第二参数 `Layout<...>` 决定"这一个 mma atom 由 thread 怎么复制":如 `<_2, _2, _1>` 是"M 上 ×2、N 上 ×2、K 上 ×1"四份 atom,合起来覆盖一个完整 tile。

> **你手写 GEMM 的对照**:你写"2 个 warp × 2 个 warp group"分担 WGMMA 64x16x16 的输出。CuTe 的 `make_tiled_mma(mma_atom, Layout<_2,_2,_1>)` 做的就是这件事,只是**用模板而不是手写 thread 分配**。一旦你写好 `TiledMma`,后续 `cute::gemm(TiledMma, ...)` 拿到正确的 thread-fragment。

### 3.6 Swizzle(B, M, S):smem 防 bank conflict

```cpp
#include <cute/swizzle.hpp>

auto swizzled = Swizzle<3, 3, 3>{};    // B=3, M=3, S=3 swizzle
// 或朱子用得多的:
auto swizzled128 = Swizzle<2, 3, 3>{};   // 128B 对齐的 swizzle
```

`sbank conflict` 是 smem 读写时同 warp 内不同 lane 命中同一 bank 的事。CuTe 的 Swizzle 是一个 layout transformation:把一个简单的 row-major layout 加一个**列方向的 XOR 置换**,让不同 lane 看到不同的 bank。

```cpp
// 原始 layout: row-major, 32×32,stride 32
auto L = Layout<Shape<_32, _32>, Stride<_32, _1>>{};

// 加上 swizzle:每 8 个 row 做一个"row index XOR"的偏移
auto Sw = Swizzle<3, 3, 3>{};
auto L_swizzled = composition(Sw, L);   // 应用 swizzle
```

主 mainloop 里你看到的所有 `SmemLayoutAtomA = composition(Swizzle<...>, Layout<Shape<...>, Stride<...>>{})` 都是这一步。Ch4 详细讲。

### 3.7 Ch3.5 — CuTe by example:走读 `examples/cute/tutorial/` 4 个文件

到这里 CuTe 的语法你应该认得出来了。但若你还没跑过任何一个 cute 文件,在 Ch4 之前强烈建议读一遍这 4 个 `examples/cute/tutorial/` 文件,作为"cute 实战的最快入门"(每个文件都很短,核心代码几十行)。

> 这一节和 `media/docs/cpp/cute/0x_gemm_tutorial.md` 平行,边读边对照。

#### 文件 1:`sgemm_1.cu`——纯 cute GEMM,不经过 kernel::GemmUniversal

最朴素的 CUTLASS-by-cute GEMM。读 4 个块:

1. 建立 full A / B / C / D tensor
2. 在 K 维循环,每个 K 切片切出 `(BlockM, BlockK)`、`(BlockN, BlockK)` 子 tile,做 `cute::gemm` 累加到 C 子 tile
3. 最后把结果拷贝回 gmem
4. `cute::print` 调试

```cpp
auto A = make_tensor(A_ptr, Layout<Shape<M, K>, Stride<K, 1>>{});    // A: M x K row-major
auto B = make_tensor(B_ptr, Layout<Shape<N, K>, Stride<1, N>>{});    // B: N x K col-major
auto C = make_tensor(C_ptr, Layout<Shape<M, N>, Stride<N, 1>>{});    // C: M x N row-major
auto D = make_tensor(D_ptr, Layout<Shape<M, N>, Stride<N, 1>>{});

int num_blocks_m = ceil_div(M, BlockM);
int num_blocks_n = ceil_div(N, BlockN);
int num_blocks_k = ceil_div(K, BlockK);

for (int bm = 0; bm < num_blocks_m; ++bm)
for (int bn = 0; bn < num_blocks_n; ++bn)
for (int bk = 0; bk < num_blocks_k; ++bk) {
  auto a_blk = local_tile(A, make_shape(BlockM, BlockK), make_coord(bm * BlockM, bk * BlockK));
  auto b_blk = local_tile(B, make_shape(BlockN, BlockK), make_coord(bn * BlockN, bk * BlockK));
  auto c_blk = local_tile(C, make_shape(BlockM, BlockN), make_coord(bm * BlockM, bn * BlockN));

  cute::gemm(a_blk, b_blk, c_blk);  // 调用 case 3 (M, K) x (N, K) => (M, N)
}

// 读 + gemm 之后:
cute::copy(C, D);
```

> **你手写 GEMM 的对照**:你的 kernel 入口就是这个结构——`for block_m ... for block_n ... for block_k ...`。CUTLASS 的"5 层抽象"是把这段循环**平铺**到 global CTA grid + warpgroup + thread。

#### 文件 2:`sgemm_2.cu`——加 TiledCopy + TiledMma

`sgemm_1` 是用 `cute::gemm` 直接做外积。`sgemm_2` 引入 `TiledCopy`(加载)和 `TiledMma`(计算),**模拟** 一个真实 mainloop:

```cpp
auto TiledCopyA = make_tiled_copy(SM80_CP_ASYNC_16x8x8_F16F16,     // Ampere cp.async
                                  Layout<Shape<_32, _8>>{});        // 32 threads × 8 lane
auto TiledMma   = make_tiled_mma(SM80_16x8x16_F16F16F16F16_F32,    // Ampere mma
                                  Layout<Shape<_2, _2, _1>>{});     // 2x2x1 thread

auto thr_copy_A = TiledCopyA.get_thread_slice(threadIdx.x);
auto thr_mma    = TiledMma.get_thread_slice(threadIdx.x);

auto tAgA = thr_copy_A.partition_S(A);     // thread 看到的 A 切片
auto tBgB = thr_copy_A.partition_S(B);
auto tCgC = thr_mma.partition_S(C);
...
```

这接近 CUTLASS mainloop 内部的样子:每个 thread 拿自己的 subview,负责一部分 copy 和 mma。

#### 文件 3:`hopper/wgmma_sm90.cu`——WGMMA + CuTe atom

只引入 Hopper WGMMA,不引入 TMA。`make_tiled_mma(SM90_64xNxK_F32F16F16F32_SS, Layout<...>)` 把一条 WGMMA 指令扩展成 CTA-wide tile。

#### 文件 4:`hopper/wgmma_tma_sm90.cu`——加 TMA 加载

最后引入 `SM90_TMA_LOAD` + `make_tiled_copy(TMA_LOAD, ...)`。这基本就是 `sm90_mma_tma_gmma_ss_warpspecialized.hpp` 内部的简化版本。

> **这 4 个文件是 Ch4 的预习**。读完这 4 个 + 上面 3.1–3.6 的语法,你再看 `sm90_mma_tma_gmma_ss_warpspecialized.hpp` 会觉得"每行都认得"。

### 3.8 章末:读完这一章你该做得到的事

- ✅ 在 Ch4 的 mainloop 代码里认出 `Shape<_M, _N>`、`make_tensor(...)`、`TiledMma`、`SmemLayoutAtomA` 这些名字各自是 CuTe 的哪个组合子。
- ✅ 读得懂 `tile_to_shape(SwizzleAtom, make_shape(...), ...)` 这种"在 mainloop 里到处出现"的复杂表达式。
- ✅ 看 `cute::gemm(tiled_mma, ...)` 时知道它在走 5-case dispatch。
- ✅ 用 `cute::print(A_tensor)` 调试 layout 时知道是在按 mainloop 的视图 print。

CuTe 不需要精通——**足够认得出来**就够读 Ch4。

---

## 第 4 章:深入 CollectiveMainloop(本教程核心价值) {#ch04-mainloop}

如果本教程只选一章细读,选这一章。Mainloop 是 CUTLASS 3.x 写得最繁的部分——也是体现"5 层抽象是否值钱"的地方。

主文件:`include/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp`。

读法:**从类头部分支开始,顺着类型别名 → helper → 主循环的 6 个函数 → SharedStorage**走。

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

**18 个模板参数**。看似吓人,分两组:

|组|内容|
|---|---|
|上下文(由 builder 推)|`MainloopSm90TmaGmmaWarpSpecialized`、`TileShape`、`ElementA/B`、`StrideA/B`、`TiledMma`|
|8 个原子(`A` 和 `B` 各 4 个)|`GmemTiledCopyA/B`(通常是 TMA)、`SmemLayoutAtomA/B`(smem 上 A/B 的 layout)、`SmemCopyAtomA/B`(smem 内部微调 copy 用的 sub-atom,通常空)、`TransformA/B`(swizzle / interleave 这类 layout 变换)|

前 7 个由 builder 推;后 8 个 A/B 原子是 "**smem A/B 各自的 4 件套**"。你写 mainloop 时不会手写这些;改 builder 输入 `ElementA * LayoutA * AlignmentA` 后,这些都会被推出来。

### 4.2 类型别名总解

```cpp
// 在 collective_mma 内部的类型,按"出现频率"排:

using DispatchPolicy = MainloopSm90TmaGmmaWarpSpecialized;  // tag (Ch7)
using TileShape      = TileShape_;
using ElementA       = ElementA_;
using StrideA        = StrideA_;
using ElementB       = ElementB_;
using StrideB        = StrideB_;
using TiledMma       = TiledMma_;
using ArchTag        = typename DispatchPolicy::ArchTag;
using PipelineStage  = ...;                                    // PipelineState 起点
using MainloopPipeline = cutlass::PipelineTmaAsync<DispatchPolicy::Stages>;
//    ↑ 来自 include/cutlass/pipeline/sm90_pipeline.hpp;producer/consumer 屏障数组

using PipelineParams = typename MainloopPipeline::Params;
using PipelineState  = typename MainloopPipeline::State;

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

> **你手写 GEMM 的对照**:你写 `__threadfence_block()` / `__syncthreads()` + 手动维护的 barrier 数组。`PipelineTmaAsync<Stages>` 就是"自动循环 barrier 的 box"。

### 4.3 实际存在的 mainloop 方法(`sm90_mma_tma_gmma_ss_warpspecialized.hpp`)

`CollectiveMma<...>` 的成员函数分三类——3 个 setup helper + 5 个 K-loop helper:

```cpp
// === setup helper(类内 static 或 const) ===
static void prefetch_tma_descriptors(Params const&);   // 单线程幂等,预取 TMA desc
static Params   to_underlying_arguments(...);           // Arguments → Params
static bool     can_implement(...);                    // 校验:形状/对齐/类型是否合法

// === K-loop helper(每个被反复调用) ===
load_init(ProblemShape_MNKL, Params const&) const;      // 初始化:smem 切 slot + pipeline state 起点

load(Params, MainloopPipeline, ProblemShape_MNKL, ...,
     TensorStorageA&, TensorStorageB&, PipelineState& write_state) const;   // producer: TMA load
mma(MainloopPipeline, cute::Tensor<...>, TensorStorageA&, TensorStorageB&,
    PipelineState& read_state, int k_tile_count) const;                       // consumer: WGMMA + cute::gemm

mma_tail(MainloopPipeline, PipelineState&, int k_tile_count);                  // K-loop 末尾: 把剩下的 mma 跑完
load_tail(MainloopPipeline, PipelineState smem_pipe_write);                    // 收尾:最后清 producer pipeline
```

> ⚠ 这里**没有** `load_wait` / `mma_next`。在早期 CUTLASS 草稿版里有这俩 helper,后来为了减少函数拆分,等价的 `consumer_wait(state)` 和 `state++` 直接内联在了 `mma(...)` 的函数体里。

读法——把这个 5 步的接力赛跑想象成:

```text
[ Producer 线程]
  load_init → while (k < K) { load → state++ } → load_tail

[ Consumer 线程]
  load_init → while (k < K) { consumer_wait(state); mma(state); state++; } → mma_tail
```

`producer_acquire(state)` / `producer_commit(state, bytes)` / `consumer_wait(state)` / `consumer_release(state)` 这些都是 `PipelineTmaAsync<N>` 类的方法,在 `include/cutlass/pipeline/sm90_pipeline.hpp` 里(`PipelineTmaAsync<Stages>::producer_acquire(PipelineState state)` 等)。

producer 和 consumer 在**同一个** smem pipeline 上协作,但**可以并行**——producer 在 stage 0 / 1 / 2 加载,consumer 在 stage 0 做 mma。Ch6 主循环代码里你会看到这两个 while 怎样在同一个 kernel entry 里交错。

#### `load`

```cpp
CUTLASS_DEVICE void
load(Params const& mainloop_params,
     MainloopPipeline mainloop_pipeline,
     MainloopPipelineParams const& mainloop_pipeline_params,
     Resource<0> /* we don't need this */,
     cute::tuple<int, int, int> const& load_offsets,
     TensorStorageA& storage_A,
     TensorStorageB& storage_B,
     PipelineState& write_state) {

  auto [m, n, k] = load_offsets;
  auto& smem_A = storage_A.smem_A;
  auto& smem_B = storage_B.smem_B;

  // producer 等 stage 0 空出来
  mainloop_pipeline.producer_acquire(write_state);

  // 用 TMA 把 A(m, *) 拉进 smem_A[stage]
  auto tma_A = mainloop_params.tma_load_a.with(make_coord(m, k), mainloop_params.problem_shape_MK);
  copy(mainloop_params.tma_load_a.with(..., smem_A(_, _, write_state.index())), ...);
  // 同上 for B
  copy(mainloop_params.tma_load_b.with(..., smem_B(_, _, write_state.index())), ...);

  // commit 这一 stage — consumer 看到后可以做 mma
  mainloop_pipeline.producer_commit(write_state);
  ++write_state;
}
```

#### `mma`

```cpp
CUTLASS_DEVICE void
mma(MainloopPipeline mainloop_pipeline,
    PipelineState& read_state,
    Accumulators& accumulators,
    TensorStorageA& storage_A, TensorStorageB& storage_B,
    ...) {

  // 等 stage 准备好(producer 已经 commit)
  mainloop_pipeline.consumer_wait(read_state);

  auto& smem_A = storage_A.smem_A;
  auto& smem_B = storage_B.smem_B;

  // 取出 smem 上 stage 位置的 sub-tile,用 TiledMma 算到 accumulators
  auto smem_A_tile = ...;  // local_tile(smem_A, TiledMma layout, ...)
  auto smem_B_tile = ...;
  cute::gemm(TiledMma, smem_A_tile, smem_B_tile, accumulators);  // → WGMMA 指令

  // release — 让 producer 可以更新这一 stage
  mainloop_pipeline.consumer_release(read_state);
  ++read_state;
}
```

#### callout:`cute::gemm` 在 mainloop 里的实例化点

这一行 `cute::gemm(TiledMma, smem_A_tile, smem_B_tile, accumulators)` 不只调用,还**实例化**——它根据 `TiledMma` 的 MMA atom / Layout,推到具体 WGMMA 指令(`wgmma.mma_async.sync.aligned.m64nXk16`). 这就是 Ch3.4 提到的 5-case dispatch 在 mainloop 的实际位置。

### 4.4 TMA descriptor(单线程预取)

```cpp
// 在 Ch6 的 kernel orchestrator 里:
if ((warp_idx == 0) && lane_predicate) {
  CollectiveMainloop::prefetch_tma_descriptors(params.mainloop);
  CollectiveEpilogue::prefetch_tma_descriptors(params.epilogue);
}
```

`prefetch_tma_descriptors` 是**单线程幂等**——只有 warp 0 的 lane 0 做实际工作;其他线程看到 if 不进。TMA descriptor 一旦就位,后面 `tma_load_a.with(coord, problem_shape)` 就不需要再传 layout 信息。

> **你手写 GEMM 的对照**:你写 `cuTensorMapEncodeTiled` 把 desc 写好,这里直接做。

### 4.5 Cluster 维度:CTA 间协作

`cute::cluster_size_v<ClusterShape>`、`cute::block_rank_in_cluster()` 出现在 `tile_to_shape` / `subtile` 等地方——具体用法是:

```cpp
// 让当前 CTA 看 DSMEM 里"邻居 CTA 的 smem"
auto neighbor_smem_A = cluster_collective_load(...);
```

> 你写 Hopper 时如果用过 cluster launch(`cudaLaunchKernelEx` + `clusterDim`),这里对应。CTA 间 DSMEM 互访省 smem 注册流量——尤其在大 BlockM × BlockN 的 tile 上。

### 4.6 SharedStorage union

回到 Ch2.6。Mainloop 用完 smem 之后,这块交给 epilogue 复用——`union { MainloopTensorStorage; EpilogueTensorStorage; }`。

变体速览:同一 `sm90_mma_tma_gmma_*` 文件名下还有这些变体,都靠 dispatch policy tag 路由(Ch7):

|文件|变体|
|---|---|
|`sm90_mma_tma_gmma_ss.hpp`|非 warp-specialized(MMA 在所有 warp 上,无 producer/consumer 切分)|
|`sm90_mma_tma_gmma_ss_warpspecialized.hpp`|**默认**:1 producer warp group + 1 consumer warp group|
|`sm90_mma_tma_gmma_ss_warpspecialized_pingpong.hpp`|1 producer + 2 consumer(交替 pingpong)|
|`sm90_mma_tma_gmma_ss_warpspecialized_cooperative.hpp`|1 producer + 多 consumer(更大 tile)|
|`sm90_mma_tma_gmma_ss_warpspecialized_fp8.hpp`|FP8|
|`sm90_mma_tma_gmma_ss_warpspecialized_fp8_blockwise_scaling.hpp`|FP8 + block scale|
|`sm90_mma_tma_gmma_rs_warpspecialized.hpp`|A 在 register(RS 而非 SS)|
|`sm90_mma_tma_gmma_rs_warpspecialized_mixed_input.hpp`|A/B 不同 dtype|
|`sm90_*_array_tma_gmma_*`|Grouped 或 ptr-array(GEMM 的"批处理")|
|`sm90_sparse_mma_tma_gmma_*`|2:4 structured sparsity|

### 4.7 图配

下面两张图都在 Ch6 也用到——这里先放感受一下 pipeline 的物理样子:

![pipeline](media/images/software-pipeline.png)

![threadblock mma pipelined](media/images/cutlass-threadblock-mma-pipelined.png)

Ch4 把 mainloop 讲完了。下一章 Ch5 看 epilogue——几乎是镜像结构。

---

## 第 5 章:深入 CollectiveEpilogue + EVT {#ch05-epilogue}

Epilogue 和 Mainloop 是镜像:同样从 gmem→smem(TMA load C)→register(TMA store D)、同用 pipeline、同 swizzle、同 swappable 接口。区别只在:

1. 数据流方向反过来:把 accumulator 通过 TMA store 写到 gmem。
2. **可能插一段融合算子**——bias、ReLU、silu、top-K softmax、scale、swizzle output 等。
3. 调度的细颗粒度更细:有专门的 `StagesC` (C load 阶段) / `StagesD` (D store 阶段),可以独立调。

主文件:`include/cutlass/epilogue/collective/sm90_epilogue_tma_warpspecialized.hpp`。

### 5.1 Dispatch policy

```cpp
template <
  int StagesC_, int StagesD_, int FragmentSize_,
  bool ReuseSmemC_, bool DelayTmaStore_,
  ...
>
struct Sm90TmaWarpSpecialized
    : ... /* 标记本身是 dispatch tag 树的一个 node */ {
  constexpr static int StagesC = StagesC_;
  constexpr static int StagesD = StagesD_;
  constexpr static int FragmentSize = FragmentSize_;
  constexpr static bool ReuseSmemC = ReuseSmemC_;
  constexpr static bool DelayTmaStore = DelayTmaStore_;
};
```

5 个参数分别在做什么:

|参数|默认值|含义|
|---|---|---|
|`StagesC`|auto|C 加载阶段数(从 C 到 register 的流水线深度)|
|`StagesD`|auto|D store 阶段数(register 到 smem 再到 gmem 的流水线深度)|
|`FragmentSize`|8|each store lane 一次写多少个元素|
|`ReuseSmemC`|true|是否把"已加载 C 的 smem slot"在 epilogue 末尾继续复用(给 output D 用)|
|`DelayTmaStore`|false|是否把 TMA store 推迟到最后(为 swizzle / fusion 留时间)|

`EpilogueScheduleAuto` builder 缺省是哪个具体 `StagesC` / `StagesD` / `FragmentSize` / `ReuseSmemC` / `DelayTmaStore` 组合,看 `include/cutlass/epilogue/collective/builders/sm90_builder.inl`(Ch8 讨论)。

### 5.2 默认 epilogue:`D = alpha * (A*B) + beta * C`

```cpp
// 在 operator() 里的 forward path(简化):
for (...) {  // tiles
  // 1. 加载 C 块
  cute::copy(TiledCopyC, C_block, C_smem_block);
  // 2. mma 累加器已经有了(从 mainloop)
  // 3. alpha/beta 系数乘法
  acc_array = alpha * acc_array + beta * (C_register loaded from C_smem_block);
  // 4. 转换:fp32 acc -> C type
  C_or_D_t_register = ConvertOp{}(acc_array);
  // 5. TMA store
  cute::copy(TiledCopyD, C_or_D_t_register, D_block);
}
```

> **你手写 GEMM 的对照**:你写 epilogue 时就是这几步——读 C、乘 β、乘 α + acc、写 D。CUTLASS 把 alpha/beta/转换类型切成一组 visitor 节点(下面)。

### 5.3 TMA Store 路径

跟 TMA Load 镜像——把 register tile 经过 smem 缓存再 store 到 gmem:

```cpp
// TMA Store atom:
auto tma_store_d = SM90_TMA_STORE{};
auto TiledCopyD = make_tiled_copy(tma_store_d, TiledMma);

// use:
cute::copy(TiledCopyD, D_register_tile, D_gmem_block);
```

TMA store 把"分散到多 lane 的 D 元素"高效打包成一个 gmem block 写。

### 5.4 EVT(Epilogue Visitor Tree)——融合算子

如果你不需要任何融合,上一节就够。需要 bias / ReLU / silu / scale 等,用 EVT。

EVT 是一个小型"小 DSL":每个 `compute` 节点代表一个算子,叶子节点是"输入源"(`Sm90SrcFetch` 是从 src 张量读,`Sm90AccFetch` 是从 accumulator 读,`Sm90ScalarBroadcast` 是标量广播)。

```cpp
#include <cutlass/epilogue/fusion/sm90_visitor_compute_tma_warpspecialized.hpp>
#include <cutlass/epilogue/fusion/sm90_visitor_tma_warpspecialized.hpp>

// 例子: D = bias + ReLU(A*B)
//                       ← Sm90EVT root:add
//                    /            \
//           bias_src_fetch     ReLU compute
//                                 /
//                          (1 arg = acc)

// silu/relu/gelu 这种 activation CUTLASS **不会**预置 — 见 §5.4 后半段,
// 这里先给用户自定义 unary functor:
struct IdentitySilu {
  template <class T>
  CUTLASS_HOST_DEVICE T operator()(T const& x) const {
    return x / (T(1) + cutlass::exp(-x));
  }
};

using MyEVT = Sm90EVT<
    Sm90Compute<homogeneous_add, /* ElementD, RoundStyle... */>,   // root: bias + ReLU(acc)
    Sm90SrcFetch,                                                    // arg 1 of add: 从 src tensor(bias)取
    Sm90EVT<                                                         // arg 2 of add: ReLU(acc)
        Sm90Compute<cutlass::identity, /* RoundStyle... */>,
        Sm90EVT<
            Sm90Compute<IdentitySilu, /* RoundStyle... */>,         // 一元 activation
            Sm90AccFetch                                            // 它的唯一 arg = accumulator
        >
    >
>;

// 用法:
CollectiveEpilogue evt_op(...);
evt_op(/* args to EVT: src tile of bias, problem shape, etc. */);
```

EVT 节点的"参数"按 visit 顺序排——根的 arg 列表从前到后,再到子节点。

#### callout:`ComputeFn` 设计约束

文件 `include/cutlass/epilogue/fusion/sm90_visitor_compute_tma_warpspecialized.hpp`(`Sm90VisitorCompute` 类附近),讲 `ComputeFn` 的 4 个成员签名约束(原文多行注释):

```cpp
// 用户要写的 ComputeFn 必须暴露这 4 个 functors(auto-detected by visitor tree):
//
// 1. auto operator()(Tensor frag, BroadcastedScalars...)  → 计算结果(frag-shape 的 tensor)
// 2. void reductionsum(...)                                  → 用于最终 reduction(如 sum of squares)
// 3. auto get_reduction_counters()                          → 用于性能计数
// 4. ... plus consumer::OrderedInnerTileVisibilityReshape  shape info
//
// 是的,你写个 sm90_visitor_compute 节点,写的是 4-5 个 functor;
// 不像 Pytorch 的 activation 是一行 lambda。
```

这一约束是 CUTLASS 3.x 决定不用 lambda fusion 的原因——**为了在 register / smem 之间有显式的形状控制**,并且 epilogue 端要保证它是 per-warp / per-tile 可拆解的。

#### 完整例子:`D = alpha * (A*B) + beta * C`(标准默认)

`examples/49_hopper_gemm_with_collective_builder/49_collective_builder.cu` 的实现是:

```cpp
// D = beta*C + alpha*(A*B),按这个树:
//     root = add
//       ├── left  = mul(alpha_scalar, acc)
//       │           ├── alpha_scalar
//       │           └── A*B (来自 accumulator)
//       └── right = mul(beta_scalar, C)
//                   ├── beta_scalar
//                   └── C

using CustomEVT =
  cutlass::epilogue::fusion::Sm90EVT<
    cutlass::epilogue::fusion::Sm90Compute<
      cutlass::homogeneous_multiply_add,
      ElementD, ElementCompute, RoundStyle>,
    cutlass::epilogue::fusion::Sm90ScalarBroadcast<ElementScalar>,  // beta
    cutlass::epilogue::fusion::Sm90SrcFetch<ElementC>,              // C
    cutlass::epilogue::fusion::Sm90EVT<
      cutlass::epilogue::fusion::Sm90Compute<
        cutlass::multiplies, ElementCompute, ElementCompute, RoundStyle>,
      cutlass::epilogue::fusion::Sm90ScalarBroadcast<ElementScalar>, // alpha
      cutlass::epilogue::fusion::Sm90AccFetch                        // A*B
    >
  >;
```

#### ComputeFn 实际可用的算子

CUTLASS 没有为 silu/relu/gelu/sigmoid 之类的 activation 单独提供 `homogeneous_silu` 这种算子。ComputeFn 模板位置写的是**任何带 `operator()(A)` 的 struct**,只要满足 §5.4 注释块里的"恰好一个模板参数"约束。

CUTLASS 内置的 functors(在 `cutlass` namespace):

- `cutlass::multiplies<A>`、`cutlass::plus<A>`、`cutlass::minus<A>`、`cutlass::divides<A>`、`cutlass::negate<A>`(数学运算)
- `cutlass::homogeneous_multiply_add<A, A, A>`(单一 FMA 节点,最常用)

如果用户需要 silu:

```cpp
// 几十行自定义一个
template <class A>
struct HomogeneousSilu {
  CUTLASS_HOST_DEVICE A operator()(A x) const {
    return x / (A(1) + cutlass::exp(-x));   // silu 公式
  }
};
// 用法:Sm90Compute<HomogeneousSilu, ...>
```

即:activation / 一元 / 二元 / 自定义算子,都需要用户自己写十几行 functor。这是 CUTLASS 3.x 故意不内建大量 activation 的取舍——为了编译期形状可控,只内建最小集(`multiplies` / `plus` 等数学)。其他 activation 通过用户 functor 扩展。

### 5.5 跨章对比

|Example|在 epilogue 上加了什么|在哪|
|---|---|---|
|`examples/48_hopper_warp_specialized_gemm/`|默认|一切从零|
|`examples/49_hopper_gemm_with_collective_builder/`|加 EVT bias + ReLU fusion + 改 schedule|Ch9|
|`examples/50_hopper_gemm_with_epilogue_swizzle/`|swizzle output(让 D 的 gmem 写入布局更好)|`examples/50_*/50_*.cu`|
|`examples/113_hopper_gemm_activation_fusion/`|多 activation fusion(bias + gelu + scale)|`examples/113_*/113_*.cu`|

![hierarchy-with-epilogue](media/images/gemm-hierarchy-with-epilogue.png)

### 5.6 章末:读完这一章你该做得到的事

- ✅ 知道 epilogue dispatch policy 5 个参数分别影响什么(StagesC、StagesD、FragmentSize、ReuseSmemC、DelayTmaStore)。
- ✅ 能看懂 `Sm90EVT<Sm90Compute<...>, ...>` 嵌套语法——它就是"小 AST"。
- ✅ 给一个 `D = alpha * (A*B) + beta * silu(C)` 这种非平凡 epilogue,你能写出对应的 EVT 树。
- ✅ 知道 EVT 跟 PyTorch 的 chain 是两回事——CUTLASS 是显式 AST 不是基于 lambda fusion。

---

## 第 6 章:Kernel orchestrator + TileScheduler(含调度器族侧栏) {#ch06-kernel-orchestrator}

这一章读 `include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized.hpp` + `sm90_tile_scheduler.hpp` + `static_tile_scheduler.hpp` + `tile_scheduler_params.h`。

把 Kernel orchestrator 和 TileScheduler 合并讲的原因:**它们在一个 kernel 的入口一起出现**——`operator()` 调 first `tile_scheduler.get_current_work()`,得到 (m, n) 切片,然后跑 mainloop + epilogue。

### 6.1 Kernel 类(`GemmUniversal<...>`)的 SFINAE 路由

```cpp
template <
  class ProblemShape_, class CollectiveMainloop_, class CollectiveEpilogue_, class TileScheduler_
>
class GemmUniversal<
  ProblemShape_, CollectiveMainloop_, CollectiveEpilogue_, TileScheduler_,
  cute::enable_if_t<
    cute::is_base_of_v<KernelTmaWarpSpecialized,
                       typename CollectiveMainloop_::DispatchPolicy::Schedule>
  >
> {
  ...
};
```

`enable_if_t<is_base_of_v<KernelTmaWarpSpecialized, Mainloop::DispatchPolicy::Schedule>>` — 这一段是"如果 mainloop 的 schedule tag 是 `KernelTmaWarpSpecialized`(或其派生),我匹配这个 partial specialization"。

为什么这样:有 5+ 种 `Kernel*` partial specialization(TmaWarpSpecialized、Pingpong、Cooperative、Pure、非 warp-spec 等)。CUTLASS 不希望用 `if-else` 选——它用 SFINAE 路由,用户写什么 mainloop schedule tag,就匹配什么 kernel。

### 6.2 Arguments / Params / SharedStorage

```cpp
struct Arguments {
  GemmUniversalMode mode;           // kGemm / kGemmSplitK / kGroupedGemm / ...
  ProblemShape problem_shape;       // (M, N, K)
  typename CollectiveMainloop::Arguments mainloop;
  typename CollectiveEpilogue::Arguments epilogue;
  KernelHardwareInfo hw_info;       // SM count / max blocks per SM
  typename TileScheduler::Arguments scheduler;  // raster / swizzle
};

struct Params {
  typename CollectiveMainloop::Params mainloop;
  typename CollectiveEpilogue::Params epilogue;
  typename TileScheduler::Params    scheduler;
  ProblemShape problem_shape;
  KernelHardwareInfo hw_info;
};

// internal SharedStorage:
struct SharedStorage {
  union TensorStorage {
    typename CollectiveMainloop::TensorStorage mainloop;
    typename CollectiveEpilogue::TensorStorage epilogue;
  } tensors;

  struct PipelineStorage : cute::aligned_struct<16, _1> {
    typename CollectiveMainloop::PipelineStorage mainloop;
    typename CollectiveEpilogue::PipelineStorage epi_load;
    typename TileScheduler::SharedStorage          scheduler;
  } pipelines;
};
```

pattern 你已经熟了: union(主要内存)、pipeline storage(流水线屏障)、scheduler 状态。

### 6.3 `operator()(Params, smem_buf)` 入口

```cpp
CUTLASS_DEVICE void operator()(Params const& params, char* smem_buf) {
  // 计算 thread / warp 角色
  int warp_idx             = threadIdx.x / 32;
  int warp_group_idx       = warp_idx / 4;     // 4 warp/warp group
  int lane_predicate       = cute::elect_one_sync();
  bool is_main_producer    = (warp_group_idx == 0);   // producer warp group
  bool is_main_consumer    = (warp_group_idx == 1);   // consumer warp group
  WarpGroupRole warp_group_role = is_main_producer ? Producer : Consumer;

  // SharedMemory 划分
  TensorStorage&   shared_storage_tensors   = *reinterpret_cast<TensorStorage*>(smem_buf);
  PipelineStorage& shared_storage_pipelines = *reinterpret_cast<PipelineStorage*>(
      smem_buf + sizeof(TensorStorage));

  // 所有 CTA 单线程预取 TMA descriptor
  if (warp_idx == 0 && lane_predicate) {
    CollectiveMainloop::prefetch_tma_descriptors(params.mainloop);
    CollectiveEpilogue::prefetch_tma_descriptors(params.epilogue);
  }

  // 准备 mainloop pipeline 参数
  typename CollectiveMainloop::PipelineParams mainloop_pipe_params;
  if (warp_group_role == Producer) {
    mainloop_pipe_params.role = MainloopPipeline::ThreadCategory::Producer;
  }

  // CTA 内的"互相 barrier"被组装好
  cutlass::arch::wait_for_dependent_instruction();
  __syncthreads();

  // 主循环:每个 CTA 处理多个 tile(persistent)
  while (true) {
    // 1) 决定本 CTA 这一轮跑哪个 tile
    auto work_tile_info = TileScheduler::get_current_work(params.scheduler);
    auto [m_coord, n_coord, k_tile_count, ...] = work_tile_info;

    if (!work_tile_info.is_valid()) {
      break;   // 全部分配完了
    }

    // 2) 计算在本 tile 内 K 维 split(用于 K-loop 边界)
    int k_tile_idx = ...;

    // 3) Producer 主循环
    if (warp_group_role == Producer) {
      CollectiveMainloop::load_init(...);
      PipelineState write_state = make_producer_start_state<MainloopPipeline>();
      // ... 详见 6.4
    }

    // 4) Consumer 主循环
    if (warp_group_role == Consumer) {
      CollectiveMainloop::load_init(...);
      PipelineState read_state{};   // default-construct,初相位与 producer 反相
      // ... 详见 6.4
    }

    // 5) 收尾:epilogue
    CollectiveEpilogue::operator()(..., work_tile_info);

    // 6) 推进 scheduler + cluster barrier(等待同 cluster 的兄弟 CTA 都完成)
    TileScheduler::advance(params.scheduler);
    cluster_arrive();
    cluster_wait();
  }
}
```

注释:

- `cute::elect_one_sync()`:32 lane 里选 1 个 candidate,(true/false)标记谁做幂等的"单线程"操作(如 TMA descriptor 预取)。
- `WarpGroupRole { Producer, Consumer }` + `ProducerWarpRole { MainloopEpilogue, Warp1, Warp2, Warp3 }`:更细的角色切分在 epilogue / 多 producer 的变体里出现。
- `cluster_arrive` / `cluster_wait`:cluster 内 CTA 一起同步(在本例子是让同 cluster 的所有 CTA 都跑完本 tile 才进入下一 tile)。

### 6.4 Producer / Consumer 主循环(俯视)

```text
[t=0]   Producer 把 K=0  拉入 smem stage 0
        Consumer 等 stage 0 准备好

[t=1]   Producer 把 K=32 拉入 smem stage 1
        Consumer 在 smem stage 0 做 mma

[t=2]   Producer 把 K=64 拉入 smem stage 2
        Consumer 在 smem stage 1 做 mma

[t=3]   Producer 等 stage 0 空出来(load_init 之初预留 producer_ahead = N stages 的空档)
        Consumer 在 smem stage 2 做 mma

        + Producer 把 K=96 拉入 smem stage 0(覆盖)
        ...
```

代码骨架:

```cpp
// Producer 主循环(简化为同步版,真实代码是非阻塞)
if (role == Producer) {
  CUTLASS_PRAGMA_NO_UNROLL  // Producer 是异步发起 TMA,不要 #pragma unroll
  for (int k_tile = 0; k_tile < K_tiles; ++k_tile) {
    // 等本 producer stage 的 smem slot 出来
    mainloop_pipeline.producer_acquire(write_state);
    // 用 TMA 拉
    CollectiveMainloop::load(params.mainloop, mainloop_pipeline, ..., write_state);
    // commit — consumer 可以 mma 这一 stage 了
    mainloop_pipeline.producer_commit(write_state);
    ++write_state;
  }
}

// Consumer 主循环
if (role == Consumer) {
  CUTLASS_PRAGMA_UNROLL  // Consumer 是同步执行 WGMMA,可以 unroll
  for (int k_tile = 0; k_tile < K_tiles; ++k_tile) {
    // 等本 consumer 的 smem slot 准备好
    mainloop_pipeline.consumer_wait(read_state);
    // 用 WGMMA 计算
    CollectiveMainloop::mma(params.mainloop, accumulators, shared_tensors, ..., read_state);
    // release — producer 可以更新这一 stage
    mainloop_pipeline.consumer_release(read_state);
    ++read_state;
  }
}
```

注意 6 个 helper 的**调用顺序**(再次强调,这对应 Ch4):

1. `load_init`(在 producer / consumer 进入 while 循环前各做一次)
2. Producer 循环:`producer_acquire → load → producer_commit → ++state` × N
3. Consumer 循环:`consumer_wait → mma → consumer_release → ++state` × N
4. `load_tail`(收尾:最后一次 consumer_wait + mma,清空 pipeline)

> **你手写 GEMM 的对照**:你大概自己写过 `is_producer = warpIdx < 4` 这种分支 + 双 queue(buffer 数组)。CUTLASS 把它"封装"成 `PipelineTmaAsync<Stages>`,你不用手写 barrier 数组,但看到的"6 个 helper 接力"完全等价。

### 6.5 `PersistentTileSchedulerSm90` —— 持续分派的调度器

文件:`include/cutlass/gemm/kernel/sm90_tile_scheduler.hpp`(默认)。persistent kernel = 每个 CTA 在一个 SM 上跑、循环摊派多个 tile。

```cpp
class PersistentTileSchedulerSm90 : public StaticPersistentTileScheduler<PersistentTileSchedulerSm90> {
  ...
};
```

CRTP 继承了一个 `StaticPersistentTileScheduler<Tag>` 公共模板,Tag 用自己(`PersistentTileSchedulerSm90`),用于 `static_cast` 反射出调度器的 tags。

#### 关键参数

```cpp
struct PersistentTileSchedulerSm90Params {
  RasterOrderOptions raster_order;            // AlongN / AlongM / Heuristic
  uint32_t            max_swizzle_size;        // 1 / 2 / 4 / 8
  int                 swizzle_log;             // log2(上面)
  uint64_t            num_blocks_in_grid;      // grid 总数
  dim3                grid;                    // 实际 grid dim
  KernelHardwareInfo  hw_info;                 // SM count / max blocks per SM
};
```

`raster_order` 决定 "tile 是按 row-major 还是 col-major 摊派到 CTA id":

- `AlongN` = N 先推进(横向扫)
- `AlongM` = M 先推进(纵向扫)
- `Heuristic` = 自动选(根据矩阵形状)

`max_swizzle_size` 决定"多大块长的 L2 reuse":1 是无 swizzle、8 是 8×8 swizzle block(swizzle 后同 L2 set 被多个 CTA 命中)。

#### 工作算法

```cpp
auto get_current_work() -> WorkTileInfo {
  // 当前是第几次 work-index
  uint64_t work_idx = work_id_counter_++;
  if (work_idx >= total_num_tiles_) {
    return WorkTileInfo::invalid();
  }
  // 把 work_idx → (m_tile, n_tile),按 swizzle + raster 算
  auto [m_tile, n_tile] = get_work_idx_m_and_n(
      blk_per_grid_dim_, work_idx, swizzle_log_, raster_order_);
  return WorkTileInfo{m_tile, n_tile, ...};
}
```

`get_work_idx_m_and_n` 是核心解调度函数,输入 work id 和 swizzle / raster 配置,输出 (m, n) 切片坐标。

### 6.6 图配

![persistent_static](media/images/persistent_static.png)

![non_persistent](media/images/non_persistent.png)

![threadblock mma pipelined](media/images/cutlass-threadblock-mma-pipelined.png)

### 6.7 调度器族侧栏

不是只 PersistentTileScheduler 一家。CUTLASS 3.x 中,TileScheduler 是一个 sumtag(`PersistentScheduler` / `StreamKScheduler` / `GroupScheduler` / `StaticPersistentScheduler` / `DynamicPersistentScheduler`),由 `TileSchedulerSelector` 按 tag 派发。

|Tag|文件|何时用|数学梗概|
|---|---|---|---|
|`PersistentScheduler` (default)|`sm90_tile_scheduler.hpp`|默认;每个 CTA 处理多 tile,持续运行|work_id → (m, n) by swizzle/raster|
|`StaticPersistentScheduler`|`static_tile_scheduler.hpp`|不需要调度复杂度的轻量版|简化版 persistent|
|`StreamKScheduler`|`sm90_tile_scheduler_stream_k.hpp`|K 维的 split 与 partial result 合并|K-parallelism,partial sum|
|`GroupScheduler`|`sm90_tile_scheduler_group.hpp`|grouped GEMM(多组不同形状的 GEMM 同 kernel 跑)|用 problem_visitor 拉多个 problem|

每种都暴露同名 `get_current_work` / `advance` 接口,所以 Ch6.3 的 kernel orchestrator 代码**完全不动**,只换 scheduler tag 就能切。

#### `StreamKScheduler` 的梗概(供"如果你好奇"用)

Stream-K 把 work 沿 K 维再切,每个 worker 处理某个 (m, n, k_partial) cube,所有 partial result 写到一个 partial buffer,最后一段 reduction kernel 把 partial result 累加成最终 D。这样可以"完全用满 GPU"——任何 K-bound 形状都能打平。

需要 partial buffer + final reduction,所以 StreamKScheduler 通常配 `StreamKReductionKernel`,由 `examples/47_ampere_gemm_universal_streamk/` 演示。

#### `GroupScheduler` 梗概

Grouped GEMM(每组 problem 的 M/N/K 不同,如 MoE),调度器由 `GroupProblemVisitor` 拉数据并按 shape 切。把 multiple GEMM 共用一次 launch。

### 6.8 章末:读完这一章你该做得到的事

- ✅ 在 kernel 入口看到 `WarpGroupRole { Producer, Consumer }`、`ProducerWarpRole`、`persistent_scheduler.get_current_work(...)` 这些,认得出各自在做什么。
- ✅ 能口述"6 个 helper 在 producer / consumer 主循环里按什么顺序被调用"。
- ✅ 把 task graph 在脑子里跑一遍:TMA load → barrier release → WGMMA → barrier release → TMA store。
- ✅ 知道 StreamK / Grouped 调度的存在和位置,以后看代码不陌生。

Ch7 看 dispatch policy——为什么这一切被自动选到。

---

## 第 7 章:DispatchPolicy——tag-inheritance 模式 {#ch07-dispatch-policy}

这是 CUTLASS 3.x 的**架构精髓**——也是 `media/docs/cpp/` 几乎完全没有覆盖、本教程必须讲的东西。

文件:`include/cutlass/gemm/dispatch_policy.hpp`(覆盖 sm70 / sm80 / sm90 / sm100 / sm120 所有 dispatch tag)。

### 7.1 核心思想:用空 struct 当标签,用继承触发 dispatch

```cpp
// 空标记
struct KernelTmaWarpSpecialized {};

// 携带数据 + 继承
template <int Stages, class ClusterShape, class KernelSchedule>
struct MainloopSm90TmaGmmaWarpSpecialized : KernelTmaWarpSpecialized {
  constexpr static int Stages = Stages;
  using ClusterShape = ClusterShape;
  using Schedule = KernelSchedule;
};

// 派生 tag 进化论(同辈,而不是互相派生)
struct KernelTmaWarpSpecializedPingpong {};
struct KernelTmaWarpSpecializedCooperative {};
```

这是一个**性质**(traits-based)的分发机制:

1. **空 struct** 充当"种类 id"。
2. **带数据的派生**把数据搬上(struct 的 `using`、constexpr 字段),同时继承父 tag。
3. 在 `CollectiveMma<...>` 的**模板特化**(partial specialization)**匹配条件**里 `is_base_of_v<KernelTmaWarpSpecialized, ...Schedule>` 区分"这是一类 dispatch policy"。
4. 在 `CollectiveMma<MainloopSm90TmaGmmaWarpSpecialized<Stages, ...>, ...>` 的 partial specialization 拿到具体变体。

### 7.2 调度 tag 进化论

**重要**:CUTLASS 的"tag 进化论"是**约定层**(大家都叫 `KernelTmaWarpSpecialized*`)上的同辈结构,在 `include/cutlass/gemm/dispatch_policy.hpp` 里是这样声明:

```cpp
struct KernelTma {};                              // 基础(TMA 大类的根)
struct KernelTmaWarpSpecialized {};               // 基础:单 consumer warp group
struct KernelTmaWarpSpecializedPingpong { ... };  // 变体:2 个 consumer 交替(pingpong)
struct KernelTmaWarpSpecializedCooperative { ... }; // 变体:多 consumer 协同(适合大 tile)
```

它们之间**没有 C++ 继承关系**——这是关键。WarpSpec、Pingpong、Cooperative 是空/近空 struct,**同辈(siblings)**,都直接是 dispatch_policy.hpp 顶层声明。

```text
                KernelTma {}                  (根标签,代指"TMA 路径")
                   │
       ┌───────────┼───────────┐
       ▼           ▼           ▼
KernelTma-    KernelTma-    KernelTma-       (三个 WarpSpecialized 变体,同辈)
WarpSpec      WarpSpec      WarpSpec
              Pingpong      Cooperative
```

对应的"Pingpong 加 FP8 后缀"等更细的 tag(`KernelTmaWarpSpecializedPingpongFP8Blockwise` 等)则**确实**用 C++ 继承(`: KernelTmaWarpSpecializedPingpong`)。这种"派生关系 vs. 同辈 tag"是两种不同的 dispatcher 风格,别混。

**为什么 builder 不靠 C++ 继承来 dispatch 这三个基础 tag?** 因为它们在源码里没有父子关系,builder 用的是 `cute::is_same_v<Schedule, KernelTmaWarpSpecialized>(或 Pingpong / Cooperative)` 在 `static_assert` 与 `if constexpr` 里硬枚举——参见 `dispatch_policy.hpp` 中 `MainloopSm90TmaGmmaRmemAWarpSpecialized` 的 `static_assert`。这意味着你写一个新 schedule 时**必须**自己在那串枚举里加一行。

### 7.3 用户如何"改 schedule"

`examples/49_hopper_gemm_with_collective_builder/49_collective_builder.cu` 把 Schedule 暴露成 template parameter:

```cpp
template <
  class MainloopScheduleType = cutlass::gemm::collective::KernelScheduleAuto,
  class EpilogueScheduleType = cutlass::epilogue::collective::EpilogueScheduleAuto,
  ...
>
struct ExampleRunner { ... };

// 用例:
ExampleRunner<cutlass::gemm::KernelTmaWarpSpecializedPingpong> runner;
```

`KernelScheduleAuto` 是一个"占位 type"——builder 看到它就用一个 default tag 替换。看到具体 `KernelTmaWarpSpecializedPingpong` 就是用户的明确选择。

### 7.4 SFINAE 接入点——`detail` 命名空间

```cpp
namespace cutlass::gemm::kernel::detail {

template <class Kernel>
struct IsCutlass3GemmKernel
    : std::bool_constant<is_base_of_v<KernelTmaWarpSpecialized, /*Schedule*/>...> {};

template <class Kernel>
using GetMmaPipeline = typename Kernel::DispatchPolicy::Schedule;  // ← 提取主调度的 alias

}  // namespace
```

这套 helper 用于:

- `GemmUniversalAdapter` 决定走"3.x path"还是"2.x path"(`IsCutlass3GemmKernel`)
- 任何想"看 kernel 当前用什么 schedule"的代码用 `GetMmaPipeline`

> 把这一对 helper 跟你写的 dispatcher 联动——比如你想写一个"自动选不同 schedule for 不同 problem shape"的下游工具,这里的 `GetMmaPipeline<MyKernel>` 就能拿到。

### 7.5 这一章的核心 takeaway

**tag-inheritance dispatch** 是 CUTLASS 3.x 实现"配置空间巨大但每个具体实现编译期最优"的关键——比 2.x 的 DefaultGemmUniversal 工厂好得多,因为编译器**完全**只编当前用户选定的那个 partial specialization,生成的代码无冗余。

如果以后你想:

- 写一个新 schedule(比如"3 个 consumer 同步")
- 写一个新 dispatcher("依问题尺寸选 schedule")
- 写一个新 builder 选项

入口就在 `dispatch_policy.hpp` + `kernel/sm90_*.hpp` 的 SFINAE 路由 + `kernel/detail/` helper。

### 7.6 章末:读完这一章你该做得到的事

- ✅ 在自己的代码里手写 `using Schedule = KernelTmaWarpSpecializedPingpong;` 替换 `KernelScheduleAuto`,看到 builder 走不同代码路径。
- ✅ 在 `dispatch_policy.hpp` 里读懂 tag 树。
- ✅ 用 `Kernel::DispatchPolicy::Schedule` 这一行 alias 找到当前选了什么 schedule。

---

## 第 8 章:CollectiveBuilder——把"形状 + 类型"压成具体实现 {#ch08-collective-builder}

文件:`include/cutlass/gemm/collective/builders/sm90_gmma_builder.inl`(~10 个 partial specialization)。

### 8.1 它的任务

`CollectiveBuilder<13 个模板参数>` 是一个 dispatcher: 给定 `(arch × op_class × element × layout × alignment × element × layout × alignment × accumulator × tile × cluster × stage × schedule)` 共 13 维空间里的一个组合,**挑出**一个 `CollectiveMma` partial specialization。

它做的事很机械化(SELECT),但靠 partial specialization 路由,完全编译期,没有运行时代价。

### 8.2 实际看一个 partial spec

最大的(开头附近)`CollectiveBuilder<arch::Sm90, arch::OpClassTensorOp, ...>`:

```cpp
template <typename ... Args>
class CollectiveBuilder<
    /* mandatory args */
    arch::Sm90,
    arch::OpClassTensorOp,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccumulator,
    TileShape_MNK,
    ClusterShape_MNK,
    /* optional args */
    StageCountType,
    KernelSchedule,
    /* derived */
    ...
> {
public:
  // ... (推导部分)

  // 1. 决定 mma atom:
  //    TiledMma = make_tiled_mma(
  //      GMMA::ss_op_selector<ElementA, ElementB, ElementAccumulator, TileShape_MNK>(),
  //      AtomLayoutMNK{}
  //    );
  //
  //    ss_op_selector 是元编程函数,按 (A_dtype, B_dtype, acc_dtype, tile) 选
  //    "SM90 64xNxK SS"(WGMMA),并把 N、K 推到 tile 的具体形状。

  // 2. 决定 pipeline stage 数:
  //    constexpr int Stages = compute_stage_count<
  //        MainloopSm90TmaGmmaWarpSpecialized, SmemSizeFunctor, TileShape_MNK,
  //        StageCountType, sizeof(Epilogue::SharedStorage), AlignmentA, AlignmentB
  //    >;
  //    如果用户给了 StageCount<N>,Stages = N。
  //    如果 StageCountAuto,Stages = 看 smem 预算 + epi carveout 后能塞几 stage。
  //    如果 StageCountAutoCarveout<bytes>,Stages = 减去 bytes 之后再 auto。

  // 3. 决定 dispatch policy:
  //    using DispatchPolicy = MainloopSm90TmaGmmaWarpSpecialized<
  //        Stages, ClusterShape_MNK, KernelSchedule
  //    >;

  // 4. 决定 Smem Layout:
  //    constexpr auto stages = Int<DispatchPolicy::Stages>{};
  //    constexpr auto tile_M = size<0>(TileShape_MNK{});
  //    constexpr auto tile_K = size<2>(TileShape_MNK{});
  //    using SmemLayoutAtomA = decltype(
  //        composition(Swizzle<...>, Layout<Shape<_tile_M, _tile_K>, Stride<_tile_K, _1>>>{})
  //    );
  //    using SmemLayoutA = decltype(
  //        tile_to_shape(SmemLayoutAtomA{}, make_shape(_tile_M, _tile_K, stages), Step<_2, _1, _3>{})
  //    );
  //    // ↑ 注意 Step<_2, _1, _3> vs Step<_1, _2, _3>:根据 LayoutA 的 major 选 K-major 还是 MN-major
  //
  //    // (B 类似)

  // 5. 决定 Smem Copy Atom:
  //    using SmemCopyAtomA = Copy_Atom<DefaultCopy, SmemElementA>;
  //    using SmemCopyAtomB = Copy_Atom<DefaultCopy, SmemElementB>;
  //    // ↑ 这两个通常是空原子,在 pipeline 中没有"内部 copy"

  // 6. 决定 Transform:
  //    using TransformA = cute::identity;  // 默认 identity,不变换
  //    using TransformB = cute::identity;

  // 7. 用这些派生参数做 CollectiveMma:
  //    using CollectiveOp = cutlass::gemm::collective::CollectiveMma<
  //        DispatchPolicy,
  //        TileShape, ElementA, StrideA, ElementB, StrideB,
  //        TiledMma,
  //        GmemTiledCopyA = SM90_TMA_LOAD, SmemLayoutAtomA, SmemCopyAtomA, TransformA,
  //        GmemTiledCopyB = SM90_TMA_LOAD, SmemLayoutAtomB, SmemCopyAtomB, TransformB
  //    >;
};
```

具体可读性,因为 builder 的代码常被选:不要被文件大小劝退——这是"在每个 partial spec 干一件略有不同的事"。读时按 1-7 步骤理解。

### 8.3 epilogue builder 是镜像

文件:`include/cutlass/epilogue/collective/builders/sm90_builder.inl`。同样结构,主要差异:

- 多接 `(ElementC, LayoutC, AlignmentC, ElementD, LayoutD, AlignmentD, StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore, ...)` 等。Caller 在 Ch2.3 看到。
- 内部推 EVT 的根节点(如果用户给 EVT 就用,否则用 `DefaultEpilogue`)。

### 8.4 "Auto" 实际上到底是什么

`StageCountAutoCarveout<epi_bytes>` 不是一个"运行时选择",是**编译期**算法:

```cpp
constexpr int compute_stage_count(...) {
  // 1. 从 default stage count(根据 smem 限制计算)开始
  // 2. 减去 epilogue::SharedStorage 占用的 bytes
  // 3. 检查 smem 剩余是否还够每个 stage 的 A+B tensor
  // 4. 选 <= 剩余预算 的最大 stages
}
```

编译期算的,程序运行时根本不知道"auto 决定过"——结果是具体某个 `Stages` 数值。

### 8.5 "怎么改默认 schedule / stages / cluster:动手清单"

|想做什么|怎么改|
|---|---|
|强制 stage 数|`StageCount<4>` 替换 `StageCountAutoCarveout<...>`|
|强制 single-pipeline|`KernelScheduleAuto` 替换成 `KernelTmaWarpSpecialized` (明确)|
|强制 pingpong|`KernelTmaWarpSpecializedPingpong`|
|强制 cooperative|`KernelTmaWarpSpecializedCooperative`|
|强制小 tile (256×128 vs 128×128)|`TileShape<_128, _128, _32>` 换成 `_256`, `_128`|
|强制 cluster 形状|`ClusterShape<_1, _2, _1>` 替换 `<_4, _2, _1>`|

实际项目里,"默认是 Pingpong 还是 Cooperative?"由 Builder 内部的经验启发式决定(`CutlassHeuristics`)。

### 8.6 章末:读完这一章你该做得到的事

- ✅ 能在 `sm90_gmma_builder.inl` 里读懂 partial specialization 的结构(挑 `KernelTmaWarpSpecialized` 那一个开始)。
- ✅ 给一个具体配置"(fp16, RowMajor × ColMajor, 128×128×32, ClusterShape<_4,_2,_1>)",你能**手算** builder 会推什么(AtomLayoutMNK、StagesSmemLayoutAtomA)。
- ✅ 知道 `Auto*` 不是"运行时决定",而是编译期算。

---

## 第 9 章:`examples/49`——一个用户故事 {#ch09-user-story}

文件:`examples/49_hopper_gemm_with_collective_builder/49_collective_builder.cu`。

这一章**不引入**新概念——是把 Ch1-8 的所有主题打包成"用户故事"。

### 9.1 这个例子做了什么

把 Ch2 的 GEMM 包装到 `ExampleRunner<...>` 模板里,允许在 **编译期** 改 4 个东西:

```cpp
template <
  class MainloopScheduleType = cutlass::gemm::collective::KernelScheduleAuto,
  class EpilogueScheduleType = cutlass::epilogue::collective::EpilogueScheduleAuto,
  class StageCountType       = cutlass::gemm::collective::StageCountAuto,
  class TileSchedulerType    = cutlass::gemm::PersistentScheduler,
  bool UseCustomEVT          = false
>
struct ExampleRunner {
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignmentA,
      ElementB, LayoutB, AlignmentB,
      ElementAccumulator,
      TileShape, ClusterShape,
      StageCountType,    // user-tunable
      MainloopScheduleType   // user-tunable
  >::CollectiveOp;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementAccumulator,
      ElementC, LayoutC, AlignmentC,
      ElementC, LayoutC, AlignmentC,
      EpilogueScheduleType  // user-tunable
  >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int>,
      CollectiveMainloop,
      CollectiveEpilogue,
      TileSchedulerType  // user-tunable
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  // ... other:run(), can_implement(), args_from_options, etc.
};
```

`main()` 用一个配置网格跑多次:

```cpp
int main() {
  // 默认 — 跟 examples/48 一样
  run<Default, Default, ..., false>();

  // 切到 Pingpong
  run<KernelTmaWarpSpecializedPingpong, Default, ..., false>();

  // 加 EVT bias + relu
  run<Default, Default, ..., true>();

  // 改 TileScheduler 为 StreamK
  run<Default, Default, ..., StreamKScheduler, false>();
}
```

### 9.2 加 EVT bias + ReLU fusion 的具体写法

```cpp
if constexpr (UseCustomEVT) {
  using namespace cutlass::epilogue::fusion;
  using CustomEVT = Sm90EVT<
      Sm90Compute<homogeneous_add, /*stage=*/0>,    // ← bias addition
      Sm90EVT<
        Sm90Compute<homogeneous_unary<ReLU>, /*stage=*/1>,  // ← ReLU
        Sm90AccFetch                              // input is the accumulator
      >,
      Sm90SrcFetch                                // bias tensor source
  >;
  // 把 EVT 接到 CollectiveEpilogue<...>::Arguments(略):
  epilogue_args = ...;  // 注入 CustomEVT
}
```

`Sm90EVT<Op, ...Args>` 是一个 nested type:Op 是当前节点的算子,Args 是该算子的输入子节点列表。叶子节点(`Sm90AccFetch` / `Sm90SrcFetch` / `Sm90ScalarBroadcast`)直接提供数据。

### 9.3 Builder 重新解析的 4 个 user 参数

当用户在 `ExampleRunner<Pingpong, ...>` 改 MainloopScheduleType:

1. **MainloopScheduleType** = `KernelTmaWarpSpecializedPingpong`——这直接传到 `CollectiveBuilder`(Ch8)。Builder 看到这是一个具体 tag(不是 Auto),直接:
   - 选 partial spec `CollectiveBuilder<Sm90, TmaWarpSpecializedPingpong, ...>`
   - 推 DispatchPolicy = `MainloopSm90TmaGmmaWarpSpecializedPingpong<Stages, ClusterShape, KernelTmaWarpSpecializedPingpong>`
   - 该 DispatchPolicy 推导具体 mainloop 实例化(就是 `sm90_mma_tma_gmma_ss_warpspecialized_pingpong.hpp` 的 partial spec)

2. **EpilogueScheduleType** 影响 epilogue builder——同 1。

3. **StageCountType** = `StageCount<4>`(强制 4 stage)。Builder 跳过 auto 算法,直接用 4。

4. **TileSchedulerType** = `StreamKScheduler`(切到 StreamK)。
   - `GemmUniversal<..., StreamKScheduler>` 的 SFINAE 路由(Ch6.1)— 因为 `StreamKScheduler` 不在 mainloop 的 kernel schedule tree 里,只影响 kernel 匹配的 TileScheduler。
   - 这会进 `sm90_tile_scheduler_stream_k.hpp` 的 partial spec。

### 9.4 这章的 takeaway

读 `examples/49` 时反复出现的 4 个 `*Type` template 参数 + `UseCustomEVT` bool,**对应 Ch7 (dispatch policy) + Ch8 (builder) + Ch5 (EVT) + Ch6 (scheduler) 这 4 件事**。

每一件都对应一个"切来切去只改一行 type alias"的具体动作:

- 切 schedule:换 mainloop pipeline 形态
- 切 stage count:换 smem 预算
- 切 epilogue schedule:换 epilogue 调度形态
- 切 TileScheduler:换持久 kernel 调度策略
- 切 EVT:加融合算子

### 9.5 章末:读完这一章你该做得到的事

- ✅ 把 `examples/49` 完整读一遍——它的代码结构跟 Ch1-8 的所有抽象都对应。
- ✅ 修改 `ExampleRunner<KernelTmaWarpSpecializedPingpong>` 跑一次,确认 mainloop 切到了双 warp group。
- ✅ 把 `UseCustomEVT = true` 跑一次,看 epilogue 加入 bias+ReLU 的 EVT 是否让性能变化。
- ✅ `TileSchedulerType = StreamKScheduler` 切过去,看是否能编译过去(可能需要不同的 problem shape,因为 StreamK 适合 K-bound)。

10 章收口——调参。

---

## 第 10 章:调参世界观 + `cutlass_profiler` {#ch10-tuning}

这一章不讲新机制——只是把 Ch1-9 给的所有"开关"整理成一张调参表,以及"怎么科学地 autotune"。

### 10.1 调参组合空间一览

`examples/48` 默认配置已经接近峰值性能。但不同 (M, N, K) shape + 不同硬件 + 不同内存层级下,还是要调。

可调的"维度"在 5 个层级上:

|调参维度|在哪|给用户切换的接口|
|---|---|---|
|**tile size**(M×N×K)|Ch2.2|`TileShape<>...>`, 一般 128/128/32 或 128/256/64|
|**cluster size**|Ch2.2|`ClusterShape<>...>`, 一般 4×2×1 / 4×4×1 / 8×8×1|
|**stage 数**|Ch2.3|`StageCount<N>` 强制;Auto 默认从 smem 推|
|**schedule 变体**|Ch7|`KernelTmaWarpSpecialized` 默认; Pingpong / Cooperative 更大 tile|
|**scheduler**|Ch6.7|`PersistentScheduler` 默认; `StreamKScheduler` 用于 K-bound|
|**raster / swizzle**|Ch2.4|`arguments.scheduler.raster_order` / `max_swizzle_size`|

> **你手写 GEMM 的对照**:这些就是你 grid_dim / block_dim / smem 切分 / cluster_dim / pipeline depth / swizzle 选型的全部决定项——CUTLASS 只是把它们"显式化"了。

### 10.2 (M, N, K) shape 决定的"匹配模式"

经验法则(可直接套):

|Shape 类型|推荐 tile|推荐 cluster|推荐 schedule|
|---|---|---|---|
|大 M、N(≥ 8192 双向 square)|`<_256,_128,_64>`|`<_2,_1,_1>` 或 `<_1,_1,_1>`|`Cooperative` (大 tile) 或 default|
|小 M(≤ 1024) + 大 N(≥ 8192)|`<_128,_256,_64>`|`<_1,_2,_1>` 或 `<_1,_1,_1>`|default 或 `Pingpong`|
|大 M、小 N 镜像|`<_256,_128,_64>`|`<_2,_1,_1>`|default|
|严格 square 中等|`<_128,_128,_32>`|`<_2,_2,_1>` 或 `<_4,_2,_1>`|default(单 warp group)|
|K-bound (K 小,M、N 大)|`<_256,_128,_32>` 或更大 K-tile|`<_1,_1,_1>`|default|
|K-bound (K 大, M、N 小)|`<_128,_128,_128>`|`<_1,_1,_1>`|`StreamKScheduler` 启用(并行化 K 维)|
|低 occupancy GPU (smem 受限)|`<_64,_64,_32>`|`<_1,_1,_1>`|smaller tile, max stages 减少|
|NVFP4 / FP8 + block scale|fp8 collector with `StageCountAuto`(stage 不变,加 block scale smem)|...|...|

> 这张表来自 `media/docs/cpp/gemm_api_3x.md` + CUTLASS profiling 多年的结果,但**实际**还要用 `cutlass_profiler` 跑一遍才知(下面)。

### 10.3 Swizzle / raster / L2 cache

CTAs 顺序与 L2 命中:

- `RasterOrderOptions::AlongN` = N direction first。**L-shape (M << N) GEMM** 时,L2 命中率更高(横向扫的 output tile 被同 L2 set 命中)。
- `RasterOrderOptions::AlongM` = M direction first。**L-shape (N << M)** 同理。
- `RasterOrderOptions::Heuristic` = 由 problem shape 自动选。
- `max_swizzle_size = 8` = 8×8 swizzle,适合"很多小 tile"。= 1 = 无 swizzle,适合超大 tile。
- `max_swizzle_size = 4` 是普适默认。

为什么 swizzle 有用:CTAs 的 tile id 不再按行/列顺序访问,而是按 swizzle 间隔访问——访问次序被打散,L2 缓存被"沿 tile 群循环用"而不是一次性被前面的 CTA 用完。

### 10.4 TMA multicast

`ClusterShape` 大了之后,sm_90 的 TMA 支持 multicast:一个 TMA load 让多个 CTA 都各拿到相同的 smem 数据,节省 gmem 带宽。**条件**:load 的 src 形状与 cluster 大小对齐。

具体看 `media/docs/cpp/dependent_kernel_launch.md` + `examples/63_hopper_gemm_with_weight_prefetch/` 的 pre-fetch + multi-cast pattern。

### 10.5 `cute::print_tensor` / `print_smem_layout` 调试

```cpp
// 调试某个 layout 的索引模式:
cute::print(smem_layout_A);              // 输出 (M, K, Stages) + stride
cute::print_latex(smem_layout_A);        // 输出 LaTeX (可视化)

// 调试某个 tensor:
cute::print_tensor(A_gmem_tensor);       // 实际打印前 1024 个元素
cute::print_tensor(A_smem_register_tile); // 同样的 register tile,按 mma view 解读
```

> 这些 hook 非常有用——尤其当你的 swizzle 后发现有些 lane 拿到错误数据时,print 一下能清晰看到"lattice"在每 lane 上的分布。

### 10.6 `cutlass_profiler` 一段最小使用

文件:`tools/profiler/src/main.cpp`。`cutlass_profiler` 是 CUTLASS 自带的 autotuner(已编译好的二进制)。

```bash
# 编译
mkdir build && cd build
cmake -DCUTLASS_NVCC_ARCHS=90a ..   # 或 100 等
make -j cutlass_profiler
# 跑一个 grid 扫描
./tools/profiler/src/cutlass_profiler \
  --operation=gemm \
  --tile=128x128_32x32 \
  --stage=4 \
  --cluster=2x2x1 \
  --iterations=10 \
  --warmup=2 \
  --problem-shape=4096x4096x4096 \
  --element=f16,f16,f16 \
  --output=csv
```

常用 flag(完整列表 `media/docs/cpp/profiler.md`):

- `--operation=gemm` / `gemm_array` / `gemm_grouped`
- `--tile=<MxN>KxK` 切换 tile
- `--stage=<N>` 强制 stage 数
- `--cluster=<MxNxK>` cluster 形状
- `--raster=<Heuristic|AlongM|AlongN>`
- `--swizzle=<1|2|4|8>` swizzle 深度
- `--iterations=<N>` 测量 N 次取平均
- `--warmup=<N>` 前 N 次不计入

输出末尾会有 GFLOPS / TFlops 数字。**profile 之前要先暖机**——`--warmup=10` 让 GPU 时钟稳定。

### 10.7 经验法则

- **经验式 1**:从 default 跑一遍 `cutlass_profiler`,得 baseline FLOPS。
- **经验式 2**:换 tile size,跑一遍;每个 tile 都"顺着 shape 推 stage 数 + 集群大小"。
- **经验式 3**:`Cooperative` schedule 在 M/N ≥ 8192 时往往最强。
- **经验式 4**:`Pingpong` 在 tile 中等(128×128) 时比 `default` 略快,在大 tile(256×256) 时反而损失。
- **经验式 5**:`StageCountAutoCarveout` 永远是对的起点,先别手动。
- **经验式 6**:`StreamK` 仅在 K-bound shape 上跑赢 persistent。
- **经验式 7**:`max_swizzle_size` 在 Hopper 上是 8 时最普适。
- **经验式 8**:alignment 128-bit 是 TMA baseline,8-byte alignment(64-bit)不够。

### 10.8 图配

![3.x gemm peak performance](media/images/cutlass-3.5.1-gemm-peak-performance.png)

(以较新一张 3.x 数据图作为参考——你看到的可能是 FP8 / FP16 不同图。)

### 10.9 章末:读完这一章你该做得到的事

- ✅ 用 `cutlass_profiler --operation=gemm --tile=128x128_32x32 --stage=4 ...` 跑一次 baseline。
- ✅ 在不同的 (M, N, K) shape 上用表格 10.2 推荐一组 tile+cluster,各自 profiler 一遍。
- ✅ 在 Ch9 的 `ExampleRunner<>` 里调 4 个 `*Type` 参数,配合 `cutlass_profiler` 评测。
- ✅ 懂得 `cutlass_profiler` 输出的几个数字怎么解读(GFLOPS、有效率、瓶颈分析)。

---

## 第 11 章:Blackwell 桥接——同样的 5 层架构,换了一组原子 {#ch11-blackwell}

最后这一章**反向**证实 Ch1-8 的 5 层抽象为什么值钱:同一个 5 层框架在 Blackwell 上几乎是逐行对应,但底层的 mma / smem 单元变了。

**核心 takeaway**:你在 Hopper 路径上学到的 5 层(CUTLASS 3.x 5-tier)架构,**无需重新学**就能进入 Blackwell。你只需重新学**第 3 层(CollectiveMainloop 的 MMA 部分)+ 几处 epilogue 接口面(TMEM)**。

### 11.1 什么没变:5 层框架完全保持

5 个类名 — `GemmUniversalAdapter` / `kernel::GemmUniversal<...>` / `collective::CollectiveMma` / `epilogue::CollectiveEpilogue` / `*TileScheduler` — 在 sm_100 上全部保留。

`examples/70_blackwell_gemm/` 的 .cu 文件跟 `examples/48_hopper_warp_specialized_gemm/` 的 .cu 文件**几乎是逐行对应**——`TileShape` / `ClusterShape` / `StageCountType` / `KernelScheduleType` 都在同一位置。读 `examples/70` 时,你会觉得"已经在 Ch2 读过一遍"。

### 11.2 什么变了:WGMMA → UMMA,smem 部分结果 → TMEM

#### MMA:`WGMMA::ss_op_selector` → `UMMA::ss_op_selector`

- **WGMMA**(sm_90):64xNxK(SM90_64x16x16_F16F16F16F32_SS 等);用户代码中=mma(0.0) 的形式由原子指令直接控制。
- **UMMA**(sm_100):`tcgen05.mma` 系列;**异步的标量协程**——你写 launch 指令,GPU 自己决定何时启动;**允许多 ct 在 cluster 内共享时打平 bandwidth**。

UMMA 7 条指令,`media/docs/cpp/blackwell_functionality.md` 列了出来(经典对比表):

|指令|等价 Hopper|用途|
|---|---|---|
|`tcgen05.mma`|`wgmma.mma_async.sync.aligned`|主 mma|
|`tcgen05.feedthrough`|n/a|TMA feedthrough(C 通过 TMA 直接进 mma,无需 smem)|
|`tcgen05.sp`|n/a|Sparse(2:4 结构化稀疏)|
|等等(7 条全列在该 doc)|||

#### 累加器:smem 部分结果 → TMEM

UMMA 的 accumulation 默认写到 **TMEM**(Tensor Memory)——一种 sm_100 新加的专用 memory,只对 mma 累加器访问友好。

```cpp
// Hopper (smem 中间结果)
SharedTmem smem_acc_buffer;  // 一个普通 smem 段
...
smem_acc_buffer = smem_acc_buffer + new_acc;  // rmem → smem → mma → smem

// Blackwell (TMEM 累加器)
TmemAllocator tmem_alloc;     // include/cute/arch/tmem_allocator_sm100.hpp
tmem_alloc.allocate(...) ;
umma(...).to(tmem_alloc);     // 直接落到 tmem
```

> 你写 Hopper 时**没有**这个 TMEM 抽象——smem 是个通用 buffer。CUTLASS 把 TMEM **隔离**到 sm100 mainloop 内部,user-facing API 不变。

#### 文件对应的迁移

|层|Hopper|Blackwell|
|---|---|---|
|Adapter|`gemm_universal_adapter.h`(完全一致)|同|
|Kernel orchestrator|`kernel/sm90_gemm_tma_warpspecialized.hpp`|`kernel/sm100_gemm_tma_warpspecialized.hpp`|
|Mainloop|`collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp`|`collective/sm100_mma_warpspecialized.hpp`(还有 sm100_mma_cpasync_warpspecialized.hpp 等变体)|
|Epilogue|`epilogue/collective/sm90_epilogue_tma_warpspecialized.hpp`|(mainloop 文件已包含 epilogue 路径;sm100 不会单写 epilogue)|
|Scheduler|`kernel/sm90_tile_scheduler.hpp`|`kernel/sm100_tile_scheduler.hpp`(多 `cluster_launch_control`)|

### 11.3 新类型宇宙:block-scaled(mx_*)与窄精度

Hopper 时代主要在 FP16/BF16/TF32/FP8。Blackwell sm_100 引入"块缩放":每 32 个元素共享一个 scale factor,把窄精度的精度损失用统计平均扳回来。

涉及类型(`media/docs/cpp/fundamental_types.md` 是完整表):

- **4-bit**:`e2m1`、`e3m2`、`e2m3`、`e4m3`(部分)
- **8-bit**:`e4m3`、`e5m2`
- **scale**:`ue8m0`(无符号 8-bit,zero-only)、`ue4m3`
- **复合**:`mx_float4_t`、`mx_float8_t`(整组代表"FP4 + scale" / "FP8 + scale")

> 这一节**只**列出类型名,不教 MXFP8 数学。要学 MXFP8 看 NVIDIA GPU 上的官方 OCaml 文档 / `cutlass/tools/library/include/cutlass/library_block_scaled.h`(本教程保持"是什么,不为什么")。

### 11.4 起步路径

读 Blackwell 路径,**4 个入口**:

|入口|文件 / 路径|内容|
|---|---|---|
|**入门 example**|`examples/70_blackwell_gemm/`|与 examples/48 几乎逐行对应|
|**数学 / 指令**|`media/docs/cpp/blackwell_functionality.md`|7 条 tcgen05.mma + 吞吐量对比 Hopper + 类型表|
|**类型**|`media/docs/cpp/fundamental_types.md`|完整 numerical type catalog|
|**Python DSL guide**|`media/docs/pythonDSL/mma_docs/tcgen05_programming.rst`|若用 CuTe Python 路径的等价入手|
|**inline 教程**|`examples/cute/tutorial/blackwell/01_mma_sm100.cu ... 05_mma_tma_epi_sm100.cu`|与 `examples/cute/tutorial/hopper/wgmma_sm90.cu` 镜像结构|

`blackwell_cluster_launch_control.md`(在 `media/docs/cpp/` 下)讲 sm_100 新加的 cluster 同步原语 — cluster 比 sm_90 更大,用新原语管理。

### 11.5 章末:从 Hopper 视角看 Blackwell 的 5 件不变 + 5 件变

|不变|变|
|---|---|
|5 层框架(`GemmUniversalAdapter` / `GemmUniversal` / `CollectiveMma` / `CollectiveEpilogue` / `*TileScheduler`)类名与方法签名|Tag 树根换:`KernelTmaWarpSpecialized*` → `KernelTmaUmmaWarpspecialized*`(`include/cutlass/gemm/dispatch_policy.hpp`)|
|Builder(`CollectiveBuilder`)的"用 13 维参数拼实例化"配方|MMA atom 来源:`WGMMA::ss_op_selector` → `UMMA::ss_op_selector`|
|`Examples/48` 的 4 步 host API 写法(`using` → builder → adapter → run)|对应 `Examples/70_blackwell_gemm/` 同样 4 步(逐行对应 Ch2)|
|Ch5.4 的 EVT 写法(`Sm90EVT<Sm90Compute<...>, ...>`)|EVT 在 sm100 上由 sm100 epilogue 接管,但 AST 语法完全一致(根节点的算子是 `Sm100*Compute` 而不是 `Sm90*Compute`)|
|`PersistentTileScheduler` 选 tile|sm100 新加 cluster 同步原语 `cluster_launch_control`(见 `media/docs/cpp/blackwell_cluster_launch_control.md`)|

另外两个**整个体系新增**的事:

|新增方向|对哪一层|
|---|---|
|Block-scaled 数值类型(`mx_float*` / `e2m1` / `e4m3` / `ue8m0` 等)|`Builder` 的输入类型空间(见 §11.3)|
|TMEM(只对 mma 累加器友好的专用 memory)|Mainloop 内部使用,通过 `tmem_allocator_sm100.hpp` 接入(**不**外露到 5 层公共接口)|

**收口**:读前 10 章,你已经掌握了"如何照搬 5 层框架到一个新架构";读本章,你已经掌握了"看到 sm_100_*.hpp 时那 5 件变对应到哪些具体文件名"。

附录补一些查表。

---

## 附录 A:Hopper 原语 ↔ CUTLASS 封装文件(速查表) {#app-a-primitives}

|原语|CUTLASS 封装路径|一行说明|
|---|---|---|
|WGMMA(`wgmma.mma_async.sync.aligned.m64nXk16...`)|`include/cute/arch/mma_sm90_gmma.hpp`|单条 WGMMA 指令的 CuTe 包装|
|WGMMA traits / atom / tiled_mma|`include/cute/atom/mma_traits_sm90_gmma.hpp` + `include/cute/atom/mma_atom.hpp`|怎么把 WGMMA 包装成 `Atom` 再包装成 `TiledMma`|
|TMA load(gmem → smem)|`include/cute/arch/copy_sm90_tma.hpp`|`copy_sm90_tma.hpp` 给 `SM90_TMA_LOAD` atom|
|TMA store(smem → gmem)|`include/cute/arch/copy_sm90_tma.hpp`(同)+ `sm90_epilogue_tma_warpspecialized.hpp`|`SM90_TMA_STORE` 由 epilogue 使用|
|TMA descriptor|`include/cute/arch/copy_sm90_desc.hpp`|`cute::TmaDescriptor` + `SubbyteTmaDescriptor`|
|cp.async.bulk|`include/cute/arch/copy_sm90.hpp`|`cp_async_bulk_*` 包装|
|SMEM swizzle|`include/cute/swizzle.hpp`|`Swizzle<B, M, S>`|
|Cluster launch / DSMEM|`include/cute/arch/cluster_sm90.hpp` + `include/cutlass/cluster_launch.hpp`|`cute::block_rank_in_cluster()` 等|
|Producer-consumer 流水线|`include/cutlass/pipeline/sm90_pipeline.hpp`|`PipelineTmaAsync<N>` + `PipelineState` / `PipelineParams` / `PipelineStorage`|
|Warp group 同步|`include/cutlass/arch/barrier.h` + `cute/arch/cluster_sm90.hpp`|`cluster_arrive` / `cluster_wait`|
|Thread ID → 角色映射|sm90_gemm_tma_warpspecialized.hpp 的 `operator()`|`WarpGroupRole { Producer, Consumer }` + `ProducerWarpRole { MainloopEpilogue, Warp1, ... }`|
|PersistentKernel|`include/cutlass/gemm/kernel/sm90_tile_scheduler.hpp`|`PersistentTileSchedulerSm90`|
|Cutlass 5 层抽象|`include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized.hpp`|`cutlass::gemm::kernel::GemmUniversal<...>`|
|Tag-inheritance 调度|`include/cutlass/gemm/dispatch_policy.hpp`|`KernelTmaWarpSpecialized` 等空 struct tag|

**怎么用**:写新 GEMM / 调参时遇到一个概念,先在本表找"哪个文件封装了它"。文件路径基本都暴露在 `examples/48` 包含头列表里。

---

## 附录 B:In-tree 散文定位("你想读 X 看 Y") {#app-b-treemap}

本文是自洽的,但仓库里 `media/docs/cpp/` 还有 ~30 篇高质量散文。本表告诉你"如果想深挖,看哪里"。

|想读什么|文件|本教程哪章引用|
|---|---|---|
|**CUTLASS 3.x 整体设计哲学**|`media/docs/cpp/cutlass_3x_design.md`|序章 + Ch1 — 5 个设计宣言的扩展阅读|
|**GEMM API 参考(3.x)**|`media/docs/cpp/gemm_api_3x.md`|Ch8 — Builder 的 4 步"装配内核"配方|
|**CUTLASS 2.x → 3.x 迁移**|`media/docs/cpp/cutlass_3x_backwards_compatibility.md`|(本文未引入)|
|**2.x GEMM API**|`media/docs/cpp/gemm_api.md`|(本文未引入)|
|**高效 GEMM 设计与伪码**|`media/docs/cpp/efficient_gemm.md`|Ch6 — 经典 5 层 nested-loop 伪码|
|**术语表**|`media/docs/cpp/terminology.md`|任何章节 — 名词定义|
|**目录地图**|`media/docs/cpp/code_organization.md`|Ch1 — 仓库组织说明|
|**完整 quickstart**|`media/docs/cpp/quickstart.md`|Ch2 — 该文件 §"Launching a GEMM kernel using CUTLASS 3.0 or newer"|
|**CUTLASS 代码风格 / Params / SharedStorage 约定**|`media/docs/cpp/programming_guidelines.md`|Ch2 + Ch4 — Params/SharedStorage 惯例|
|**CUTLASS ↔ 数据类型支持表**|`media/docs/cpp/functionality.md`|(查询用)|
|**数值类型 catalog**|`media/docs/cpp/fundamental_types.md`|Ch11 — sm_100 mx_* 类型说明|
|**性能测量方法学**|`media/docs/cpp/gemm_performance_measurement_methodology_guidelines.md`|Ch10 — 测量 cuBLAS 对比时的科学方法|
|**profiler CLI**|`media/docs/cpp/profiler.md`|Ch10.6|
|**heuristics / autotune**|`media/docs/cpp/heuristics.md`|Ch10.6|
|**Blackwell 功能清单**|`media/docs/cpp/blackwell_functionality.md`|Ch11.2 — 7 条 tcgen05.mma + 类型表|
|**Blackwell cluster launch control**|`media/docs/cpp/blackwell_cluster_launch_control.md`|Ch11.5|
|**依赖 kernel launch(PDL)**|`media/docs/cpp/dependent_kernel_launch.md`|Ch10.4 / 附录 D|
|**Grouped scheduler**|`media/docs/cpp/grouped_scheduler.md`|附录 D|
|**Implicit GEMM Convolution**|`media/docs/cpp/implicit_gemm_convolution.md`|附录 D|
|**CuTe 入门**|`media/docs/cpp/cute/00_quickstart.md`|Ch3|
|**CuTe Layout primer**|`media/docs/cpp/cute/01_layout.md`|Ch3 — 教材级长文,可复读|
|**CuTe Layout 代数**|`media/docs/cpp/cute/02_layout_algebra.md`|Ch3 — composition / coalesce / zipped-divide 等|
|**CuTe Tensor = Engine + Layout**|`media/docs/cpp/cute/03_tensor.md`|Ch3|
|**CuTe Algorithms (copy / gemm / clear)**|`media/docs/cpp/cute/04_algorithms.md`|Ch3.4 — `cute::gemm` 的来源|
|**CuTe MMA atom naming**|`media/docs/cpp/cute/0t_mma_atom.md`|Ch3.5|
|**CuTe GEMM 实战教程**|`media/docs/cpp/cute/0x_gemm_tutorial.md`|Ch3.5 — 配 `examples/cute/tutorial/sgemm_*.cu`|
|**CuTe 残差 / predication**|`media/docs/cpp/cute/0y_predication.md`|(本文未引入)|
|**CuTe + TMA**|`media/docs/cpp/cute/0z_tma_tensors.md`|Ch4 — 读懂 TMA + CuTe 联合用法|

**怎么用**:读完本文后,如果某个章节里某段你有"想再深一层"的冲动,在本表对应行找到该深读的散文。**不要**先看散文再回来看本文——本文是按"够用"写的,看散文是为了"扩展"。

---

## 附录 C:你手写 GEMM 的 X 行 ↔ CUTLASS 哪里(对照表) {#app-c-mapping}

这一章是"用户视角的快速跳转表"——给你在写 CUTLASS 3.x GEMM 时,看到自己手写 GEMM 的某一段,就跳到 CUTLASS 的对应位置。

|你手写 Hopper GEMM 中的功能 / 代码|CUTLASS 3.x 中的对应位置|备注|
|---|---|---|
|`-arch=sm_90a`|`using ArchTag = cutlass::arch::Sm90;`|Ch2.2|
|选 WGMMA(不是 cp.async.mma)|`using OperatorClass = cutlass::arch::OpClassTensorOp;`|Ch2.2|
|`dim3 block_dim(...)`|`using TileShape = Shape<_M, _N, _K>;`|Ch2.2|
|`dim3 cluster_dim(...)`|`using ClusterShape = Shape<_cluster_M, _cluster_N, _cluster_K>;`|Ch2.2|
|`cudaLaunchKernel(...)`|`Gemm gemm; gemm.can_implement(...); gemm.initialize(...); gemm.run();`|Ch2.5|
|`if (!can_handle_residue(m, n, k)) return;`|`gemm.can_implement(arguments)`|Ch2.5|
|`__shared__ float smem_A[TileM][TileK];` + swizzle|`SmemLayoutAtomA = composition(Swizzle<B,M,S>, Layout<Shape<_TileM, _TileK>, Stride<_TileK, _1>>{})`|Ch8|
|`prepare_gmem_desc(...)` 写 TMA desc|`collective::CollectiveBuilder` 内部 + `prefetch_tma_descriptors(...)`|Ch2.5 + Ch4.4|
|`cuTensorMapEncodeTiled(...)`|`cute::TmaDescriptor` 包装|附录 A|
|TMA load `cp.async.bulk.tensor.*`|`cute::copy(SM90_TMA_LOAD, ...)`|Ch4.3|
|`wgmma.mma_async.sync.aligned.m64n...k16...`(单条)|`cute::gemm(TiledMma, A_frag, B_frag, acc)` (内部 dispatch 到该指令)|Ch3.4|
|`is_producer = warpIdx < 4` 分支|`WarpGroupRole { Producer, Consumer }` + `kernel/.../operator()`|Ch6.3|
|producer/consumer 屏障数组(`bar[N]`)|`cutlass::PipelineTmaAsync<N>`|Ch4.2 + Ch6.4|
|`__syncthreads()` 同步 producer/consumer|`PipelineTmaAsync::producer_acquire / producer_commit / consumer_wait / consumer_release`|Ch4.3|
|cluster 同步 `cluster_arrive + cluster_wait`|`cute::cluster_arrive + cute::cluster_wait`|Ch6.3|
|`grid = ceil_div(M, BlockM) * ceil_div(N, BlockN);`|`PersistentTileSchedulerSm90Params::num_blocks_in_grid`|Ch6.5|
|`blockIdx.x` 决定本 CTA 处理哪个 tile|`PersistentTileSchedulerSm90::get_current_work(...)`|Ch6.5|
|swizzle 步进(swizzle_pattern[blockIdx.x])|`max_swizzle_size = ...` 在 `arguments.scheduler` 中|Ch2.4 + Ch6.5|
|tiled row-major store D|`CollectiveEpilogue::operator()` 内部 TMA store|Ch5.2|
|加 bias bias[b], epilogue 时 `D = alpha * acc + beta * C + bias[m] + bias[n]`|EVT 树 = `Sm90EVT<Sm90Compute<...>, AccFetch, SrcFetch,...>`|Ch5.4|
|加 ReLU|`homogeneous_unary<ReLU>` 在 EVT 节点中|Ch5.4|
|加 silu|自定义 unary functor(例如 §5.4 中的 `IdentitySilu`)在 `Sm90Compute<...>` 节点中|Ch5.4|
|输出 swizzle(`D_swizzled[i,j] = D[i^swizz_b, j^swizz_m]`)|`examples/50_hopper_gemm_with_epilogue_swizzle/`||
|Stream-K(把 K 维继续切,partial sum)|`TileSchedulerType = StreamKScheduler`|Ch6.7|
|Grouped GEMM(多 problem 一次 launch)|`examples/57_hopper_grouped_gemm/`|附录 D|
|2:4 structured sparse GEMM|`examples/62_hopper_sparse_gemm/`|附录 D|
|im2col / col2im Conv|`examples/cute/tutorial/sgemm_*` + `media/docs/cpp/implicit_gemm_convolution.md`|附录 D|
|Pull weight prefetch (overlap weight load 与 compute)|`examples/63_hopper_gemm_with_weight_prefetch/`|附录 D|

**怎么用**:从你手写 GEMM 的视角查表(而不是"我从 CUTLASS 视角要改什么"反着查)。这是反向索引。

---

## 附录 D:再之后(Grouped / Sparse / Conv / SSD / PDL) {#app-d-future}

本教程主线是 GEMM。但 CUTLASS 还在以下方向延伸,**每个主题 1–2 段**,告诉你"文件名 + 一个指针"。

### D.1 Grouped GEMM

文件:`include/cutlass/gemm/kernel/sm90_tile_scheduler_group.hpp` + `examples/57_hopper_grouped_gemm/` + `media/docs/cpp/grouped_scheduler.md`。

用于:**一次 launch 跑多个不同形状的 GEMM**(典型:MoE 的 experts 前向)。`GroupScheduler` 由 `GroupProblemVisitor` 牵引多个 problem。example 57 演示。

### D.2 Sparse GEMM(2:4 structured sparsity)

文件:`include/cutlass/gemm/collective/sm90_sparse_mma_tma_gmma_ss_warpspecialized.hpp` + `examples/62_hopper_sparse_gemm/`。

用于:A / B 中每 4 元素有 2 个 0 的稀疏矩阵;cutlass 通过索引元数据把"非零 2 元素"读进 mma。**一般用于 Ampere TF32,但 Hopper 上也有 sparse variant**。

### D.3 Convolution(im2col / col2im)

文件:`media/docs/cpp/implicit_gemm_convolution.md` + `examples/16_ampere_tensorop_conv2dfprop/`。

将 conv2d `y[n,p,q,k] = Σ_c Σ_r Σ_s x[n,c,p+r,q+s] · w[k,c,r,s]` 重写为 implicit GEMM。**Conv 不是切 GEMM 做的——是把 conv 当成一个"非常特定的 GEMM"来编译**(im2col → GEMM → col2im)。

具体看 `media/docs/cpp/implicit_gemm_convolution.md` 的 im2col 形式。

### D.4 SSD(Sparse + SD,DSS-style decoupling)

文件:`examples/111_hopper_ssd/` + `examples/112_blackwell_ssd/`,以及 `media/images/13_example_*`。

Sparse + Dst-Sparse Decomposition:用稀疏权重 + dense activation 算 sparse dense matmul,但反着——dense-side 同样可以"反稀疏"。

### D.5 PDL(Dependent Kernel Launch)

文件:`media/docs/cpp/dependent_kernel_launch.md` + `examples/63_hopper_gemm_with_weight_prefetch/`。

PDL 是 sm_90+ 的硬件特性:**让 host 在 kernel A 还没结束时启动 kernel B**,只要 kernel A 满足"init done"信号就行。CUTLASS 提供 `epilogue_aux` 这种 server-style kernel 来利用 PDL。

### D.6 Conv 路径内的 GEMM

`examples/cute/tutorial/sgemm_*` 系列也是为 conv 教程铺垫。`16_ampere_tensorop_conv2dfprop/` 是 Ampere 时代 Conv + fused epilogue 真实例。

### D.7 Sparse + Block-scaled Quantization

`examples/95_blackwell_gemm_blockwise/*` 演示 sm_100 上的 block-scale + sparse。`examples/67_hopper_fp8_warp_specialized_gemm_with_blockwise_scaling/` 同主题 Hopper 版本。

---
