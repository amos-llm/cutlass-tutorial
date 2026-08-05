## 第 2 章:跑通一个最小 GEMM——走读 `examples/48`

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
using ElementA   = float;                  // 类型:FP32(smem 里仍是 FP32;TF32 精度截断在 WGMMA 执行时发生)
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

