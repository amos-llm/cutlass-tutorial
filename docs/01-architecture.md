## 第 1 章:5 层架构图(整体鸟瞰)

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
  // 在 kernel 主循环被调用,决定这一轮跑哪个 tile — 返回 pair<next, increment_pipe>
  cute::tuple<WorkTileInfo, bool> fetch_next_work(WorkTileInfo prev);

  // sm100 CLC 路径专用:用 clc pipeline + state 推进
  PipelineState advance_to_next_work(Pipeline&, PipelineState) const;
};
```

Scheduler 也有自己的参数(由 host 端 `Arguments.scheduler` 填):

```cpp
arguments.scheduler.raster_order     = options.raster;       // AlongN / AlongM / Heuristic
arguments.scheduler.max_swizzle_size = options.swizzle;      // 1 / 2 / 4 / 8
```

> **你手写 GEMM 的对照**:你的 `grid_dim` 计算 + `blockIdx` 分配策略。CUTLASS 默认走"persistent + swizzle + raster"——见 Ch6 详细。

### 1.6 三句话侧栏:这些 `sm90_*_warpspecialized` 文件不是"5 个不同实现"

看到 `include/cutlass/gemm/collective/` 下有几十个 `sm90_*_warpspecialized*.hpp` 不要慌——它们是**同一个 partial specialization 在 4 个维度上的变体**:

|维度|取值|
|---|---|
|mma 形态|GMMA(SS:smem↔smem) / RS(A 在 register) / Array(多 CTA 协同)|
|调度|WarpSpecialized(1 个消费者 warp group) / Pingpong(2 个交替) / Cooperative(多 warp group)|
|类型|f16 / bf16 / tf32 / fp8 / fp8+block-scaling / mixed-input / sparse|
|输入|A/B 都是 gmem(A gmem→smem) / A register(A gmem→register→mma)|

第 7 章会讲——这种变体是怎么靠"tag-inheritance"模式**同构**地生成的,你改 dispatch policy tag 就换实现,不用"换文件"。

---

