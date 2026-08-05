## 第 4 章:深入 CollectiveMainloop(本教程核心价值)

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

**15 个模板参数**。看似吓人,分两组:

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

#### 8 个方法——逐个拆解

每个方法的「签名 / 谁调 / 什么时候 / 做什么 / 前置条件」。**每个 entry 都对应 `sm90_mma_tma_gmma_ss_warpspecialized.hpp` 一处,行号是对应**。

##### 1. `to_underlying_arguments`(static,host 端)

```cpp
template <class ProblemShape>
static constexpr Params
to_underlying_arguments(ProblemShape const& problem_shape,
                        Arguments const& args, void* workspace);
```

- **行号**: L202-240
- **谁调**: Device 层 `GemmUniversalAdapter` 在 `initialize()` 里调一次,**kernel 启动前**
- **做什么**: 把 host 端的 runtime `Arguments`(`ptr_A / ptr_B / dA / dB` 之类)固化进 `Params`,并在 **这一步** 用 `make_tma_copy_A_sm90` / `make_tma_copy_B_sm90` 把 TMA descriptor 编好。返回的 `Params` 会被传给 kernel,device 端只读
- **前置条件**: `workspace` 必须指向 builder 在 `get_workspace_size()` 里声明的地址(虽然 sm90 这份实现 `(void) workspace;` 没用,但 `sm100_umma_builder.inl` 会用——比如 ptr-array batch 要把 tensor map 数组放 workspace)
- **特点**: `static constexpr` 但内部其实是运行时构造 TMA descriptor(TMA 描述符涉及运行时计算 box dimensions)。这就是 Ch0.2「默认正确」那段说的"为了避免意外触发编译而拆 3 段"——`to_underlying_arguments` 在 `initialize()` 里跑,出问题不会让编译炸

##### 2. `can_implement`(static,host 端)

```cpp
template<class ProblemShape>
static bool
can_implement(ProblemShape const& problem_shape, Arguments const& args);
```

- **行号**: L242-261
- **谁调**: Device 层 `GemmUniversalAdapter::can_implement(arguments)` 转发到这里
- **做什么**: 校验 (problem_shape × args) 是不是**合法组合**——目前 sm90 这份只查 **TMA 对齐**(`check_alignment<min_tma_aligned_elements_A/B>` 看 `M, K` 维 stride 是否满足 128-bit 对齐)
- **前置条件**: 无
- **返回值**: `true` 表示能跑,`false` 表示 builder 选了不支持的组合。调用方拿 false 时会抛 `CUTLASS_CHECK` 错误
- **注意**: 这里的 `can_implement` **不查 smem 是否够**——smem 检查在 builder 里(Ch8 `compute_stage_count_or_override`)。也就是说 `can_implement` 通过不代表一定跑得起来,还得看 builder 推出来的 `PipelineStages` 是否 > 0

##### 3. `prefetch_tma_descriptors`(static,device 端)

```cpp
static void prefetch_tma_descriptors(Params const& mainloop_params);
```

- **行号**: L271-276
- **谁调**: kernel orchestrator(Ch6 的 `GemmUniversal::device_kernel`),只让 warp 0 lane 0 真正调,其他线程在 `if (lane_predicate)` 外直接跳过
- **做什么**: `cute::prefetch_tma_descriptor(tma_load_a)` + 同 b——把 TMA descriptor 提到 L1/Tex cache。**这步必须在 kernel 第一条 TMA 指令之前足够早**,否则首次 TMA 会有 stall
- **前置条件**: `mainloop_params.tma_load_a/b` 必须有效(由 `to_underlying_arguments` 填好)
- **幂等性**: 标了 `static`,但因为是 device 端方法,**不是**编译期常量;真正幂等靠 caller 的 `lane_predicate` 保护(每个 warp 一份工作,warp 0 lane 0 做,其他人 skip)

##### 4. `load_init`(instance const,device 端,K-loop 前调一次)

```cpp
template <class ProblemShape_MNKL>
CUTLASS_DEVICE auto
load_init(ProblemShape_MNKL const& problem_shape_MNKL,
          Params const& mainloop_params) const;
```

- **行号**: L284-301
- **谁调**: kernel orchestrator 在 producer 和 consumer K-loop **开始前各调一次**——producer 拿到 gmem 视图用来发 TMA,consumer 一般不需要但为了接口一致也调
- **做什么**: 把 `mainloop_params.tma_load_a/b` 里藏着的 full gmem tensor 拿出来,用 `local_tile(..., TileShape{}, make_coord(_,_,_), Step<...>{})` 切成"CTA 级"视图,返回 `tuple<gA_mkl, gB_nkl>`。注意返回的不是 smem 布局——是 **gmem 的 tile 视图**,TMA 后续要靠它
- **前置条件**: `mainloop_params` 必须有效
- **返回**: `cute::tuple<Tensor, Tensor>`,shape 分别 `(BLK_M, BLK_K, m, k, l)` 和 `(BLK_N, BLK_K, n, k, l)`。**这是 collective 和 kernel 层的契约**——`load()` 用 `get<0>(load_inputs)` / `get<1>(load_inputs)` 解构。L280-282 注释明确说 "first two elements must be gA_mkl and gB_nkl"

##### 5. `load`(instance,device 端,K-loop 每 tile 调一次,producer)

```cpp
template <class TensorA, class TensorB, class KTileIterator, class BlockCoord>
CUTLASS_DEVICE void
load(Params const& mainloop_params,
     MainloopPipeline pipeline,
     PipelineState smem_pipe_write,
     cute::tuple<TensorA, TensorB> const& load_inputs,
     BlockCoord const& blk_coord,
     KTileIterator k_tile_iter, int k_tile_count,
     int thread_idx,
     uint32_t block_rank_in_cluster,
     TensorStorage& shared_tensors);
```

- **行号**: L309-392
- **谁调**: producer warpgroup 的 K-loop 体内,每个 K-tile 调一次
- **做什么**: 用 TMA 把 gmem 的 `(BLK_M, BLK_K)` 和 `(BLK_N, BLK_K)` 写到 smem 的当前 stage。**单线程**(`cute::elect_one_sync()`)做实际 TMA 发起,其他线程直接 skip;multicast 时构造 mcast_mask
- **前置条件**: `smem_pipe_write` 当前 stage 必须已被 consumer `release`(由 `pipeline.producer_acquire()` 检查);`load_inputs` 是 `load_init` 返回的 tuple
- **循环里做的事**(per K-tile):
  1. `pipeline.producer_acquire(smem_pipe_write)` —— 等 stage 空
  2. `pipeline.producer_get_barrier(smem_pipe_write)` —— 拿当前 stage 的 mbarrier
  3. `copy(tma_load_a.with(barrier, mcast_mask), gmem_partition, smem_partition(_,_,_,write_stage))` —— 发 TMA
  4. `++k_tile_iter` —— 推到下一个 K 切片
  5. `++smem_pipe_write` —— 推到下一个 stage(parity 自动翻转)
- **关键**: TMA 完成时**自动 arrive barrier**(`producer_commit` 在 TMA 路径是 NoOp,见 Ch4 4-方法语义框);所以 load 里**没有显式 commit**
- **`CUTLASS_PRAGMA_NO_UNROLL`**: L371 显式阻止编译器 unroll,producer / consumer 必须按 runtime `k_tile_count` 推进

##### 6. `load_tail`(instance,device 端,K-loop 收尾,producer 调一次)

```cpp
CUTLASS_DEVICE void
load_tail(MainloopPipeline pipeline, PipelineState smem_pipe_write);
```

- **行号**: L394-409
- **谁调**: producer K-loop 退出后调一次
- **做什么**: `pipeline.producer_tail(smem_pipe_write)` —— 等所有 stage 都被 consumer `release`(或初始 `make_producer_start_state` 的 phase 让 acquire 直接成功)。**关键目的**: 防止 cluster 里 producer 跑得快的 CTA 提前 `cudaDeviceSynchronize` 退出,L2 缓存被踢,导致慢的 CTA 拿不到数据
- **前置条件**: K-loop 体内 producer 已推进 `smem_pipe_write` 到「最后推进 + 1」位置(就是 K-loop 退出时的 state)
- **单线程**: `cute::elect_one_sync()` 保护,只有 warp 0 lane 0 真的调

##### 7. `mma`(instance,device 端,K-loop 每 tile 调一次,consumer)

```cpp
template <class FrgTensorC>
CUTLASS_DEVICE void
mma(MainloopPipeline pipeline,
    PipelineState smem_pipe_read,
    FrgTensorC& accum,
    int k_tile_count,
    int thread_idx,
    TensorStorage& shared_tensors,
    Params const& mainloop_params);
```

- **行号**: L416-559
- **谁调**: consumer warpgroup 的 K-loop 体内,每个 K-tile 调一次
- **做什么**: 1) 等 smem 当前 stage 可读,2) 把 smem 上 `(BLK_M, BLK_K, PIPE)` 和 `(BLK_N, BLK_K, PIPE)` 的当前 stage 通过 `partition_A/B` 摊到 `TiledMma` 的 thread fragment,3) `cute::gemm(TiledMma, ..., accum)` 推到 `wgmma.mma_async.sync.aligned.m64n...k16...`,4) `consumer_release` 把 stage 让回给 producer
- **前置条件**: `accum` 必须是 rmem 驻留的 `cute::Tensor`(`static_assert(is_rmem<FrgTensorC>::value)`);`smem_pipe_read` 当前 stage 必须已被 producer TMA 完成(由 `pipeline.consumer_wait()` 检查)
- **循环里做的事**(per K-tile):
  1. `pipeline.consumer_wait(smem_pipe_read)` —— 等 stage 写好
  2. 取当前 stage 的 smem sub-tile,做 `cute::gemm(TiledMma, frag_A, frag_B, accum)`
  3. `pipeline.consumer_release(smem_pipe_read)` —— 释放 stage
  4. `++smem_pipe_read` —— 推到下一个 stage(parity 翻转)
- **关键**: `mma` 内层 `cute::gemm` 实际是 PTX `wgmma.mma_async`(Hopper 上是异步 mma,warpgroup 指令),一次发 64×16×16 sub-tile。多次 sub-tile 累加到同一个 `accum`,直到 K-loop 退出
- **`K_PIPE_MMAS = 1`** (L264): WGMMA 一次只取 1 个 K-tile,所以 consumer/producer K-loop 长度严格相等

##### 8. `mma_tail`(instance,device 端,K-loop 收尾,consumer 调一次)

```cpp
CUTLASS_DEVICE void
mma_tail(MainloopPipeline pipeline,
         PipelineState smem_pipe_release, int k_tile_count);
```

- **行号**: L561-577
- **谁调**: consumer K-loop 退出后调一次
- **做什么**: 1) 处理 K-loop 没跑完的「prologue 剩余 mma」(`prologue_mma_count = min(K_PIPE_MMAS, k_tile_count)`,K_PIPE_MMAS=1,这里基本就是 0 或 1),2) `smem_pipe_release.advance(k_tile_count)` 推到对应 stage,3) `warpgroup_wait<0>()` 等所有 async mma 完成,4) 对 `prologue_mma_count` 个 stage 做 `consumer_release`
- **前置条件**: K-loop 已退出,`accum` 不再被修改
- **目的**: **确保所有 WGMMA 指令 retire**(异步 mma,提交后还在 GPU 内部流水)。`warpgroup_wait<0>` 是硬件级 wait,把 warpgroup 全部 mma 指令 fence 完,才能保证 `accum` 写完可被 epilogue 读

#### 方法之间的契约

把 8 个方法串起来看,**consumer 和 producer 之间的契约**只通过两类东西传:

1. **smem 上的数据**(通过 pipeline barrier 同步可见性)
2. **`PipelineState`**(推进 stage、parity 翻转)

所以 `mma` 和 `load` 各自带一个独立的 `PipelineState`(`smem_pipe_read` / `smem_pipe_write`),**初始位置**也不同:

- producer 起点:`make_producer_start_state<MainloopPipeline>()`(phase 让首次 `acquire` 直接成功)
- consumer 起点:默认构造(`smem_pipe_read` 默认从 stage 0 开始)

两个 state 在 K-loop 里各自 `++`,互相不依赖——通过 barrier 的 arrive/wait 间接同步。这是 **Ch0「5 层互不依赖」的体现**:mainloop 内部怎么编排 producer/consumer 都是 CollectiveMma 的事,kernel 层只调 5 个方法 + 维护两块 PipelineState。

`producer_acquire(state)` / `producer_commit(state, bytes)` / `consumer_wait(state)` / `consumer_release(state)` 这些都是 `PipelineTmaAsync<N>` 类的方法,在 `include/cutlass/pipeline/sm90_pipeline.hpp` 里(`PipelineTmaAsync<Stages>::producer_acquire(PipelineState state)` 等)。

#### 四方法的语义——Ch4 必须先校准的 4 个词

`PipelineTmaAsync<N>` 的 4 个方法每个有明确的「阻塞 / 非阻塞 / 可能在某些条件下退化」语义,写 mainloop 时不能搞混:

| 方法 | 谁调用 | 阻塞? | 作用 |
|---|---|---|---|
| `producer_acquire(state)` | producer(TMA 加载 warp) | **阻塞** | 等到 `state` 对应的 stage 被 consumer `release` 后才返回——也就是确认这个 smem 槽位空出来了,可以写入。 |
| `producer_commit(state)` | producer | **非阻塞**(但在 TMA 场景下可能被吞掉,见下) | 把当前 stage 标记成「producer 已完成」,通知 consumer。 |
| `consumer_wait(state)` | consumer(WGMMA warp) | **阻塞** | 等到 `state` 对应的 stage 被 producer `commit` 后才返回——也就是确认 smem 槽位里已有可读数据,可以发起 mma。 |
| `consumer_release(state)` | consumer | **非阻塞** | 把当前 stage 标记成「consumer 已用完」,让 producer 可以重新写这个槽位。 |

两个「会被吞掉」的关键细节,手写时容易踩:

1. **`producer_commit` 在 TMA 场景下可能是 NoOp**——因为 TMA 自身完成时会自动 arrive barrier 并 arrive bytes,所以你再调一次 `producer_commit` 是冗余的(但无害)。`PipelineTmaAsync` 故意保留了 API 形式一致;真正的 producer-commit 语义由 TMA 完成事件承担。
2. **`make_producer_start_state<MainloopPipeline>()` 不是装饰**——pipeline 在起点「空」的时候,producer 的第一次 `acquire` 会直接成功;但你必须显式构造「让首轮 acquire 成功」的 state,而不是用默认构造的 state。这是 `smem_pipe_write = make_producer_start_state<MainloopPipeline>()` 在 Ch6 主循环出现的原因。

> 一句话总结:**acquire / wait 是「等我需要的东西就绪」,commit / release 是「通知对方我这边就绪/用完」。** 两个阻塞方法(acquire、wait)是真正让线程「停下来等」的同步点;两个非阻塞方法只是更新 barrier 状态、不卡线程。

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

这一行 `cute::gemm(TiledMma, smem_A_tile, smem_B_tile, accumulators)` 不只调用,还**实例化**——它根据 `TiledMma` 的 MMA atom / Layout,推到具体 WGMMA 指令(`wgmma.mma_async.sync.aligned.m64nXk16`). 这就是 Ch3.5 提到的 5-case dispatch 在 mainloop 的实际位置。

具体两步:

1. **从 `TiledMma` 拿到 atom**: `TiledMma` 内部存的就是一个 `MMA_Atom<...>`(Ch3.4 定义的),比如 `SM90_64x16x16_F32F16F16F32_SS`,以及 `AtomLayout`(`<_2,_2,_1>` 之类,Ch3.4 第二段)。
2. **从 atom + Layout 推到具体 mma 指令**: `cute::gemm` 走 Ch3.5 的 5-case dispatch,根据 smem tile 的 layout rank(2D / 3D / V-mode)选 case;case 落到具体实现时,**从 atom 类型本身就推到了 PTX 字符串**(对 WGMMA atom,模板参数 `m64n16k16` 直接对应 `wgmma.mma_async.sync.aligned.m64n16k16`)。**所以同一行 `cute::gemm` 模板,换 atom 就换指令**——不需要 if-else。

主 mainloop 里**没有 if-else 选 WGMMA / cp.async.mma / fp8**——所有「用哪条指令」的信息都封装在 `TiledMma` 内部的 atom 里。这就是为什么 builder 推 atom 是关键步骤(Ch8 §8.1 步骤 1)。

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

注意: Ch4.6 上面那张表里的「变体」(SS/RS/FP8/sparse/grouped)改的是 **mainloop 内部**(数据从哪里来、什么 dtype);这里的 3 个变体改的是 **kernel 编排逻辑**(几个 warp group 一起算一个 CTA 的事)。后者是「**producer / consumer 怎么分工**」,跟 collective mainloop 解耦:

| 变体 | 编排 | 适合 | 文件 |
|---|---|---|---|
| `KernelTmaWarpSpecialized` | **1 producer warpgroup + 1 consumer warpgroup**。producer 走 TMA 加载,consumer 走 WGMMA。固定分工,简单 | 中等 tile(BM/BN ≤ 128)| `kernel/sm90_gemm_tma_warpspecialized.hpp` |
| `KernelTmaWarpSpecializedPingpong` | **1 producer + 2 consumer 交替**(软硬流水线重叠加倍)| 中大 tile(BM/BN ≈ 128) | `kernel/sm90_gemm_tma_warpspecialized_pingpong.hpp` |
| `KernelTmaWarpSpecializedCooperative` | **1 producer + N 个 consumer 协同**算一个超大 tile(每个 consumer 算 tile 的一部分,WGMMA 之间互不重叠)| **巨型 tile**(BM=256 / BN=256 这种) | `kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp` |

机制差别一句话:

- **WarpSpec**: 1 个 consumer warpgroup 负责整个 tile 的 WGMMA,K-loop 串行。最简单、stage 数最少、对 smem 压力最小。问题是单 consumer 算 128×128 的 tile 已经是它的带宽上限,再大就瓶颈。
- **Pingpong**: 2 个 consumer **交替**——一个跑 K-step 0、另一个跑 K-step 1,软硬流水线叠加。每个 consumer 算**整个** tile,所以 K-loop 总吞吐 = 2 × 单 consumer 带宽。代价:smem pipeline 要 **double buffer**(stage 数 × 2)、smem 压力大。适合中等 tile 想再压一压。
- **Cooperative**: N 个 consumer **协同**算一个 tile,每个 consumer 只算 tile 的 **1/N**(比如 4 个 consumer 每人算 128×32)。tile 整体很大时,这种切分让每个 consumer 的 smem footprint 变小,可以塞更多 stage;同时 K-loop 总吞吐 = N × 单 consumer 带宽。代价:consumer 之间需要协调(谁负责哪个子区域),dispatch policy 里要算清楚 fragment 切分。

**怎么选?** 经验启发式(具体见 Ch8 的 `media/docs/cpp/heuristics.md` + `python/cutlass_library/heuristics.py`):

- BM × BN ≤ 128 × 128:WarpSpec 够用,smem 压力最小。
- 128 × 128 ~ 256 × 128:Pingpong 收益明显。
- ≥ 256 × 256:Cooperative 才喂得满。

`KernelScheduleAuto` 就是按这条启发式选;用户写 `KernelTmaWarpSpecialized` / `_Pingpong` / `_Cooperative` 之一就是显式覆盖。**这些 tag 之间没有 C++ 继承关系**(Ch7)——是同辈空 struct,builder 用 `is_same_v` 硬枚举路由。

### 4.7 图配

下面两张图都在 Ch6 也用到——这里先放感受一下 pipeline 的物理样子:

![pipeline](../media/images/software-pipeline.png)

![threadblock mma pipelined](../media/images/cutlass-threadblock-mma-pipelined.png)

Ch4 把 mainloop 讲完了。下一章 Ch5 看 epilogue——几乎是镜像结构。

---

