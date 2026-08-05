## 第 6 章:Kernel orchestrator + TileScheduler(含调度器族侧栏)

这一章读 `include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized.hpp` + `sm90_tile_scheduler.hpp` + `static_tile_scheduler.hpp` + `tile_scheduler_params.h`。

把 Kernel orchestrator 和 TileScheduler 合并讲的原因:**它们在一个 kernel 的入口一起出现**——`operator()` 调 first `tile_scheduler.fetch_next_work(...)`,得到 (m, n) 切片,然后跑 mainloop + epilogue。

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
    // 1) 决定本 CTA 这一轮跑哪个 tile — fetch_next_work 返回 pair
    auto [work_tile_info, increment_pipe] =
        TileScheduler::fetch_next_work(params.scheduler);
    auto [m_coord, n_coord, l_coord] = work_tile_info;     // WorkTileInfo 只有 M/N/L idx

    if (!work_tile_info.is_valid()) {
      break;   // 全部分配完了
    }

    // 2) 计算在本 tile 内 K 维 split(用于 K-loop 边界) — K 不在 WorkTileInfo 里
    int k_tile_count = ...;   // 例如:size<3>(gA_mkl) from mainloop load
    int k_tile_idx   = ...;

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

    // 6) cluster barrier(等待同 cluster 的兄弟 CTA 都跑完本 tile 才进入下一 tile)
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

`Arguments`(用户填,`tile_scheduler_params.h`)只暴露两个字段:

```cpp
struct Arguments {
  int max_swizzle_size = 0;                            // 1 / 2 / 4 / 8
  RasterOrderOptions raster_order = RasterOrderOptions::Heuristic;  // Heuristic / AlongM / AlongN
};
```

实际 `Params`(scheduler 内部算完后固化,`PersistentTileSchedulerSm90Params` 在 `tile_scheduler_params.h:87`)大致是:

```cpp
struct PersistentTileSchedulerSm90Params {
  FastDivmodU64Pow2 divmod_cluster_shape_major_{};    // cluster major 维快速除/模
  FastDivmodU64Pow2 divmod_cluster_shape_minor_{};
  FastDivmodU64    divmod_batch_{};
  FastDivmodU64    divmod_cluster_blk_major_{};

  uint64_t blocks_per_problem_ = 0;                   // 总 CTA 数
  int32_t  log_swizzle_size_   = 0;                   // log2(max_swizzle_size)
  RasterOrder raster_order_    = RasterOrder::AlongN; // 解析后的 raster
  uint32_t problem_tiles_m_, problem_tiles_n_, problem_tiles_l_;
  uint32_t cluster_shape_m_,   cluster_shape_n_;
};
```

`raster_order` 决定 "tile 是按 row-major 还是 col-major 摊派到 CTA id":

- `AlongN` = N 先推进(横向扫)
- `AlongM` = M 先推进(纵向扫)
- `Heuristic` = 自动选(根据矩阵形状)

`max_swizzle_size` 决定"多大块长的 L2 reuse":1 是无 swizzle、8 是 8×8 swizzle block(swizzle 后同 L2 set 被多个 CTA 命中)。

#### 工作算法

```cpp
// fetch_next_work(WorkTileInfo prev) 返回 pair<next, increment_pipe>
auto fetch_next_work(WorkTileInfo work_tile_info) -> tuple<WorkTileInfo, bool> {
  // 当前是第几次 work-index
  uint64_t work_idx = work_id_counter_++;
  if (work_idx >= total_num_tiles_) {
    return {WorkTileInfo::invalid_work_tile(), false};
  }
  // 把 work_idx → (m_tile, n_tile),按 swizzle + raster 算
  auto [m_tile, n_tile] = get_work_idx_m_and_n(
      blk_per_grid_dim_, work_idx, swizzle_log_, raster_order_);
  return {WorkTileInfo{m_tile, n_tile, ...}, increment_pipe};
}
```

`get_work_idx_m_and_n` 是核心解调度函数,输入 work id 和 swizzle / raster 配置,输出 (m, n) 切片坐标。

### 6.6 图配

上面 §6.5 提到的两种调度模式:

![persistent_static](../media/images/persistent_static.png)

![non_persistent](../media/images/non_persistent.png)

### 6.7 调度器族侧栏

不是只 PersistentTileScheduler 一家。CUTLASS 3.x 中,TileScheduler 是一个 sumtag(`PersistentScheduler` / `StreamKScheduler` / `GroupScheduler` / `StaticPersistentScheduler` / `DynamicPersistentScheduler`),由 `TileSchedulerSelector` 按 tag 派发。

|Tag|文件|何时用|数学梗概|
|---|---|---|---|
|`PersistentScheduler` (default)|`sm90_tile_scheduler.hpp`|默认;每个 CTA 处理多 tile,持续运行|work_id → (m, n) by swizzle/raster|
|`StaticPersistentScheduler`|`static_tile_scheduler.hpp`|不需要调度复杂度的轻量版|简化版 persistent|
|`StreamKScheduler`|`sm90_tile_scheduler_stream_k.hpp`|K 维的 split 与 partial result 合并|K-parallelism,partial sum|
|`GroupScheduler`|`sm90_tile_scheduler_group.hpp`|grouped GEMM(多组不同形状的 GEMM 同 kernel 跑)|用 problem_visitor 拉多个 problem|

每种都暴露同名 `fetch_next_work` 接口(以及 sm100 CLC 用的 `advance_to_next_work`),所以 Ch6.3 的 kernel orchestrator 代码**完全不动**,只换 scheduler tag 就能切。

#### `StreamKScheduler` 的梗概(供"如果你好奇"用)

Stream-K 把 work 沿 K 维再切,每个 worker 处理某个 (m, n, k_partial) cube,所有 partial result 写到 partial buffer(smem 或 gmem),**reduction 由同一个 kernel 在 tile 边界处 atomic-add 完成**(CUTLASS 3.x 的 `PersistentTileSchedulerSm90StreamK` 在 `sm90_tile_scheduler_stream_k.hpp`)。这样可以"完全用满 GPU"——任何 K-bound 形状都能打平。

注意 `examples/47_ampere_gemm_universal_streamk/` 是 CUTLASS **2.x** 的 Stream-K 演示(走 `UniversalGemm` + 单独的 reduction kernel),跟 3.x 的 `StreamKScheduler` tag 是两套实现,不要混。

#### `GroupScheduler` 梗概

Grouped GEMM(每组 problem 的 M/N/K 不同,如 MoE),调度器由 `GroupProblemVisitor` 拉数据并按 shape 切。把 multiple GEMM 共用一次 launch。

### 6.8 章末:读完这一章你该做得到的事

- ✅ 在 kernel 入口看到 `WarpGroupRole { Producer, Consumer }`、`ProducerWarpRole`、`persistent_scheduler.fetch_next_work(...)` 这些,认得出各自在做什么。
- ✅ 能口述"6 个 helper 在 producer / consumer 主循环里按什么顺序被调用"。
- ✅ 把 task graph 在脑子里跑一遍:TMA load → barrier release → WGMMA → barrier release → TMA store。
- ✅ 知道 StreamK / Grouped 调度的存在和位置,以后看代码不陌生。

Ch7 看 dispatch policy——为什么这一切被自动选到。

---

