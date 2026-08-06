## 第 5 章:CollectiveMainloop 逐方法——5 个核心函数拆解

Ch4 讲了 mainloop 的**骨架**(类头、类型别名、三阶段、契约、Cluster、SharedStorage)。这一章**逐方法钻进去**——`load_init` / `load` / `mma` / `load_tail` / `mma_tail` 5 个函数,每个函数体里"每行 CuTe 计算在做什么"。

主文件:`include/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp`。

### 本章涉及 CUTLASS 源文件

- `include/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp:286/310/396/417/543` — 5 个方法 `load_init` / `load` / `load_tail` / `mma` / `mma_tail` 函数体
- `include/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp:281-298` — `gA_mkl` / `gB_nkl` 契约注释
- `include/cute/atom/mma_atom.hpp:531-541` — `make_tiled_mma` 入口
- `include/cutlass/pipeline/sm90_pipeline.hpp:271` — `PipelineTmaAsync` 的 4 个 helper(producer_acquire / producer_commit / consumer_wait / consumer_release)

### 5.0 本章导航

```text
§5.1  load_init:CuTe gmem 视图的构造                       ← 起点
§5.2  load:producer 的 CuTe 流程                          ← TMA + cluster 切分
§5.3  mma:consumer 的 CuTe 流程                           ← WGMMA + fragment
§5.4  load_tail + mma_tail:K-loop 收尾                    ← cluster 早退保护 + mma retire
§5.5  Pipeline 4 方法语义:必须先校准的 4 个词             ← producer_acquire / consumer_wait 等
§5.6  章末:读完这一章你该做得到的事                       ← 自检 checklist
```

读法建议:

- **第一次**(顺着读):§5.1 → §5.5 一把过,跟着 producer/consumer 走完 K-loop。
- **第二次**(按需查):§5.5 当速查表。

### 5.1 `load_init`:CuTe gmem 视图的构造

源码对应段:**`load_init`** 函数体,**kernel 入口各调一次**(producer / consumer 都要拿到 gmem 视图)。

```cpp
template <class ProblemShape_MNKL>
CUTLASS_DEVICE auto
load_init(ProblemShape_MNKL const& problem_shape_MNKL,
          Params const& mainloop_params) const {
  using X = Underscore;
  auto [M, N, K, L] = problem_shape_MNKL;

  // ① 从 TMA descriptor 拿 full gmem tensor
  Tensor mA_mkl = mainloop_params.tma_load_a.get_tma_tensor(make_shape(M,K,L));   // (m, k, l)
  Tensor mB_nkl = mainloop_params.tma_load_b.get_tma_tensor(make_shape(N,K,L));   // (n, k, l)

  // ② local_tile: 把 full tensor 切成 CTA 级
  Tensor gA_mkl = local_tile(mA_mkl, TileShape{}, make_coord(_,_,_), Step<_1, X, _1>{});  // (BLK_M, BLK_K, m, k, l)
  Tensor gB_nkl = local_tile(mB_nkl, TileShape{}, make_coord(_,_,_), Step< X, _1, _1>{});  // (BLK_N, BLK_K, n, k, l)

  return cute::make_tuple(gA_mkl, gB_nkl);
}
```

#### 5.1.1 `get_tma_tensor(make_shape(M,K,L))` — 从 TMA desc 拿 gmem Tensor

`mainloop_params.tma_load_a` 是 `Params::TMA_A`(由 `to_underlying_arguments` 里 `make_tma_copy_A_sm90` 构造),本质是个 **TMA descriptor + 对应的 CuTe Atom**。`get_tma_tensor(shape)` 返回一个 `cute::Tensor`,**底层指针来自 TMA descriptor**,layout 来自 desc 编进去的 stride:

- **底层 ptr**: TMA desc 编进去的 `ptr_A`
- **layout**: TMA desc 编进去的 stride + 用户给的 `make_shape(M, K, L)`
- **数据类型**: `ElementA`(fp16 / bf16 / fp8 等)

**这一步不是「重新构造 tensor」**,是「从 desc 把已经登记好的全局视图拿回来」。如果 `tma_load_a` 没编好(没 `make_tma_copy_A_sm90`),这一步会编译失败。

#### 5.1.2 `local_tile(...)` — 把 full tensor 切成 CTA 级

```cpp
local_tile(mA_mkl, TileShape{}, make_coord(_,_,_), Step<_1, X, _1>{})
//        ^full   ^shape       ^占位          ^stride: 在哪些 mode 上分 tile
```

`local_tile` 把一个 tensor 按 `TileShape` 切成 5D:第 0/1 维是 `BLK_M / BLK_K`(tile 内的位置),第 2 维是 `m`(tile 索引),第 3 维 `k` 保留,第 4 维 `l` 保留。**`Step<_1, X, _1>` 决定「哪个 mode 是 tile 索引、哪个 mode 保留」**:Step 里出现的 mode 顺序对应 TileShape 里被切的 mode,Step 里 `_1` 表示这个 mode 切成 tile,`X` 表示这个 mode **不切**(整个 tile 内是同一个值)。

`Step<_1, X, _1>` 的含义:

```text
Mode index:  0      1      2
对应 mkl:    m      k      l
含义:      切 tile  不切  切 tile
         (BLK_M)  (k)  (l)
```

所以 A 的 tile 把 M 和 L 维切成 `(BLK_M, l)` 两个 tile 索引,K 维作为 tile 内的连续遍历轴。**B 同理**但 `Step<X, _1, _1>` —— 把 N 和 L 切成 tile 索引。

> 一句话:**`local_tile` 之后,gmem tensor 变成 5D `(BLK, BLK, tile_idx, kept, tile_idx)`;后续 `load` 用 `(_, _, m_coord, _, l_coord)` 切片到当前 CTA 的 `(BLK_M, BLK_K)`。**

#### 5.1.3 返回的 `tuple<gA_mkl, gB_nkl>` 是 collective / kernel 之间的契约

`load_init` 函数体末尾的注释明确:**"first two elements must be gA_mkl and gB_nkl"**。这是 **kernel 层能依赖的契约**——kernel orchestrator(Ch6)只知道这俩名字,具体 layout 怎么切由 collective 推。**契约的好处**:换 collective 实现(比如从 SS 换 RS,或 sm90 换 sm100),kernel 代码不动。

### 5.2 `load`:producer 的 CuTe 流程

源码对应段:**`load`** 函数体,**单线程执行**(靠 `cute::elect_one_sync()`),每个 K-tile 调一次。

```cpp
CUTLASS_DEVICE void
load(Params const& mainloop_params,
     MainloopPipeline pipeline,
     PipelineState smem_pipe_write,
     cute::tuple<TensorA, TensorB> const& load_inputs,
     BlockCoord const& blk_coord,
     KTileIterator k_tile_iter, int k_tile_count,
     int thread_idx,
     uint32_t block_rank_in_cluster,
     TensorStorage& shared_tensors) {

  int lane_predicate = cute::elect_one_sync();
  if (lane_predicate) {
    // ① smem Tensor 构造
    Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.data()), SmemLayoutA{});
    Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.data()), SmemLayoutB{});

    // ② cluster 内分 CTA:本 CTA 看到 TMA desc 的哪一片
    constexpr uint32_t cluster_shape_x = get<0>(typename DispatchPolicy::ClusterShape());
    uint2 cluster_local_block_id = {block_rank_in_cluster % cluster_shape_x,
                                    block_rank_in_cluster / cluster_shape_x};
    auto block_tma_a = mainloop_params.tma_load_a.get_slice(cluster_local_block_id.y);
    auto block_tma_b = mainloop_params.tma_load_b.get_slice(cluster_local_block_id.x);

    // ③ 切到当前 CTA 的 gmem / smem 视图
    auto [m_coord, n_coord, k_coord, l_coord] = blk_coord;
    Tensor gA = gA_mkl(_, _, m_coord, _, l_coord);     // (BLK_M, BLK_K, k)
    Tensor gB = gB_nkl(_, _, n_coord, _, l_coord);     // (BLK_N, BLK_K, k)
    Tensor tAgA = block_tma_a.partition_S(gA);         // gmem 摊到 TMA thread
    Tensor tAsA = block_tma_a.partition_D(sA);         // smem 摊到 TMA thread

    Tensor tBgB = block_tma_b.partition_S(gB);
    Tensor tBsB = block_tma_b.partition_D(sB);

    // ④ multicast mask(只有当 GmemTiledCopy = SM90_TMA_LOAD_MULTICAST 时构造)
    uint16_t mcast_mask_a = 0;
    uint16_t mcast_mask_b = 0;
    if constexpr (cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD_MULTICAST>) {
      auto block_layout = Layout<typename DispatchPolicy::ClusterShape>{};
      for (int n = 0; n < size<1>(block_layout); ++n) {
        mcast_mask_a |= (uint16_t(1) << block_layout(cluster_local_block_id.x, n, Int<0>{}));
      }
    }
    // (B 同理)

    // ⑤ K-loop:每 tile 一次
    CUTLASS_PRAGMA_NO_UNROLL
    for ( ; k_tile_count > 0; --k_tile_count) {
      pipeline.producer_acquire(smem_pipe_write);                     // 等 stage 空
      auto* tma_barrier = pipeline.producer_get_barrier(smem_pipe_write);  // 拿 mbarrier
      int write_stage = smem_pipe_write.index();

      // ⑥ 把 barrier + mcast_mask 绑到 TMA atom,然后发 TMA
      copy(mainloop_params.tma_load_a.with(*tma_barrier, mcast_mask_a),
           tAgA(_, _, _, *k_tile_iter), tAsA(_, _, _, write_stage));
      copy(mainloop_params.tma_load_b.with(*tma_barrier, mcast_mask_b),
           tBgB(_, _, _, *k_tile_iter), tBsB(_, _, _, write_stage));
      ++k_tile_iter;

      ++smem_pipe_write;     // 推到下一 stage(parity 自动翻转)
    }
  }
}
```

#### 5.2.1 ① smem Tensor 构造:`make_smem_ptr + make_tensor`

```cpp
Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.data()), SmemLayoutA{});
```

- `make_smem_ptr(ptr)` 包一个普通指针成「我知道我在 smem」的 smart pointer——后续按 `SmemLayoutA` 算 swizzle / bank conflict 时,这个 tag 会被识别
- `make_tensor(ptr, layout)` 把指针 + layout 合成 `cute::Tensor`(Ch2.2 讲过)
- **`SmemLayoutA` 来自 builder 推导**(Ch9 §9.2 步骤 4),形状 `(BLK_M, BLK_K, Stages)`,stride 已经把 swizzle 编进去了

`SharedStorage` 是 `union { MainloopTensorStorage; EpilogueTensorStorage; }`(Ch4 §4.6 详讲),producer / consumer 共享同一块 smem,**靠 union 复用**——mainloop 阶段只用到 `TensorStorage`,epilogue 阶段才能用 epilogue 的 smem layout。

#### 5.2.2 ② `get_slice(cluster_local_block_id)` — cluster 内 CTA 分摊

```cpp
uint2 cluster_local_block_id = {block_rank_in_cluster % cluster_shape_x,
                                block_rank_in_cluster / cluster_shape_x};
auto block_tma_a = mainloop_params.tma_load_a.get_slice(cluster_local_block_id.y);
auto block_tma_b = mainloop_params.tma_load_b.get_slice(cluster_local_block_id.x);
```

`block_rank_in_cluster` 是当前 CTA 在 cluster 内的 ID(0 ~ cluster_size - 1)。Hopper cluster 是 2D(M×N 方向),所以拆成 `cluster_local_block_id.x` 和 `.y`。

`get_slice` 把 TMA atom 按 cluster 内的 CTA 切一份:**每个 CTA 看到 TMA desc 的不同切片**。A 沿 N 方向分(`cluster_local_block_id.y`),B 沿 M 方向分(`.x`),这样 cluster 内 8 个 CTA(典型 `<_4,_2,_1>`)能把同一个 `(BLK_M, BLK_N)` tile 拆 8 份加载,**总带宽 × 8**。

> 这是 Ch4 §4.5 cluster 协作的物理实现:TMA 本身不会自动 multicast,得 builder / kernel 显式算「我这个 CTA 写哪一片」,再用 `get_slice` 取出来。

#### 5.2.3 ③ `partition_S / partition_D` — gmem / smem 摊分到 TMA thread

TMA 的发起方其实是 **single thread**(warp 0 lane 0),但它操作的是一个 thread group 的 smem 区域。`partition_S(gA)` 和 `partition_D(sA)` 返回的 tensor 形状都是 `(TMA, TMA_M, TMA_K, K)`(gmem 侧)和 `(TMA, TMA_M, TMA_K, PIPE)`(smem 侧)——**前 4 维是 TMA 内部的「thread/mode 摊分」**,最后一维是 K / PIPE 维度。

`partition_S` 算「gmem 上每个 TMA 子请求要访问哪一段」,`partition_D` 算「smem 上每个 TMA 子请求写到哪个槽位」。两者**形状对得上**——`copy(...)` 才能把它们一一对应发起。

#### 5.2.4 ④ multicast mask —— 只对 `SM90_TMA_LOAD_MULTICAST` 构造

`if constexpr (cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD_MULTICAST>)` 编译期分支。如果 builder 没选 multicast(`SM90_TMA_LOAD`),mask 留 0,TMA 不会广播。multicast mask 是个位图——告诉 TMA「这次加载完成后,**还要**给 cluster里其他哪些 CTA 也喂同样数据」,让他们能直接用,不用各自重新加载。

#### 5.2.5 ⑤ K-loop 内部:6 步接力

每 K-tile 都做这 6 步:

| 步 | 调用 | 做什么 |
|---|---|---|
| 1 | `pipeline.producer_acquire(smem_pipe_write)` | 等当前 stage 被 consumer `release`(barrier 阻塞)|
| 2 | `pipeline.producer_get_barrier(smem_pipe_write)` | 拿当前 stage 的 mbarrier 指针,绑定给 TMA |
| 3 | `copy(tma_load_a.with(*tma_barrier, mcast_mask), tAgA, tAsA)` | 把 barrier + mcast_mask 绑到 TMA atom,**发 TMA**|
| 4 | `copy(... same for B ...)` | 同上 for B |
| 5 | `++k_tile_iter` | 推到下一个 K 切片 |
| 6 | `++smem_pipe_write` | 推到下一 stage(parity 自动翻转)|

#### 5.2.6 关键:`producer_commit` 在 `load` 里**不存在**

注意上面 6 步**没有 `producer_commit`**——因为 TMA 自身的完成事件会自动 arrive barrier(`PipelineTmaAsync` 的 `producer_commit` 在 TMA 路径是 NoOp,见 §5.5 4-方法语义框)。如果手写 GEMM 移植过 cp.async 路径,会习惯性地在 `load` 末尾加 `producer_commit`,**Hopper TMA 路径上这一行冗余**,可省。

#### 5.2.7 `CUTLASS_PRAGMA_NO_UNROLL` —— 不能 unroll 的关键

K-loop 的循环次数 `k_tile_count` 是**运行时**(K / Stages_t大小),不是编译期常量。如果编译器把循环 unroll,会按 K=2048/BlockK=32 这种「典型值」展开,**但实际 GEMM 可能 K=2000**(只跑 62 次),展开的代码就有 stale state。`NO_UNROLL` 强制按 runtime 推进,处理 K 任意大小。

### 5.3 `mma`:consumer 的 CuTe流程

源码对应段:**`mma`** 函数体,**全部 consumer warpgroup 线程都执行**(不只是单线程)。这是 Ch4 最长、最重要的函数。

```cpp
template <class FrgTensorC>
CUTLASS_DEVICE void
mma(MainloopPipeline pipeline,
    PipelineState smem_pipe_read,
    FrgTensorC& accum,
    int k_tile_count,
    int thread_idx,
    TensorStorage& shared_tensors,
    Params const& mainloop_params) {

  // ① smem Tensor 构造
  Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.data()), SmemLayoutA{});   // (BLK_M, BLK_K, PIPE)
  Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.data()), SmemLayoutB{});   // (BLK_N, BLK_K, PIPE)

  // ② TiledMma 按 warpgroup 切
  constexpr int MmaWarpGroups = size(TiledMma) / NumThreadsPerWarpGroup;
  Layout warp_group_thread_layout = make_layout(Int<MmaWarpGroups>{}, Int<NumThreadsPerWarpGroup>{});
  int warp_group_idx = __shfl_sync(0xFFFFFFFF, thread_idx / NumThreadsPerWarpGroup, 0);
  TiledMma tiled_mma;
  auto thread_mma = tiled_mma.get_slice(warp_group_thread_layout(warp_group_idx));

  // ③ smem 按 thread fragment 摊分
  Tensor tCsA = thread_mma.partition_A(sA);                              // (MMA, MMA_M, MMA_K, PIPE)
  Tensor tCsB = thread_mma.partition_B(sB);                              // (MMA, MMA_N, MMA_K, PIPE)

  // ④ 给 rmem 分配 fragment
  Tensor tCrA = thread_mma.make_fragment_A(tCsA);                        // (MMA, MMA_M, MMA_K, PIPE)
  Tensor tCrB = thread_mma.make_fragment_B(tCsB);                        // (MMA, MMA_N, MMA_K, PIPE)

  static_assert(... /* shape 校验, 见 §5.3 */);

  // ⑤ prologue:第一次 mma,accumulate_ = Zero(从 0 开始)
  PipelineState smem_pipe_release = smem_pipe_read;
  int prologue_mma_count = min(K_PIPE_MMAS, k_tile_count);   // = min(1, k_tile_count), 通常 1
  assert(k_tile_count >= 1);
  tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;
  warpgroup_fence_operand(accum);
  {
    auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
    pipeline.consumer_wait(smem_pipe_read, barrier_token);
    int read_stage = smem_pipe_read.index();
    warpgroup_arrive();
    tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;
    CUTLASS_PRAGMA_UNROLL
    for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
      cute::gemm(tiled_mma, tCrA(_,_,k_block,read_stage), tCrB(_,_,k_block,read_stage), accum);
      tiled_mma.accumulate_ = GMMA::ScaleOut::One;
    }
    warpgroup_commit_batch();
    ++smem_pipe_read;
  }
  tiled_mma.accumulate_ = GMMA::ScaleOut::One;

  // ⑥ prologue 剩余 tile (k_tile_prologue = prologue_mma_count - 1, 通常 0 次)
  warpgroup_fence_operand(accum);
  CUTLASS_PRAGMA_UNROLL
  for (int k_tile_prologue = prologue_mma_count - 1; k_tile_prologue > 0; --k_tile_prologue) {
    auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
    pipeline.consumer_wait(smem_pipe_read, barrier_token);
    int read_stage = smem_pipe_read.index();
    warpgroup_arrive();
    cute::gemm(tiled_mma, tCrA(_,_,_,read_stage), tCrB(_,_,_,read_stage), accum);
    warpgroup_commit_batch();
    ++smem_pipe_read;
  }

  // ⑦ mainloop:每次切到一个新 stage,做 mma + consumer_release
  warpgroup_fence_operand(accum);
  k_tile_count -= prologue_mma_count;
  CUTLASS_PRAGMA_NO_UNROLL
  for ( ; k_tile_count > 0; --k_tile_count) {
    auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
    pipeline.consumer_wait(smem_pipe_read, barrier_token);
    int read_stage = smem_pipe_read.index();
    warpgroup_fence_operand(accum);
    warpgroup_arrive();
    cute::gemm(tiled_mma, tCrA(_,_,_,read_stage), tCrB(_,_,_,read_stage), accum);
    warpgroup_commit_batch();

    // 等 K_PIPE_MMAS 个 mma 提交完,确保这 stage 算完能安全 release
    warpgroup_wait<K_PIPE_MMAS>();
    warpgroup_fence_operand(accum);

    pipeline.consumer_release(smem_pipe_release);    // 让 producer 重新写这 stage
    ++smem_pipe_read;
    ++smem_pipe_release;
  }
  warpgroup_fence_operand(accum);
}
```

#### 5.3.1 ① smem Tensor 构造(同 load)

和 `load` 那一步一样,只是这里 consumer **所有** warpgroup 线程都构造自己的 smem view——每个 thread 用同一个 `SmemLayoutA` 但通过不同 `partition` 拿不同 smem 子区域。

#### 5.3.2 ② TiledMma 按 warpgroup 切

```cpp
Layout warp_group_thread_layout = make_layout(Int<MmaWarpGroups>{}, Int<NumThreadsPerWarpGroup>{});
int warp_group_idx = __shfl_sync(0xFFFFFFFF, thread_idx / NumThreadsPerWarpGroup, 0);
auto thread_mma = tiled_mma.get_slice(warp_group_thread_layout(warp_group_idx));
```

`NumThreadsPerWarpGroup = 128`(Hopper 的 warpgroup = 4 个 warp × 32 lane = 128 thread)。`MmaWarpGroups` 是 TiledMma 覆盖多少个 warpgroup——典型 WarpSpec 是 1 个 consumer warpgroup,Cooperative 是 4 个。

`warp_group_idx = thread_idx / 128` 拿到当前 thread 属于哪个 warpgroup。**`__shfl_sync` 不是装饰**——`thread_idx / 128` 在 warp 内不同 lane 值不同,但所有 warpgroup 内的 lane 应该有同一个 `warp_group_idx`(同一个 warpgroup = 同一个 warpgroup idx),所以用 `__shfl_sync(0xFFFFFFFF, value, 0)` 在 warp 内广播 lane 0 的值,保证全 warpgroup 一致。

`get_slice(layout(warp_group_idx))` 返回当前 warpgroup 看到的 TiledMma 视图——**`tiled_mma` 是 CTA 级的,`thread_mma` 是单个 warpgroup 级的**。

#### 5.3.3 ③ `partition_A / partition_B` — smem 按 thread fragment 摊分

```cpp
Tensor tCsA = thread_mma.partition_A(sA);   // (MMA, MMA_M, MMA_K, PIPE)
Tensor tCsB = thread_mma.partition_B(sB);   // (MMA, MMA_N, MMA_K, PIPE)
```

把 `sA = (BLK_M, BLK_K, PIPE)` 摊到当前 warpgroup 的 thread fragment。返回的 4D tensor:

- `MMA`:该 thread 拥有的 mma instructions(WGMMA 一条指令对应一组 fragment)
- `MMA_M`:该 thread 在 M 维上覆盖的 fragment(配合 WGMMA m64n...k16 的 64 × 16)
- `MMA_K`:K 维上的 sub-block 数(典型 = `BlockK / 16`,因为 WGMMA 一次 16)
- `PIPE`:pipeline stage 数

这一步把「CTA 级 smem layout」降到「单个 thread 的 fragment」,**所有 fragment 加起来正好覆盖整个 (BLK_M, BLK_K, PIPE) tensor**。

#### 5.3.4 ④ `make_fragment_A / B` — rmem 分配

```cpp
Tensor tCrA = thread_mma.make_fragment_A(tCsA);   // 形状同 tCsA
Tensor tCrB = thread_mma.make_fragment_B(tCsB);
```

`make_fragment_A` 内部其实是 `make_tensor_like`——**按 `tCsA` 的 layout 分配 rmem**。返回的 tensor 形状和 `tCsA` 一致,但 storage 在 rmem。

**关键区别**: `tCsA` 是 smem view(`make_smem_ptr` 标了 tag),`tCrA` 是 rmem view。`tCsA` 用于「输入位置」描述,`tCrA` 用于「WGMMA 指令操作的寄存器」描述。

#### 5.3.5 shape 校验的 5 个 `CUTE_STATIC_ASSERT_V`

对应的 5 个 static_assert 块:

```cpp
CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(accum));    // M 维度匹配 accum
CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<2>(accum));    // N 维度匹配 accum
CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCsB));     // A 和 B K 维相等
CUTE_STATIC_ASSERT_V(size<3>(tCsA) == size<3>(tCsB));     // A 和 B PIPE 相等
CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sA));   // smem PIPE 等于 Stages
```

这些**编译期静态断言**——任何 mismatch 直接拒编译。这就是 Ch0「默认正确」的硬保证:不是「编译通过就基本对」,而是「编译期把不对的可能性全排掉」。

#### 5.3.6 ⑤ prologue:第一次 mma `accumulate_ = Zero`

```cpp
tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;     // ← 第一次: 从 0 开始
cute::gemm(tiled_mma, tCrA(_,_,k_block,read_stage), tCrB(_,_,k_block,read_stage), accum);
tiled_mma.accumulate_ = GMMA::ScaleOut::One;      // ← 后续: 累加
```

WGMMA 指令有个 scale-out 控制位 `accumulate?`(PTX 层 `A` bit):0 = `D = A*B + 0`(从 0 写),1 = `D = A*B + C`(累加到 C)。**第一次 mma 必须 `Zero`**——`accum` 没初始化过,累加垃圾。**之后的所有 mma 都 `One`**——把 `(M,N)` sub-tile 累加到同一个 accum 上。

**注意**: `tiled_mma.accumulate_` 是 TiledMma 的运行时成员,**`tiled_mma` 是按值赋值的**(`TiledMma tiled_mma;` 这一行),所以改 `accumulate_` 不会影响其它 thread / warpgroup / 之前创建的 thread_mma view。

`CUTLASS_PRAGMA_UNROLL for k_block`: 内层 K 维 sub-block 循环**强制 unroll**(因为 `size<2>(tCrA) = BlockK / 16` 是编译期常量)。每个 k_block 对应一次 `cute::gemm` → 一条 `wgmma.mma_async.sync.aligned.m64n...k16...`。

#### 5.3.7 ⑥ prologue 剩余 tile + ⑦ mainloop:`warpgroup_wait` 是关键

mainloop 里 K-loop 体的关键 3 步:

```cpp
cute::gemm(...);                 // 发 WGMMA(异步,提交后 GPU 内部流水)
warpgroup_commit_batch();        // 标记「提交了一个 mma batch」

warpgroup_wait<K_PIPE_MMAS>();   // ← 等 K_PIPE_MMAS = 1 个 mma 完成
                                  //   K_PIPE_MMAS 是「允许 in-flight 的 mma 数」

pipeline.consumer_release(smem_pipe_release);   // 等 mma 完成才能 release!
```

**为什么要 `warpgroup_wait<K_PIPE_MMAS>()`?**

WGMMA 是**异步**——`cute::gemm` 调完不代表算完,GPU 还在内部流水。如果不 wait 就 `consumer_release` smem,producer 立刻往同一 stage 写新数据,**会冲掉正在读的 WGMMA 输入**。

`K_PIPE_MMAS = 1`: 一次只允许 1 个 mma in-flight,所以 consumer 一次 release 一个 stage 就够。**如果有更多 stage,可以把 K_PIPE_MMAS 调大,smem pipeline 更深,但 smem 占用也更大**。

`warpgroup_fence_operand(accum)`: mma 提交前后各加一次,**保证 `accum` 寄存器对 warpgroup fence 的可见性**——这是 Hopper warpgroup 指令的特化 fence,跟 `__threadfence_block()` 不是一回事。

#### 5.3.8 `cute::gemm(tiled_mma, tCrA, tCrB, accum)` —— 这一行做了什么

Ch2.5 讲过 5-case dispatch。这里具体说:

```cpp
cute::gemm(tiled_mma, tCrA(_,_,k_block,read_stage), tCrB(_,_,k_block,read_stage), accum)
```

- 第 1 参数 `tiled_mma` = `TiledMma`(Ch2.4 讲的 atom + AtomLayout 的组合)
- 第 2/3 参数是 4D fragment 的「K 维 sub-block 投影 + smem stage 投影」——`tCrA(_,_,k_block,read_stage)` 取出当前 stage 的当前 k_block 子片
- 第 4 参数 `accum` 是 rmem 累加器

`cute::gemm` 走 Ch2.5 的 5-case dispatch:**根据 `tiled_mma.accumulate_` 决定 scale-out(Zero/One);根据 atom 类型(m64n16k16)推到 PTX 字符串;根据 thread layout 决定哪些 lane 实际参与本次 mma**。最终生成的指令:

```text
wgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16.f32
  {%fa, %fa+1, ..., %fa+7},   ← 32 lane 各自的 A fragment(rmem)
  {%fb, ...},                  ← B fragment
  {%fc, ...}                   ← accum(rmem)
```

**「实例化」在这一步**——`cute::gemm` 不是普通函数调用,是模板元编程,根据 `tiled_mma` 类型生成对应 PTX。**改 `tiled_mma` 类型 = 改指令**,不需要 if-else。

#### 5.3.9 `accum` 的生命周期

`accum` 是 kernel 层(Ch6)传进来的 rmem tensor,大小 `(M, N) = (BLK_M, BLK_N)`(因为每个 consumer warpgroup 算整个 tile 的 M、N 维,K 维在 K-loop 里累加)。**调用前是 garbage**——所以 prologue 必须 `Zero` scale-out,第一次 mma 才不会累加垃圾。

> **你手写 GEMM 的对照**:你写一个 `float acc[BLOCK_M][BLOCK_N] = {0}` 或 `__shared__` accumulator。这里 `accum` 是 rmem 视图,prologue `Zero` 起到「等价清零」的作用。

### 5.4 `load_tail` + `mma_tail`:K-loop 收尾

源码对应段:**`load_tail`** + **`mma_tail`** 两个函数体。

#### 5.4.1 `load_tail` —— 防 cluster 早退

```cpp
CUTLASS_DEVICE void
load_tail(MainloopPipeline pipeline, PipelineState smem_pipe_write) {
  int lane_predicate = cute::elect_one_sync();
  if (lane_predicate) {
    pipeline.producer_tail(smem_pipe_write);
  }
}
```

producer K-loop 退出后调一次。`producer_tail` 等所有 stage 被 consumer `release`(或初始 phase 让 acquire 直接成功)。

**为什么必要**: cluster 里 producer 快的 CTA 可能比 consumer 慢的 CTA 早退。如果早退时 producer 直接 `cudaDeviceSynchronize` 退出,L2 缓存被踢,**慢的 CTA 重新读 gmem 时命中不到 L2**,带宽骤降。`producer_tail` 把 producer 阻塞到「cluster 里所有 CTA 的 K-loop 都退完」,避免早退踢 L2。

#### 5.4.2 `mma_tail` —— 等所有 WGMMA retire

```cpp
CUTLASS_DEVICE void
mma_tail(MainloopPipeline pipeline, PipelineState smem_pipe_release, int k_tile_count) {
  int prologue_mma_count = min(K_PIPE_MMAS, k_tile_count);
  k_tile_count -= prologue_mma_count;
  smem_pipe_release.advance(k_tile_count);
  warpgroup_wait<0>();      // ← 关键:等所有 in-flight mma 完成
  for (int count = 0; count < prologue_mma_count; ++count) {
    pipeline.consumer_release(smem_pipe_release);
    ++smem_pipe_release;
  }
}
```

`warpgroup_wait<0>()` —— 等**所有** outstanding mma batch 完成。K-loop 里 `warpgroup_wait<K_PIPE_MMAS>()` 只等 1 个;**`mma_tail` 里 wait<0> 是等全部**。**只有 `accum` 全部 retire,epilogue 才能安全读它**(Ch6 epilogue 第一节)。

`smem_pipe_release.advance(k_tile_count)` —— 把 `smem_pipe_release` state 推到对应 stage。K-loop 已经 `++smem_pipe_release` 推进过 mainloop 部分,这里再 `advance(k_tile_count)` 把 prologue 剩余 + mainloop 剩余的 stage 全部 cover 到。

### 5.5 Pipeline 4 方法语义:必须先校准的 4 个词

`PipelineTmaAsync<N>` 的 4 个方法每个有明确的「阻塞 / 非阻塞 / 可能在某些条件下退化」语义,写 mainloop 时不能搞混:

| 方法 | 谁调用 | 阻塞? | 作用 |
|---|---|---|---|
| `producer_acquire(state)` | producer(TMA 加载 warp) | **阻塞** | 等到 `state` 对应的 stage 被 consumer `release` 后才返回——也就是确认这个 smem 槽位空出来了,可以写入。 |
| `producer_commit(state)` | producer | **非阻塞**(但在 TMA 场景下可能被吞掉,见下) | 把当前 stage 标记成「producer 已完成」,通知 consumer。 |
| `consumer_wait(state)` | consumer(WGMMA warp) | **阻塞** | 等到 `state` 对应的 stage 被 producer `commit` 后才返回——也就是确认 smem 槽位里已有可读数据,可以发起 mma。 |
| `consumer_release(state)` | consumer | **非阻塞** | 把当前 stage 标记成「consumer 已用完」,让 producer 可以重新写这个槽位。 |

两个「会被吞掉」的关键细节,手写时容易踩:

1. **`producer_commit` 在 TMA 场景下是 NoOp**——因为 TMA 自身完成时会自动 arrive barrier 并 arrive bytes,所以你再调一次 `producer_commit` 是冗余的(但无害)。`PipelineTmaAsync` 故意保留了 API 形式一致;真正的 producer-commit 语义由 TMA 完成事件承担。
2. **`make_producer_start_state<MainloopPipeline>()` 不是装饰**——pipeline 在起点「空」的时候,producer 的第一次 `acquire` 会直接成功;但你必须显式构造「让首轮 acquire 成功」的 state,而不是用默认构造的 state。这是 `smem_pipe_write = make_producer_start_state<MainloopPipeline>()` 在 Ch7 主循环出现的原因。

> 一句话总结:**acquire / wait 是「等我需要的东西就绪」,commit / release 是「通知对方我这边就绪/用完」。** 两个阻塞方法(acquire、wait)是真正让线程「停下来等」的同步点;两个非阻塞方法只是更新 barrier 状态、不卡线程。

### 5.6 章末:读完这一章你该做得到的事

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
- ✅ 知道本章重点不是把 CollectiveMma 整段代码抄下来——重点是 5 个函数每个在 mainloop 轨迹里不同位置上的"做的事"。**Ch8 (DispatchPolicy) 讲 tag 怎么路由到这些变体,Ch9 (Builder) 讲策略怎么选**。

---