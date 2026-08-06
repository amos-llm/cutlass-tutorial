## 附录 C:你手写 GEMM 的 X 行 ↔ CUTLASS 哪里(对照表)

这一章是"用户视角的快速跳转表"——给你在写 CUTLASS 3.x GEMM 时,看到自己手写 GEMM 的某一段,就跳到 CUTLASS 的对应位置。

|你手写 Hopper GEMM 中的功能 / 代码|CUTLASS 3.x 中的对应位置|备注|
|---|---|---|
|`-arch=sm_90a`|`using ArchTag = cutlass::arch::Sm90;`|Ch2.2|
|选 WGMMA(不是 cp.async.mma)|`using OperatorClass = cutlass::arch::OpClassTensorOp;`|Ch2.2|
|`dim3 block_dim(...)`|`using TileShape = Shape<_M, _N, _K>;`|Ch2.2|
|`dim3 cluster_dim(...)`|`using ClusterShape = Shape<_cluster_M, _cluster_N, _cluster_K>;`|Ch2.2|
|`cudaLaunchKernel(...)`|`Gemm gemm; gemm.can_implement(...); gemm.initialize(...); gemm.run();`|Ch2.5|
|`if (!can_handle_residue(m, n, k)) return;`|`gemm.can_implement(arguments)`|Ch2.5|
|`__shared__ float smem_A[TileM][TileK];` + swizzle|`SmemLayoutAtomA = composition(Swizzle<B,M,S>, Layout<Shape<_TileM, _TileK>, Stride<_TileK, _1>>{})`|Ch11|
|`prepare_gmem_desc(...)` 写 TMA desc|`collective::CollectiveBuilder` 内部 + `prefetch_tma_descriptors(...)`|Ch2.5 + Ch5.4|
|`cuTensorMapEncodeTiled(...)`|`cute::TmaDescriptor` 包装|附录 A|
|TMA load `cp.async.bulk.tensor.*`|`cute::copy(SM90_TMA_LOAD, ...)`|Ch5.3|
|`wgmma.mma_async.sync.aligned.m64n...k16...`(单条)|`cute::gemm(TiledMma, A_frag, B_frag, acc)` (内部 dispatch 到该指令)|Ch3.5|
|`is_producer = warpIdx < 4` 分支|`WarpGroupRole { Producer, Consumer }` + `kernel/.../operator()`|Ch8.3|
|producer/consumer 屏障数组(`bar[N]`)|`cutlass::PipelineTmaAsync<N>`|Ch5.2 + Ch8.4|
|`__syncthreads()` 同步 producer/consumer|`PipelineTmaAsync::producer_acquire / producer_commit / consumer_wait / consumer_release`|Ch5.3|
|cluster 同步 `cluster_arrive + cluster_wait`|`cute::cluster_arrive + cute::cluster_wait`|Ch8.3|
|`grid = ceil_div(M, BlockM) * ceil_div(N, BlockN);`|`PersistentTileSchedulerSm90Params::num_blocks_in_grid`|Ch8.5|
|`blockIdx.x` 决定本 CTA 处理哪个 tile|`PersistentTileSchedulerSm90::fetch_next_work(...)`|Ch8.5|
|swizzle 步进(swizzle_pattern[blockIdx.x])|`max_swizzle_size = ...` 在 `arguments.scheduler` 中|Ch2.4 + Ch8.5|
|tiled row-major store D|`CollectiveEpilogue::operator()` 内部 TMA store|Ch6.2|
|加 bias bias[b], epilogue 时 `D = alpha * acc + beta * C + bias[m] + bias[n]`|EVT 树 = `Sm90EVT<Sm90Compute<...>, AccFetch, SrcFetch,...>`|Ch6.4|
|加 ReLU|`homogeneous_unary<ReLU>` 在 EVT 节点中|Ch6.4|
|加 silu|自定义 unary functor(例如 §5.4 中的 `IdentitySilu`)在 `Sm90Compute<...>` 节点中|Ch6.4|
|输出 swizzle(`D_swizzled[i,j] = D[i^swizz_b, j^swizz_m]`)|`examples/50_hopper_gemm_with_epilogue_swizzle/`||
|Stream-K(把 K 维继续切,partial sum)|`TileSchedulerType = StreamKScheduler`|Ch8.7|
|Grouped GEMM(多 problem 一次 launch)|`examples/57_hopper_grouped_gemm/`|附录 D|
|2:4 structured sparse GEMM|`examples/62_hopper_sparse_gemm/`|附录 D|
|im2col / col2im Conv|`examples/cute/tutorial/sgemm_*` + `media/docs/cpp/implicit_gemm_convolution.md`|附录 D|
|Pull weight prefetch (overlap weight load 与 compute)|`examples/63_hopper_gemm_with_weight_prefetch/`|附录 D|

**怎么用**:从你手写 GEMM 的视角查表(而不是"我从 CUTLASS 视角要改什么"反着查)。这是反向索引。

---

