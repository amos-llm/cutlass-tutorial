## 附录 A:Hopper 原语 ↔ CUTLASS 封装文件(速查表)

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

