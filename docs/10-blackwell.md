## 第 13 章:Blackwell 桥接——同样的 5 层架构,换了一组原子

最后这一章**反向**证实 Ch1-8 的 5 层抽象为什么值钱:同一个 5 层框架在 Blackwell 上几乎是逐行对应,但底层的 mma / smem 单元变了。

**核心 takeaway**:你在 Hopper 路径上学到的 5 层(CUTLASS 3.x 5-tier)架构,**无需重新学**就能进入 Blackwell。你只需重新学**第 3 层(CollectiveMainloop 的 MMA 部分)+ 几处 epilogue 接口面(TMEM)**。

### 11.1 什么没变:5 层框架完全保持

5 个类名 — `GemmUniversalAdapter` / `kernel::GemmUniversal<...>` / `collective::CollectiveMma` / `epilogue::CollectiveEpilogue` / `*TileScheduler` — 在 sm_100 上全部保留。

`examples/70_blackwell_gemm/` 的 .cu 文件跟 `examples/48_hopper_warp_specialized_gemm/` 的 .cu 文件**几乎是逐行对应**——`TileShape` / `ClusterShape` / `StageCountType` / `KernelScheduleType` 都在同一位置。读 `examples/70` 时,你会觉得"已经在 Ch2 读过一遍"。

### 11.2 什么变了:WGMMA → UMMA,smem 部分结果 → TMEM

#### MMA:`cute::GMMA::ss_op_selector` → `cute::UMMA::Major`(不再有 selector)

- **WGMMA**(sm_90):64xNxK(如 `SM90_64x16x16_F16F16F16F32_SS`);以 `wgmma.mma_async.sync.aligned.m64n...k16...` 的 PTX 指令形式发出,初始累加器 `(0.0)` 由调用方显式传入。
- **UMMA**(sm_100):`tcgen05.mma` 系列;**异步 tensor 协程**——你写 launch 指令,GPU 自己决定何时启动;允许多 CTA 在 cluster 内共享时打平 bandwidth(`tcgen05.mma` 自身是 tensor 指令,不是 scalar 协程)。

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
|Epilogue|`epilogue/collective/sm90_epilogue_tma_warpspecialized.hpp`|`epilogue/collective/sm100_epilogue_tma_warpspecialized.hpp`(sm100 有独立 epilogue 文件,以及 `sm100_epilogue_array_tma_warpspecialized.hpp` 等变体)|
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
|5 层框架(`GemmUniversalAdapter` / `GemmUniversal` / `CollectiveMma` / `CollectiveEpilogue` / `*TileScheduler`)类名与方法签名|Tag 树根换:`KernelTmaWarpSpecialized*` → `KernelTmaWarpSpecializedSm100*` / `KernelTmaWarpSpecialized1SmSm100` / `KernelTmaWarpSpecialized2SmSm100`(`include/cutlass/gemm/dispatch_policy.hpp`)|
|Builder(`CollectiveBuilder`)的"用 13 维参数拼实例化"配方|MMA atom 来源:sm90 走 `cute::GMMA::ss_op_selector<ElementA, ElementB, ElementAccumulator, TileShape_MNK>()`(`include/cute/atom/mma_traits_sm90_gmma.hpp`);sm100 走 `cute::UMMA::Major` + `tag_to_umma_major_A/B<GmemLayoutATag/B>()`,builder 内联构造 atom(`include/cute/atom/mma_traits_sm100.hpp`)|
|`Examples/48` 的 4 步 host API 写法(`using` → builder → adapter → run)|对应 `Examples/70_blackwell_gemm/` 同样 4 步(逐行对应 Ch2)|
|Ch6.4 的 EVT 写法(`Sm90EVT<Sm90Compute<...>, ...>`)|EVT 在 sm100 上由 sm100 epilogue 接管,但 AST 语法完全一致(根节点的算子是 `Sm100*Compute` 而不是 `Sm90*Compute`)|
|`PersistentTileScheduler` 选 tile|sm100 新加 cluster 同步原语 `cluster_launch_control`(见 `media/docs/cpp/blackwell_cluster_launch_control.md`)|

另外两个**整个体系新增**的事:

|新增方向|对哪一层|
|---|---|
|Block-scaled 数值类型(`mx_float*` / `e2m1` / `e4m3` / `ue8m0` 等)|`Builder` 的输入类型空间(见 §11.3)|
|TMEM(只对 mma 累加器友好的专用 memory)|Mainloop 内部使用,通过 `tmem_allocator_sm100.hpp` 接入(**不**外露到 5 层公共接口)|

**收口**:读前 10 章,你已经掌握了"如何照搬 5 层框架到一个新架构";读本章,你已经掌握了"看到 sm_100_*.hpp 时那 5 件变对应到哪些具体文件名"。

附录补一些查表。

---

