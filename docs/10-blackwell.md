## 第 10 章:Blackwell 桥接——同样的 5 层架构,换了一组原子

最后这一章**反向**证实 Ch1-9 的 5 层抽象为什么值钱:同一个 5 层框架在 Blackwell 上几乎是逐行对应,但底层的 mma / smem 单元变了。

**核心 takeaway**:你在 Hopper 路径上学到的 5 层(CUTLASS 3.x 5-tier)架构,**无需重新学**就能进入 Blackwell。你只需重新学**第 3 层(CollectiveMainloop 的 MMA 部分)+ 几处 epilogue 接口面(TMEM)**——详细见 **Ch11 Blackwell 物理基础**(UMMA 指令族 + TMEM 累加器 + CLC)。

### 本章涉及 CUTLASS 源文件

- `examples/70_blackwell_gemm/70_blackwell_*.cu` — Sm100 入口 example(对照 examples/48)
- `include/cutlass/gemm/dispatch_policy.hpp:461/471/714/715` — Sm100 dispatch tag(`KernelTmaWarpSpecializedSm100` 等)
- `include/cutlass/gemm/collective/builders/sm1xx_common.inl:95/117` — `tag_to_umma_major_A/B<GmemLayoutATag>()`
- `include/cutlass/gemm/collective/builders/sm100_*_umma_builder.inl` — Sm100 builder 系列(8+ partial spec)
- `include/cutlass/arch/mma_sm100.h` + `include/cute/atom/mma_traits_sm100.hpp` + `mma_sm100_frag.hpp` — UMMA atom + traits
- `include/cutlass/arch/mma_sm100_umma.hpp` — `SM100_MMA_*` 原子族
- `include/cutlass/arch/mma_sm100_desc.hpp:62` — `UMMA::Major` enum(K / MN)
- `include/cutlass/float8.h:411/1081/1161` — `float_e4m3_t` / `float_ue4m3_t` / `float_ue8m0_t`
- `include/cutlass/float_subbyte.h:79/160/481/494` — `float_e2m1_t` / `float_e2m3_t` / `mx_float6_t` / `mx_float4_t`
- `include/cute/atom/mma_traits_sm100_frag.hpp` — Sm100 fragment 形状

详细硬件原语(TMEM / CLC)见 **Ch11**。

### 10.0 本章导航

```text
§10.1  什么没变:5 层框架完全保持
§10.2  文件对应的迁移(从 sm90_*.hpp 到 sm100_*.hpp)
§10.3  新类型宇宙:block-scaled(mx_*)与窄精度
§10.4  examples/70_blackwell_gemm/ 走查(对应 examples/48)
§10.5  起步路径(5 个入口文件)
§10.6  章末:从 Hopper 视角看 Blackwell 的 5 件不变 + 5 件变
§10.7  章末:读完这一章你该做得到的事
```

读法建议:

- **如果你是 Hopper 老手**:§10.1 确认 5 层不变 → §10.2 看清文件路径 → 跳过 §10.3 / §10.4 / §10.5,直接看 §10.6 那张 5 件不变 + 5 件变对照表。需要深入物理细节再翻 Ch11。
- **如果你是新接触 Blackwell**:§10.1 → §10.4 走查 examples/70 → §10.3 类型表 → §10.5 备查清单 → Ch11 看物理基础(TMEM + UMMA + CLC)。

### 10.1 什么没变:5 层框架完全保持

5 个类名 — `GemmUniversalAdapter` / `kernel::GemmUniversal<...>` / `collective::CollectiveMma` / `epilogue::CollectiveEpilogue` / `*TileScheduler` — 在 sm_100 上全部保留。

`examples/70_blackwell_gemm/` 的 .cu 文件跟 `examples/48_hopper_warp_specialized_gemm/` 的 .cu 文件**几乎是逐行对应**——`TileShape` / `ClusterShape` / `StageCountType` / `KernelScheduleType` 都在同一位置。读 `examples/70` 时,你会觉得"已经在 Ch2 读过一遍"。


### 10.2 文件对应的迁移(从 sm90_*.hpp 到 sm100_*.hpp)

|层|Hopper|Blackwell|
|---|---|---|
|Adapter|`gemm_universal_adapter.h`(完全一致)|同|
|Kernel orchestrator|`kernel/sm90_gemm_tma_warpspecialized.hpp`|`kernel/sm100_gemm_tma_warpspecialized.hpp`|
|Mainloop|`collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp`|`collective/sm100_mma_warpspecialized.hpp`(还有 sm100_mma_cpasync_warpspecialized.hpp 等变体)|
|Epilogue|`epilogue/collective/sm90_epilogue_tma_warpspecialized.hpp`|`epilogue/collective/sm100_epilogue_tma_warpspecialized.hpp`(sm100 有独立 epilogue 文件,以及 `sm100_epilogue_array_tma_warpspecialized.hpp` 等变体)|
|Scheduler|`kernel/sm90_tile_scheduler.hpp`|`kernel/sm100_tile_scheduler.hpp`(多 `cluster_launch_control`)|

#### 一句话总结

> **5 层骨架不动,5 个文件名改了**。从 sm90 到 sm100 的迁移,改 5 个文件路径就够了——你脑子里的 5 层架构不必重学。

### 10.3 新类型宇宙:block-scaled(mx_*)与窄精度

Hopper 时代主要在 FP16/BF16/TF32/FP8。Blackwell sm_100 引入"块缩放":每 32 个元素共享一个 scale factor,把窄精度的精度损失用统计平均扳回来。

涉及类型(`media/docs/cpp/fundamental_types.md` 是完整表;本节只列名称,定义在 cutlass 源码):

- **4-bit**(`cutlass/float_subbyte.h`):
  - `float_e2m1_t`(FP4,最常见的 MXFP4 元素类型,line 79)
  - `float_e3m2_t`(FP6,line 267)
  - `float_e2m3_t`(FP6,line 160)
- **8-bit**(`cutlass/float8.h`):
  - `float_e4m3_t`(FP8 E4M3,line 411)
  - `float_e5m2_t`(FP8 E5M2)
- **scale factor**(`cutlass/float8.h`):
  - `float_ue8m0_t`(无符号 8-bit power-of-2 scale,line 1161;`0` 表示 scale=1,这是为什么通常说"zero-only")
  - `float_ue4m3_t`(无符号 4-bit 偏置 scale,line 1081)
- **复合块缩放包**:
  - `mx_float4_t<float_e2m1_t>`(MXFP4 = 1 个 FP4 + 1 个 `float_ue8m0_t` scale,`float_subbyte.h:494`)
  - `mx_float6_t<float_e2m3_t>`(MXFP6,`float_subbyte.h:481`)
  - `mx_float8_t<float_e4m3_t>`(MXFP8 = 1 个 FP8 + 1 个 scale,`float8.h:1320`)
  - `mx_float8_t<float_e5m2_t>`(MXFP8 e5m2 变体)

> 这一节**只**列出类型名,不教 MXFP 数学。要学 MXFP 看 NVIDIA GPU 上的官方文档 / `cutlass/tools/library/include/cutlass/library_block_scaled.h`(本教程保持"是什么,不为什么")。

#### 跟 builder 的关系

类型扩展影响 `CollectiveBuilder` 的输入参数——`ElementA` / `ElementB` 不再只是 dtype 倍数(`float_e4m3_t` 等),还有"FP4 + scale factor"这种复合包(`mx_float8_t<...>`)。Builder 看到 block-scaled 类型输入会走专门的 partial spec(`sm100_blockscaled_mma_*` 系列,见 `cutlass/gemm/collective/builders/sm100_blockscaled_umma_builder.inl:144-145` 里调 `cutlass::gemm::collective::detail::tag_to_umma_major_A/B<...>()`)。这是为什么 builder 在 sm100 路径上比 sm90 多了 ~10 个 spec 的原因。


### 10.4 examples/70_blackwell_gemm/ 走查(对应 examples/48)

走查 examples/70 跟 examples/48 的对应关系,帮 Hopper 用户"对照着读":

```text
examples/48_hopper_warp_specialized_gemm/      examples/70_blackwell_gemm/
  48_hopper_warp_specialized_gemm.cu            70_blackwell_gemm.cu
       ↓                                            ↓
  // 5 步 host API                              // 5 步 host API(同一份代码)
  using ElementA = ...;                         using ElementA = ...;
  using LayoutA = ...;                          using LayoutA = ...;
  using TileShape = Shape<_128, _128, _32>;      using TileShape = Shape<_128, _128, _32>;  // 形状可以一样
  using ClusterShape = Shape<_2, _1, _1>;       using ClusterShape = Shape<_2, _1, _1>;   // 形状可以一样
                                                using ArchTag = cutlass::arch::Sm100;      // ← 唯一差别
  using CollectiveMainloop = CollectiveBuilder<
      cutlass::arch::Sm90, ...                  // Sm90 → Sm100
  >;
  using CollectiveEpilogue = CollectiveBuilder<
      cutlass::arch::Sm90, ...                  // Sm90 → Sm100
  >;
  using GemmKernel = GemmUniversal<...>;        // 同
  using Gemm = GemmUniversalAdapter<GemmKernel>; // 同
```

**关键观察**:

- 5 行 `using` (ElementA/B/C/D + LayoutA/B/C/D 之类)**完全相同**——`examples/70` 跟 `examples/48` 的差别**只有** `ArchTag` 是 `Sm100` 还是 `Sm90`。
- CollectiveBuilder 看到 `Sm100` 自动路由到 sm100 族的 partial spec——你**不需要**手动选 `KernelTmaWarpSpecializedSm100` 这类 tag(让 `KernelScheduleAuto` 替你选)。
- `examples/70_blackwell_gemm/` 物理文件跟 `examples/48_hopper_warp_specialized_gemm/` 几乎一样长,逐行能对照。

### 10.5 起步路径(5 个入口文件)

读 Blackwell 路径,**5 个入口**:

|入口|文件 / 路径|内容|
|---|---|---|
|**入门 example**|`examples/70_blackwell_gemm/`|与 examples/48 几乎逐行对应|
|**数学 / 指令**|`media/docs/cpp/blackwell_functionality.md`|7 条 tcgen05.mma + 吞吐量对比 Hopper + 类型表|
|**类型**|`media/docs/cpp/fundamental_types.md`|完整 numerical type catalog|
|**Python DSL guide**|`media/docs/pythonDSL/mma_docs/tcgen05_programming.rst`|若用 CuTe Python 路径的等价入手|
|**inline 教程**|`examples/cute/tutorial/blackwell/01_mma_sm100.cu ... 05_mma_tma_epi_sm100.cu`|与 `examples/cute/tutorial/hopper/wgmma_sm90.cu` 镜像结构|

`blackwell_cluster_launch_control.md`(在 `media/docs/cpp/` 下)讲 sm_100 新加的 cluster 同步原语 — cluster 比 sm_90 更大,用新原语管理。详细 CLC 入门见 §10.5。

### 10.6 章末:从 Hopper 视角看 Blackwell 的 5 件不变 + 5 件变

|不变|变|
|---|---|
|5 层框架(`GemmUniversalAdapter` / `GemmUniversal` / `CollectiveMma` / `CollectiveEpilogue` / `*TileScheduler`)类名与方法签名|Tag 树根换:`KernelTmaWarpSpecialized*` → `KernelTmaWarpSpecializedSm100*` / `KernelTmaWarpSpecialized1SmSm100` / `KernelTmaWarpSpecialized2SmSm100`(`include/cutlass/gemm/dispatch_policy.hpp`)|
|Builder(`CollectiveBuilder`)的"用 13 维参数拼实例化"配方|MMA atom 来源:sm90 走 `cute::ss_op_selector<TileShape_MNK, ElementA, ElementB, ElementAccumulator, ...>()`(`include/cute/arch/mma_sm90.hpp:366`,按 dtype + tile 选 GMMA 指令);sm100 走 `cute::UMMA::Major::K/MN` + `cutlass::gemm::collective::detail::tag_to_umma_major_A/B<GmemLayoutATag/B>()`(`include/cutlass/gemm/collective/builders/sm1xx_common.inl:95/117`),builder 内联构造 atom(`include/cute/atom/mma_traits_sm100.hpp`)|
|`Examples/48` 的 4 步 host API 写法(`using` → builder → adapter → run)|对应 `Examples/70_blackwell_gemm/` 同样 4 步(逐行对应 Ch2)|
|Ch6 的 EVT 写法(`Sm90EVT<Sm90Compute<...>, ...>`)|EVT 在 sm100 上由 sm100 epilogue 接管,但 AST 语法完全一致(根节点的算子是 `Sm100*Compute` 而不是 `Sm90*Compute`)|
|`PersistentTileScheduler` 选 tile|sm100 新加 cluster 同步原语 `cluster_launch_control`(见 Ch11 §11.3)|

另外两个**整个体系新增**的事:

|新增方向|对哪一层|
|---|---|
|Block-scaled 数值类型(`mx_float*` / `e2m1` / `e4m3` / `ue8m0` 等)|`Builder` 的输入类型空间(见 §10.4)|
|TMEM(只对 mma 累加器友好的专用 memory)|Mainloop 内部使用,通过 `tmem_allocator_sm100.hpp` 接入(**不**外露到 5 层公共接口)|
|CLC(cluster launch control)|Mainloop 内部使用,通过 `PipelineCLCFetchAsync` 接入(**不**外露到 5 层公共接口)|

**收口**:读前 10 章,你已经掌握了"如何照搬 5 层框架到一个新架构";读本章,你已经掌握了"看到 sm_100_*.hpp 时那 5 件变对应到哪些具体文件名"。

附录 D 补一些查表和后续方向。

### 10.7 章末:读完这一章你该做得到的事

- ✅ 看到 `KernelTmaWarpSpecializedSm100*` / `_1SmSm100` / `_2SmSm100` 这类 tag 知道这些是 sm100 dispatcher 的根 tag,跟 Hopper 端的 `KernelTmaWarpSpecialized*` / `*Pingpong` / `*Cooperative` 是**同一族但不同根**。
- ✅ 看 `cute::UMMA::Major` 不再让"工具查表"——而是 builder 内联构造 atom,知道这是 sm100 改"config 一处"为"config 接口"的体现。
- ✅ 区分 sm100 累加器从 smem → **TMEM**(tensor memory,只对 mma 累加器友好的专用 memory),知道这是 sm100 重新设计 ISA 的核心物理基础。
- ✅ 区分 sm100 上 `tmem_allocator_sm100.hpp` 是 **内部**,5 层公共接口不外露——mainloop 走 union shared storage 这件事没变。
- ✅ 知道 `examples/70_blackwell_gemm/` 跟 `examples/48_hopper_warp_specialized_gemm/` 是**逐行对应**——读到 70 应该能"在脑子里并行对比"48 那一遍。
- ✅ 通过 Ch10.6 那张"5 件不变 + 5 件变"对照表,能用 5 个具体文件名(Tag 树根 / MMA atom / 调度器 / 数据类型 / 累加器)说出 sm100 变了什么。
- ✅ 理解 CLC 是 sm100 独有机制,让 `DynamicPersistentScheduler` 存在——静态调度器不需要 CLC,动态调度器必须用 CLC。
- ✅ 知道 `PipelineCLCFetchAsync<Stages, ClusterShape>` 是 sm100 调度器专用 pipeline 包装,在 `sm100_pipeline.hpp` 里。
- ✅ 知道 ch10 末尾的"附录 D"是查表与后续方向(Grouped / Sparse / Conv / SSD / PDL 路线图),而 ch10 之前的 9 章是"必备主干"。
- ✅ 区分 H100 上跑 `arch=sm_90a` fallback,s100 上跑 `arch=sm_100a` 启用新指令;这是 build-system 决定,不是 CUTLASS 显式分支。

---



