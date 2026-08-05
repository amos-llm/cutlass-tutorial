## 第 8 章:CollectiveBuilder——把"形状 + 类型"压成具体实现

### 8.0 builder 不是一份——它是 33 份的 dispatcher

`include/cutlass/gemm/collective/builders/` 下不是一份文件,而是 **33 份 `.inl`**。读 Ch8 之前先认清这张「地图」,否则会被文件名劝退,也不知道自己该读哪一份。

#### 二维拆分:arch × feature

每一份 builder 对应一个 (arch, feature) 组合。两个轴:

**arch 轴(决定 mma atom 来源)**:

| arch | 文件前缀 | mma atom 来源 | smem 上限 |
|---|---|---|---|
| **sm90**(Hopper)| `sm90_*` | `cute::GMMA::ss_op_selector<...>`(WGMMA,smem-smem) | `detail::sm90_smem_capacity_bytes` |
| **sm100**(Blackwell)| `sm100_*` | `cute::UMMA::Major` + `tag_to_umma_major_A/B<...>`,atom 内联构造(Ch11)| sm100 smem 上限 |
| **sm103**(Blackwell refresh)| `sm103_blockscaled_umma_builder.inl` | 同 sm100,但用 sm103 特有的指令 | 同 sm100 |
| **sm120**(consumer Blackwell)| `sm120_*` | sm120 MMA atom(走 TMA,不是 cluster launch)| sm120 smem 上限 |

**feature 轴(决定数据类型 / 数据通路)**:

| feature | 后缀 | 加了什么 |
|---|---|---|
| **basic** | `_gmma_builder.inl` / `_umma_builder.inl` / `_mma_builder.inl` | 单纯 fp16 / bf16 / tf32 / fp32 GEMM |
| **sparse** | `_sparse_gmma_builder.inl` / `_sparse_umma_builder.inl` / `_sparse_mma_builder.inl` | 2:4 结构化稀疏(atom 是 `ss_op_selector_sparse` 或对应 UMMA sparse)|
| **blockscaled** | `_blockscaled_*_builder.inl` | NVFP4 / MXFP8 这类「每个 block 有自己的 scale」(atom 走 `compute_stage_count_with_blockwise_scale`,smem layout 多了 `SmemLayoutAtomSFA/B` 来放 per-block scale) |
| **blockwise** | `_blockwise_umma_builder.inl` / `_blockwise_mma_builder.inl` | blockwise scaling(sm100 + sm120) |
| **mixed_input** | `_mixed_input_umma_builder.inl` / `_mixed_tma_cpasync_umma_builder.inl` | A 和 B 不同 dtype,或 gmem 路径走 cp.async + TMA 混合 |
| **array** | `_array_*_builder.inl` | grouped GEMM / ptr-array batch(每个 batch 一对指针)|
| **simt** | `_simt_builder.inl` | 非 tensor core,走 CUDA core(`sm100_simt_builder.inl` 是 SM100 上唯一一条非 tensor core 路径)|
| **complex** | `_planar_complex_*` / `_interleaved_complex_*` / `_9xBF16_*` | 复数 GEMM(planar = 实部虚部分开存;interleaved = 同一 buffer 交替存)|

#### 实际清单(33 份)

按 arch 分组的完整清单(`ls include/cutlass/gemm/collective/builders/`):

```
sm90_common.inl                          ← sm90 公共 helper
sm90_gmma_builder.inl                    ← sm90 basic(Ch8 主要读这一份)
sm90_sparse_config.inl                   ← sm90 sparse 公共 config
sm90_sparse_gmma_builder.inl             ← sm90 sparse(2:4)

sm100_common.inl                         ← sm100 公共 helper
sm100_umma_builder.inl                   ← sm100 basic(Ch11 主要读这一份)
sm100_sparse_umma_builder.inl            ← sm100 sparse
sm100_blockscaled_umma_builder.inl       ← sm100 blockscaled(NVFP4 / MXFP8)
sm100_blockscaled_sparse_umma_builder.inl← sm100 sparse + blockscaled
sm100_blockscaled_mixed_tma_cpasync_umma_builder.inl ← sm100 blockscaled + TMA/cp.async 混合
sm100_blockwise_umma_builder.inl         ← sm100 blockwise scaling
sm100_mixed_input_umma_builder.inl       ← sm100 A/B 不同 dtype
sm100_mixed_tma_cpasync_umma_builder.inl ← sm100 TMA/cp.async 混合
sm100_cpasync_umma_builder.inl           ← sm100 纯 cp.async 路径(无 TMA)
sm100_planar_complex_umma_builder.inl    ← sm100 planar complex
sm100_interleaved_complex_umma_builder.inl ← sm100 interleaved complex
sm100_9xBF16_umma_builder.inl            ← sm100 9xBF16 特殊格式
sm100_9xBF16_interleaved_complex_umma_builder.inl ← sm100 9xBF16 + complex
sm100_simt_builder.inl                   ← sm100 CUDA core(非 tensor core)
sm100_pipeline_carveout.inl              ← sm100 pipeline stage 推导

sm103_blockscaled_umma_builder.inl       ← sm103 (Blackwell refresh) blockscaled

sm120_common.inl                         ← sm120 公共 helper
sm120_mma_builder.inl                    ← sm120 basic
sm120_sparse_mma_builder.inl             ← sm120 sparse
sm120_blockscaled_mma_builder.inl        ← sm120 blockscaled
sm120_blockscaled_sparse_mma_builder.inl ← sm120 sparse + blockscaled
sm120_blockwise_mma_builder.inl          ← sm120 blockwise
sm120_array_mma_builder.inl              ← sm120 grouped/ptr-array

sm1xx_common.inl                         ← sm100/sm103 共享 helper
sm1xx_sparse_config.inl                  ← sm100/sm103 sparse config
```

#### 为什么不是一份

「为什么 CUTLASS 不写一个 `CollectiveBuilder` 处理所有情况?」三条理由:

1. **mma atom 来源不一样**:Hopper 走 `cute::GMMA::ss_op_selector<ElementA, ElementB, ElementAccumulator, TileShape_MNK>()`,Blackwell 走 `cute::UMMA::Major` + `tag_to_umma_major_A/B<...>()` 内联构造。**写 if-else 选哪一个会让模板深度爆炸**,partial specialization 各管一份是 C++ 模板元编程最干净的分流。
2. **smem layout 多余字段不一样**:basic 没有 scale tensor,blockscaled 有 `SmemLayoutAtomSFA/B`(per-block scale 的 smem 布局),sparse 多了 E(2:4 元数据)的 smem 布局。**派生 `SmemLayoutA/B` 的代码不一样**,塞同一份 builder 会让 static_assert 拒编条件写得很乱。
3. **compute_stage_count 不一样**:basic 用 `compute_stage_count_or_override`,blockscaled 用 `compute_stage_count_with_blockwise_scale`(每个 stage 字节数里多了 scale tensor),sparse 用 `compute_stage_count_or_override_sparse`。**3 个不同公式**,硬塞一个 builder 就要 `if constexpr` 分流。

所以每一份 `.inl` = 一份「(arch, feature) 的 partial specialization 集合」。CUTLASS 在 `collective_builder.hpp` 顶层把所有 `*.inl` 用 `#include` 串起来,让编译器看到所有 partial spec,按 13 个模板参数做 dispatcher 路由。

#### 读哪一份

按你的项目需要:

| 你在做什么 | 读哪一份 |
|---|---|
| Hopper 普通 fp16/bf16 GEMM | `sm90_gmma_builder.inl`(Ch8 默认走这一份) |
| Hopper 2:4 稀疏 | `sm90_sparse_gmma_builder.inl` |
| Blackwell 普通 bf16/fp16 | `sm100_umma_builder.inl` |
| Blackwell NVFP4 / MXFP8 | `sm100_blockscaled_umma_builder.inl`(`SmemLayoutAtomSFA/B` 在这) |
| Blackwell 2:4 稀疏 | `sm100_sparse_umma_builder.inl` |
| Blackwell A/B 不同 dtype | `sm100_mixed_input_umma_builder.inl` |
| Blackwell grouped GEMM | `sm100_array_umma_builder.inl`(注意:array 在 sm100 上是独立的 builder,**不是** tag 选项) |
| Blackwell CUDA core 路径 | `sm100_simt_builder.inl`(唯一非 tensor core 入口) |
| 复数 GEMM | `sm100_planar_complex_umma_builder.inl` 或 `_interleaved_complex_umma_builder.inl`,按你存法选 |
| SM120(consumer Blackwell)| `sm120_mma_builder.inl`(起步) |
| SM120 grouped | `sm120_array_mma_builder.inl` |

**debug 提示**:遇到「builder 推不出 partial spec」或「static_assert 报 `static_assert failed: ... CollectiveBuilder ...`」,**先翻这份清单**——多半是你要的 (arch, feature) 组合没在对应的 `.inl` 里,而你给的 tag 落到了错误的 spec。

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
  //    static constexpr int PipelineStages =
  //      detail::compute_stage_count_or_override<
  //        Sm90ReducedSmemCapacityBytes,
  //        ElementA, ElementB, TileShape_MNK, /*alignment=*/128
  //      >(StageCountType{});
  //    如果用户给了 StageCount<N> → 直接返回 N(走「override」overload,见 sm90_gmma_builder.inl)。
  //    如果 StageCountAuto → 走 auto 重载:用 smem 容量上限(228KB,详见
  //    `sm90_smem_capacity_bytes`)减去 epi carveout 后再除以每个 stage 字节,得最大 stage 数。

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
// 来源:include/cutlass/gemm/collective/builders/sm90_gmma_builder.inl
// namespace cutlass::gemm::collective::detail
template<int capacity_bytes_, class ElementA, class ElementB, class TileShapeMNK,
         int alignment = 128, int carveout_bytes_>
constexpr int
compute_stage_count_or_override(StageCountAutoCarveout<carveout_bytes_> stage_count) {
  constexpr auto mainloop_pipeline_bytes =
      sizeof(typename cutlass::PipelineTmaAsync<1>::SharedStorage);
  constexpr auto a_bits = cute::sizeof_bits_v<ElementA>;
  constexpr auto b_bits = cute::sizeof_bits_v<ElementB>;
  constexpr int stage_bytes_ =
      cutlass::bits_to_bytes(a_bits * size<0>(TileShapeMNK{}) * size<2>(TileShapeMNK{})) +
      cutlass::bits_to_bytes(b_bits * size<1>(TileShapeMNK{}) * size<2>(TileShapeMNK{}));
  constexpr int stage_bytes = cutlass::round_up(stage_bytes_, alignment)
                            + static_cast<int>(mainloop_pipeline_bytes);
  constexpr int carveout_bytes = cutlass::round_up(carveout_bytes_, alignment);
  constexpr int capacity_bytes = capacity_bytes_ / alignment * alignment;
  return (capacity_bytes - carveout_bytes) / stage_bytes;
}
```

注意几点:

1. **三套 overload 都在 `detail::` 命名空间里**,每个 builder `.inl` 文件(`sm90_gmma_builder.inl`、`sm100_*_umma_builder.inl`、`sm120_*_mma_builder.inl` 等)各有一份。Hopper / Blackwell / SM120 各自有「smem 容量上限」(sm90 是 `Sm90ReducedSmemCapacityBytes`,详细常量在 `cute/arch/sm90*.hpp`)。其它变体还有 `compute_stage_count_or_override_sparse`、`compute_stage_count_or_override_blockwise`、`compute_stage_count_or_override_blockscaled`、`compute_stage_count_or_override_interleaved_complex_tf32`、`compute_stage_count_with_blockwise_scale` 等。
2. `StageCount<N>` 走另一个 overload 直接返回 `N`(就是「override」分支,不计算)。只有 `StageCountAutoCarveout<bytes>` 走上面这个公式。
3. 公式核心:`(smem_cap - carveout) / (per_stage_A_bytes + per_stage_B_bytes + pipeline_barrier_bytes)`。**减去的「per_stage 字节」只算 A+B tensor 体积,不算 epilogue** —— epilogue 是在外层参数 `carveout_bytes_` 里单独扣除的。pipeline barrier 字节(`PipelineTmaAsync<1>::SharedStorage`)按 stage 数线性增长,所以算每个 stage 的「全部成本」。
4. 编译期 `constexpr`,程序运行时根本不知道「auto 决定过」——结果是 `PipelineStages` 这个具体数值,被烧到 dispatch policy 里。

> **debug 提示**: `PipelineStages < 你设的 StageCount<N>` 通常意味着 builder 选择了一个 fallback / 简化版本;`PipelineStages == 0` 几乎肯定是 `carveout_bytes > capacity_bytes`(epi SharedStorage 把整个 smem 吃光了)——静态断言会直接拒编译。

### 8.5 "怎么改默认 schedule / stages / cluster:动手清单"

|想做什么|怎么改|
|---|---|
|强制 stage 数|`StageCount<4>` 替换 `StageCountAutoCarveout<...>`|
|强制 single-pipeline|`KernelScheduleAuto` 替换成 `KernelTmaWarpSpecialized` (明确)|
|强制 pingpong|`KernelTmaWarpSpecializedPingpong`|
|强制 cooperative|`KernelTmaWarpSpecializedCooperative`|
|强制小 tile (256×128 vs 128×128)|`TileShape<_128, _128, _32>` 换成 `_256`, `_128`|
|强制 cluster 形状|`ClusterShape<_1, _2, _1>` 替换 `<_4, _2, _1>`|

实际项目里,"默认是 Pingpong 还是 Cooperative?"由 Builder 内部的经验启发式决定(参考 `media/docs/cpp/heuristics.md`,实际 Python 端在 `python/cutlass_library/heuristics.py`)。

### 8.6 章末:读完这一章你该做得到的事

- ✅ 看清 `include/cutlass/gemm/collective/builders/` 下 33 份 `.inl` 的二维拆分(arch × feature),知道「为什么不是一份」(mma atom 来源、smem layout 多余字段、`compute_stage_count` 公式三者都随 (arch, feature) 变)。
- ✅ 给一个具体配置"(fp16, RowMajor × ColMajor, 128×128×32, ClusterShape<_4,_2,_1>)",能从 33 份里挑出对应那份(`sm90_gmma_builder.inl`),并能**手算** builder 会推什么(AtomLayoutMNK、PipelineStages、SmemLayoutAtomA)。
- ✅ 能在 `sm90_gmma_builder.inl` 里读懂 partial specialization 的结构(挑 `KernelTmaWarpSpecialized` 那一个开始)。
- ✅ 知道 `Auto*` 不是"运行时决定",而是编译期算。

---

