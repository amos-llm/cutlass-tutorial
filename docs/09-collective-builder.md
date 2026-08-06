## 第 9 章:CollectiveBuilder——把"形状 + 类型"压成具体实现

### 本章涉及 CUTLASS 源文件

- `include/cutlass/gemm/collective/builders/sm90_gmma_builder.inl:209/327/524/633/772/875/984/1064` — 8 个 `CollectiveBuilder` partial spec
- `include/cutlass/gemm/collective/builders/sm90_gmma_builder.inl:230/264/347/553` — `is_use_rmem_A` + `SmemLayoutAtomA` 推导
- `include/cutlass/gemm/collective/builders/sm90_gmma_builder.inl:998` — `is_same_v<..., KernelScheduleAuto>` 路由
- `include/cutlass/epilogue/collective/builders/sm90_builder.inl` — epilogue builder 镜像(在 Ch6 §6.10 引用)
- `python/cutlass_library/heuristics.py:415` — heuristic 实现(`KernelScheduleAuto` 启发式)

### 9.0 sm90_gmma_builder.inl 里的 8 个 partial spec 怎么分

`sm90_gmma_builder.inl` 这**一份文件**里有 8 个 `struct CollectiveBuilder` 的 partial specialization,**不是一份覆盖所有情况**。本章 §8.2 给你看的「最大那个」只是 8 个里的 1 个,其他 7 个按相同结构存在(只是分流条件不同)。读这章之前先认清 8 个 spec 全表——否则你看 §8.2 会以为 builder 只能 cover 那个场景。

每个 spec 都被一段 `cute::enable_if_t<...>` 在模板参数末尾分流。**13 个模板参数里,所有 spec 都共用前 12 个(arch, op_class, element, layout, alignment × 2, accumulator, tile, cluster, stage_count),唯一的分流维度是第 13 个 —— `KernelScheduleType`**。具体说,3 个开关决定落到哪个 spec:

#### 三个开关

| 开关 | 哪个模板参数决定 | 取值 |
|---|---|---|
| **gmem 路径**(TMA 还是 cp.async)| `KernelScheduleType` 的 tag 前缀 | `KernelTma*` 走 TMA;`KernelCpAsync*` 走 cp.async(Ampere 风格)|
| **warp-specialized**(producer/consumer 分工)| 同上 — tag 后缀 | `WarpSpecialized` / `WarpSpecializedPingpong` / `WarpSpecializedCooperative` = 切了;`KernelTma`(裸)| = 没切;`KernelMultistage` = 旧式多 stage |
| **A 在 smem 还是 register**(SS / RS)| `ElementA` 的 type trait (`detail::is_use_rmem_A<...>()`)| 命中 = A 留 register(混合精度 / ConvertAndScale / DirectConvert / 不同 dtype);不命中 = A 走 smem |

#### 8 个 spec 的全表

`sm90_gmma_builder.inl` 按文件里的 `// 名字` 注释标注:

| # | 文件里名字 | 三个开关组合 | 备注 |
|---|---|---|---|
| 1 | `GMMA_TMA_WS_SS` | **TMA + WS + SS** | **默认路径** — `examples/48` 落这。`KernelTmaWarpSpecialized`(及 Pingpong / Cooperative、PtrArray 变体)+ A 走 smem |
| 2 | `GMMA_TMA_WS_RS` | **TMA + WS + RS** | 同 WS tag 5 个 + `is_use_rmem_A` 命中。A 留 register 跨 mma step(mixed precision,ConvertAndScale,DirectConvert,A/B 不同 dtype)|
| 3 | `GMMA_TMA_WS_FP8_FAST_ACCUM_SS` | **TMA + WS + SS + FP8 fast-accum** | tag 5 个 `_FP8FastAccum` 后缀 + 输入必须 FP8。走硬件 fast-accum(不经过 fp32 中转)|
| 4 | `GMMA_TMA_SS` | **TMA + 非 WS + SS** | 旧式 `KernelTma`(所有人又加载又计算)+ SS。Ampere 时代的「非 warp-specialized」,Hopper 上仍能编译但**没优化收益** |
| 5 | `GMMA_CpAsync` | **cp.async + `KernelMultistage`** | **`[[deprecated]]` 标记** — 内部直接转发到 `KernelCpAsyncWarpSpecialized`,用户写 `KernelMultistage` 会触发 deprecation 警告,新代码不该用 |
| 6 | `GMMA_CpAsync_WS_SS` | **cp.async + WS + SS** | `KernelCpAsyncWarpSpecialized`(及 Pingpong / Cooperative)+ SS。**Ampere 上**`examples/14_ampere_tf32_tensorop_gemm/` 落这;Hopper 上也能用但**不推荐**(TMA 路径收益更大) |
| 7 | `GMMA_CpAsync_WS_RS` | **cp.async + WS + RS** | 同 3 个 cp.async WS tag + `is_use_rmem_A` 命中 |
| 8 | `GMMA auto kernel schedule` | **Auto picker** | `KernelScheduleType == KernelScheduleAuto`。这是**唯一一个「占位 type」spec**——内部按 (arch, dtype, tile) 启发式选 1-7 之一。`examples/48` 写 `KernelScheduleAuto` 落这 |

> ⚠ 一个常见误解:8 号 spec 是**唯一一个**——「`KernelScheduleAuto` 走哪条具体路径」在这里决定,不是在外层 dispatcher。auto 的逻辑写在那个 partial spec 内部。

#### dispatcher 怎么挑

给定 13 个模板参数,编译器按以下顺序匹配:

```text
先看 KernelScheduleType:
  ├─ == KernelScheduleAuto        → spec 8(Auto picker,内部再选 1-7)
  ├─ 含 FP8FastAccum 后缀 + FP8 input → spec 3
  ├─ 含 CpAsync 前缀              → spec 6 或 7(看 is_use_rmem_A)
  ├─ == KernelMultistage          → spec 5(已 deprecated)
  ├─ == KernelTma(裸,无 WS 后缀)  → spec 4
  └─ 含 WarpSpecialized* 后缀     → spec 1 或 2(看 is_use_rmem_A)
```

**`is_use_rmem_A` 的判定**(切到 RS 的 4 个条件):在 `detail::deduce_mixed_width_dtype_t` 里查 `ElementA` 的类型——命中下列任一即走 RS:

1. 命中 `is_use_rmem_A<ElementA, GmemLayoutATag, ElementB, GmemLayoutBTag>()`(Hopper 上的 RS-friendly layout 组合)
2. `cute::is_tuple<ElementA>::value`(ConvertAndScale / ConvertAndScaleWithZero 这类带 scale 的复合 type)
3. `cute::is_tuple<ElementB>::value`(同上,B 侧)
4. `sizeof_bits<ElementA>::value != sizeof_bits<ElementB>::value`(A 和 B 不同 dtype,DirectConvert)

> 一句话:**SS 还是 RS 不由你显式选**,由「A 是 fp16 + B 是 fp8?」「A 是 tuple?」这些 type trait 自动决定——你写 `KernelTmaWarpSpecialized`,builder 看 element types 决定落到 spec 1 还是 2。

#### 读哪个 spec

| 你的 GEMM | 落到 spec | 主要新增字段 |
|---|---|---|
| `examples/48`(fp32 + `KernelScheduleAuto`)| 8 → 1 | — |
| `examples/48` 写 `KernelTmaWarpSpecializedPingpong` | 1 | `IsCooperative = false`, `AtomLayoutMNK = Shape<_1,_1,_1>` |
| fp16 × fp8 混合精度(`mixed_input.h`)| 2 | `ScaleA`, `ScaleB`, `ZeroA`, `ZeroB`(`deduce_mixed_width_dtype_t` 推)|
| fp8 × fp8 + `KernelTmaWarpSpecializedFP8FastAccum` | 3 | `IsArrayOfPointersGemm` + `IsCooperative`,`AtomLayoutMNK` 同 spec 1,`static_assert(detail::is_input_fp8<...>)` |
| 故意用 `KernelTma`(非 WS) | 4 | `DispatchPolicy = MainloopSm90TmaGmma`(而非 `*WarpSpecialized`)|

> **debug 提示**:遇到「`static_assert failed: ... CollectiveBuilder ...`」,**先翻这张表**——多半是你的 tag 落到了意料之外的 spec(例如你以为写的是 WS 但其实 spec 4 接管了,意味着「WS 路径被吞掉了」)。再看 auto picker 那段的注释,理解 auto 选了什么。

### 9.1 它的任务

`CollectiveBuilder<13 个模板参数>` 是一个 dispatcher: 给定 `(arch × op_class × element × layout × alignment × element × layout × alignment × accumulator × tile × cluster × stage × schedule)` 共 13 维空间里的一个组合,**挑出**一个 `CollectiveMma` partial specialization。

它做的事很机械化(SELECT),但靠 partial specialization 路由,完全编译期,没有运行时代价。

### 9.2 实际看一个 partial spec

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

### 9.3 epilogue builder 是镜像

文件:`include/cutlass/epilogue/collective/builders/sm90_builder.inl`。同样结构,主要差异:

- 多接 `(ElementC, LayoutC, AlignmentC, ElementD, LayoutD, AlignmentD, StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore, ...)` 等。Caller 在 Ch2.3 看到。
- 内部推 EVT 的根节点(如果用户给 EVT 就用,否则用 `DefaultEpilogue`)。

### 9.4 "Auto" 实际上到底是什么

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

### 9.5 "怎么改默认 schedule / stages / cluster:动手清单"

|想做什么|怎么改|
|---|---|
|强制 stage 数|`StageCount<4>` 替换 `StageCountAutoCarveout<...>`|
|强制 single-pipeline|`KernelScheduleAuto` 替换成 `KernelTmaWarpSpecialized` (明确)|
|强制 pingpong|`KernelTmaWarpSpecializedPingpong`|
|强制 cooperative|`KernelTmaWarpSpecializedCooperative`|
|强制小 tile (256×128 vs 128×128)|`TileShape<_128, _128, _32>` 换成 `_256`, `_128`|
|强制 cluster 形状|`ClusterShape<_1, _2, _1>` 替换 `<_4, _2, _1>`|

实际项目里,"默认是 Pingpong 还是 Cooperative?"由 Builder 内部的经验启发式决定(参考 `media/docs/cpp/heuristics.md`,实际 Python 端在 `python/cutlass_library/heuristics.py`)。

### 9.6 章末:读完这一章你该做得到的事

- ✅ 认清 `sm90_gmma_builder.inl` 里 8 个 partial specialization 的分流维度(gmem 路径 TMA/cp.async × WS/非 WS × SS/RS × FP8 fast-accum),并能根据 13 维模板参数推出自己的 GEMM 落到哪个 spec。
- ✅ 知道 `is_use_rmem_A` 怎么决定 A 走 smem 还是 register,以及 `KernelScheduleAuto` 走的是 spec 8(internal auto picker)而非具体 spec。
- ✅ 给一个具体配置"(fp16, RowMajor × ColMajor, 128×128×32, ClusterShape<_4,_2,_1>)",你能**手算** builder 会推什么(AtomLayoutMNK、PipelineStages、SmemLayoutAtomA)。
- ✅ 能在 `sm90_gmma_builder.inl` 里读懂 partial specialization 的结构(挑 `KernelTmaWarpSpecialized` 那一个开始)。
- ✅ 知道 `Auto*` 不是"运行时决定",而是编译期算。

### 9.7 全配置空间地图(8 章读完的总结)

到这里 8 章走完一遍。下面这张图把"5 层 + 所有 dispatch tag + 所有 scheduler tag + builder 路由"放在一张总图里,看一眼就知道 CUTLASS 3.x 的配置空间有多大。

```text
┌──────────────────────── CUTLASS 3.x 配置空间 ────────────────────────┐
│                                                                     │
│  5 层固定骨架(永远不变)                                              │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ 1. GemmUniversalAdapter(§1.1)                              │  │
│  │ 2. GemmUniversal<ProblemShape, Mainloop, Epilogue, Sched>(§6.1)│  │
│  │ 3. CollectiveMma(§4)                                       │  │
│  │ 4. CollectiveEpilogue(§5 上半) + EVT(§5 下半)                  │  │
│  │ 5. TileScheduler(§6.5)                                     │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  配置维度(每个维度对应一族 tag)                                       │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ 架构 ArchTag:Sm80 / Sm90 / Sm100 / Sm120                   │  │
│  │ dtype:fp16 / bf16 / tf32 / fp8 / int8 / mx_* / e2m1 / ...   │  │
│  │ 内存路径 TMA vs cp.async                                     │  │
│  │ warp specialization:WS / 非 WS                              │  │
│  │ A 走 smem (SS) vs register (RS)                              │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  Mainloop schedule tags(§7.2 / §7.6)                                  │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ KernelTmaWarpSpecialized    (default,1 consumer)            │  │
│  │ KernelTmaWarpSpecializedPingpong(2 consumer 交替)            │  │
│  │ KernelTmaWarpSpecializedCooperative(N consumer 协同)         │  │
│  │ KernelTmaWarpSpecializedSm100* (1Sm / 2Sm)                  │  │
│  │ KernelTmaWarpSpecializedSm120*                                 │  │
│  │ KernelScheduleAuto (builder 启发式挑选)                       │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  Epilogue fusion tags(§5.9)                                          │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ LinearCombination            (默认,D = α*acc + β*C)          │  │
│  │ LinCombEltAct                (加 activation)                  │  │
│  │ LinCombPerRowBias / PerColBias (加 bias)                      │  │
│  │ LinCombPerRowBiasEltAct / PerColBiasEltAct (bias + activation)│  │
│  │ LinCombTopKSoftmaxCol        (MoE gating)                    │  │
│  │ ScaledLinCombPerRowBiasEltActAmaxAux(block-scale)            │  │
│  │ ~25 种 predefined + 手写 Sm90EVT<...>                         │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  TileScheduler tags(§6.5)                                            │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ PersistentScheduler         (sm90 / sm100 默认)               │  │
│  │ StreamKScheduler            (K-bound partial sum)            │  │
│  │ GroupScheduler               (Grouped GEMM / MoE)             │  │
│  │ DynamicPersistentScheduler   (sm100 CLC 动态)                 │  │
│  │ StaticPersistentScheduler    (sm100 轻量版)                  │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  Builder 路由(§8)                                                    │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ CollectiveBuilder<13 个 template 参数>                       │  │
│  │   ↓ is_same_v 静态枚举                                      │  │
│  │ 8 partial specializations(§8.0):                              │  │
│  │   - 1: GMMA + TMA + WS + SS                                 │  │
│  │   - 2: GMMA + TMA + WS + RS                                 │  │
│  │   - 3: GMMA + TMA + WS + SS + FP8 fast-accum                │  │
│  │   - 4: GMMA + TMA + WS + SS + FP8 blockwise                 │  │
│  │   - 5: GMMA + cp.async + WS + SS                             │  │
│  │   - 6: GMMA + cp.async + 非 WS + SS                          │  │
│  │   - 7: GMMA + cp.async + 非 WS + RS                          │  │
│  │   - 8: KernelScheduleAuto (auto picker)                     │  │
│  │   ↓ 编译期只编选中那个 partial spec,运行时无开销             │  │
│  │ 产出具体 CollectiveMma 实例                                  │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**配置空间大小估算**(全部组合相乘):

- ArchTag × dtype × memory-path × WS × SS/RS × schedule × epilogue × scheduler ≈ `4 × 8 × 2 × 2 × 2 × 5 × 25 × 5` ≈ **~160,000 个有效组合**

但 **CUTLASS 不是 160,000 个代码**——它把每个维度拆成 ~25 个 partial specialization,每个 partial spec 编译期只编选中的那一个,生成 1 份代码。整个"配置空间"在编译期被 builder 用 `is_same_v` 路由到 1 个具体的 partial spec,**生成的二进制文件 size 跟 1 份代码一样,没有冗余**。

这就是 5 层抽象 + tag-inheritance dispatch 的价值:**配置空间巨大,但每个具体实现编译期最优**。

**Ch10(Blackwell)在这张图上**只换了 ArchTag / memory-path / schedule 三处——5 层骨架不动。这印证了序章的承诺:"同样 5 层框架,换一组原子就能跨架构"。

---

