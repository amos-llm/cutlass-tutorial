## 第 8 章:CollectiveBuilder——把"形状 + 类型"压成具体实现

文件:`include/cutlass/gemm/collective/builders/sm90_gmma_builder.inl`(~10 个 partial specialization)。

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
  //    constexpr int Stages = compute_stage_count<
  //        MainloopSm90TmaGmmaWarpSpecialized, SmemSizeFunctor, TileShape_MNK,
  //        StageCountType, sizeof(Epilogue::SharedStorage), AlignmentA, AlignmentB
  //    >;
  //    如果用户给了 StageCount<N>,Stages = N。
  //    如果 StageCountAuto,Stages = 看 smem 预算 + epi carveout 后能塞几 stage。
  //    如果 StageCountAutoCarveout<bytes>,Stages = 减去 bytes 之后再 auto。

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
constexpr int compute_stage_count(...) {
  // 1. 从 default stage count(根据 smem 限制计算)开始
  // 2. 减去 epilogue::SharedStorage 占用的 bytes
  // 3. 检查 smem 剩余是否还够每个 stage 的 A+B tensor
  // 4. 选 <= 剩余预算 的最大 stages
}
```

编译期算的,程序运行时根本不知道"auto 决定过"——结果是具体某个 `Stages` 数值。

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

- ✅ 能在 `sm90_gmma_builder.inl` 里读懂 partial specialization 的结构(挑 `KernelTmaWarpSpecialized` 那一个开始)。
- ✅ 给一个具体配置"(fp16, RowMajor × ColMajor, 128×128×32, ClusterShape<_4,_2,_1>)",你能**手算** builder 会推什么(AtomLayoutMNK、StagesSmemLayoutAtomA)。
- ✅ 知道 `Auto*` 不是"运行时决定",而是编译期算。

---

