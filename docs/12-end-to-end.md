## 第 12 章:`examples/49`——一个 End-to-End 集成示例

文件:`examples/49_hopper_gemm_with_collective_builder/49_collective_builder.cu`。

这一章**不引入**新概念——是把 Ch1-8 的所有主题打包成"用户故事"。

### 12.1 这个例子做了什么

把 Ch2 的 GEMM 包装到 `ExampleRunner<...>` 模板里,允许在 **编译期** 改 4 个东西:

```cpp
template <
  class MainloopScheduleType = cutlass::gemm::collective::KernelScheduleAuto,
  class EpilogueScheduleType = cutlass::epilogue::collective::EpilogueScheduleAuto,
  class StageCountType       = cutlass::gemm::collective::StageCountAuto,
  class TileSchedulerType    = cutlass::gemm::PersistentScheduler,
  bool UseCustomEVT          = false
>
struct ExampleRunner {
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignmentA,
      ElementB, LayoutB, AlignmentB,
      ElementAccumulator,
      TileShape, ClusterShape,
      StageCountType,    // user-tunable
      MainloopScheduleType   // user-tunable
  >::CollectiveOp;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementAccumulator,
      ElementC, LayoutC, AlignmentC,
      ElementC, LayoutC, AlignmentC,
      EpilogueScheduleType  // user-tunable
  >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int>,
      CollectiveMainloop,
      CollectiveEpilogue,
      TileSchedulerType  // user-tunable
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  // ... other:run(), can_implement(), args_from_options, etc.
};
```

`main()` 用一个配置网格跑多次:

```cpp
int main() {
  // 默认 — 跟 examples/48 一样
  run<Default, Default, ..., false>();

  // 切到 Pingpong
  run<KernelTmaWarpSpecializedPingpong, Default, ..., false>();

  // 加 EVT bias + relu
  run<Default, Default, ..., true>();

  // 改 TileScheduler 为 StreamK
  run<Default, Default, ..., StreamKScheduler, false>();
}
```

### 12.2 加 EVT bias + ReLU fusion 的具体写法

```cpp
// ReLU 不是 `cutlass::ReLU` 直接可用;CUTLASS 预置的是模板化的
// `cutlass::epilogue::thread::ThresholdReLU<T>`(template, `threshold=0` 即 ReLU)。
// ComputeFn 必须是模板 functor,所以 `Sm90Compute<ThresholdReLU<T>, ...>` 里要带模板参数。
if constexpr (UseCustomEVT) {
  using namespace cutlass::epilogue::fusion;
  using CustomEVT = Sm90EVT<
      Sm90Compute<cutlass::plus,
                  ElementD, ElementCompute, RoundStyle>,    // ← bias addition (root)
      Sm90EVT<
        Sm90Compute<cutlass::epilogue::thread::ThresholdReLU<ElementCompute>,
                    ElementD, ElementCompute, RoundStyle>,  // ← ReLU(模板 functor,带 ElementCompute)
        Sm90AccFetch                                        // input is the accumulator
      >,
      Sm90SrcFetch<ElementC>                                // bias tensor source(ElementC 是 src 类型)
  >;
  // 把 EVT 接到 CollectiveEpilogue<...>::Arguments(略):
  epilogue_args = ...;  // 注入 CustomEVT
}
```

`Sm90EVT<Op, ...Args>` 是一个 nested type:Op 是当前节点的算子,Args 是该算子的输入子节点列表。叶子节点(`Sm90AccFetch` / `Sm90SrcFetch` / `Sm90ScalarBroadcast`)直接提供数据。

### 12.3 Builder 重新解析的 4 个 user 参数

当用户在 `ExampleRunner<Pingpong, ...>` 改 MainloopScheduleType:

1. **MainloopScheduleType** = `KernelTmaWarpSpecializedPingpong`——这直接传到 `CollectiveBuilder`(Ch11)。Builder 看到这是一个具体 tag(不是 Auto),直接:
   - 选 partial spec `CollectiveBuilder<Sm90, TmaWarpSpecializedPingpong, ...>`
   - 推 DispatchPolicy = `MainloopSm90TmaGmmaWarpSpecializedPingpong<Stages, ClusterShape, KernelTmaWarpSpecializedPingpong>`
   - 该 DispatchPolicy 推导具体 mainloop 实例化(就是 `sm90_mma_tma_gmma_ss_warpspecialized_pingpong.hpp` 的 partial spec)

2. **EpilogueScheduleType** 影响 epilogue builder——同 1。

3. **StageCountType** = `StageCount<4>`(强制 4 stage)。Builder 跳过 auto 算法,直接用 4。

4. **TileSchedulerType** = `StreamKScheduler`(切到 StreamK)。
   - `GemmUniversal<..., StreamKScheduler>` 的 SFINAE 路由(Ch8.1)— 因为 `StreamKScheduler` 不在 mainloop 的 kernel schedule tree 里,只影响 kernel 匹配的 TileScheduler。
   - 这会进 `sm90_tile_scheduler_stream_k.hpp` 的 partial spec。

### 12.4 这章的 takeaway

读 `examples/49` 时反复出现的 4 个 `*Type` template 参数 + `UseCustomEVT` bool,**对应 Ch9 (dispatch policy) + Ch11 (builder) + Ch6 (EVT) + Ch8 (scheduler) 这 4 件事**。

每一件都对应一个"切来切去只改一行 type alias"的具体动作:

- 切 schedule:换 mainloop pipeline 形态
- 切 stage count:换 smem 预算
- 切 epilogue schedule:换 epilogue 调度形态
- 切 TileScheduler:换持久 kernel 调度策略
- 切 EVT:加融合算子

### 12.5 章末:读完这一章你该做得到的事

- ✅ 把 `examples/49` 完整读一遍——它的代码结构跟 Ch1-8 的所有抽象都对应。
- ✅ 修改 `ExampleRunner<KernelTmaWarpSpecializedPingpong>` 跑一次,确认 mainloop 切到了双 warp group。
- ✅ 把 `UseCustomEVT = true` 跑一次,看 epilogue 加入 bias+ReLU 的 EVT 是否让性能变化。
- ✅ `TileSchedulerType = StreamKScheduler` 切过去,看是否能编译过去(可能需要不同的 problem shape,因为 StreamK 适合 K-bound)。

10 章收口——调参。

---

