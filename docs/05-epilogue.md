## 第 5 章:深入 CollectiveEpilogue + EVT

Epilogue 和 Mainloop 是镜像:同样从 gmem→smem(TMA load C)→register(TMA store D)、同用 pipeline、同 swizzle、同 swappable 接口。区别只在:

1. 数据流方向反过来:把 accumulator 通过 TMA store 写到 gmem。
2. **可能插一段融合算子**——bias、ReLU、silu、top-K softmax、scale、swizzle output 等。
3. 调度的细颗粒度更细:有专门的 `StagesC` (C load 阶段) / `StagesD` (D store 阶段),可以独立调。

主文件:`include/cutlass/epilogue/collective/sm90_epilogue_tma_warpspecialized.hpp`。

### 5.1 Dispatch policy

```cpp
template <
  int StagesC_, int StagesD_, int FragmentSize_,
  bool ReuseSmemC_, bool DelayTmaStore_,
  ...
>
struct Sm90TmaWarpSpecialized
    : ... /* 标记本身是 dispatch tag 树的一个 node */ {
  constexpr static int StagesC = StagesC_;
  constexpr static int StagesD = StagesD_;
  constexpr static int FragmentSize = FragmentSize_;
  constexpr static bool ReuseSmemC = ReuseSmemC_;
  constexpr static bool DelayTmaStore = DelayTmaStore_;
};
```

5 个参数分别在做什么:

|参数|默认值|含义|
|---|---|---|
|`StagesC`|auto|C 加载阶段数(从 C 到 register 的流水线深度)|
|`StagesD`|auto|D store 阶段数(register 到 smem 再到 gmem 的流水线深度)|
|`FragmentSize`|8|each store lane 一次写多少个元素|
|`ReuseSmemC`|true|是否把"已加载 C 的 smem slot"在 epilogue 末尾继续复用(给 output D 用)|
|`DelayTmaStore`|false|是否把 TMA store 推迟到最后(为 swizzle / fusion 留时间)|

`EpilogueScheduleAuto` builder 缺省是哪个具体 `StagesC` / `StagesD` / `FragmentSize` / `ReuseSmemC` / `DelayTmaStore` 组合,看 `include/cutlass/epilogue/collective/builders/sm90_builder.inl`(Ch8 讨论)。

### 5.2 默认 epilogue:`D = alpha * (A*B) + beta * C`

```cpp
// 在 operator() 里的 forward path(简化):
for (...) {  // tiles
  // 1. 加载 C 块
  cute::copy(TiledCopyC, C_block, C_smem_block);
  // 2. mma 累加器已经有了(从 mainloop)
  // 3. alpha/beta 系数乘法
  acc_array = alpha * acc_array + beta * (C_register loaded from C_smem_block);
  // 4. 转换:fp32 acc -> C type
  C_or_D_t_register = ConvertOp{}(acc_array);
  // 5. TMA store
  cute::copy(TiledCopyD, C_or_D_t_register, D_block);
}
```

> **你手写 GEMM 的对照**:你写 epilogue 时就是这几步——读 C、乘 β、乘 α + acc、写 D。CUTLASS 把 alpha/beta/转换类型切成一组 visitor 节点(下面)。

### 5.3 TMA Store 路径

跟 TMA Load 镜像——把 register tile 经过 smem 缓存再 store 到 gmem:

```cpp
// TMA Store atom:
auto tma_store_d = SM90_TMA_STORE{};
auto TiledCopyD = make_tiled_copy(tma_store_d, TiledMma);

// use:
cute::copy(TiledCopyD, D_register_tile, D_gmem_block);
```

TMA store 把"分散到多 lane 的 D 元素"高效打包成一个 gmem block 写。

### 5.4 EVT(Epilogue Visitor Tree)——融合算子

如果你不需要任何融合,上一节就够。需要 bias / ReLU / silu / scale 等,用 EVT。

EVT 是一个小型"小 DSL":每个 `compute` 节点代表一个算子,叶子节点是"输入源"(`Sm90SrcFetch` 是从 src 张量读,`Sm90AccFetch` 是从 accumulator 读,`Sm90ScalarBroadcast` 是标量广播)。

```cpp
#include <cutlass/epilogue/fusion/sm90_visitor_compute_tma_warpspecialized.hpp>
#include <cutlass/epilogue/fusion/sm90_visitor_tma_warpspecialized.hpp>

// 例子: D = bias + ReLU(A*B)
//                       ← Sm90EVT root:add
//                    /            \
//           bias_src_fetch     ReLU compute
//                                 /
//                          (1 arg = acc)

// silu/relu/gelu 这种 activation CUTLASS **不会**预置 — 见 §5.4 后半段,
// 这里先给用户自定义 unary functor:
struct IdentitySilu {
  template <class T>
  CUTLASS_HOST_DEVICE T operator()(T const& x) const {
    return x / (T(1) + cutlass::exp(-x));
  }
};

using MyEVT = Sm90EVT<
    Sm90Compute<homogeneous_add, /* ElementD, RoundStyle... */>,   // root: bias + ReLU(acc)
    Sm90SrcFetch,                                                    // arg 1 of add: 从 src tensor(bias)取
    Sm90EVT<                                                         // arg 2 of add: ReLU(acc)
        Sm90Compute<cutlass::identity, /* RoundStyle... */>,
        Sm90EVT<
            Sm90Compute<IdentitySilu, /* RoundStyle... */>,         // 一元 activation
            Sm90AccFetch                                            // 它的唯一 arg = accumulator
        >
    >
>;

// 用法:
CollectiveEpilogue evt_op(...);
evt_op(/* args to EVT: src tile of bias, problem shape, etc. */);
```

EVT 节点的"参数"按 visit 顺序排——根的 arg 列表从前到后,再到子节点。

#### callout:`ComputeFn` 设计约束

文件 `include/cutlass/epilogue/fusion/sm90_visitor_compute_tma_warpspecialized.hpp`(`Sm90VisitorCompute` 类附近),讲 `ComputeFn` 的 4 个成员签名约束(原文多行注释):

```cpp
// 用户要写的 ComputeFn 必须暴露这 4 个 functors(auto-detected by visitor tree):
//
// 1. auto operator()(Tensor frag, BroadcastedScalars...)  → 计算结果(frag-shape 的 tensor)
// 2. void reductionsum(...)                                  → 用于最终 reduction(如 sum of squares)
// 3. auto get_reduction_counters()                          → 用于性能计数
// 4. ... plus consumer::OrderedInnerTileVisibilityReshape  shape info
//
// 是的,你写个 sm90_visitor_compute 节点,写的是 4-5 个 functor;
// 不像 Pytorch 的 activation 是一行 lambda。
```

这一约束是 CUTLASS 3.x 决定不用 lambda fusion 的原因——**为了在 register / smem 之间有显式的形状控制**,并且 epilogue 端要保证它是 per-warp / per-tile 可拆解的。

#### 完整例子:`D = alpha * (A*B) + beta * C`(标准默认)

`examples/49_hopper_gemm_with_collective_builder/49_collective_builder.cu` 的实现是:

```cpp
// D = beta*C + alpha*(A*B),按这个树:
//     root = add
//       ├── left  = mul(alpha_scalar, acc)
//       │           ├── alpha_scalar
//       │           └── A*B (来自 accumulator)
//       └── right = mul(beta_scalar, C)
//                   ├── beta_scalar
//                   └── C

using CustomEVT =
  cutlass::epilogue::fusion::Sm90EVT<
    cutlass::epilogue::fusion::Sm90Compute<
      cutlass::homogeneous_multiply_add,
      ElementD, ElementCompute, RoundStyle>,
    cutlass::epilogue::fusion::Sm90ScalarBroadcast<ElementScalar>,  // beta
    cutlass::epilogue::fusion::Sm90SrcFetch<ElementC>,              // C
    cutlass::epilogue::fusion::Sm90EVT<
      cutlass::epilogue::fusion::Sm90Compute<
        cutlass::multiplies, ElementCompute, ElementCompute, RoundStyle>,
      cutlass::epilogue::fusion::Sm90ScalarBroadcast<ElementScalar>, // alpha
      cutlass::epilogue::fusion::Sm90AccFetch                        // A*B
    >
  >;
```

#### ComputeFn 实际可用的算子

CUTLASS 没有为 silu/relu/gelu/sigmoid 之类的 activation 单独提供 `homogeneous_silu` 这种算子。ComputeFn 模板位置写的是**任何带 `operator()(A)` 的 struct**,只要满足 §5.4 注释块里的"恰好一个模板参数"约束。

CUTLASS 内置的 functors(在 `cutlass` namespace):

- `cutlass::multiplies<A>`、`cutlass::plus<A>`、`cutlass::minus<A>`、`cutlass::divides<A>`、`cutlass::negate<A>`(数学运算)
- `cutlass::homogeneous_multiply_add<A, A, A>`(单一 FMA 节点,最常用)

如果用户需要 silu:

```cpp
// 几十行自定义一个
template <class A>
struct HomogeneousSilu {
  CUTLASS_HOST_DEVICE A operator()(A x) const {
    return x / (A(1) + cutlass::exp(-x));   // silu 公式
  }
};
// 用法:Sm90Compute<HomogeneousSilu, ...>
```

即:activation / 一元 / 二元 / 自定义算子,都需要用户自己写十几行 functor。这是 CUTLASS 3.x 故意不内建大量 activation 的取舍——为了编译期形状可控,只内建最小集(`multiplies` / `plus` 等数学)。其他 activation 通过用户 functor 扩展。

### 5.5 跨章对比

|Example|在 epilogue 上加了什么|在哪|
|---|---|---|
|`examples/48_hopper_warp_specialized_gemm/`|默认|一切从零|
|`examples/49_hopper_gemm_with_collective_builder/`|加 EVT bias + ReLU fusion + 改 schedule|Ch9|
|`examples/50_hopper_gemm_with_epilogue_swizzle/`|swizzle output(让 D 的 gmem 写入布局更好)|`examples/50_*/50_*.cu`|
|`examples/113_hopper_gemm_activation_fusion/`|多 activation fusion(bias + gelu + scale)|`examples/113_*/113_*.cu`|

![hierarchy-with-epilogue](../media/images/gemm-hierarchy-with-epilogue.png)

### 5.6 章末:读完这一章你该做得到的事

- ✅ 知道 epilogue dispatch policy 5 个参数分别影响什么(StagesC、StagesD、FragmentSize、ReuseSmemC、DelayTmaStore)。
- ✅ 能看懂 `Sm90EVT<Sm90Compute<...>, ...>` 嵌套语法——它就是"小 AST"。
- ✅ 给一个 `D = alpha * (A*B) + beta * silu(C)` 这种非平凡 epilogue,你能写出对应的 EVT 树。
- ✅ 知道 EVT 跟 PyTorch 的 chain 是两回事——CUTLASS 是显式 AST 不是基于 lambda fusion。

---

