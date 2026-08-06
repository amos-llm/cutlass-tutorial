## 第 6 章:深入 CollectiveEpilogue

Epilogue 和 Mainloop 是镜像:同样从 gmem→smem(TMA load C)→register(算子)→smem(D 缓存)→gmem(TMA store D)、同用 pipeline、同 swizzle、同 swappable 接口。区别只在:

1. 数据流方向反过来:把 accumulator 通过 TMA store 写到 gmem(TMA store 本身是从 smem 到 gmem,所以 epilogue 还要先把转换后的 D 元素写到 smem)。
2. **可能插一段融合算子**——bias、ReLU、silu、top-K softmax、scale、swizzle output 等。
3. 调度的细颗粒度更细:有专门的 `StagesC` (C load 阶段) / `StagesD` (D store 阶段),可以独立调。

主文件:`include/cutlass/epilogue/collective/sm90_epilogue_tma_warpspecialized.hpp`。

### 6.1 Dispatch policy

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

|参数|默认(由 builder 算)|含义|
|---|---|---|
|`StagesC`|`min(EpiTiles, 4)` (ReuseSmem 时 `max(min(EpiTiles, 4), StagesD+1)`)|C 加载阶段数(从 C 到 register 的流水线深度)|
|`StagesD`|`min(EpiTiles, 2)`|D store 阶段数(register 到 smem 再到 gmem 的流水线深度)|
|`FragmentSize`|`size(EpilogueTileMN{}) / (cooperative ? 256 : 128)`|each store lane 一次写多少个元素|
|`ReuseSmemC`|true iff `sizeof(C)==sizeof(D) && sizeof(D)>8`|是否把"已加载 C 的 smem slot"在 epilogue 末尾继续复用(给 output D 用)|
|`DelayTmaStore`|true iff `C==void && !ptr-array schedule`|是否把 TMA store 推迟到最后(为 swizzle / fusion 留时间)|

`EpilogueScheduleAuto` builder 缺省是哪个具体 `StagesC` / `StagesD` / `FragmentSize` / `ReuseSmemC` / `DelayTmaStore` 组合,看 `include/cutlass/epilogue/collective/builders/sm90_builder.inl`(Ch11 讨论)。

### 6.2 默认 epilogue:`D = alpha * (A*B) + beta * C`

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

### 6.3 TMA Store 路径

跟 TMA Load 镜像——把 register tile 经过 smem 缓存再 store 到 gmem:

```cpp
// TMA Store atom:
auto tma_store_d = SM90_TMA_STORE{};
auto TiledCopyD = make_tiled_copy(tma_store_d, TiledMma);

// use:
cute::copy(TiledCopyD, D_register_tile, D_gmem_block);
```

TMA store 把"分散到多 lane 的 D 元素"高效打包成一个 gmem block 写。

### 6.4 EVT(Epilogue Visitor Tree)——融合算子

如果你不需要任何融合,上一节就够。需要 bias / ReLU / silu / scale 等,用 EVT。

EVT 是一个小型"小 DSL":每个 `compute` 节点代表一个算子,叶子节点是"输入源"(`Sm90SrcFetch` 是从 src 张量读,`Sm90AccFetch` 是从 accumulator 读,`Sm90ScalarBroadcast` 是标量广播)。

```cpp
#include <cutlass/epilogue/fusion/sm90_visitor_compute_tma_warpspecialized.hpp>
#include <cutlass/epilogue/fusion/sm90_visitor_tma_warpspecialized.hpp>

// 例子: D = bias + silu(A*B)
//                       ← Sm90EVT root:plus
//                    /            \
//           bias_src_fetch     silu compute
//                                 /
//                          (1 arg = acc)

// CUTLASS 在 `cutlass::epilogue::thread::activation.h` 里预置了 GELU / Sigmoid /
// Tanh / ThresholdReLU / LeakyReLU 等 activation;silu 这种没有,需要自己写 functor。
// ComputeFn 必须是 `template <class> class`,所以 functor 本身要写成模板:
template <class T>
struct HomogeneousSilu {
  CUTLASS_HOST_DEVICE T operator()(T const& x) const {
    return x / (T(1) + cutlass::exp(-x));   // silu 公式
  }
};

using MyEVT = Sm90EVT<
    Sm90Compute<cutlass::plus, /* ElementD, RoundStyle... */>,       // root: bias + silu(acc)
    Sm90SrcFetch<ElementC>,                                           // arg 1 of plus: 从 src tensor(bias)取(ElementC 是 src 类型)
    Sm90EVT<                                                          // arg 2 of plus: silu(acc)
        Sm90Compute<HomogeneousSilu, /* RoundStyle... */>,           // 一元 activation(模板 functor)
        Sm90AccFetch                                                  // 它的唯一 arg = accumulator (Sm90AccFetch 无模板参数)
    >
>;

// 用法:
CollectiveEpilogue evt_op(...);
evt_op(/* args to EVT: src tile of bias, problem shape, etc. */);
```

EVT 节点的"参数"按 visit 顺序排——根的 arg 列表从前到后,再到子节点。

#### callout:`ComputeFn` 设计约束

`include/cutlass/epilogue/fusion/sm90_visitor_compute_tma_warpspecialized.hpp` 第 65-83 行,讲 `ComputeFn` 的实际约束(原文翻译):

```cpp
// ComputeFn 必须能接受**恰好一个**模板参数。在标准 C++ 里,ComputeFn
// 也可以有其他模板参数,只要那些参数都有默认值(例如 template<class A,
// class B = A> struct Foo)。但 Clang 这类编译器有时要求**只能有一个**
// 模板参数——此时常见 workaround 是写一个单模板参数的子类继承原类,再
// 用子类作为 Sm90Compute 的 ComputeFn 参数:
//
//   template<class A>
//   struct FooHomogeneous : public Foo<A, A> {};
```

实际 `Sm90Compute::visit()` 内部只调 `ComputeFn<Array<ElementCompute, FragmentSize>>::operator()(cvt_frg_inputs...)`(见同文件第 184 行)——只需要实现这一个 `operator()`。要不要广播参数、可选 `Arguments` 都是 per-functor 扩展点,不是强制 4 个 functor。

这是 CUTLASS 3.x 决定不用 lambda fusion 的原因——**为了在 register / smem 之间有显式的形状控制**,并且 epilogue 端要保证它是 per-warp / per-tile 可拆解的。

#### 完整例子:`D = alpha * (A*B) + beta * C`(标准默认)

`examples/49_hopper_gemm_with_collective_builder/49_collective_builder.cu` 的实现是:

```cpp
// D = beta*C + alpha*(A*B),按这个树(单一 FMA 节点作 root):
//     root = homogeneous_multiply_add   (算 X*Y + Z)
//       ├── X = beta_scalar   (Sm90ScalarBroadcast)
//       ├── Y = C             (Sm90SrcFetch)
//       └── Z = mul(alpha_scalar, acc)
//                  ├── alpha_scalar
//                  └── A*B      (来自 accumulator,Sm90AccFetch)

using CustomEVT =
  cutlass::epilogue::fusion::Sm90EVT<
    cutlass::epilogue::fusion::Sm90Compute<
      cutlass::homogeneous_multiply_add,                              // root: X*Y + Z
      ElementD, ElementCompute, RoundStyle>,
    cutlass::epilogue::fusion::Sm90ScalarBroadcast<ElementScalar>,  // X = beta
    cutlass::epilogue::fusion::Sm90SrcFetch<ElementC>,              // Y = C
    cutlass::epilogue::fusion::Sm90EVT<
      cutlass::epilogue::fusion::Sm90Compute<
        cutlass::multiplies, ElementCompute, ElementCompute, RoundStyle>,
      cutlass::epilogue::fusion::Sm90ScalarBroadcast<ElementScalar>, // alpha_scalar
      cutlass::epilogue::fusion::Sm90AccFetch                        // A*B
    >
  >;
```

#### ComputeFn 实际可用的算子

`Sm90Compute<ComputeFn, ...>` 的 `ComputeFn` 必须是**模板 functor**(`template <class> class`),`Sm90Compute` 内部会用 `ComputeFn<Array<ElementCompute, FragmentSize>>` 实例化它(见 `sm90_visitor_compute_tma_warpspecialized.hpp` 的 `Sm90Compute` 定义)。所以任何想塞进 EVT 的算子都得写成模板形式。

CUTLASS 内置的算子(在 `cutlass` namespace):

- 数学类:`cutlass::multiplies<A>`、`cutlass::plus<A>`、`cutlass::minus<A>`、`cutlass::divides<A>`、`cutlass::negate<A>`
- FMA 类:`cutlass::homogeneous_multiply_add<A, A, A>`(单一 FMA 节点,最常用)
- Activation(`cutlass::epilogue::thread::activation.h`):`GELU<T>`、`Sigmoid<T>`、`Tanh<T>`、`LeakyReLU<T>`、`ThresholdReLU<T>` 等(注意这些都是**模板类**,直接喂给 `Sm90Compute` 即可)

如果用户需要 silu / ReLU(CUTLASS 没预置 ReLU 本身;只有 `ThresholdReLU`):

```cpp
// 模板 functor(必须 `template <class T>` 形式)
template <class T>
struct HomogeneousSilu {
  CUTLASS_HOST_DEVICE T operator()(T const& x) const {
    return x / (T(1) + cutlass::exp(-x));   // silu 公式
  }
};
// 用法:Sm90Compute<HomogeneousSilu, ElementD, ElementCompute, RoundStyle>
```

即:自定义算子都需要用户写十几行模板 functor。CUTLASS 故意不内建所有 activation——为了编译期形状可控,只内建常见数学运算 + 几个常用 activation;其他通过用户 functor 扩展。

### 6.5 跨章对比

|Example|在 epilogue 上加了什么|在哪|
|---|---|---|
|`examples/48_hopper_warp_specialized_gemm/`|默认|一切从零|
|`examples/49_hopper_gemm_with_collective_builder/`|加 EVT bias + ReLU fusion + 改 schedule|Ch12|
|`examples/50_hopper_gemm_with_epilogue_swizzle/`|swizzle output(让 D 的 gmem 写入布局更好)|`examples/50_*/50_*.cu`|
|`examples/113_hopper_gemm_activation_fusion/`|多 activation fusion(bias + gelu + scale)|`examples/113_*/113_*.cu`|

![hierarchy-with-epilogue](../media/images/gemm-hierarchy-with-epilogue.png)

### 6.6 章末:读完这一章你该做得到的事

- ✅ 知道 epilogue dispatch policy 5 个参数分别影响什么(StagesC、StagesD、FragmentSize、ReuseSmemC、DelayTmaStore)。
- ✅ 能看懂 `Sm90EVT<Sm90Compute<...>, ...>` 嵌套语法——它就是"小 AST"。
- ✅ 给一个 `D = alpha * (A*B) + beta * silu(C)` 这种非平凡 epilogue,你能写出对应的 EVT 树。
- ✅ 知道 EVT 跟 PyTorch 的 chain 是两回事——CUTLASS 是显式 AST 不是基于 lambda fusion。

---

