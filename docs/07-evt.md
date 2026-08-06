## 第 9 章:Epilogue Visitor Tree(EVT)——融合算子 DSL

这一章从 Ch8 epilogue 中拆出来单独讲——EVT 是 CUTLASS 3.x 的**第二个核心创新**(第一个是 5 层抽象)。它把"epilogue 阶段可做的所有事"组织成一个**小型 DSL**,用户写一段"小表达式"就能组合 bias、ReLU、silu、scale、swizzle output、top-K softmax 等任意 fusion。

主文件:`include/cutlass/epilogue/fusion/sm90_visitor_*.hpp` 一族(共 5 个:`tma_warpspecialized`、`compute_tma_warpspecialized`、`load_tma_warpspecialized`、`store_tma_warpspecialized`、`topk_softmax`)。

### 7.1 思路:把 epilogue 看成"小 AST"

epilogue 阶段从 mainloop 拿到 fp32 accumulator,然后做这些事的某种组合:

```text
D = activation( alpha * (A*B) + beta * C + bias + scale_per_row + ... + SiLU(top_k(...)) )
```

每件事都是一个 **node**;节点之间靠"输入输出"连成**树**(其实是 DAG)。叶子节点 = 输入源(`acc` / `C` / `bias` / `scalar`),内部节点 = 算子(`alpha * acc` / `bias + ...` / `silu(...)`)。

EVT 的设计就是:每个 node 是一个 struct,有标准 6 个 hook(`is_producer_load_needed` / `is_C_load_needed` / `get_producer_load_callbacks` / `get_consumer_store_callbacks` 等),编译器自动按 tree 形态生成 epilogue 的 producer/consumer 代码。

### 7.2 三类节点:compute / load / store

CUTLASS 把 epilogue 拆成 3 种角色,每种是一组 struct:

#### Compute 节点(`sm90_visitor_compute_tma_warpspecialized.hpp`)

| 节点 | 作用 | 典型算子 |
|---|---|---|
| `Sm90Compute<Fn, ...>` | 一个算子 `Fn`,接 0-N 个输入,产出一个输出 | `homogeneous_multiply_add` (= `alpha * acc + beta * C`)、`mul_add`、自定义 `silu` 等 |
| `Sm90TopkSoftmax<...>` | 算 top-K + softmax,专用于 MoE gating 输出 | topk → softmax |

#### Load 节点(`sm90_visitor_load_tma_warpspecialized.hpp`)

| 节点 | 作用 | 数据来源 |
|---|---|---|
| `Sm90AccFetch` | 拿 mainloop 的 accumulator | register(rmem) |
| `Sm90SrcFetch<Element>` | 拿源矩阵 C | gmem(走 TMA load) |
| `Sm90AccFetchGroupedWgrad` | grouped GEMM 反向专用(拿 wgrad 的 acc)| rmem |
| `Sm90ScalarBroadcast<Element, Value>` | 标量广播(常数 α / β) | 编译期常量 |
| `Sm90ScalarBroadcastPtrArray<...>` | ptr-array 调度下的标量 | gmem 指针数组 |
| `Sm90RowBroadcast<...>` | 行向量广播(bias per row)| gmem |
| `Sm90ColBroadcast<...>` | 列向量广播(bias per col)| gmem |
| `Sm90AuxLoad<Element, Stride, ...>` | 任意 stride 的辅助矩阵加载 | gmem |
| `Sm90AuxArrayLoad<...>` | ptr-array 下的辅助矩阵 | gmem 指针数组 |

#### Store 节点(`sm90_visitor_store_tma_warpspecialized.hpp`)

| 节点 | 作用 |
|---|---|
| `Sm90AuxStore<Element, Stride, ...>` | 主输出 D 写到 gmem |
| `Sm90AuxArrayStore<...>` | ptr-array 下的主输出 |
| `Sm90ScalarReduction<...>` | 把一个向量 reduce 成标量(写到 host) |
| `Sm90RowReduction<...>` | 按行 reduce(partial row-sum,典型用作后续 norm)|
| `Sm90ColReduction<...>` | 按列 reduce |
| `Sm90MatrixReduction<...>` | 全矩阵 reduce |

### 7.3 一个完整的 EVT 例子

经典 fusion `D = silu(A * B + bias)`:

```cpp
// 写法 1:手写树节点(Sm90EVT<...>)
using Sm90EVT = Sm90EVT<
  Sm90Compute<homogeneous_multiply_add, ElementOutput, ElementCompute, RoundStyle>,
  Sm90ScalarBroadcast<ElementScalar, ElementCompute>,    // 叶子:alpha
  Sm90AccFetch,                                            // 叶子:accumulator
  Sm90SrcFetch<ElementC>,                                 // 叶子:C
  Sm90Compute<silu, ElementOutput, ElementCompute, RoundStyle>,  // 内部: silu
  Sm90AuxLoad<ElementBias, StrideBias, ...>               // 叶子:bias
>;

// 写法 2:用 `cutlass::epilogue::fusion::EvtreeOp` builder(更短)
```

**(注意**:源码里实际写法是 `Sm90EVT<ComputeFn, Child0, Child1, ...>`,第一个模板参数是 root 节点,后面是它的输入。`homogeneous_multiply_add` 是 `alpha * acc + beta * C`;这里写的是简化示意。实际 EVT 构造在 `examples/49_hopper_gemm_with_collective_builder/` / `examples/68_hopper_fp8_*_blockwise_scaling/` 之类。)

### 7.4 tree 的"求值"过程

跟普通的 AST 求值不一样——EVT 在 epilogue 阶段是按 producer / consumer warp 分别触发不同节点的回调:

```cpp
// Producer warp 端(在 epilogue 开始前):
void begin_prologue() {
  for (auto& node : tree.post_order()) {
    if (node.is_producer_load_needed()) {
      auto callbacks = node.get_producer_load_callbacks(args);
      // 比如 C 节点 / bias 节点需要 TMA 加载,这里发起
      cute::copy(node.tiled_copy(), node.partition_S(src), node.partition_D(smem));
    }
  }
}

// Consumer warp 端(在每个 tile 的 epilogue):
Array<ElementOutput, FragmentSize> result = tree.visit(frg_acc, epi_v, epi_m, epi_n);
//   ↑ 这是一个递归调用:从 root 开始,root 的 visit() 调它自己的 children's visit(),
//     children's visit() 再调它们的 children 的,直到叶子(从 accumulator / src / scalar / 加载好的 smem 拿值)
//     最后从 leaf 一路返回,每个内部节点做自己的算子
```

`tree.visit()` 的递归形态让**算子组合性极强**——你可以把任意 depth 的 fusion 树塞进去,只要每个节点都有 6 个标准 hook,编译器就能生成 epilogue 代码。

### 7.5 为什么不在 epilogue 章顺手讲?

3 个原因,各自够分量:

1. **节点种类 30+**——光是"叶子节点"就有 8 种(scalar broadcast / row broadcast / col broadcast / acc fetch / src fetch / aux load × 2 / ptr-array 变体),"内部节点"再加 Sm90Compute + Sm90TopkSoftmax。塞进 epilogue 章会让 epilogue 走样。
2. **partial specialization 多**——CUTLASS 3.x 给常见 fusion 模式(`alpha * acc + beta * C`、`alpha * acc + beta * C + bias`、`alpha * acc + beta * C + per-row scale`、`alpha * acc + bias + silu` 等)都写了**专门的 partial specialization**(在 `sm90_visitor_compute_tma_warpspecialized.hpp` 见到 `struct Sm90TreeVisitor<...> : Sm90VisitorImpl<...>` 这种)。这种"特定 fusion 的 fast path"是 EVT 性能保证的核心,讲不到就抓不到要点。
3. **典型 fusion 不止一个**——典型用户写 `D = alpha * (A*B) + bias`;写 MoE gating `topk + softmax`;写 `D = silu(A*B)`;写 `D = (A*B) .* per_token_scale`(这是 `Sm90AuxStore` 的"element-wise multiply with broadcast"模式)——每种都是不同的 partial spec。

### 7.6 怎么写 EVT

#### 写法 A:手写 EVT 节点(精细控制)

```cpp
// 在 CollectiveEpilogue 的 Arguments 里:
using EpilogueEVT = Sm90EVT<
  Sm90Compute<silu, ElementOutput, ElementCompute, FloatRoundStyle::round_to_nearest>,  // root
  Sm90Compute<homogeneous_multiply_add, ElementOutput, ElementCompute, FloatRoundStyle::round_to_nearest>,  // child: alpha*acc + beta*C
  Sm90ScalarBroadcast<ElementScalar, ElementCompute>,  // alpha
  Sm90AccFetch,                                         // acc
  Sm90SrcFetch<ElementC>,                              // C
  Sm90AuxLoad<ElementBias, StrideBias, AlignmentBias>  // bias
>;
```

root = `silu( alpha*acc + beta*C + bias )`,child 是 `alpha*acc + beta*C`,它的 children 是 alpha/acc/C,加上 bias 作为 `silu` 的另一个 input。

#### 写法 B:用 builder(推荐,新手)

`Cutlass 3.x` 提供 `cutlass::epilogue::fusion::EvtOp` builder,但实际项目里**最常见的还是手写**(因为 partial specialization 多,builder 不能 cover 所有组合)。手写更直观。

### 7.7 典型 fusion 速查

| fusion 描述 | 节点写法 |
|---|---|
| `D = alpha * acc + beta * C` | `Sm90EVT<Sm90Compute<homogeneous_multiply_add, ...>, Sm90ScalarBroadcast, Sm90AccFetch, Sm90ScalarBroadcast, Sm90SrcFetch>`(默认 epilogue 就是这个) |
| `D = acc + bias` | `Sm90EVT<Sm90Compute<plus, ...>, Sm90AccFetch, Sm90AuxLoad<bias>>` |
| `D = silu(acc)` | `Sm90EVT<Sm90Compute<silu, ...>, Sm90AccFetch>` |
| `D = silu(alpha*acc + beta*C + bias)` | 嵌套 `Sm90Compute<homogeneous_multiply_add>` + `Sm90Compute<silu>` |
| `D = acc + per_row_scale * C` | `Sm90EVT<Sm90Compute<homogeneous_multiply_add, ...>, Sm90AccFetch, Sm90Compute<homogeneous_multiply, ...>, Sm90RowBroadcast<scale>, Sm90SrcFetch<C>>` |
| `topk_softmax(acc)` | `Sm90TopkSoftmax<...>` |
| `D = acc, aux = sum_rows(D)` | `Sm90EVT<Sm90AuxStore<D>, ..., Sm90RowReduction<aux>>` |

### 7.8 EVT 与 builder 的关系

EVT 节点是 epilogue 的**输入**。builder 看 EVT 类型,落到对应的 partial specialization:

- 默认 epilogue = `Sm90EVT<Sm90Compute<homogeneous_multiply_add, ...>, ScalarBroadcast, AccFetch, ScalarBroadcast, SrcFetch>` —— builder 走默认 epilogue。
- 含 EVT 的 epilogue = builder 看你给的 EVT 类型,落到对应的 `CollectiveEpilogue` partial spec(`include/cutlass/epilogue/collective/builders/sm90_builder.inl` 里)。

Ch10 builder 章会讲 builder 怎么 dispatch 到 EVT-aware 的 partial spec。

### 7.9 章末:读完这一章你该做得到的事

- ✅ 看到 `Sm90EVT<Sm90Compute<...>, Child0, Child1, ...>` 能讲出树结构。
- ✅ 区分 compute / load / store 三类节点各 30+ 种,知道什么时候用哪类。
- ✅ 能手写一个 `D = silu(alpha * acc + beta * C + bias)` 的 EVT。
- ✅ 看 `examples/68`(`hopper_fp8_warp_specialized_grouped_gemm_with_blockwise_scaling`)的 epilogue 能讲清哪些节点在做什么。
- ✅ 知道 builder 怎么按 EVT 类型选 partial spec(细节看 Ch10)。

Ch11 看 kernel orchestrator + TileScheduler——它把 mainloop(Ch6)+ epilogue(Ch8)+ EVT(Ch9)+ scheduler 拼成一个完整 kernel。

---