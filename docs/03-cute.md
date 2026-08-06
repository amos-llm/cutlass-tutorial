## 第 3 章:CuTe——CUTLASS 真正的"语言"

你可能认为 CuTe 是装饰用的工具库。其实不是——它是 CUTLASS **写 mainloop / epilogue 的语言**。mainloop 文件里那一串 `Shape` / `make_tensor` / `make_tiled_mma` 不是装饰,是 mainloop 在描述"我这一 stage 的 smem 上 A 是什么样的、warp 怎么拿自己的 fragment"。

读 Ch5 之前,你需要 CuTe 的"语法熟悉"。这一章不讲 CuTe 怎么写 sgemm(那是 `media/docs/cpp/cute/0x_gemm_tutorial.md` 的事),只讲所有 CUTLASS mainloop **一定会用到**的 4 个核心抽象:`Shape` / `Stride` / `Layout` / `Tensor`,以及 6 个常用组合子:`make_shape` / `make_layout` / `make_tensor` / `composition` / `coalesce` / `tile_to_shape`。

读完之后,你应该能在 `sm90_mma_tma_gmma_ss_warpspecialized.hpp` 里把 `SmemLayoutAtomA` / `GmemTiledCopyA` / `TiledMma` 这些名字各自认出来——它们是 `cute::` 类型还是 CollectiveMma 内部的 type alias、各自在 mainloop 里在做什么。

### 3.1 思想:布局代数

CuTe 把"内存布局"看成一个数学结构:`Layout: Coord → Index`。给定一个整数坐标 `(i, j, k)`,Layout 把它映射到一个一维 index `n`,n 是这个元素在内存中的字节偏移除以 sizeof(elem)。

```cpp
// "4 行 8 列,行主序,fp32"
using LayoutA = Layout<Shape<_4, _8>, Stride<_8, _1>>;

constexpr int idx(int i, int j) {
  // (i, j) → idx 展开(编译期常量)
  return i*8 + j;
}
```

> **你手写 GEMM 的对照**:你写 `A[i][j]` 在 row-major 下算字节偏移 `i * (cols * sizeof(E)) + j * sizeof(E)`。CuTe 的 Layout 就是把这个"展开"放到模板里——所有偏移都编译期算完。

### 3.2 4 个最常用类型

```cpp
#include <cute/layout.hpp>   // Shape / Stride / Layout
#include <cute/int_tuple.hpp> // Int<_N>
#include <cute/tensor.hpp>    // Tensor / make_tensor
#include <cute/pointer.hpp>   // make_gmem_ptr / make_smem_ptr / make_rmem_ptr
```

#### `Shape<a, b, c, ...>` 与 `_N`

```cpp
// Shape 展开 = cute::tuple<Int<...>, ...>(本质是 "tuple of compile-time ints")
template <class... Ts> using Shape = cute::tuple<Ts...>;

using MyShape              = Shape<_128, _128, _32>;     // 编译期 128×128×32
using DynamicShape         = Shape<int, int, int>;       // 运行期 3-tuple
using MixedShape           = Shape<_128, _128, int>;     // 前两者编译期,最后一个动态
```

`_N` 是 `Int<N>` 的 alias,语义是"一个值为 N 的类型"。`Shape` 是 `tuple<...>`,所以你只能 `get<i>` 访问,CUTLASS 大量这种"tuple of compile-time ints"模式,用模板元编程做下标计算。

> 你写 `TileShape = Shape<_128, _128, _32>`,CUDA 编译器会**完全**把这些数字 bake 进代码——没有运行时开销。

#### `Stride<a, b, c, ...>` 与 `_N`

```cpp
template <int M, int N>             // 给 _M / _N 一个明确上下文
using RowMajor2D = Stride<Int<N>, _1>;   // (i, j) → i*N + j
using ColMajor2D = Stride<_1, Int<M>>;   // (i, j) → i + j*M
```

实际写 mainloop 时,你**不需要自己 typedef RowMajor / ColMajor**——CUTLASS 已经用 `cutlass::layout::RowMajor` / `ColumnMajor` 命名暴露出来,内部展开成 CuTe 的 `Stride<...>`。这层是"高层 API 复用底层 Layout"。

> `LayoutLeft::Apply<Shape>` 是 **col-major**(`Stride<1, M, ...>`,最左维度 stride=1),`LayoutRight::Apply<Shape>` 是 **row-major**(`Stride<N, 1, ...>`,最右维度 stride=1)— 见 `cute/stride.hpp:277,279` 的 forward decl 注释。直接写 row-major / col-major 不需要你填 Stride。

组合 `Shape<_M, _N> + Stride<_1, _M>` 就是 `(i,j) → i + j*M`——列主序(LayoutLeft)。`Shape<_M, _N> + Stride<_N, _1>` 是 `(i,j) → i*N + j`——行主序(LayoutRight)。

#### `Layout`

```cpp
template <class Shape, class Stride = LayoutLeft::Apply<Shape>>
struct Layout : private cute::tuple<Shape, Stride> { ... };

using A_Layout = Layout<Shape<_4, _8>, Stride<_1, _4>>;   // col-major (LayoutLeft 默认)
```

`LayoutLeft::Apply<Shape>` 是默认 stride:**col-major**(`Stride<1, M, ...>`)。`LayoutRight::Apply<Shape>` 是 **row-major**(`Stride<N, 1, ...>`)。直接写 row-major / col-major 不需要你填 Stride。

`Layout` 是一个轻量包装——内部就是一个 pair(shape, stride) tuple,所有功能通过 free function 提供。Layout 本身不存数据,只存"如何从 coord 算 index"。

#### `Tensor = Pointer + Layout`

```cpp
#include <cute/tensor.hpp>

// 假设有指针
ElementA* A_ptr = ...;

auto A = make_tensor(make_gmem_ptr(A_ptr),
                     make_layout(make_shape(M, N),     // 运行时尺寸
                                 make_stride(N, 1)));   // RowMajor

// 把上面合成(常见,等价于上面两行的 A):
auto A_alt = make_tensor(A_ptr,                  // 原始指针(自动包为 gmem_ptr)
                         Layout<Shape<M, N>, Stride<N, 1>>{});
```

Tensor 的语义是"一段连续内存 + 怎么解读它"。同样的指针,不同的 Layout = 不同的视图:

```cpp
// 同一个 16 元素 buffer,横看作 4×4,纵看作 4×4,或看作 16
auto x_row = make_tensor(p, Layout<Shape<_4, _4>, Stride<_4, _1>>{}); // row-major
auto x_col = make_tensor(p, Layout<Shape<_4, _4>, Stride<_1, _4>>{}); // col-major
auto x_seq = make_tensor(p, Layout<Shape<_16>>{});                   // 1-D
```

> **你手写 GEMM 的对照**:你的 `cudaMemcpy2D(...)` 给定 pitch / ptr / width / height——Layout 做的就是同一个事,但**编译期全部固化**。

#### pointer 三件套

```cpp
ElementA* raw;
auto gptr = make_gmem_ptr(raw);   // 在 gmem 上的指针
auto sptr = make_smem_ptr(smem_addr);  // 在 smem 上的指针(__shared__ void* 转过来)
auto rptr = make_rmem_ptr(reg_addr);   // 在 register 上的指针
```

这三种 engine 的区别不在指针本身,而在 CuTe 的 layout 变换对它们的处理方式:

- `gmem_ptr`:**只能做普通偏移**——layout 算出来的 offset 就是字节偏移。`make_gmem_ptr` 实质只是个 type tag,几乎不做事。
- `smem_ptr`:**会做 swizzle 修正**——如果你把这个 Tensor 配上一个 `Swizzle<...>` 的 layout(见 3.7),CuTe 会真的按 swizzle 改写到 smem 的访问模式(避免 bank conflict)。这是 mainloop 里重头戏。
- `rmem_ptr`:**layout 直接决定 register 分量**——比如 TiledMma 的 thread-fragment,每个 lane 拿到哪些寄存器,完全由 layout 推。

> **你手写 GEMM 的对照**:你写"threadIdx.x / warpIdx.x 算 smem 偏移"——CUTLASS 里这步就是 `partition_S(thr_mma, smem_layout)`(3.6 略讲),里面靠 `smem_ptr` 的 engine 知道"我现在在 smem,需要走 swizzle"。

### 3.3 6 个核心组合子

#### `make_shape` / `make_stride` / `make_layout`

```cpp
auto s = make_shape(_128, _128, _32);                   // = Shape<_128, _128, _32>
auto d = make_stride(_128, _1);                         // = Stride<_128, _1>
auto l = make_layout(make_shape(_128, _128),            // M × N tile
                     make_stride(_128, _1));            // row-major stride
// 等价 Layout<Shape<_128, _128>, Stride<_128, _1>>
```

#### `make_tensor`

上面 3.2 已经讲过。

#### `composition`:Layout 的"函数复合"

Layout 就是个映射函数;`composition(A, B)` = "先按 B 映射,再按 A 映射"——这相当于数学的 `f ∘ g`,后到先应用。生成的 layout 形状跟 B 一样(读 B 写的 coord),但 index 解释按 A 走。

```cpp
auto L1 = make_layout(make_shape(_8, _8), make_stride(_8, _1));   // row-major 8x8
auto L2 = make_layout(make_shape(_2, _2), make_stride(_1, _8));   // (i, j) → i + 8j

auto L = composition(L1, L2);
// L 是 2×2 的 layout,每元素 (i, j) 先 L2→ (i, 8j) 坐标,再 L1 → i + 8j
// 效果:"一个 2×2 tile,沿行放 1,沿列放 8",嵌进 8×8 大矩阵
```

> ⚠ **mainloop 里 `composition` 的实际用途几乎全是 3.7 的 Swizzle ∘ BaseLayout**——把 swizzle 叠到 smem 的 base layout 上。`tile of tile` 这种用法在 mainloop 里是隐式的(被 `local_tile` 封装了),所以你读 mainloop 时**真正需要认识 `composition` 的场景就一种:smem 那个 swizzle composition**。

> **你手写 GEMM 的对照**:你写 `threadIdx` + `warpIdx` 算 smem 偏移。`composition` 就是把"warp 内偏移"和"warp 在 CTA 内的偏移"复合。

#### `coalesce`:合并相邻同 shape 的 mode

```cpp
auto L = make_layout(make_shape(_2, _3, _4), make_stride(_6, _2, _1));
coalesce(L);  // → Shape<_6, _4>, Stride<_2, _1>(把前两个 mode 合并)
```

`coalesce` 等价于一个 layout simplification——把连续访问的 mode 合成一个 mode。这对编译器做循环优化非常友好。

#### `tile_to_shape`:把一个 tile 排到大矩阵里

```cpp
auto Tile = Layout<Shape<_8, _8>, Stride<_8, _1>>{};   // 8x8 tile
auto Placed = tile_to_shape(Tile,                     // tile 的内部 layout
                            make_shape(_4, _4),       // 在大矩阵里 4×4 个 tile
                            Step<_1, _2>{});          // 大矩阵 first-mode major
// → 在 32×32 的大矩阵里,沿第一维排 4 个 Tile,再沿第二维排 4 个 Tile
```

mainloop 里 `SmemLayoutAtomA = tile_to_shape(SwizzleAtom, make_shape(...block_M, ...block_K, Stages), ...)` 就是这个模式:smem 上 A 是 SwizzleAtom 的 layout,再 tile 到 "`BlockM × BlockK × Stages`"的 3D 上。Ch5 会具体讲到。

#### 其他常用(略讲,你会在 Ch5/Ch8 看到)

- `local_tile`:从"大矩阵 + tile 大小 + 起始坐标"切成子 tile。Mainloop 里获取"本 CTA 当前 stage 的 A":

  ```cpp
  auto A_smem_tile = local_tile(A_smem_layout, /*coord=*/make_coord(block_m_coord, k_coord, stage), /*proj=*/...);
  ```

- `slice`:把 `Tile (M_tile, N_tile, K_tile, ...)` 切成"由 thread / warp 看的子 view"。
- `make_tiled_mma` / `make_tiled_copy`:把一个 mma atom / copy atom 复制成 CTA / warp / thread 分工——见 3.6。

### 3.4 atom:CUTLASS / CuTe 最小单位,三种类型,三层嵌套

读 CUTLASS 源码时,你第一眼看到的是一堆 `XXX_Atom`、`YYY_Atom`、`TiledMma`、`TiledCopy`、`AtomLayout`、`TileShape`、`ClusterShape`——它们之间什么关系?这一节一次性讲清。

#### 什么是 atom

**atom = 一条硬件指令 + 它对应的 thread/value layout,是 CUTLASS/CuTe 里不可再拆的最小命名单位。**

比如:

- `SM90_64x16x16_F32F16F16F32_SS` 是一个 mma atom——它就是 Hopper 的 `wgmma.mma_async.sync.aligned.m64n16k16` 那条 PTX,加上「32 lane 怎么分这 64×16 的输入/输出」的全部 layout 信息。
- `SM90_TMA_LOAD` 是一个 copy atom——它就是 TMA load 指令,加上「TMA descriptor 怎么编、哪个 thread 发起的」metadata。
- `Swizzle<3,3,3>` 是一个 swizzle atom——它**不是指令**,而是一个 layout 算子(把一个 base layout 改写成 swizzled 版本,防 bank conflict)。

**为什么叫 atom(原子)?** 因为你拿到 atom 后,理论上只剩「把这个 atom 复制几份、拼成更大的 tile」这一种组合——这是 3.6 的 `make_tiled_mma` / `make_tiled_copy`。**主 mainloop 写不出比 atom 更细的指令。**

#### 三种 atom:Gmem / Smem Layout / Smem Copy

源码里你看到的 atom 大致分成 3 类,**为什么这么分是 CUTLASS 框架级的命名规则**,不从硬件直觉就能推出:

| 类型 | 谁持有 | 含义 | mainloop 里典型名字 |
|---|---|---|---|
| `Copy_Atom<...>`(gmem copy)| 每边各一个 | **gmem 侧**的拷贝单元(TMA load / cp.async / ldmatrix.x4) | `GmemTiledCopyA`, `GmemTiledCopyB` |
| `SmemLayoutAtom` | 每边各一个 | **smem 上**数据「怎么排」——**这是 layout 描述符,不是指令!** | `SmemLayoutAtomA`, `SmemLayoutAtomB` |
| `Copy_Atom<...>`(smem sub-copy) | 每边各一个 | smem **内部**微调 copy 的 sub-atom;**几乎总是 `DefaultCopy` 空** | `SmemCopyAtomA`, `SmemCopyAtomB` |

一个典型 mainloop 里,**A 和 B 各自有上面 3 类**,加起来 8 个名字(A 侧 4 + B 侧 4):

```text
GmemTiledCopyA   SmemLayoutAtomA   SmemCopyAtomA   TransformA
GmemTiledCopyB   SmemLayoutAtomB   SmemCopyAtomB   TransformB
                       ↑ layout 描述    ↑ 通常空        ↑ layout 变换(可选)
```

注意几个微妙点:

- `GmemTiledCopy` 是「真指令」(`SM90_TMA_LOAD`),`SmemLayoutAtom` 是「layout 描述」(`composition(Swizzle, Layout<...>)`),`SmemCopyAtom` 是「真指令但通常是空」。三者**不是同一种东西**,只是都叫 "atom"。
- 主 mainloop 真正的 smem 写入者是 `GmemTiledCopy` + `SmemLayoutAtom`:**TMA load 按 `GmemTiledCopy` 的指令,把数据写进 `SmemLayoutAtom` 描述的 smem 位置**。`SmemCopyAtom` 在 sm90 上是空,因为 Hopper 直接走 TMA,不需要 smem 内部再 cp。
- `TransformA/B` 是可选的 layout 变换(identity / interleave / swizzle),严格说不是 atom,但放在同一组模板参数里。

Ch5 的 `CollectiveMma<...>` 模板参数恰好是这 8 个;Ch11 的 builder 推导会一个个把它们算出来。

#### 三层 shape 嵌套:AtomLayout / TileShape / ClusterShape

读者看到 Ch2.2 给的两个 `Shape<...>`:

```cpp
using TileShape    = Shape<_128, _128, _32>;   // BlockM × BlockN × BlockK
using ClusterShape = Shape<_4, _2, _1>;        // block 集群大小
```

和 Ch3.6(原 §3.5)的 `make_tiled_mma` 第二参数:

```cpp
make_tiled_mma(mma_atom, Layout<Shape<_2, _2, _1>>{});  // AtomLayout
```

是**同一族**东西——三种并行维度,各自决定不同层级上的事:

| 名字 | 形状 | 决定什么 | 谁定义 |
|---|---|---|---|
| **AtomLayout**(atom 内) | `Layout<Shape<_2, _2, _1>>{}` | **一个 atom 内部,thread 怎么分摊这条指令的输入/输出**——典型 `<_2,_2,_1>` 是「2×2×1」份 atom 拼成一个 warpgroup-level tile | `make_tiled_mma` 第二参数 |
| **TileShape**(CTA 内 / block 内) | `Shape<_128, _128, _32>` | **一个 CTA / threadblock 一次算多大块**——`(_M, _N, _K)` 是 BlockM × BlockN × BlockK | builder 顶层 user 给的 `TileShape` |
| **ClusterShape**(CTA 间 / cluster 内) | `Shape<_4, _2, _1>` | **几个 CTA 协作分担同一个 tile**——`<_4,_2,_1>` 是「4×2×1=8 个 CTA」分摊 M、N、K 维 | builder 顶层 user 给的 `ClusterShape` |

三者的关系是**嵌套**,不是并列:**atom 决定「一条指令多大」,TileShape 决定「一个 CTA 算多大」,ClusterShape 决定「几个 CTA 协作」**。你写 GEMM 时给后两个(顶层 `TileShape` + `ClusterShape`),builder 推第一个(`AtomLayout`);atom 自己要么是 user 写要么是 builder 从 dtype 推。

> 一句话总结:**atom 是「这条 mma/copy 指令长什么样」,AtomLayout 是「这条 atom 在 CTA 内复制几份」,TileShape 是「一个 CTA 负责多大块」,ClusterShape 是「几个 CTA 合作」。** 三层都按 (M, N, K) 维度分摊,但分摊的范围不同。

### 3.5 callout:`cute::gemm` 的 5-case dispatch(原始出处)

文件:`include/cute/algorithm/gemm.hpp` 的 5-case dispatch 表所在位置(原文):

```cpp
/** The gemm algorithm takes four (or three) tensors and computes

 *   D = A * B + C
 * It dispatches based on the number of modes each tensor has:
 *

 * 1. (V)        x (V)        => (V)        . The element-wise product of vectors. Dispatches to FMA or MMA.
 * 2. (M)        x (N)        => (M, N)     . The outer product of vectors.    Dispatches to [3] with new mode K=(1).
 * 3. (M, K)     x (N, K)     => (M, N)     . The product of matrices.         Dispatches to [5] with MMA vector-mode V.
 * 4. (V, M)     x (V, N)     => (V, M, N)  . The batched outer product of vectors.
 * 5. (V, M, K)  x (V, N, K)  => (V, M, N)  . The batched product of matrices.
 */
```

这是 `cute::gemm` 的语义总览——5 种 tensor 形态,5 种 dispatch 路径,具体哪条路径根据 dim / mode rank 选。最后都落到 `cute::gemm(...)` 的元编程调度。

**作用**:Ch5 里会出现 `cute::gemm(tiled_mma, A_frag_smem, B_frag_smem, acc)`。这一行的 dispatch 就是根据 `A / B / acc` 的 Layout rank(2D / 3D / V-mode)选具体走哪条 case。**真正的 mma 形态是 WGMMA 还是 cp.async.mma 还是 fp16 还是 fp8,都被吸进 `TiledMma` 内部**——`cute::gemm` 看到的是一致的 Layout 接口。

> **3-arg vs 4-arg**:`cute::gemm` 有两种签名,容易混。
>
> - **3-arg** `cute::gemm(A, B, C)` —— A / B / C 都是裸 layout,对应上面 5-case 里的纯 layout dispatch。3.8 的 `sgemm_1.cu` 就是这种。
> - **4-arg** `cute::gemm(TiledMma, A, B, C)` —— 第一个参数是 `TiledMma`(3.6),把"用哪条 mma 指令、thread 怎么分"也告诉 gemm。**mainloop 里实际见到的是这种**——因为 mma 形态 (WGMMA / cp.async.mma / fp8) 都封装在 TiledMma 里。
>
> 5-case dispatch 的注释对应 3-arg 形式;4-arg 形式把 mma 信息也吃了,内部仍走 5-case 但第一步是"按 TiledMma 把 A/B 摊到 thread 自己的 fragment"。

### 3.6 TiledMma / TiledCopy:从 atom 到 tile

```cpp
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/copy_atom.hpp>

// 单条 mma 指令的抽象
auto mma_atom = SM90_64x16x16_F32F16F16F32_SS;     // WGMMA 64x16x16 fp16→fp32, smem→smem
auto copy_atom_a = SM90_TMA_LOAD;                   // TMA load (gmem→smem)
auto copy_atom_b = SM90_TMA_LOAD;                   // 同上

// 把一个 mma atom 复制成"由线程分摊"的 TiledMma
auto TiledMma_ = make_tiled_mma(mma_atom,
                                Layout<Shape<_2, _2, _1>>{});  // thread 排布

// 同样,copy atom 也用 make_tiled_copy 复制
// 第二参数传 TiledMma_ 而非独立 Layout:从 mma 的 thread layout 派生,
// 保证"每 thread 拷进 smem 的区域"和"它之后 mma 读 smem 的区域"一致
auto TiledCopyA = make_tiled_copy(copy_atom_a,
                                  TiledMma_);                  // copy 配 mma
```

`make_tiled_mma` 的第二参数 `Layout<...>` 决定"这一个 mma atom 由 thread 怎么复制":如 `<_2, _2, _1>` 是"M 上 ×2、N 上 ×2、K 上 ×1"四份 atom,合起来覆盖一个完整 tile。`make_tiled_copy` 也能传一个独立 `Layout<...>`——但 mainloop 里几乎总是传 `TiledMma`(或单独算出来的"thread layout")让 copy / mma 对齐,这样写/读 smem 时 lane 看到的子区域天然一致,**避免 smem 读写的 lane mis-align**。

> **你手写 GEMM 的对照**:你写"2 个 warp × 2 个 warp group"分担 WGMMA 64x16x16 的输出。CuTe 的 `make_tiled_mma(mma_atom, Layout<_2,_2,_1>)` 做的就是这件事,只是**用模板而不是手写 thread 分配**。一旦你写好 `TiledMma`,后续 `cute::gemm(TiledMma, ...)` 拿到正确的 thread-fragment。

### 3.7 Swizzle(B, M, S):smem 防 bank conflict

```cpp
#include <cute/swizzle.hpp>

auto Sw = Swizzle<3, 3, 3>{};          // B=3, M=3, S=3 标准 swizzle
// 或 fp16 / bf16 的 smem layout 里更常见的 128B 对齐版:
auto Sw128 = Swizzle<2, 3, 3>{};       // 128B 对齐的 swizzle
```

**smem bank conflict** 是 smem 读写时同 warp 内不同 lane 命中同一 bank 的事。CuTe 的 Swizzle 是一个 layout transformation:把一个简单的 row-major layout 加一个**列方向的 XOR 置换**,让不同 lane 看到不同的 bank。

```cpp
// 原始 layout: row-major, 32×32,stride 32
auto L = Layout<Shape<_32, _32>, Stride<_32, _1>>{};

// 加上 swizzle:每 8 个 row 做一个"row index XOR"的偏移
auto L_swizzled = composition(Sw, L);   // 应用 swizzle
```

主 mainloop 里你看到的所有 `SmemLayoutAtomA = composition(Swizzle<...>, Layout<Shape<...>, Stride<...>>{})` 都是这一步。Ch5 详细讲。

### 3.8 CuTe by example:走读 `examples/cute/tutorial/` 4 个文件

到这里 CuTe 的语法你应该认得出来了。但若你还没跑过任何一个 cute 文件,在 Ch5 之前强烈建议读一遍这 4 个 `examples/cute/tutorial/` 文件,作为"cute 实战的最快入门"(每个文件都很短,核心代码几十行)。

> 这一节和 `media/docs/cpp/cute/0x_gemm_tutorial.md` 平行,边读边对照。

#### 文件 1:`sgemm_1.cu`——纯 cute GEMM,不经过 kernel::GemmUniversal

最朴素的 CUTLASS-by-cute GEMM。读 4 个块:

1. 建立 full A / B / C / D tensor
2. 在 K 维循环,每个 K 切片切出 `(BlockM, BlockK)`、`(BlockN, BlockK)` 子 tile,做 `cute::gemm` 累加到 C 子 tile
3. 最后把结果拷贝回 gmem
4. `cute::print` 调试

```cpp
auto A = make_tensor(A_ptr, Layout<Shape<M, K>, Stride<K, 1>>{});    // A: M x K row-major
auto B = make_tensor(B_ptr, Layout<Shape<N, K>, Stride<1, N>>{});    // B: N x K col-major
auto C = make_tensor(C_ptr, Layout<Shape<M, N>, Stride<N, 1>>{});    // C: M x N row-major
auto D = make_tensor(D_ptr, Layout<Shape<M, N>, Stride<N, 1>>{});

int num_blocks_m = ceil_div(M, BlockM);
int num_blocks_n = ceil_div(N, BlockN);
int num_blocks_k = ceil_div(K, BlockK);

for (int bm = 0; bm < num_blocks_m; ++bm)
for (int bn = 0; bn < num_blocks_n; ++bn)
for (int bk = 0; bk < num_blocks_k; ++bk) {
  auto a_blk = local_tile(A, make_shape(BlockM, BlockK), make_coord(bm * BlockM, bk * BlockK));
  auto b_blk = local_tile(B, make_shape(BlockN, BlockK), make_coord(bn * BlockN, bk * BlockK));
  auto c_blk = local_tile(C, make_shape(BlockM, BlockN), make_coord(bm * BlockM, bn * BlockN));

  cute::gemm(a_blk, b_blk, c_blk);  // 调用 case 3 (M, K) x (N, K) => (M, N)
}

// 读 + gemm 之后:
cute::copy(C, D);
```

> **你手写 GEMM 的对照**:你的 kernel 入口就是这个结构——`for block_m ... for block_n ... for block_k ...`。CUTLASS 的"5 层抽象"是把这段循环**平铺**到 global CTA grid + warpgroup + thread。

#### 文件 2:`sgemm_2.cu`——加 TiledCopy + TiledMma

`sgemm_1` 是用 `cute::gemm` 直接做外积。`sgemm_2` 引入 `TiledCopy`(加载)和 `TiledMma`(计算),**模拟** 一个真实 mainloop:

```cpp
auto TiledCopyA = make_tiled_copy(SM80_CP_ASYNC_16x8x8_F16F16,     // Ampere cp.async
                                  Layout<Shape<_32, _8>>{});        // 32 threads × 8 lane
auto TiledMma   = make_tiled_mma(SM80_16x8x16_F16F16F16F16_F32,    // Ampere mma
                                  Layout<Shape<_2, _2, _1>>{});     // 2x2x1 thread

auto thr_copy_A = TiledCopyA.get_thread_slice(threadIdx.x);
auto thr_mma    = TiledMma.get_thread_slice(threadIdx.x);

auto tAgA = thr_copy_A.partition_S(A);     // thread 看到的 A 切片
auto tBgB = thr_copy_A.partition_S(B);
auto tCgC = thr_mma.partition_S(C);
...
```

这接近 CUTLASS mainloop 内部的样子:每个 thread 拿自己的 subview,负责一部分 copy 和 mma。

#### 文件 3:`hopper/wgmma_sm90.cu`——WGMMA + CuTe atom

只引入 Hopper WGMMA,不引入 TMA。`make_tiled_mma(SM90_64xNxK_F32F16F16F32_SS, Layout<...>)` 把一条 WGMMA 指令扩展成 CTA-wide tile。

#### 文件 4:`hopper/wgmma_tma_sm90.cu`——加 TMA 加载

最后引入 `SM90_TMA_LOAD` + `make_tiled_copy(TMA_LOAD, ...)`。这基本就是 `sm90_mma_tma_gmma_ss_warpspecialized.hpp` 内部的简化版本。

> **这 4 个文件是 Ch5 的预习**。读完这 4 个 + 上面 3.1–3.7 的语法,你再看 `sm90_mma_tma_gmma_ss_warpspecialized.hpp` 会觉得"每行都认得"。

### 3.9 章末:读完这一章你该做得到的事

- ✅ 在 Ch5 的 mainloop 代码里认出 `Shape<_M, _N>`、`make_tensor(...)`、`TiledMma`、`SmemLayoutAtomA` 这些名字各自是 CuTe 的哪个组合子。
- ✅ 读得懂 `tile_to_shape(SwizzleAtom, make_shape(...), ...)` 这种"在 mainloop 里到处出现"的复杂表达式。
- ✅ 看 `cute::gemm(tiled_mma, ...)` 时知道它在走 5-case dispatch。
- ✅ 用 `cute::print(A_tensor)` 调试 layout 时知道是在按 mainloop 的视图 print。

CuTe 不需要精通——**足够认得出来**就够读 Ch5。

### 3.10 `partition_S` / `partition_D`:把 tile 切成 thread 视图

Ch3.8 的 `sgemm_2.cu` 示例里有一行:

```cpp
auto tAgA = thr_copy_A.partition_S(A);     // thread 看到的 A 切片
auto tBgB = thr_copy_A.partition_S(B);
auto tCgC = thr_mma.partition_S(C);
```

这是 CuTe **最关键的 thread 分工原语**——mainloop 里每个 thread 拿到自己的 subview,靠的就是它。

#### 语义

`partition_S(Tensor)` = "把这个 Tensor 按我所在的 thread / warp 的位置切出我自己的一片"。

具体说,`thr_mma` 是一个 `ThrMma`(从 `TiledMma.get_thread_slice(threadIdx.x)` 得到),它知道自己"在 TiledMma 整个布局里的什么位置"——`partition_S(C)` 就把这个位置反推到 source Tensor C 上,**返回的 tensor 形状是本 thread 实际要读的子区域**。

```text
TiledMma                    Source Tensor C
  ┌─────┬─────┐                ┌─────────┐
  │Thr 0│Thr 1│   partition_S  │         │
  ├─────┼─────┤   ─────────►   │         │
  │Thr 2│Thr 3│                │         │
  └─────┴─────┘                └─────────┘
                              ↑ 每个 thread 拿自己那片
```

类似的 `partition_D(Tensor)` 是"切出 thread 自己作为 destination 的那片"——`tAgA = thr_copy.partition_S(A)`(source)而 `tAsA = thr_copy.partition_D(A_smem)`(destination,smem 上的目标视图)。

#### 为什么这套 API 这么重要

1. **layout 自动推导**——你不用手算 "thread `lane_id` 应该从 smem 的哪个位置读"。CuTe 看 `ThrMma` 内部的 layout 帮你算完。
2. **copy / mma 对齐**——`thr_copy_A.partition_S(A)` 切出的源 layout 和 `thr_mma.partition_S(A_smem)` 切出的 smem 视图在内存里是**一一对齐的**——`cute::copy(tiled_copy_A, tAgA, tAsA)` 就是把 lane 视角的 A 拷到 lane 视角的 smem A,不需要任何 thread 间的数据交换。
3. **swizzle 兼容**——smem 上的 `tAsA` 已经把 swizzle 算进去了,partition_D 返回的 layout 里 swizzle 是隐式的(在 smem_ptr engine 上自动应用)。

#### `local_partition`:跨 thread/warp 切

`partition_S` 是单 thread 视图;`local_partition` 是按"thread group"(warp 或 warpgroup)切:

```cpp
// 把整个 tensor 按 warp 切,本 warp 拿一片
auto warp_smem = local_partition(smem_layout, warp_idx, Layout<Shape<_4, _1>>{});
//   ↑ 在 (M, N) 维里,4 个 warp 沿 M 各拿 1/4
```

`partition_S` 是 `local_partition` 的特例(当 thread group 大小 = 1 时)。mainloop 里**绝大多数**用法是 `partition_S`,因为每个 thread 自己处理自己的 lane fragment。

### 3.11 `cute::copy`:copy 算法 dispatch

`cute::copy` 是 `cute::gemm` 的"孪生兄弟"——dispatch 形态不同:

```text
cute::copy(src, dst): 把 src 按 src layout 拷到 dst layout 的位置
cute::copy(tiled_copy, src, dst): 用 TiledCopy 描述的 copy atom,所有 lane 并行拷
```

#### 单 tensor 版(cute::copy(src, dst))

```cpp
auto A = make_tensor(A_ptr, Layout<Shape<_128, _128>, Stride<_128, _1>>{});  // gmem row-major
auto A_smem_view = make_tensor(smem_ptr, smem_layout);                        // smem swizzled
cute::copy(A, A_smem_view);   // 等价于 memcpy
```

**不推荐**——这种用法 lane 间不做并行,实际上就是 lane 0 拷全部。性能差。

#### TiledCopy 版(cute::copy(tiled_copy, src, dst))

```cpp
auto TiledCopyA = make_tiled_copy(SM90_TMA_LOAD, TiledMma_);
auto thr_copy_A = TiledCopyA.get_thread_slice(threadIdx.x);
auto tAgA = thr_copy_A.partition_S(A);       // gmem 视图
auto tAsA = thr_copy_A.partition_D(sA);      // smem 视图

// 每个 lane 用 SM90_TMA_LOAD 的一条指令拷自己那一片
cute::copy(TiledCopyA, tAgA, tAsA);
//   ↑ 编译期展开成 lane-parallel copy;每 lane 一条 LDG / cp.async / TMA
```

**这是 mainloop 写法**。`TiledCopy` 内部的 `Copy_Atom` 决定具体指令:
- `SM90_TMA_LOAD` → 一条 TMA descriptor load
- `SM80_CP_ASYNC_16x8x8_F16F16` → cp.async
- `SM75_U16x8_LD_T` → 普通 LDG

切哪个 `Copy_Atom` 由 builder / dispatch policy 决定。

### 3.12 engine 类型深讲(gmem / smem / rmem)

`make_gmem_ptr` / `make_smem_ptr` / `make_rmem_ptr` 这三种"指针包装"看起来差不多,但**走不同 layout 变换路径**——这是 mainloop 跑对的根本。

#### gmem_ptr:朴素指针

```cpp
auto gptr = make_gmem_ptr(raw_ptr);   // type tag = "gmem"
auto A = make_tensor(gptr, Layout<M, N>{});
//   ↑ gmem ptr 上的 layout:index 就是字节偏移 / sizeof(elem)
```

layout 算出来的 index 就是从 raw_ptr 出发的字节 offset,**没有修饰**。

#### smem_ptr:swizzle-aware

```cpp
auto sptr = make_smem_ptr(smem_addr);   // type tag = "smem"
auto A_smem = make_tensor(sptr, composition(Swizzle<3,3,3>{}, Layout<...>{}));
//   ↑ smem ptr 上的 Swizzle 会被实际应用到访问模式
```

**关键差异**:当你把 `A_smem` 的 layout 里含有 Swizzle 时,cute::copy / cute::gemm 会按 Swizzle **真的改写访问的 smem 地址**——同一个物理 smem buffer,Swizzle 让不同 lane 看到不同的偏移,避免 bank conflict。如果用 `make_gmem_ptr` + 同样的 layout,Swizzle 不会被解释——它只是 layout 的一个抽象,没有"落到真实访问"这一步。

#### rmem_ptr:寄存器视图

```cpp
auto rptr = make_rmem_ptr(register_ptr);   // type tag = "rmem"
auto frag = make_tensor(rptr, ...layout...);
//   ↑ layout 直接决定 lane 拿到哪些寄存器分量
```

`TiledMma` 里 partition_S(C) 返回的 tensor 用 `make_rmem_ptr` 包装——layout 描述了"32 lane 各拿哪个 fragment 的哪个寄存器"。`cute::gemm(tiled_mma, ...)` 看到的是 rmem tensor,直接展开成 mma 指令。

> **你手写 GEMM 的对照**:你写"lane_id 算 smem 偏移时考虑 swizzle"、"lane_id 算 fragment 寄存器"——CUTLASS 把这两步**隐式包进 smem_ptr / rmem_ptr 的 type tag 里**。所以 mainloop 代码完全没出现手动算偏移。

#### 选错会怎样

```cpp
// 错误:smem buffer + gmem_ptr → swizzle 被吞掉,跑出来 bank conflict 严重
auto A_smem = make_tensor(make_gmem_ptr(smem_addr), swizzled_layout);

// 正确:smem buffer + smem_ptr + swizzled layout
auto A_smem = make_tensor(make_smem_ptr(smem_addr), swizzled_layout);
```

这条 mainloop 代码里几乎**不会写错**——builder 帮你生成的 `SmemLayoutA` 都用 `make_smem_ptr` 包装,你直接拿来用就行。但如果你在 smem 上做 smem-to-smem copy(比如把 data 从一个 swizzled layout 拷到另一个),记得用 smem_ptr engine。

### 3.13 layout 函数 zoo:zipped_divide / blocked_product / flat_product

除 `composition` / `coalesce` / `tile_to_shape` 之外,mainloop 还常用这 3 个 layout 函数,这里一并讲清。

#### `zipped_divide`:把高维 layout 拆成"高维 × 低维"

```cpp
// 假设 layout 是 (M, N, K) = (16, 16, 8)
auto L = Layout<Shape<_16, _16, _8>, Stride<...>>{};

// 沿 K 维 split 成 "16*16 个 K-tile"
auto L_zipped = zipped_divide(L, make_shape(_1, _1, _8));
//   → shape: (16, 16, 8)   (K 维被保留为独立 mode)
//   → 解读:第一个 (16, 16) 是 outer,后面的 8 是 inner
```

`zipped_divide` 比 `tile_to_shape` 更"对称"——保留所有 mode,只把一个 mode 拆成"outer × inner"两段。

#### `blocked_product`:两个 layout 的"块化笛卡尔积"

```cpp
auto A = Layout<Shape<_128, _128>, Stride<_128, _1>>{};        // 128x128 row-major
auto B = Layout<Shape<_8, _8>, Stride<_8, _1>>{};             // 8x8 row-major tile

auto P = blocked_product(A, B);   // shape = (128/8, 128/8, 8, 8) = (16, 16, 8, 8)
//   ↑ 16x16 个 8x8 tile 排成 128x128 大矩阵
```

跟 `tile_to_shape` 区别:`tile_to_shape` 把 tile 摊到大矩阵上,`blocked_product` 同时保留 inner / outer 两套 stride 信息。

#### `flat_product`:outer × inner 但 inner 摊平

```cpp
auto P = flat_product(A, B);     // shape = (128/8, 128/8, 64)
//   ↑ inner 的 8x8 = 64 个元素被摊平成 1 维
```

`flat_product` 用于"内层按 1 维连续访问"的场景——典型是把 tile 的所有元素看作 1 个连续的 inner 块。

#### 这 3 个跟 `composition` / `tile_to_shape` 的关系

| 函数 | 干什么 | 典型用法 |
|---|---|---|
| `composition(A, B)` | "先 B 后 A" 的函数复合 | swizzle composition |
| `coalesce(L)` | 合并相邻同 shape mode | compiler 友好 |
| `tile_to_shape(T, shape, step)` | 把 tile T 摊到 shape 大矩阵 | smem layout |
| `zipped_divide(L, factor)` | 沿某个 mode split outer/inner | mainloop K 维 split |
| `blocked_product(A, B)` | 两 layout 笛卡尔积 + 块化 | wgmma tile 嵌套 |
| `flat_product(A, B)` | blocked 但 inner 摊平 | 1 维连续访问 |

mainloop 里你看到这些函数,知道它们在表达什么就行——具体怎么组合是 builder / atom 的事。

### 3.14 章末(扩充):读完这一章你该做得到的事

- ✅ 在 Ch5 的 mainloop 代码里认出 `Shape<_M, _N>`、`make_tensor(...)`、`TiledMma`、`SmemLayoutAtomA` 这些名字各自是 CuTe 的哪个组合子。
- ✅ 读得懂 `tile_to_shape(SwizzleAtom, make_shape(...), ...)` 这种"在 mainloop 里到处出现"的复杂表达式。
- ✅ 看 `cute::gemm(tiled_mma, ...)` 时知道它在走 5-case dispatch。
- ✅ 看 `cute::copy(tiled_copy, ...)` 时知道它在走 TMA / cp.async / LDG 之一(由 atom 决定)。
- ✅ 看到 `partition_S` / `partition_D` 知道是"切出本 thread 的子 view"——`cute::copy` 用 src / dst 配对保证 lane 间对齐。
- ✅ 区分 gmem_ptr / smem_ptr / rmem_ptr 在 layout 解释上的差异(smem_ptr 应用 swizzle,rmem_ptr 决定 fragment 寄存器)。
- ✅ 用 `cute::print(A_tensor)` 调试 layout 时知道是在按 mainloop 的视图 print。

CuTe 不需要精通——**足够认得出来**就够读 Ch5。

---

