## 第 3 章:CuTe——CUTLASS 真正的"语言"

你可能认为 CuTe 是装饰用的工具库。其实不是——它是 CUTLASS **写 mainloop / epilogue 的语言**。Cutlass 文件里那一串 `cute::Shape` / `cute::Tile` / `cute::make_tensor` 不是装饰,是 mainloop 在描述"我这一 stage 的 smem 上 A 是什么样的"。

读 Ch4 之前,你需要 CuTe 的"语法熟悉"。这一章不讲 CuTe 怎么写 sgemm(那是 `media/docs/cpp/cute/0x_gemm_tutorial.md` 的事),只讲所有 CUTLASS mainloop **一定会用到**的 4 个核心抽象:`Shape` / `Stride` / `Layout` / `Tensor`,以及 6 个常用组合子:`make_shape` / `make_layout` / `make_tensor` / `composition` / `coalesce` / `tile_to_shape`。

读完之后,你应该能在 `sm90_mma_tma_gmma_ss_warpspecialized.hpp` 里把 `SmemLayoutAtomA` / `GmemTiledCopyA` / `TiledMma` 这些 `cute::` 类型认出来、能理解它们在 mainloop 里在做什么。

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
using Shape<_M, _N, _K>    = cute::tuple<Int<M>, Int<N>, Int<K>>;
using MyShape              = Shape<_128, _128, _32>;     // 编译期 128×128×32
using DynamicShape         = Shape<int, int, int>;       // 运行期 3-tuple
using MixedShape           = Shape<_128, _128, int>;     // 1+2 前两者编译期
```

`_N` 是 `Int<N>` 的 alias,语义是"一个值为 N 的类型"。`Shape` 是 `tuple<...>`,所以你只能 `get&lt;i&gt;` 访问,CUTLASS 大量这种"tuple of compile-time ints"模式,用模板元编程做下标计算。

> 你写 `TileShape = Shape<_128, _128, _32>`,CUDA 编译器会**完全**把这些数字 bake 进代码——没有运行时开销。

#### `Stride<a, b, c, ...>` 与 `_N`

```cpp
using RowMajor    = Stride<_N, _1>;     // (i, j) → i*N + j
using ColMajor    = Stride<_1, _M>;     // (i, j) → i + j*M
```

组合 `Shape<_M, _N> + Stride<_N, _1>` 就是 `(i,j) → i*N + j`——行主序。

#### `Layout`

```cpp
template <class Shape, class Stride = LayoutLeft::Apply<Shape>>
struct Layout : private cute::tuple<Shape, Stride> { ... };

using A_Layout = Layout<Shape<_4, _8>, Stride<_8, _1>>;
```

`LayoutLeft::Apply<Shape>` 是默认 stride:行主序。`LayoutRight::Apply<Shape>` 是列主序。直接写 row-major / col-major 不需要你填 Stride。

`Layout` 是一个轻量包装——>内部就是一个 pair(shape, stride) tuple,所有功能通过 free function 提供。

#### `Tensor = Pointer + Layout`

```cpp
#include <cute/tensor.hpp>

// 假设有指针
ElementA* A_ptr = ...;

auto A = make_tensor(make_gmem_ptr(A_ptr),
                     make_layout(make_shape(M, N),     // 运行时尺寸
                                 make_stride(N, 1)));   // RowMajor

// 把上面合成(常见):
auto A = make_tensor(A_ptr,                      // 原始指针(自动包为 gmem_ptr)
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

这三种 engine 唯一区别是 `Tensor` 自己做下标计算时,会按位置(memory space)做不同的"一致化假设"——比如 smem 上的指针在 swizzle 时会做 swizzle 修正。

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

Layout 就是个映射函数;`composition(A, B)` = "先按 B 映射,再按 A 映射"——这相当于数学的 `f ∘ g`,后到先应用。

```cpp
auto L1 = make_layout(make_shape(_8, _8), make_stride(_8, _1));  // row-major 8x8
auto L2 = make_layout(make_shape(_2, _2), make_stride(_2, _8));  // block of 2x2

auto L = composition(L1, L2);
// L 把一个 2×2 tile 排进 8×8 大矩阵的某个 block 位置
// 这是 mainloop 里经常出现的"tile of tile"组合
```

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

mainloop 里 `SmemLayoutAtomA = tile_to_shape(SwizzleAtom, make_shape(...block_M, ...block_K, Stages), ...)` 就是这个模式:smem 上 A 是 SwizzleAtom 的 layout,再 tile 到 "`BlockM × BlockK × Stages`"的 3D 上。Ch4 会具体讲到。

#### 其他常用(略讲,你会在 Ch4/Ch6 看到)

- `local_tile`:从"大矩阵 + tile 大小 + 起始坐标"切成子 tile。Mainloop 里获取"本 CTA 当前 stage 的 A":

  ```cpp
  auto A_smem_tile = local_tile(A_smem_layout, /*coord=*/make_coord(block_m_coord, k_coord, stage), /*proj=*/...);
  ```

- `slice`:把 `Tile (M_tile, N_tile, K_tile, ...)` 切成"由 thread / warp 看的子 view"。
- `make_tiled_mma` / `make_tiled_copy`:把一个 mma atom / copy atom 复制成 CTA / warp / thread 分工——见 3.5。

### 3.4 callout:`cute::gemm` 的 5-case dispatch(原始出处)

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

**作用**:Ch4 里会出现 `cute::gemm(tiled_mma, A_frag_smem, B_frag_smem, acc)`。这一行的 dispatch 就是根据 `A / B / acc` 的 Layout rank(2D / 3D / V-mode)选具体走哪条 case。**真正的 mma 形态是 WGMMA 还是 cp.async.mma 还是 fp16 还是 fp8,都被吸进 `TiledMma` 内部**——`cute::gemm` 看到的是一致的 Layout 接口。

### 3.5 TiledMma / TiledCopy:从 atom 到 tile

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
auto TiledCopyA = make_tiled_copy(copy_atom_a,
                                  TiledMma_);                  // copy 配 mma
```

`make_tiled_mma` 的第二参数 `Layout<...>` 决定"这一个 mma atom 由 thread 怎么复制":如 `<_2, _2, _1>` 是"M 上 ×2、N 上 ×2、K 上 ×1"四份 atom,合起来覆盖一个完整 tile。

> **你手写 GEMM 的对照**:你写"2 个 warp × 2 个 warp group"分担 WGMMA 64x16x16 的输出。CuTe 的 `make_tiled_mma(mma_atom, Layout<_2,_2,_1>)` 做的就是这件事,只是**用模板而不是手写 thread 分配**。一旦你写好 `TiledMma`,后续 `cute::gemm(TiledMma, ...)` 拿到正确的 thread-fragment。

### 3.6 Swizzle(B, M, S):smem 防 bank conflict

```cpp
#include <cute/swizzle.hpp>

auto swizzled = Swizzle<3, 3, 3>{};    // B=3, M=3, S=3 swizzle
// 或朱子用得多的:
auto swizzled128 = Swizzle<2, 3, 3>{};   // 128B 对齐的 swizzle
```

`sbank conflict` 是 smem 读写时同 warp 内不同 lane 命中同一 bank 的事。CuTe 的 Swizzle 是一个 layout transformation:把一个简单的 row-major layout 加一个**列方向的 XOR 置换**,让不同 lane 看到不同的 bank。

```cpp
// 原始 layout: row-major, 32×32,stride 32
auto L = Layout<Shape<_32, _32>, Stride<_32, _1>>{};

// 加上 swizzle:每 8 个 row 做一个"row index XOR"的偏移
auto Sw = Swizzle<3, 3, 3>{};
auto L_swizzled = composition(Sw, L);   // 应用 swizzle
```

主 mainloop 里你看到的所有 `SmemLayoutAtomA = composition(Swizzle<...>, Layout<Shape<...>, Stride<...>>{})` 都是这一步。Ch4 详细讲。

### 3.7 Ch3.5 — CuTe by example:走读 `examples/cute/tutorial/` 4 个文件

到这里 CuTe 的语法你应该认得出来了。但若你还没跑过任何一个 cute 文件,在 Ch4 之前强烈建议读一遍这 4 个 `examples/cute/tutorial/` 文件,作为"cute 实战的最快入门"(每个文件都很短,核心代码几十行)。

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

> **这 4 个文件是 Ch4 的预习**。读完这 4 个 + 上面 3.1–3.6 的语法,你再看 `sm90_mma_tma_gmma_ss_warpspecialized.hpp` 会觉得"每行都认得"。

### 3.8 章末:读完这一章你该做得到的事

- ✅ 在 Ch4 的 mainloop 代码里认出 `Shape<_M, _N>`、`make_tensor(...)`、`TiledMma`、`SmemLayoutAtomA` 这些名字各自是 CuTe 的哪个组合子。
- ✅ 读得懂 `tile_to_shape(SwizzleAtom, make_shape(...), ...)` 这种"在 mainloop 里到处出现"的复杂表达式。
- ✅ 看 `cute::gemm(tiled_mma, ...)` 时知道它在走 5-case dispatch。
- ✅ 用 `cute::print(A_tensor)` 调试 layout 时知道是在按 mainloop 的视图 print。

CuTe 不需要精通——**足够认得出来**就够读 Ch4。

---

