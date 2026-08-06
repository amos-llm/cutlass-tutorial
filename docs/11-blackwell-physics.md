## 第 11 章:Blackwell 物理基础——TMEM + UMMA + CLC

Ch10 讲了 Blackwell 桥接的**心智模型**——5 件不变 + 5 件变。这章讲"具体换了的硬件原子"——`TMEM` 累加器、`UMMA` 指令族、`CLC` 同步原语。这些是 sm100 mainloop 内部用、**不对外暴露到 5 层公共接口**的硬件细节。

主文件:
- `include/cute/arch/tmem_allocator_sm100.hpp` — `cute::TMEM::Allocator1Sm` / `Allocator2Sm`
- `include/cute/arch/mma_sm100_umma.hpp` — `SM100_MMA_*` 原子
- `include/cute/arch/mma_sm100_desc.hpp` — `cute::UMMA::Major` enum + `make_umma_desc`
- `include/cutlass/gemm/kernel/sm100_tile_scheduler.hpp` — CLC PTX 内联

### 本章涉及 CUTLASS 源文件

- `include/cute/arch/tmem_allocator_sm100.hpp:60/117` — `cute::TMEM::Allocator1Sm` / `Allocator2Sm`(`allocate(num_columns, dst_ptr)`)
- `include/cute/arch/mma_sm100_umma.hpp:46/86/127/171/214/258/299/341/386/430/471/512/552/593` — `SM100_MMA_*` 14 个原子族(FP16 / TF32 / Sparse / Scaled / 2x1SM 等变体)
- `include/cute/arch/mma_sm100_desc.hpp:62/67/72` — `UMMA::Major` enum(K / MN) + `ScaleIn` + `ScaleOut`
- `include/cute/arch/mma_sm100.hpp` — MMA traits + atom wrappers
- `include/cutlass/gemm/kernel/sm100_tile_scheduler.hpp:400/421/423` — CLC PTX 内联(`clusterlaunchcontrol.try_cancel.*` / `query_cancel.*`)
- `include/cutlass/pipeline/sm100_pipeline.hpp:935` — `PipelineCLCFetchAsync<Stages, ClusterShape>`(被 §7.5.4 动态调度器用)

### 11.0 本章导航

```text
§11.1  WGMMA → UMMA:tcgen05.mma 指令族             ← 主 mma 指令变化
§11.2  smem 部分结果 → TMEM(Tensor Memory)累加器    ← 累加器从 smem 搬到专用 memory
§11.3  Cluster Launch Control(CLC)新原语           ← sm100 同步新机制
§11.4  章末:读完这一章你该做得到的事                ← 自检 checklist
```

读法建议:这一章**只读一遍**,作为 Ch10 心智模型的"物理支撑"。不用背细节——需要时回查即可。

### 11.1 WGMMA → UMMA:`tcgen05.mma` 指令族

#### WGMMA(Hopper)与 UMMA(Blackwell)的对比

- **WGMMA**(sm_90):`wgmma.mma_async.sync.aligned.m64n...k16...` PTX 指令,如 `SM90_64x16x16_F16F16F16F32_SS`;初始累加器 `(0.0)` 由调用方显式传入。
- **UMMA**(sm_100):`tcgen05.mma` 系列;**异步 tensor 协程**——你写 launch 指令,GPU 自己决定何时启动;允许多 CTA 在 cluster 内共享时打平 bandwidth(`tcgen05.mma` 自身是 tensor 指令,不是 scalar 协程)。

#### UMMA 指令族(7 条,见 `media/docs/cpp/blackwell_functionality.md`)

| 指令 | 等价 Hopper | 用途 |
|---|---|---|
| `tcgen05.mma` | `wgmma.mma_async.sync.aligned` | 主 mma |
| `tcgen05.feedthrough` | n/a | TMA feedthrough(C 通过 TMA 直接进 mma,无需 smem)|
| `tcgen05.sp` | n/a | Sparse(2:4 结构化稀疏)|
| 等等(7 条全列在该 doc) |||

CUTLASS 在 `include/cute/atom/mma_traits_sm100.hpp:58-125` 里定义了 `namespace cute::UMMA` 的 layout atom,加 `mma_traits_sm100_frag.hpp` 提供 fragment 形状。

#### 关键 API

```cpp
// 真实接口(include/cute/arch/mma_sm100_umma.hpp:46+)
//   struct SM100_MMA_F16BF16_SS : MMA_Atom<MMA_Traits>;  // 标准 mma 原子
//   struct SM100_MMA_F16BF16_SS_SCALED;                  // 带 block-scale 的 mma
//   struct SM100_MMA_F16BF16_2x1SM_SS;                    // 2-SM 协作(巨型 tile)
// ...

// Major enum(include/cute/arch/mma_sm100_desc.hpp:62-65)
namespace UMMA {
  enum class Major : uint8_t {
    K  = 0,    // K-major
    MN = 1     // M/N-major
  };
  // enum class ScaleIn, ScaleOut 也在这里
}
```

builder 通过 `cute::UMMA::Major` + `tag_to_umma_major_A/B<GmemLayoutATag>()`(`include/cutlass/gemm/collective/builders/sm1xx_common.inl:95/117`) 推 layout atom。

### 11.2 smem 部分结果 → TMEM(Tensor Memory)累加器

UMMA 的 accumulation 默认写到 **TMEM**——一种 sm_100 新加的专用 memory,只对 mma 累加器访问友好。

#### 真实 API(见 `include/cute/arch/tmem_allocator_sm100.hpp`)

```cpp
namespace cute::TMEM {
  // 1Sm(单 SM 协作)和 2Sm(双 SM 协作)两个变体
  class Allocator1Sm {  // line 60
  public:
    static constexpr int ColumnsPerAllocationSlice = 32;
    static constexpr int Sm100Tmemusage 容量Columns = 512;
    __device__ Allocator1Sm() { }

    // 申请 num_columns 个 column 的 TMEM(必须是 32 的倍数, ≤ 512)
    CUTE_HOST_DEVICE void
    allocate(int num_columns, uint32_t* dst_ptr) {
      // dst_ptr 是 smem,写的是 tmem 指针
      // 必须由 1 个 fully active warp 调用,且后续 allocation 都由同一 warp
    }
  };

  class Allocator2Sm { /* line 117,双 SM 协作 */ };
}
```

#### 使用流程

```cpp
// 1. CTA 内单 warp 调 allocate() 申请 TMEM column 区间
//    allocate(int num_columns, uint32_t* dst_ptr);

// 2. mma 原子 (SM100_MMA_* in mma_sm100_umma.hpp) 自动把 accumulator 写到 TMEM
//    - mma 完成后,accumulator 就在 TMEM 里

// 3. epilogue 阶段从 TMEM 读 → 写回 gmem
//    - 通过 cute::TMEM::load(tmem_ptr, ...) 把 TMEM 数据搬到 smem
//    - epilogue 的 write pipeline 再 store 到 gmem
```

#### Allocator1Sm vs Allocator2Sm

| 变体 | 用途 | 何时用 |
|---|---|---|
| `Allocator1Sm` | 单 SM 协作 | 1 个 CTA 算整个 tile |
| `Allocator2Sm` | 双 SM 协作 | 2 个 CTA 协作算巨型 tile(配合 `KernelTmaWarpSpecialized2SmSm100` 调度)|

> 你写 Hopper 时**没有**这个 TMEM 抽象——smem 是个通用 buffer。CUTLASS 把 TMEM **隔离**到 sm100 mainloop 内部,user-facing API 不变(`cute::TMEM::Allocator1Sm/2Sm` 仅 mainloop 内部用,不对外暴露)。

### 11.3 Cluster Launch Control(CLC)新原语

Blackwell sm100 引入了一个 Hopper 没有的同步原语:**cluster launch control**。它让 cluster 内某个 CTA 可以"等 cluster 内所有 CTA 完成某个 phase 后,再决定下一步走哪个 path"——这是动态调度器(`DynamicPersistentScheduler`,Ch7 §7.5.4)能存在的前提。

#### 为什么需要 CLC

Hopper 上 `PersistentTileSchedulerSm90` 是**静态**的:`fetch_next_work` 在编译期决定下一个 tile 是什么;CTA 之间的"我要做什么"是 grid launch 之前就分配好的。

Blackwell 上需要**动态**派工:按 cluster 整组同步地决定下一步——比如某 CTA 算完了,它想知道"cluster 内其他 CTA 是不是也算完了这个 stage",再决定"我现在该 fetch 哪个 tile"。**这必须有一个 cluster-wide barrier**。

#### CLC 指令族(实际见 `cutlass/gemm/kernel/sm100_tile_scheduler.hpp:400-423`)

| 指令(实际 PTX)|作用|对比 Hopper|
|---|---|---|
|`clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.multicast::cluster::all.b128`|尝试"取消"一次 cluster-wide 等待并 multicast 一段数据|无 / Hopper 走 `cute::cluster_arrive + cluster_wait`|
|`clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_result`|query 当前 cluster 是否被取消(结果写 pred 寄存器)|无|
|`clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 {...}, clc_result`|取被取消那一次 cluster 第一个 CTA 的 id|无|

#### 在 CollectiveMainloop 里的位置

CLC 接入点在 `load` 末尾和 `mma` 末尾——`load` 末尾可以做一次 multicast-with-c-cancel,允许"这个 CTA 写这一段 smem,如果 cluster 内其他 CTA 还没准备好,就把这个 multicast 撤销"。

这是 sm100 **独有的优化机制**:Hopper 走"全部 multicast + 等所有 CTA ready"路径,sm100 走"按需 multicast + 取消"路径,后者在 cluster 形状不规则时更省 bandwidth。

#### Pipeline 集成

`PipelineCLCFetchAsync<Stages, ClusterShape>`(`include/cutlass/pipeline/sm100_pipeline.hpp:935`)是 sm100 调度器专用 pipeline——它包装 CLC,让 `fetch_next_work` 在 stage 满时走 CLC 路径,空时走默认路径。这就是 Ch3 §3.2 提到的"Sm100 调度器的 CLC pipeline"。

#### 一句话总结

> **CLC = cluster-wide 的"我准备好了吗?"应答机制**。Hopper 走 `cluster_arrive + cluster_wait` 隐式同步;sm100 走 CLC 显式应答。动态调度器只能用 CLC,静态调度器两者都能用。

### 11.4 章末:读完这一章你该做得到的事

- ✅ 看到 `tcgen05.mma` 指令知道这是 sm100 上的主 mma 指令,跟 Hopper `wgmma.mma_async.sync.aligned` 等价但底层是 tensor 协程。
- ✅ 知道 UMMA 7 条指令族的大致用途(`tcgen05.mma` / `.feedthrough` / `.sp` 等),具体见 `media/docs/cpp/blackwell_functionality.md`。
- ✅ 区分 sm100 累加器从 smem → **TMEM**(tensor memory,只对 mma 累加器友好的专用 memory),知道这是 sm100 重新设计 ISA 的核心物理基础。
- ✅ 区分 sm100 上 `tmem_allocator_sm100.hpp` 是 **内部**,5 层公共接口不外露——mainloop 走 union shared storage 这件事没变。
- ✅ 理解 CLC 是 sm100 独有机制,让 `DynamicPersistentScheduler` 存在——静态调度器不需要 CLC,动态调度器必须用 CLC。
- ✅ 知道 `PipelineCLCFetchAsync<Stages, ClusterShape>` 是 sm100 调度器专用 pipeline 包装,在 `sm100_pipeline.hpp:935` 里。
- ✅ 知道 sm100 上 `Allocator1Sm` (单 SM) 和 `Allocator2Sm` (双 SM) 的区别,跟 `KernelTmaWarpSpecialized1SmSm100` / `_2SmSm100` 调度器对应。

---