## 第 8 章:DispatchPolicy——tag-inheritance 模式

这是 CUTLASS 3.x 的**架构精髓**——也是 `media/docs/cpp/` 几乎完全没有覆盖、本教程必须讲的东西。

文件:`include/cutlass/gemm/dispatch_policy.hpp`(覆盖 sm70 / sm80 / sm90 / sm100 / sm120 所有 dispatch tag)。

### 本章涉及 CUTLASS 源文件

- `include/cutlass/gemm/dispatch_policy.hpp:118-155` — `KernelTmaWarpSpecialized` / `KernelTmaWarpSpecializedPingpong` / `KernelTmaWarpSpecializedCooperative` 同辈 tag + FP8 派生 tag
- `include/cutlass/gemm/dispatch_policy.hpp:265/279/297` — `MainloopSm90TmaGmmaWarpSpecialized<Stages, ClusterShape, Schedule>` 派生结构
- `include/cutlass/gemm/dispatch_policy.hpp:461/471/714/715` — sm100 tag(`KernelTmaWarpSpecializedSm100` / `_BlockScaledSm100` / `_1SmSm100` / `_2SmSm100`)
- `include/cutlass/gemm/gemm.h` — `detail::IsCutlass3GemmKernel` SFINAE 探测
- `include/cutlass/gemm/device/gemm_universal_adapter.h:125` — Adapter 用 IsCutlass3GemmKernel 区分 2.x/3.x

### 8.1 核心思想:用空 struct 当标签,用继承触发 dispatch

```cpp
// 空标记
struct KernelTmaWarpSpecialized {};

// 携带数据 + 继承
template <int Stages, class ClusterShape, class KernelSchedule>
struct MainloopSm90TmaGmmaWarpSpecialized : KernelTmaWarpSpecialized {
  constexpr static int Stages = Stages;
  using ClusterShape = ClusterShape;
  using Schedule = KernelSchedule;
};

// 派生 tag 进化论(同辈,而不是互相派生)
struct KernelTmaWarpSpecializedPingpong {};
struct KernelTmaWarpSpecializedCooperative {};
```

这是一个**性质**(traits-based)的分发机制:

1. **空 struct** 充当"种类 id"。
2. **带数据的派生**把数据搬上(struct 的 `using`、constexpr 字段),同时继承父 tag。
3. 在 `CollectiveMma<...>` 的**模板特化**(partial specialization)**匹配条件**里 `is_base_of_v<KernelTmaWarpSpecialized, ...Schedule>` 区分"这是一类 dispatch policy"。
4. 在 `CollectiveMma<MainloopSm90TmaGmmaWarpSpecialized<Stages, ...>, ...>` 的 partial specialization 拿到具体变体。

### 8.2 调度 tag 进化论

**重要**:CUTLASS 的"tag 进化论"是**约定层**(大家都叫 `KernelTmaWarpSpecialized*`)上的同辈结构,在 `include/cutlass/gemm/dispatch_policy.hpp` 里是这样声明:

```cpp
struct KernelTma {};                              // 基础(TMA 大类的根)
struct KernelTmaWarpSpecialized {};               // 基础:单 consumer warp group
struct KernelTmaWarpSpecializedPingpong { ... };  // 变体:2 个 consumer 交替(pingpong)
struct KernelTmaWarpSpecializedCooperative { ... }; // 变体:多 consumer 协同(适合大 tile)
```

它们之间**没有 C++ 继承关系**——这是关键。WarpSpec、Pingpong、Cooperative 是空/近空 struct,**同辈(siblings)**,都直接是 dispatch_policy.hpp 顶层声明。

```text
                KernelTma {}                  (根标签,代指"TMA 路径")
                   │
       ┌───────────┼───────────┐
       ▼           ▼           ▼
KernelTma-    KernelTma-    KernelTma-       (三个 WarpSpecialized 变体,同辈)
WarpSpec      WarpSpec      WarpSpec
              Pingpong      Cooperative
```

对应的"Pingpong 加 FP8 后缀"等更细的 tag(`KernelTmaWarpSpecializedPingpongFP8Blockwise` 等)则**确实**用 C++ 继承(`: KernelTmaWarpSpecializedPingpong`)。这种"派生关系 vs. 同辈 tag"是两种不同的 dispatcher 风格,别混。

**为什么 builder 不靠 C++ 继承来 dispatch 这三个基础 tag?** 因为它们在源码里没有父子关系,builder 用的是 `cute::is_same_v<Schedule, KernelTmaWarpSpecialized>(或 Pingpong / Cooperative)` 在 `static_assert` 与 `if constexpr` 里硬枚举——参见 `dispatch_policy.hpp` 中 `MainloopSm90TmaGmmaRmemAWarpSpecialized` 的 `static_assert`。这意味着你写一个新 schedule 时**必须**自己在那串枚举里加一行。

### 8.3 用户如何"改 schedule"

`examples/49_hopper_gemm_with_collective_builder/49_collective_builder.cu` 把 Schedule 暴露成 template parameter:

```cpp
template <
  class MainloopScheduleType = cutlass::gemm::collective::KernelScheduleAuto,
  class EpilogueScheduleType = cutlass::epilogue::collective::EpilogueScheduleAuto,
  ...
>
struct ExampleRunner { ... };

// 用例:
ExampleRunner<cutlass::gemm::KernelTmaWarpSpecializedPingpong> runner;
```

`KernelScheduleAuto` 是一个"占位 type"——builder 看到它就用一个 default tag 替换。看到具体 `KernelTmaWarpSpecializedPingpong` 就是用户的明确选择。

### 8.4 SFINAE 接入点——`detail` 命名空间

`cutlass::gemm::detail`(`include/cutlass/gemm/gemm.h`)里只提供了 **1 个** 真正存在的 helper——`IsCutlass3GemmKernel`,靠 SFINAE 探测 `Kernel::ProblemShape` 别名是否存在来区分 3.x 和 2.x kernel:

```cpp
namespace cutlass::gemm::detail {

// 探测 Kernel 是否有 ProblemShape 别名 — 有就当 3.x 处理
template <class GemmKernel, class = void>
struct IsCutlass3GemmKernel : cute::false_type { };

template <class GemmKernel>
struct IsCutlass3GemmKernel<GemmKernel, cute::void_t<typename GemmKernel::ProblemShape>>
    : cute::true_type { };

}  // namespace detail
```

这套 helper 用于:

- `GemmUniversalAdapter`(在 `gemm_universal_adapter.h:125`)用 `IsCutlass3GemmKernel<...>::value` 决定走 3.x path 还是 2.x path,分别 SFINAE 到不同的 `operator()` 实现

> 想"看 kernel 当前用什么 schedule"是合法的 — 直接写 `Kernel::DispatchPolicy::Schedule`(`MainloopSm90*GmmaWarpSpecialized` 已经 `using Schedule = KernelSchedule;`),不需要 helper。但 CUTLASS 没有单独再 typedef 成 `GetMmaPipeline` 这种便利别名。

### 8.5 实战范例:手写一个 "3-consumer 同步" 新 schedule

这是 Ch7 最值钱的演练。**假设**你想加一个 `KernelTmaWarpSpecializedTriConsumer`(3 个 consumer warpgroup 同步,适合超巨型 tile 还要 K-loop 满载)。需要改 3 个地方:

#### 1. 在 `dispatch_policy.hpp` 加 tag

```cpp
// 新加:空 struct 作为种类 id
struct KernelTmaWarpSpecializedTriConsumer {};

// 派生:带数据 + 继承
template <int Stages, class ClusterShape>
struct MainloopSm90TmaGmmaWarpSpecializedTriConsumer
    : KernelTmaWarpSpecializedTriConsumer {
  constexpr static int Stages = Stages;
  using ClusterShape = ClusterShape;
  using Schedule = KernelTmaWarpSpecializedTriConsumer;  // ← 关键
};
```

#### 2. 在 `CollectiveMma` 加 partial specialization

```cpp
// 现存:处理单 consumer
template <class TileShape, class ClusterShape, int Stages>
class CollectiveMma<MainloopSm90TmaGmmaWarpSpecialized<Stages, ClusterShape, KernelTmaWarpSpecialized>, ...> {
  // ... 1 producer + 1 consumer 主循环
};

// 新加:处理 3 consumer
template <class TileShape, class ClusterShape, int Stages>
class CollectiveMma<MainloopSm90TmaGmmaWarpSpecializedTriConsumer<Stages, ClusterShape>, ...> {
  // ... 1 producer + 3 consumer 同步主循环
};
```

partial specialization 的**匹配条件**是 `MainloopSm90TmaGmmaWarpSpecializedTriConsumer<Stages, ClusterShape>` 的具体类型——用户写 `MainloopScheduleType = KernelTmaWarpSpecializedTriConsumer` 时,`CollectiveBuilder` 会用它构造 `MainloopSm90TmaGmmaWarpSpecializedTriConsumer<...args...>`,**编译器自动**匹配到新 partial specialization。

#### 3. 在 `CollectiveBuilder` 加 is_same_v 路由

```cpp
// 在 sm90_gmma_builder.inl 的 dispatch 段:
if constexpr (is_same_v<MainloopScheduleType, KernelTmaWarpSpecialized>) {
  using DispatchPolicy = MainloopSm90TmaGmmaWarpSpecialized<StageCount, ClusterShape, KernelTmaWarpSpecialized>;
} else if constexpr (is_same_v<MainloopScheduleType, KernelTmaWarpSpecializedPingpong>) {
  using DispatchPolicy = MainloopSm90TmaGmmaWarpSpecialized<StageCount, ClusterShape, KernelTmaWarpSpecializedPingpong>;
} else if constexpr (is_same_v<MainloopScheduleType, KernelTmaWarpSpecializedCooperative>) {
  using DispatchPolicy = MainloopSm90TmaGmmaWarpSpecialized<StageCount, ClusterShape, KernelTmaWarpSpecializedCooperative>;
} else if constexpr (is_same_v<MainloopScheduleType, KernelTmaWarpSpecializedTriConsumer>) {
  // ← 新加这一行
  using DispatchPolicy = MainloopSm90TmaGmmaWarpSpecializedTriConsumer<StageCount, ClusterShape>;
} else {
  static_assert(/* 错误信息 */);
}
```

**关键观察**:

- 加 1 个 schedule 改 3 处:tag 定义 + CollectiveMma partial spec + Builder 枚举。
- 编译期**完全**只编选中的那个 partial specialization,其他 7 个 spec 一个字节都不进二进制。
- 老用户代码不会破——只要他们不写新 tag,枚举走原来路径。

### 8.6 跨架构 dispatch tag 族:sm80 / sm90 / sm100 / sm120

`dispatch_policy.hpp` 不是只有 sm90 一族。完整一支:

```text
KernelTmaWarpSpecialized*        ← sm90 标准族(WarpSpec / Pingpong / Cooperative)
KernelTmaWarpSpecializedSm100*   ← sm100 标准族(1Sm / 2Sm / Extended)
KernelTmaWarpSpecializedSm120*   ← sm120(未来的 120 系列)
KernelMultistage / KernelCpAsyncWarpSpecialized ← sm80 旧式(cp.async 路径)
```

每一族都用"根 + 变体"同辈结构,但**根不同**——所以 `KernelTmaWarpSpecializedCooperative` 跟 `KernelTmaWarpSpecializedSm100Cooperative` 是两种东西(builder 路由时**先看 sm_arch,再看 schedule**)。

#### 调用约定

`CollectiveBuilder<ArchTag, OpClass, ...>` 第一参数 `ArchTag` 决定走哪一族 mainloop 文件:

- `Sm80` → `collective/sm80_mma_*.hpp`(cp.async 路径,无 TMA)
- `Sm90` → `collective/sm90_mma_tma_gmma_*.hpp`(TMA + WGMMA)
- `Sm100` → `collective/sm100_mma_tma_umma_*.hpp`(TMA + UMMA + TMEM)
- `Sm120` → `collective/sm120_*.hpp`(未来)

**注意**:sm80 的 `KernelMultistage` **不是** sm90 的 `KernelTmaWarpSpecialized`——它们是不同时代的实现, dispatcher 完全是两套。

### 8.7 跟 2.x `if constexpr` 模式对比:为什么 tag-inheritance 比 2.x 好

2.x 的"DefaultGemmUniversal 工厂"长这样:

```cpp
// 2.x 风格的"伪代码"
template <int Stages, bool Pingpong>
struct MmaPipelined {
  void operator()(...) {
    if constexpr (Pingpong) {
      // 2 consumer 交替逻辑
    } else {
      // 1 consumer 逻辑
    }
  }
};
```

这条路 3 个问题:

1. **代码冗余**:即使 Pingpong 是 false,2 consumer 那一支的代码**也会被编译进二进制**(只是 unreachable branch)。3.x 的 partial specialization **编译期只编**选中那个。
2. **调度组合爆炸**:N 个开关在 2.x 是 `2^N` 个组合(每个 `if constexpr` 编一份),3.x 是 N 个 partial spec(独立,只编 N 个)。
3. **配置空间可见性**:2.x 的 `if constexpr` 散落在多个函数体里,reader 看不到"这代码到底有几种形态";3.x 的 partial spec 集中在 builder 文件里,一眼能扫完。

**这是 3.x 比 2.x 显著进步的一处**——不是"略好",而是"配置空间支持 10 倍以上"。

### 8.8 这一章的核心 takeaway

**tag-inheritance dispatch** 是 CUTLASS 3.x 实现"配置空间巨大但每个具体实现编译期最优"的关键——比 2.x 的 DefaultGemmUniversal 工厂好得多,因为编译器**完全**只编当前用户选定的那个 partial specialization,生成的代码无冗余。

如果以后你想:

- 写一个新 schedule(比如"3 个 consumer 同步")
- 写一个新 dispatcher("依问题尺寸选 schedule")
- 写一个新 builder 选项

入口就在 `dispatch_policy.hpp` + `kernel/sm90_*.hpp` 的 SFINAE 路由 + `cutlass::gemm::detail` helper(在 `include/cutlass/gemm/gemm.h`,见 §8.4)。

### 8.9 章末:读完这一章你该做得到的事

- ✅ 在自己的代码里手写 `using Schedule = KernelTmaWarpSpecializedPingpong;` 替换 `KernelScheduleAuto`,看到 builder 走不同代码路径。
- ✅ 在 `dispatch_policy.hpp` 里读懂 tag 树——区分"同辈 tag"(`KernelTmaWarpSpecialized` / `KernelTmaWarpSpecializedPingpong` / `KernelTmaWarpSpecializedCooperative`)和"派生 tag"(`KernelTmaWarpSpecializedPingpongFP8Blockwise` 这种真的 `: KernelTmaWarpSpecializedPingpong`)。
- ✅ 用 `Kernel::DispatchPolicy::Schedule` 这一行 alias 找到当前选了什么 schedule。
- ✅ 能在 `sm90_gmma_builder.inl` 的 `is_same_v` 枚举段中加 1 个新 schedule(走 3 步:tag 定义 + CollectiveMma partial spec + Builder 枚举)。
- ✅ 区分 sm80 / sm90 / sm100 / sm120 各自的根 tag(`KernelMultistage` / `KernelTmaWarpSpecialized*` / `KernelTmaWarpSpecializedSm100*` / `KernelTmaWarpSpecializedSm120*`)。
- ✅ 能口述"3.x 的 partial specialization 比 2.x 的 `if constexpr` 好在哪"——3 个理由:无冗余代码、N 不爆炸、配置空间可见性。
- ✅ 知道 `detail::IsCutlass3GemmKernel` 是 2.x ↔ 3.x 兼容入口,不是新 dispatch 路。

---



