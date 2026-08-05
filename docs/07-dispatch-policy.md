## 第 7 章:DispatchPolicy——tag-inheritance 模式

这是 CUTLASS 3.x 的**架构精髓**——也是 `media/docs/cpp/` 几乎完全没有覆盖、本教程必须讲的东西。

文件:`include/cutlass/gemm/dispatch_policy.hpp`(覆盖 sm70 / sm80 / sm90 / sm100 / sm120 所有 dispatch tag)。

### 7.1 核心思想:用空 struct 当标签,用继承触发 dispatch

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

### 7.2 调度 tag 进化论

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

### 7.3 用户如何"改 schedule"

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

### 7.4 SFINAE 接入点——`detail` 命名空间

```cpp
namespace cutlass::gemm::kernel::detail {

template <class Kernel>
struct IsCutlass3GemmKernel
    : std::bool_constant<is_base_of_v<KernelTmaWarpSpecialized, /*Schedule*/>...> {};

template <class Kernel>
using GetMmaPipeline = typename Kernel::DispatchPolicy::Schedule;  // ← 提取主调度的 alias

}  // namespace
```

这套 helper 用于:

- `GemmUniversalAdapter` 决定走"3.x path"还是"2.x path"(`IsCutlass3GemmKernel`)
- 任何想"看 kernel 当前用什么 schedule"的代码用 `GetMmaPipeline`

> 把这一对 helper 跟你写的 dispatcher 联动——比如你想写一个"自动选不同 schedule for 不同 problem shape"的下游工具,这里的 `GetMmaPipeline<MyKernel>` 就能拿到。

### 7.5 这一章的核心 takeaway

**tag-inheritance dispatch** 是 CUTLASS 3.x 实现"配置空间巨大但每个具体实现编译期最优"的关键——比 2.x 的 DefaultGemmUniversal 工厂好得多,因为编译器**完全**只编当前用户选定的那个 partial specialization,生成的代码无冗余。

如果以后你想:

- 写一个新 schedule(比如"3 个 consumer 同步")
- 写一个新 dispatcher("依问题尺寸选 schedule")
- 写一个新 builder 选项

入口就在 `dispatch_policy.hpp` + `kernel/sm90_*.hpp` 的 SFINAE 路由 + `kernel/detail/` helper。

### 7.6 章末:读完这一章你该做得到的事

- ✅ 在自己的代码里手写 `using Schedule = KernelTmaWarpSpecializedPingpong;` 替换 `KernelScheduleAuto`,看到 builder 走不同代码路径。
- ✅ 在 `dispatch_policy.hpp` 里读懂 tag 树。
- ✅ 用 `Kernel::DispatchPolicy::Schedule` 这一行 alias 找到当前选了什么 schedule。

---

