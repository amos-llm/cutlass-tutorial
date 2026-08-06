## 序章:为什么 CUTLASS 长成这样——5 件事、5 层类、5 个理由

你手写 Hopper GEMM 时,脑子里并行地想了这些事:

1. **问题切片**:M、N、K 三个维度各自切多大块(`BlockM × BlockN × BlockK`,你可能叫 tile size 或 threadblock shape)。
2. **CTA 调度**:`gridDim.x/y/z` 该怎么排,几个 CTA 跑同一个 tile、要不要做 L2-locality swizzle、要不要 persistent(每个 CTA 处理多 tile)。
3. **加载 A/B**:gmem → smem 走 TMA 还是 cp.async,描述符怎么准备、smem 怎么 swizzle 防 bank conflict、几条 stage 流水线。
4. **mma 计算**:A、B 在 smem 里怎么布局 → WGMMA,fragment 累积器怎么分给 warp / lane、流水线推进。
5. **写回 C/D**:做完的 accumulator 怎么搬到 gmem,要不要顺手做 bias / ReLU / swizzle。

CUTLASS 3.x 把这件事**正好**切成 5 个 C++ 类,每类一一对应一件事:

```text
┌─ 切层类(对应 GEMM 5 件事)
│
├─ 1. cutlass::gemm::device::GemmUniversalAdapter       ──  host 句柄:开工、收尾
├─ 2. cutlass::gemm::kernel::GemmUniversal<...>          ──  CTA 调度 + 主循环 orchestrator
├─ 3. cutlass::gemm::collective::CollectiveMma           ──  加载 A/B → mma(事 3 + 事 4)
├─ 4. cutlass::epilogue::collective::CollectiveEpilogue  ──  写回 C/D + 融合算子(事 5)
└─ 5. cutlass::gemm::kernel::*TileScheduler              ──  切 tile + 排队(事 1 + 事 2)
```

![CUTLASS 五层架构图](../media/images/cutlass-layered-organization.png)

> **如果你只能记住这张图,就够了。** 后面每一章只是把这一张图"哪个方框里住着什么、代码长什么样、对应你手写的哪一段"展开。

### 5 个设计理由(为什么这样切、不那样切)

CUTLASS 3.x 设计文档 `media/docs/cpp/cutlass_3x_design.md` 里列了 5 条:

1. **可组合性**:5 层之间用模板参数互不依赖,可以单独替换某一层而不动其他层。比如你想改 epilogue 加 swizzle,只换第 4 个模板参数,不动 mainloop。
2. **可配置性**:每个具体实现都用"dispatch policy" tag(空 struct)标记,builder(第 11 章)用 tag 路由到对应 partial specialization——用户写 `Auto` 而不是写具体类名。
3. **关注点分离**:mma 编程模型(`CollectiveMma`)和 epilogue 融合(`CollectiveEpilogue`)是两个完全不同的领域专家的工作,CUTLASS 把它们解耦,各自演化。
4. **硬件可移植**:同一套 5 层框架可以在 Hopper、Blackwell、Volta、Ampere、Turing 上工作;具体某一层(比如 mma atom)从 `WGMMA` 换到 `UMMA` 不影响其他 4 层。
5. **默认正确**:`Auto*`(`StageCountAuto`、`KernelScheduleAuto`、`EpilogueScheduleAuto`)总是选一个合理的实现,你得手写错才能跑错——这是 3.x 比 2.x 显著进步的一处。

### 2.x 的归类破了一个洞——这才是重写的真正动机

上面 5 条理由听起来像「设计目标」,但 CUTLASS 3.x 不是为了好看重写,是被 Hopper 逼的。具体说:2.x 把 GEMM 的所有部件按 **GPU 硬件层次**(thread / warp / threadblock)分层——`gemm::threadblock::MmaMultistage`、`gemm::warp::MmaTensorOp` 等等。这套组织假设了「每一层都有同名的硬件单元」。

Hopper 的 WGMMA 打破了这个假设。WGMMA 的指令粒度是 **4-warp × 4-warp warpgroup**(32 lane × 32 lane = 1024 thread 的指令),既不属于 `warp` 层,也不属于 `threadblock` 层——它在 2.x 的层次里没有「家」。同样,Volta 上的 quad-pair mma 指令也是先在 warp 层 tile 一遍再用,本来就是硬塞进去的。TMA 也不是「threadblock 级 memcpy」,它有自己的 descriptor 和跨 cluster 能力,塞进 2.x 的 threadblock iterator 又是一处硬塞。

CUTLASS 3.x 的解法是**重新切分层次**,不再按「硬件哪一层」切,而是按 **GEMM 算法本身的自然结构**切:`CollectiveMma`(一次迭代里 group 协作做的事)、`CollectiveEpilogue`(产物出来后要做什么)、`Kernel`(orchestrator + scheduler)、`Device`(host 句柄)。这样硬件变了,WGMMA → UMMA,只是换一个 dispatch tag,5 层骨架不动。

> 一句话:**2.x 把 GEMM 绑在了 GPU 拓扑上,所以每来一种新指令都得拆骨架;3.x 把 GEMM 绑在算法结构上,硬件只是参数。**

### 「默认正确」不只是 slogan——CuTe 给你了硬保证

2.x 的「默认正确」是社区约定:default config 用对了就跑对,用错了就崩。3.x 的「默认正确」是**编译期就能查**:

- CuTe 的 `Layout` 始终保持坐标一致性,所有静态内层循环的 pre/post-condition 都在编译期检查。
- `if (compile-time 条件满足)` → 否则直接 static_assert 拒绝编译。
- 这意味着:**内层循环如果编译通过,大概率是正确的**;2.x 要靠运行时 reference 对比才能确认。

副作用:编译时间变长,出错信息变陡峭。`Auto*` tag 的便利和编译期硬保证是同一件事的两面。

### 本教程的承诺

- **重写、自成一体**——不依赖读者已经读过 `media/docs/cpp/` 任何一篇文章。但附录 B 给"如果你想深挖,看哪里"的导航。
- **单文件**——便于全文搜、git 单 file diff、转 PDF。章末锚点 `<a name>` 可直接跳转。
- **类名不省略**——每个 `using`、`constexpr`、`typename`、`operator()` 前缀都不缩。
- **你不熟悉的 CUTLASS 概念 ↔ 你已经会的 Hopper 概念**——每个新类、每个调度 tag、每个 `SharedStorage` 段都有手写对照 callout。

### 章节地图与阅读顺序

教程共 13 章 + 4 附录。按"对 CUTLASS 3.x 抽象的覆盖深度"分三档:

```text
必修 (Core Path)        ── 9 章构成主干,读完即能读懂 examples/48 + 99% 的 mainloop
  Ch1  5 层架构图(整体鸟瞰)
  Ch2  跑通一个最小 GEMM(走读 examples/48)
  Ch3  CuTe 实战(写法 / layout / swizzle / make_tiled_*)
  Ch4  smem pipeline 与 barrier 抽象(PipelineTmaAsync 6 个 helper)
  Ch5  深入 CollectiveMainloop
  Ch6  深入 CollectiveEpilogue
  Ch7  Epilogue Visitor Tree(EVT)
  Ch8  Kernel orchestrator + TileScheduler(调度器族对比)
  Ch9  DispatchPolicy(tag-inheritance 模式)

进阶 (Advanced)         ── 学完必修之后按需读
  Ch10 Blackwell 桥接(sm100 / sm103 架构差异 + tcgen05)
  Ch11 CollectiveBuilder(8 个 partial spec 路由)
  Ch13 调参世界观 + cutlass_profiler

选修 (Case Study)       ── 不在主线,但读完会拼起完整工程图
  Ch12 `examples/49`——一个 End-to-End 集成示例
```

#### 章节依赖图

```text
                            Ch1 (架构)
                              │
                              ▼
                            Ch2 (最小 GEMM)
                              │
            ┌─────────────────┼─────────────────┐
            ▼                 ▼                 ▼
       Ch3 (CuTe)       Ch3 (CuTe)        Ch3 (CuTe)
            │                 │                 │
            ▼                 ▼                 ▼
            └──────► Ch4 (smem pipeline) ◄─────┘
                          │
              ┌───────────┼───────────┐
              ▼           ▼           ▼
        Ch5 (mainloop) Ch6 (epilogue)
              │           │
              │           ▼
              │      Ch7 (EVT)
              │           │
              └─────┬─────┘
                    ▼
              Ch8 (kernel orchestrator + TileScheduler)
                    │
                    ▼
              Ch9 (DispatchPolicy)
                    │
            ┌───────┼───────┐
            ▼               ▼
       Ch10 (Blackwell) Ch11 (CollectiveBuilder)
                            │
                            ▼
                       Ch12 (End-to-End)
                            │
                            ▼
                       Ch13 (Tuning)
```

读法建议:

- **第一次读**:从 Ch1 顺序读到 Ch9(必修主干)。
- **第二次读**:补 Ch10(Blackwell,如果你用 sm100)和 Ch11(Builder,理解 dispatch 怎么自动选)。
- **第三次读**:Ch13(Tuning,实战用)+ Ch12(End-to-End,看工程怎么组织)。
- **附录 A-D**:随时查阅——`primitives ↔ CUTLASS 封装文件` 表、`media/docs/cpp/` 导航、`code map`、`future` 路线图。

每章末尾有 checklist——读完能自检"这一章到底讲清没讲清"。如果某条 checklist 卡住,回到对应章节的图 + 代码段重读。

---

