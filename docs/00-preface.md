### 章节地图与阅读顺序

教程共 10 章 + 1 附录(附录 D:再之后)。按"对 CUTLASS 3.x 抽象的覆盖深度"分两档:

```text
必修 (Core Path)        ── 9 章构成主干,读完即能读懂 examples/48 + 99% 的 mainloop
  Ch1  Overview(examples/48 走读 + 5 层架构图 + 4 个 *Type 开关)
  Ch2  CuTe 实战(写法 / layout / swizzle / make_tiled_*)
  Ch3  smem pipeline 与 barrier 抽象(PipelineTmaAsync 6 个 helper)
  Ch4  CollectiveMainloop 概览(类头 + 类型别名 + 三阶段 + 契约 + Cluster)
  Ch5  CollectiveMainloop 逐方法(load_init / load / mma / *_tail 5 个函数)
  Ch6  CollectiveEpilogue & EVT(融合算子 DSL——上半 epilogue 机制,下半 EVT)
  Ch7  Kernel orchestrator + TileScheduler(调度器族对比)
  Ch8  DispatchPolicy(tag-inheritance 模式)
  Ch9  CollectiveBuilder(8 个 partial spec 路由 + §9.7 全配置空间地图)

进阶 (Advanced)         ── 学完必修之后按需读
  Ch10 Blackwell 桥接(sm100 / sm103 架构差异 + tcgen05)
  Ch11 Blackwell 物理基础(TMEM + UMMA + CLC,深入 sm100 硬件细节)
```

#### 章节依赖图

```text
                            Ch1 (Overview)
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
        Ch2 (CuTe)      Ch4 (mainloop 概览)  Ch6 (epilogue + EVT)
              │               │               │
              ▼               ▼               │
        Ch3 (smem pipeline)  Ch5 (mainloop 逐方法)
              │               │               │
              └───────────────┴───────────────┘
                              │
                              ▼
                    Ch7 (kernel orchestrator + TileScheduler)
                              │
                              ▼
                      Ch8 (DispatchPolicy)
                              │
                              ▼
                      Ch9 (CollectiveBuilder)
                              │
                              ▼
                      Ch10 (Blackwell 桥接)
                              │
                              ▼
                      Ch11 (Blackwell 物理基础)
```

读法建议:

- **第一次读**(必修主干):从 Ch1 顺序读到 Ch9。Ch1 是 entry,看完脑子里有"5 层骨架 + 一份代码 + 4 个开关";Ch2-Ch9 逐层展开。
- **第二次读**(按需):补 Ch10(Blackwell 桥接,如果你用 sm100 / sm103)。**深入物理细节**再看 Ch11。
- 附录 D(再之后):随时查阅——Grouped / Sparse / Conv / SSD / PDL 路线图与对应 example 入口。

每章末尾有 checklist——读完能自检"这一章到底讲清没讲清"。如果某条 checklist 卡住,回到对应章节的图 + 代码段重读。

---