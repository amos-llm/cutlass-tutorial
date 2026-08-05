# CUTLASS 3.x 教程:从手写 Hopper GEMM 到读懂工业级 GEMM 库

> **适用对象**:你已经手写过 Hopper GEMM 并达到 ~100% cuBLAS 性能,熟悉 WGMMA、cp.async / TMA、warp specialization、shared memory swizzle、thread block cluster。
>
> **本文不教**:GPU 入门 / GEMM 算法基础 / WGMMA 编码细节 / mma intrinsic 编程。
>
> **本文教**:CUTLASS 3.x 为什么要切片成 5 个 C++ 抽象,每个抽象在你已经熟悉的机制里**对应**什么,从哪里读起,怎么改默认值,以及为什么 Blackwell 上同样的 5 层抽象依然能复用。

本教程按章节拆分到 `docs/` 下独立文件,跨章跳转走相对链接,GitHub 网页 / 移动 App / 桌面浏览器 / 本地编辑器一致可用。

---

## 目录

- [序章:为什么 CUTLASS 长成这样——5 件事、5 层类、5 个理由](docs/00-preface.md)
- [第 1 章:5 层架构图(整体鸟瞰)](docs/01-architecture.md)
- [第 2 章:跑通一个最小 GEMM——走读 examples/48](docs/02-minimal-gemm.md)
- [第 3 章:CuTe——CUTLASS 真正的「语言」](docs/03-cute.md)
- [第 4 章:深入 CollectiveMainloop(本教程核心价值)](docs/04-mainloop.md)
- [第 5 章:深入 CollectiveEpilogue + EVT](docs/05-epilogue.md)
- [第 6 章:Kernel orchestrator + TileScheduler(含调度器族侧栏)](docs/06-kernel-orchestrator.md)
- [第 7 章:DispatchPolicy——tag-inheritance 模式](docs/07-dispatch-policy.md)
- [第 8 章:CollectiveBuilder——把「形状 + 类型」压成具体实现](docs/08-collective-builder.md)
- [第 9 章:examples/49——一个用户故事](docs/09-user-story.md)
- [第 10 章:调参世界观 + cutlass_profiler](docs/10-tuning.md)
- [第 11 章:Blackwell 桥接——同样的 5 层架构,换了一组原子](docs/11-blackwell.md)
- [附录 A:Hopper 原语 ↔ CUTLASS 封装文件(速查表)](docs/app-a-primitives.md)
- [附录 B:In-tree 散文定位(「你想读 X 看 Y」)](docs/app-b-treemap.md)
- [附录 C:你手写 GEMM 的 X 行 ↔ CUTLASS 哪里(对照表)](docs/app-c-mapping.md)
- [附录 D:再之后(Grouped / Sparse / Conv / SSD / PDL)](docs/app-d-future.md)

---

## 阅读顺序建议

1. [序章](docs/00-preface.md) 把 5 层抽象和手写 GEMM 的对应关系建起来。
2. [第 1 章](docs/01-architecture.md) → [第 2 章](docs/02-minimal-gemm.md) 看一次最小 GEMM 跑通。
3. [第 3 章](docs/03-cute.md) → [第 6 章](docs/06-kernel-orchestrator.md)——本教程核心价值。
4. [第 7 章](docs/07-dispatch-policy.md) / [第 8 章](docs/08-collective-builder.md) 是「工厂」层。
5. [第 9 章](docs/09-user-story.md) 是用户视角的端到端故事。
6. [第 10 章](docs/10-tuning.md) / [第 11 章](docs/11-blackwell.md) 收尾,讨论调参与 Blackwell 兼容。
7. 附录是速查表,不需要从前往后读。

---

## 约定

- 代码示例以 `examples/48_hopper_warp_specialized_gemm/` 为主;数值(BM=128, BN=128 等)用于说明,实际以 `examples/` 中默认值为准。
- 代码约定 C++17;命名空间 `cutlass::gemm::…` 路径以 `include/cutlass/…` 为准。
- 所有图片位于 `media/images/`,章节内引用形如 `![alt](../media/images/foo.png)`。
