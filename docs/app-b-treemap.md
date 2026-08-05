## 附录 B:In-tree 散文定位("你想读 X 看 Y")

本文是自洽的,但仓库里 `media/docs/cpp/` 还有 ~30 篇高质量散文。本表告诉你"如果想深挖,看哪里"。

|想读什么|文件|本教程哪章引用|
|---|---|---|
|**CUTLASS 3.x 整体设计哲学**|`media/docs/cpp/cutlass_3x_design.md`|序章 + Ch1 — 5 个设计宣言的扩展阅读|
|**GEMM API 参考(3.x)**|`media/docs/cpp/gemm_api_3x.md`|Ch8 — Builder 的 4 步"装配内核"配方|
|**CUTLASS 2.x → 3.x 迁移**|`media/docs/cpp/cutlass_3x_backwards_compatibility.md`|(本文未引入)|
|**2.x GEMM API**|`media/docs/cpp/gemm_api.md`|(本文未引入)|
|**高效 GEMM 设计与伪码**|`media/docs/cpp/efficient_gemm.md`|Ch6 — 经典 5 层 nested-loop 伪码|
|**术语表**|`media/docs/cpp/terminology.md`|任何章节 — 名词定义|
|**目录地图**|`media/docs/cpp/code_organization.md`|Ch1 — 仓库组织说明|
|**完整 quickstart**|`media/docs/cpp/quickstart.md`|Ch2 — 该文件 §"Launching a GEMM kernel using CUTLASS 3.0 or newer"|
|**CUTLASS 代码风格 / Params / SharedStorage 约定**|`media/docs/cpp/programming_guidelines.md`|Ch2 + Ch4 — Params/SharedStorage 惯例|
|**CUTLASS ↔ 数据类型支持表**|`media/docs/cpp/functionality.md`|(查询用)|
|**数值类型 catalog**|`media/docs/cpp/fundamental_types.md`|Ch11 — sm_100 mx_* 类型说明|
|**性能测量方法学**|`media/docs/cpp/gemm_performance_measurement_methodology_guidelines.md`|Ch10 — 测量 cuBLAS 对比时的科学方法|
|**profiler CLI**|`media/docs/cpp/profiler.md`|Ch10.6|
|**heuristics / autotune**|`media/docs/cpp/heuristics.md`|Ch10.6|
|**Blackwell 功能清单**|`media/docs/cpp/blackwell_functionality.md`|Ch11.2 — 7 条 tcgen05.mma + 类型表|
|**Blackwell cluster launch control**|`media/docs/cpp/blackwell_cluster_launch_control.md`|Ch11.5|
|**依赖 kernel launch(PDL)**|`media/docs/cpp/dependent_kernel_launch.md`|Ch10.4 / 附录 D|
|**Grouped scheduler**|`media/docs/cpp/grouped_scheduler.md`|附录 D|
|**Implicit GEMM Convolution**|`media/docs/cpp/implicit_gemm_convolution.md`|附录 D|
|**CuTe 入门**|`media/docs/cpp/cute/00_quickstart.md`|Ch3|
|**CuTe Layout primer**|`media/docs/cpp/cute/01_layout.md`|Ch3 — 教材级长文,可复读|
|**CuTe Layout 代数**|`media/docs/cpp/cute/02_layout_algebra.md`|Ch3 — composition / coalesce / zipped-divide 等|
|**CuTe Tensor = Engine + Layout**|`media/docs/cpp/cute/03_tensor.md`|Ch3|
|**CuTe Algorithms (copy / gemm / clear)**|`media/docs/cpp/cute/04_algorithms.md`|Ch3.4 — `cute::gemm` 的来源|
|**CuTe MMA atom naming**|`media/docs/cpp/cute/0t_mma_atom.md`|Ch3.5|
|**CuTe GEMM 实战教程**|`media/docs/cpp/cute/0x_gemm_tutorial.md`|Ch3.5 — 配 `examples/cute/tutorial/sgemm_*.cu`|
|**CuTe 残差 / predication**|`media/docs/cpp/cute/0y_predication.md`|(本文未引入)|
|**CuTe + TMA**|`media/docs/cpp/cute/0z_tma_tensors.md`|Ch4 — 读懂 TMA + CuTe 联合用法|

**怎么用**:读完本文后,如果某个章节里某段你有"想再深一层"的冲动,在本表对应行找到该深读的散文。**不要**先看散文再回来看本文——本文是按"够用"写的,看散文是为了"扩展"。

---

