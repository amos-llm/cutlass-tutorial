## 附录 D:再之后(Grouped / Sparse / Conv / SSD / PDL)

本教程主线是 GEMM。但 CUTLASS 还在以下方向延伸,**每个主题 1–2 段**,告诉你"文件名 + 一个指针"。

### D.1 Grouped GEMM

文件:`include/cutlass/gemm/kernel/sm90_tile_scheduler_group.hpp` + `examples/57_hopper_grouped_gemm/` + `media/docs/cpp/grouped_scheduler.md`。

用于:**一次 launch 跑多个不同形状的 GEMM**(典型:MoE 的 experts 前向)。`GroupScheduler` 由 `GroupProblemVisitor` 牵引多个 problem。example 57 演示。

### D.2 Sparse GEMM(2:4 structured sparsity)

文件:`include/cutlass/gemm/collective/sm90_sparse_mma_tma_gmma_ss_warpspecialized.hpp` + `examples/62_hopper_sparse_gemm/`。

用于:A / B 中每 4 元素有 2 个 0 的稀疏矩阵;cutlass 通过索引元数据把"非零 2 元素"读进 mma。**一般用于 Ampere TF32,但 Hopper 上也有 sparse variant**。

### D.3 Convolution(im2col / col2im)

文件:`media/docs/cpp/implicit_gemm_convolution.md` + `examples/16_ampere_tensorop_conv2dfprop/`。

将 conv2d `y[n,p,q,k] = Σ_c Σ_r Σ_s x[n,c,p+r,q+s] · w[k,c,r,s]` 重写为 implicit GEMM。**Conv 不是切 GEMM 做的——是把 conv 当成一个"非常特定的 GEMM"来编译**(im2col → GEMM → col2im)。

具体看 `media/docs/cpp/implicit_gemm_convolution.md` 的 im2col 形式。

### D.4 SSD(Sparse + SD,DSS-style decoupling)

文件:`examples/111_hopper_ssd/` + `examples/112_blackwell_ssd/`,以及 `media/images/13_example_*`。

Sparse + Dst-Sparse Decomposition:用稀疏权重 + dense activation 算 sparse dense matmul,但反着——dense-side 同样可以"反稀疏"。

### D.5 PDL(Dependent Kernel Launch)

文件:`media/docs/cpp/dependent_kernel_launch.md` + `examples/63_hopper_gemm_with_weight_prefetch/`。

PDL 是 sm_90+ 的硬件特性:**让 host 在 kernel A 还没结束时启动 kernel B**,只要 kernel A 满足"init done"信号就行。CUTLASS 提供 `epilogue_aux` 这种 server-style kernel 来利用 PDL。

### D.6 Conv 路径内的 GEMM

`examples/cute/tutorial/sgemm_*` 系列也是为 conv 教程铺垫。`16_ampere_tensorop_conv2dfprop/` 是 Ampere 时代 Conv + fused epilogue 真实例。

### D.7 Sparse + Block-scaled Quantization

`examples/81_blackwell_gemm_blockwise/*` 演示 sm_100 上的 block-scale GEMM。`examples/67_hopper_fp8_warp_specialized_gemm_with_blockwise_scaling/` 同主题 Hopper 版本。

---
