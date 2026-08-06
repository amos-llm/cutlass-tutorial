## 第 3 章:smem pipeline 与 barrier 抽象

这一章是 CUTLASS 3.x **真正的"概念孵化器"**——mainloop / epilogue / kernel orchestrator / scheduler 都依赖这一套 `PipelineTmaAsync` / `PipelineAsync` / `PipelineTmaStore` 抽象,但它**没有任何一个章节真正把它当主语讲过**。

读完 Ch4,你会明白为什么 `producer_acquire` / `consumer_wait` 是必须的、为什么 `PipelineState` 要分 producer 和 consumer 两份、`Stages` 是什么、为什么 TMA 路径上 `producer_commit` 是 NoOp、cluster barrier 怎么跟 single-CTA barrier 区分。

主文件:`include/cutlass/pipeline/sm90_pipeline.hpp` + `pipeline.hpp`(`PipelineAsync` 与 `PipelineTmaStore`)。

### 4.1 问题的提出:smem 上的 producer / consumer 怎么同步

mainloop 里有 producer warpgroup(TMA 把数据从 gmem 拉入 smem)和 consumer warpgroup(WGMMA 从 smem 读数据做 mma)两个角色。它们共享同一块 smem 的 Stages 个 slot,典型 3-5 个 stage 流水线:

```text
smem stage:   [0]  [1]  [2]  [3]
              ^      ^      ^
              |      |      |
producer      write  write  write   (TMA 写)
consumer      read   read   read    (WGMMA 读)
```

producer 写第 N 个 stage 时,consumer 可能正在读第 N-1 个 stage——它们不能撞车。

CUTLASS 用 **mbarrier**(NVIDIA 的 hardware barrier,Hopper+ 引入)实现这套同步。原语大致是:

- `bar.arrive()`:当前线程"我已经到了"
- `bar.wait(phase)`:当前线程"等到 phase 翻面再走"
- `bar.arrive_and_expect_tx(bytes)`:TMA 配套——"我即将发起一个 bytes 大小的 TMA transaction"
- `bar.complete_transaction(bytes)`:TMA 完成时由硬件自动调用——"之前 arrive_and_expect_tx 承诺的 bytes 已经到了"

整个 CUTLASS 的 pipeline 抽象,就是把这套原语包成"看起来像 smem 队列"的 API。

### 4.2 5 类 pipeline

`include/cutlass/pipeline/pipeline.hpp` + `sm90_pipeline.hpp` + `sm100_pipeline.hpp` 里定义了 5 类 pipeline,选哪个看数据流方向 + smem 类型:

| 类 | 文件 | 用在哪 | 数据方向 |
|---|---|---|---|
| `PipelineTmaAsync<Stages>` | `sm90_pipeline.hpp:271` | **mainloop 的 smem ↔ TMA 路径**(默认) | gmem → smem(TMA producer)、smem → register(consumer)|
| `PipelineTmaStore<Stages>` | `sm90_pipeline.hpp:655` | **epilogue 的 register → smem → gmem**(TMA store) | register → smem(producer)、smem → gmem(TMA)|
| `PipelineAsync<Stages>` | `sm90_pipeline.hpp:1015` | **不用 TMA 的 smem 路径**(cp.async,sm80 旧式)| gmem → smem(cp.async producer)、smem → register(consumer)|
| `PipelineTransactionAsync<Stages>` | `sm90_pipeline.hpp:766` | 多生产者协同时的 transactional 路径(避免同一 stage 被多个 producer 抢)| 同 PipelineAsync,但带 transaction semantics |
| `PipelineCLCFetchAsync<Stages, ClusterShape>` | `sm100_tile_scheduler.hpp` | **Sm100 调度器的 CLC pipeline**(Ch9 Sm100 节)| 调度器专用 |

注意 Hopper 上**默认 mainloop 走 `PipelineTmaAsync`**——TMA 的 `arrive_and_expect_tx` 由 producer 发起,`complete_transaction` 由硬件在 TMA 写完时自动调用,**producer 那边不需要显式 `commit`**。这跟 Ampere 时代的 cp.async 路径不一样。

### 4.3 6 个 helper 的语义

`PipelineTmaAsync` 暴露 6 个 helper,**严格成对使用**(producer 2 个,consumer 2 个,辅助 2 个):

#### Producer 侧

```cpp
// producer_acquire(state): 等 smem slot 空出来
void producer_acquire(PipelineState state) {
  empty_barrier_ptr_[state.index()].wait(state.phase());  // 等 consumer 释放
  if (params_.is_leader) {
    full_barrier_ptr_[state.index()].arrive_and_expect_tx(params_.transaction_bytes);
    // ↑ "我即将发起 transaction_bytes 字节的 TMA"
  }
}

// producer_commit(state, bytes): 在 PipelineTmaAsync 里是 NoOp
void producer_commit(PipelineState state, uint32_t bytes) {
  // TMA 路径:硬件在 TMA 完成时自动调 complete_transaction,这里什么都不做
  // Ampere PipelineAsync 路径:这里真的 arrive full_barrier
  #if CUTLASS_UNIT_TEST_PIPELINE
    // 测试用
  #endif
}
```

为什么 TMA 路径 `producer_commit` 是 NoOp:TMA descriptor 启动时已经 `arrive_and_expect_tx` 了 barrier,硬件完成 TMA 时会自动 `complete_transaction`,**这条 helper 在 TMA 路径上就是个 placeholder**。如果你从 cp.async 移植过来会习惯性地写一行 `producer_commit`,**Hopper 上这一行冗余**,可省(Ch5 mainloop §5.4 会强调)。

#### Consumer 侧

```cpp
// consumer_wait(state): 等 producer 写完这个 stage
void consumer_wait(PipelineState state) {
  full_barrier_ptr_[state.index()].wait(state.phase());  // 等 producer arrive
}

// consumer_release(state): "我读完了,producer 可以覆盖这个 stage 了"
void consumer_release(PipelineState state) {
  empty_barrier_ptr_[state.index()].arrive(...);  // 通知 producer
}
```

#### 辅助

- `producer_try_acquire` / `consumer_try_wait`:非阻塞版本,带 timeout(用于 prefetch 优化)。
- `producer_get_barrier`:把当前 stage 的 barrier 指针给 TMA(TMA descriptor 内部要用)。

### 4.4 `PipelineState` 与"双 state"模式

```cpp
template <int Stages_>
struct PipelineState {
  static constexpr uint32_t Stages = Stages_;
  uint32_t phase_ = 0;
  uint32_t index_ = 0;

  PipelineState& operator++() {
    if (++index_ == Stages_) { index_ = 0; phase_ ^= 1; }
    return *this;
  }

  uint32_t index() const { return index_; }
  uint32_t phase() const { return phase_; }
};
```

每个 pipeline 实例用**两份 `PipelineState`**——一份 producer 的,一份 consumer 的,初始相位**反相**:

```cpp
// Ch5 mainloop §5.3 producer/consumer 主循环的真实代码:

// Producer 起点 state
PipelineState write_state = make_producer_start_state<MainloopPipeline>();
// = {phase=1, index=0} —— "我已经假设 producer_ahead 个 stage 已经写满,
// 起点 phase=1 是为了第一次 producer_acquire 立刻成功"

// Consumer 起点 state
PipelineState read_state{};   // {phase=0, index=0} 默认
```

为什么反相:producer 已经"预先 arrive"了 `producer_ahead` 个 stage(默认值 = Stages - 1),第一次 `producer_acquire` 时这些 stage 的 empty barrier 已经是 producer-arrived phase,**consumer 必须用 phase=0 开始才匹配**。

每次 `++state`:
- `index_` 走到 `Stages` 时回 0
- `phase_ ^= 1` 翻面

barrier 的 phase 翻转跟 state 同步——smem stage 上"哪个 phase 算空、哪个算满"严格按 `state.phase()` 决定。

### 4.5 6 个 helper 在 K-loop 里的精确顺序

Ch5 / Ch7 会反复用到这一套,这里把骨架抽出来单独讲:

```cpp
// 起点
PipelineState write_state = make_producer_start_state<MainloopPipeline>();
PipelineState read_state  = {};

// 主循环:每个 iteration 算一个 K-tile
for (int k_tile = 0; k_tile < K_tiles; ++k_tile) {

  // -------- producer 写第 k_tile 个 stage --------
  pipeline.producer_acquire(write_state);   // 1. 等 empty barrier 翻面(consumer 已经 release 上一轮)
  //    隐含 producer.leader 上 arrive_and_expect_tx(transaction_bytes)
  launch_tma_load(...);                     // 2. 发起 TMA(把 smem slot 写满)
  pipeline.producer_commit(write_state, ...);  // 3. TMA 路径上是 NoOp,这里没有 barrier op
  ++write_state;                            // 4. 推进 producer state

  // -------- consumer 读第 k_tile 个 stage --------
  pipeline.consumer_wait(read_state);       // 5. 等 full barrier 翻面(producer 已经 arrive_and_expect_tx)
  launch_wgmma(...);                        // 6. 发起 WGMMA(把 smem slot 读完)
  pipeline.consumer_release(read_state);    // 7. arrive empty barrier(通知 producer 这 stage 空了)
  ++read_state;                             // 8. 推进 consumer state
}
```

关键观察:

- `producer_acquire` 和 `consumer_release` **不在同一个 stage 上操作**——producer acquire 的 empty barrier 是 consumer release 的;consumer wait 的 full barrier 是 producer arrive_and_expect_tx 的。
- `producer_commit` 在 TMA 路径上是 **NoOp**(硬件完成 TMA 时自动 `complete_transaction`)。
- consumer 在 iteration k 时读的是 iteration k 的 stage;producer 在 iteration k 时写的是 iteration k 的 stage——**两边是同一个 stage**。但时序上 producer acquire 比 consumer release 早 K_loop_pipeline_mmas 个 iteration(典型 2 个),实现 K-loop 流水线。
- 第一次 iteration 时,producer 已经预先 arrive 了一些 stage(`producer_ahead`);consumer 第一次 wait 等的就是这些 pre-arrived 的 stage。

### 4.6 Stages 怎么选

`Stages` 决定 smem 上有几个 slot、pipeline 能"预取"几个 tile:

| Stages | smem 用量 | 流水线深度 | 适用 |
|---|---|---|---|
| 2 | 最小 | 1 stage 预取 | smem 紧张 |
| 3 | 中等 | 2 stage 预取 | **Hopper 默认** |
| 4-5 | 大 | 3-4 stage 预取 | 大 tile + 充足 smem |
| 6+ | 很大 | 5+ stage 预取 | 罕见,通常 Stages 受 smem 限制 |

经验法则:

```text
Stages smem 用量 (per operand, fp16):
  Stages × tile_M × tile_K × sizeof(fp16) = Stages × 128 × 32 × 2B = Stages × 8 KB

A 和 B 各自一份,smem 总占用 = 2 × Stages × tile_M × tile_K × sizeof(elem)
```

`Sm90_tma_warpspecialized` 的 dispatch policy 默认 `Stages = 3`(mainloop 文件顶部常量),smem 上 A + B + Stages × tile = 适合典型 128×128×32 tile。

### 4.7 cluster barrier 与 multicast

Hopper 引入 **threadblock cluster**——多个 CTA 共享一个 barrier,允许 multicast TMA + cluster 内 CTA 同步。`PipelineTmaAsync` 的 barrier 是 **ClusterBarrier**(不是普通的 NamedBarrier):

```cpp
using FullBarrier  = cutlass::arch::ClusterTransactionBarrier;  // ← cluster 内可见
using EmptyBarrier = cutlass::arch::ClusterBarrier;             // ← cluster 内可见
```

普通的 `NamedBarrier` 只在单个 CTA 内可见;cluster barrier 可以让多个 CTA 同步。**这意味着 cluster 内所有 CTA 的 producer / consumer 共享同一套 barrier 数组**。

mainloop 实际用时,builder 会算:
- `producer_arv_cnt = num_producers`(同 cluster 内所有 CTA 的 producer warp 数)
- `multicast_consumer_arrival_count = num_consumers`(同 cluster 内所有 CTA 的 consumer warp 数)
- barrier 的 init 在 cluster_shape 算完后才能定

看 `PipelineTmaAsync::init_barriers` 的逻辑。

### 4.8 sm100 上的 pipeline

`include/cutlass/pipeline/sm100_pipeline.hpp` 跟 sm90 同构,但 cluster barrier 升级为 **distributed shared memory barrier**(Blackwell 引入)。具体变化:

- `clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.multicast::cluster::all.b128` PTX 指令(Ch9 Sm100 节会详讲 CLC)
- Distributed shared memory 允许 CTA 直接读其他 CTA 的 smem,**不需要走 gmem**

API 形态跟 sm90 一样,6 个 helper 一一对应,用法不变。

### 4.9 章末:读完这一章你该做得到的事

- ✅ 在 Ch5 mainloop / Ch7 epilogue / Ch9 kernel orchestrator 看到 `producer_acquire` / `consumer_wait` 时,知道每个 helper 在改哪个 barrier。
- ✅ 明白 `PipelineState` 的 phase 翻转跟 barrier 的 phase 翻转同步;producer / consumer 两份 state 起点反相的意义。
- ✅ 能解释"TMA 路径上 `producer_commit` 为什么是 NoOp"。
- ✅ 看懂 `PipelineTmaAsync<Stages>` 的 Stages 参数怎么影响 smem 用量。
- ✅ 区分 `ClusterBarrier` / `NamedBarrier` / 普通 `mbarrier` 三类 barrier 的可见性范围。
- ✅ 知道 sm90 / sm100 pipeline 的对应关系。

下一章 Ch5 看 mainloop——`CollectiveMma` 怎么用这一套 pipeline 把 TMA + WGMMA + producer/consumer 编排成一个 K-loop。

---