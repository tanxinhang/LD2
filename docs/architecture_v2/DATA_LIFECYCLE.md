# 数据生命周期 V2

## 1. 数据分级

| 等级 | 内容 | 策略 |
|---|---|---|
| source | 代码、活动配置、seed bank、schema | Git 管理 |
| raw | 原始 trace/checkpoint/solver 输出 | 不可变、按 run_id 存储 |
| derived | 指标、表格、图、聚合数据 | 可由 raw + manifest 重建 |
| published | 论文或正式审计引用的结果索引 | 只保存引用与校验和 |
| scratch | smoke、崩溃探针、重复 console log | 有 TTL，可清理 |
| legacy | 缺 manifest 或旧物理语义的数据 | 只读隔离，不参与新结论 |

## 2. 新产物布局

```text
artifacts/
  runs/<run_id>/manifest.json
  runs/<run_id>/raw/
  runs/<run_id>/derived/
  published/<release>.yaml
  legacy/catalog.jsonl
  cleanup/candidates.jsonl
```

`run_id` 由代码提交/dirty patch、解析配置哈希、输入 checkpoint、seed 集合和运行类型共同
确定。manifest 至少记录命令、依赖锁、平台、开始/结束时间、schema、输入输出哈希和状态。

## 3. 清理规则

当前架构审计与打包复现已经通过，`destructive_data_cleanup=true`。清理仍采用最小授权：
以下任一条件成立即不得删除：被文档引用、属于 frozen/blind 证据、没有保留副本、没有
校验和、或无法确认生成命令。

优先清理候选是可重建的缓存、重复 console log、失败 smoke 的中间张量和已过 TTL 的临时
探针。缺 provenance 的大文件先标为 legacy，不因“看起来过时”直接删除。

正式清理必须执行：候选清单 → 引用扫描 → 哈希/保留副本验证 → 明确授权 → 精确路径删除 →
删除报告。目录通配符和递归根目录删除不允许进入清理脚本。

当前 metadata catalog 写入 `artifacts/legacy/catalog.jsonl`，清理候选写入
`artifacts/cleanup/candidates.jsonl`。候选项默认 `deletion_authorized=false`；完成引用复核与
SHA-256 校验前不得改变。

## 4. 结果刷新隔离

V2 全量刷新写入新 run_id，旧结果保持只读。只有审计通过的 run_id 才能进入
`published/`；文档引用逻辑名称和哈希，不引用“latest”目录。
