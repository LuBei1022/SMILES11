# 人工诊断与自然样本最终验证报告

## 结论

本次离线验证完成 60 条人工标注受控样本和 500 对自然样本复核。自然基线复现状态为 `valid`，五项产物差异总数为 0。

本次未重新运行 GLM，也未重新执行检索；受控样本仅在本地重算确定性指标与诊断，自然样本只使用已有 Trace、Metric 和正式诊断基线进行离线复现。

## 人工标注验证

- 严格完全一致率：31/60 （51.67%；分母：evaluated_count）
- 支持类别条件准确率：31/48 （64.58%；分母：gold_supported_count）
- 诊断覆盖率：47/60 （78.33%；分母：evaluated_count）
- 弃权率：13/60 （21.67%；分母：evaluated_count）

### 按检索器

#### BM25

- 严格完全一致率：15/30 （50.00%；分母：evaluated_count）
- 支持类别条件准确率：15/24 （62.50%；分母：gold_supported_count）
- 诊断覆盖率：24/30 （80.00%；分母：evaluated_count）
- 弃权率：6/30 （20.00%；分母：evaluated_count）

#### DENSE

- 严格完全一致率：16/30 （53.33%；分母：evaluated_count）
- 支持类别条件准确率：16/24 （66.67%；分母：gold_supported_count）
- 诊断覆盖率：23/30 （76.67%；分母：evaluated_count）
- 弃权率：7/30 （23.33%；分母：evaluated_count）

### 固定轴混淆矩阵

| 金标签 \ 预测 | chunking | retrieval | context | generation | STATUS:no_decisive_fault | STATUS:unknown | STATUS:input_error |
|---|---:|---:|---:|---:|---:|---:|---:|
| chunking | 6 | 0 | 0 | 4 | 7 | 0 | 1 |
| retrieval | 0 | 11 | 0 | 4 | 1 | 0 | 0 |
| generation | 0 | 0 | 0 | 14 | 0 | 0 | 0 |
| out_of_scope | 0 | 0 | 0 | 6 | 0 | 0 | 0 |
| unknown | 0 | 0 | 0 | 2 | 4 | 0 | 0 |

## 自然 BM25/Dense 成对复核

- 可配对样本：500
- 诊断状态/根因不一致：251/500（50.20%）
- 检索故障桶：both=98，neither=186，only_bm25=107，only_dense=109
- 五个正式基线文件差异：english_bm25.jsonl=0，english_bm25_summary.json=0，english_dense.jsonl=0，english_dense_summary.json=0，english_bm25_vs_dense.json=0

## 人工复核队列与限制

复核队列共 44 条，来自预测不一致、标注置信度不高于 4、非 diagnosed 状态、不受支持的金标签或标注解析警告的并集。裁决列保持为空。

当前人工标签仅来自一名标注者，尚未完成第二位人员的独立裁决。因此本报告给出单人标注版自动评估结果，不将其表述为双人一致性结论；后续可直接使用 `human_review_queue.csv` 完成人工裁决。

受控 Trace 中有 1 条被诊断输入合约标记为 `input_error`；该状态作为系统预测保留并进入复核队列，没有被修补、删除或映射为人工类别。

## 可复现性

- 验证 schema：`human_diagnosis_validation.v1`
- 诊断规则：`deterministic_rules.v1`
- 所有自然输入与基线均记录 SHA-256，且运行前后必须保持一致。
- 本报告中的计数均由同目录结构化产物生成。
