# 结构改造测试体系网络调研：推荐 EGFR–gefitinib（4WKQ）

查询日期：2026-09-27

## 1. 结论

建议把当前 CDK2/NU6094 测试体系替换为：

- **靶点**：人 EGFR kinase，ChEMBL `CHEMBL203`，UniProt `P00533`
- **结构**：PDB **4WKQ**，WT EGFR–gefitinib，分辨率 **1.85 Å**
- **配体**：gefitinib，ChEMBL `CHEMBL939`，PDB ligand code `IRE`
- **主实验 SAR 文献**：Barker 等，*Studies leading to the identification of ZD1839 (IRESSA)*，DOI `10.1016/S0960-894X(01)00344-4`，ChEMBL `CHEMBL1134804`
- **辅助陡峭 SAR 文献**：Bridges 等，PD153035 analogues，DOI `10.1021/jm9503613`，ChEMBL `CHEMBL1129291`

在本次调查的 10 个常用药物靶点中，EGFR 的公开实验改造记录最多：**11,318 个具有精确 IC50/Ki/Kd 记录的独立 ChEMBL 分子、21,315 条活性记录、1,093 个关联文档**。这显著高于当前 CDK2 的 2,753 个分子和 342 个文档。

这不是“全球所有文献”的绝对证明，而是依据同一 ChEMBL 查询规则得到的可复现比较；在被比较的常用结构药化体系中，EGFR 排名第一。

## 2. 候选体系比较

统一纳入标准：ChEMBL target exact match，`standard_relation = '='`，`standard_type in {IC50, Ki, Kd}`。

| 排名 | 靶点 | 独立实验分子 | 活性记录 | 文档 | 含非聚合物的 RCSB 条目（结构丰富度代理） |
|---:|---|---:|---:|---:|---:|
| 1 | EGFR | 11,318 | 21,315 | 1,093 | 352 |
| 2 | BACE1 | 10,183 | 15,543 | 511 | 411 |
| 3 | BRD4 | 9,377 | 16,198 | 446 | 603 |
| 4 | Factor Xa | 6,087 | 7,694 | 381 | 188 |
| 5 | Thrombin | 5,929 | 6,906 | 510 | 437 |
| 6 | HIV-1 protease | 4,936 | 6,718 | 346 | 425 |
| 7 | ABL1 | 3,127 | 6,290 | 339 | 69 |
| 8 | CDK2 | 2,753 | 3,556 | 342 | 488 |
| 9 | HSP90α | 1,690 | 2,198 | 126 | 423 |
| 10 | DHFR | 1,301 | 1,903 | 173 | 88 |

完整数据见 `target_survey.csv` 和 `target_survey.json`。

## 3. 文献统计

EGFR 全量 ChEMBL 文档表见：

- `egfr_all_chembl_documents.csv`
- `egfr_all_chembl_documents.json`

它们覆盖本次查询命中的全部 **1,093 个 ChEMBL 文档**：

- 988 篇 publication
- 93 项 patent
- 12 个 dataset
- 997 个文档具有 DOI
- 959 个文档具有 PubMed ID

高密度药化文献子集见：

- `egfr_publication_series_ge25.csv`
- `egfr_publication_series_ge25.json`

该子集包含 **117 篇 publication**，每篇至少关联 25 个直接 EGFR 精确效力分子，适合后续逐篇抽取 SAR 表。

注意：文档级全量表覆盖 ChEMBL 收录范围，不包含未进入 ChEMBL 的论文、补充材料中未结构化的记录或只有定性活性的化合物。

## 4. 主测试集：gefitinib discovery series

来源：ChEMBL `CHEMBL1134804`，DOI `10.1016/S0960-894X(01)00344-4`。

### 4.1 数据规模

- 同一直接 EGFR-TK assay 中共有 **28 个分子**
- 以 gefitinib 为参考，相当于 **27 个替代结构改造方案**
- gefitinib IC50：**23 nM**
- 全系列 IC50：**2–310 nM**
- 相对 gefitinib：
  - 12 个更强
  - 15 个更弱
  - 1 个参考分子

逐分子数据、SMILES、R-group、改造方向、fold change 和 ΔpChEMBL 见：

- `gefitinib_discovery_series_28.csv`
- `gefitinib_discovery_series_28.json`

### 4.2 已确认的 gefitinib 衍生扩展文献

除原始 discovery series 外，已确认至少两套可以直接扩充结构参考库的实验系列：

1. **C6 carbon-linked 4-anilinoquinazolines**，DOI `10.1016/j.bmcl.2006.02.025`，ChEMBL `CHEMBL1147701`：35 个分子、69 条直接 EGFR/EGF-stimulated KB 活性记录。gefitinib 在同文献中的 biochemical EGFR IC50 为 33 nM，在 KB assay 中为 54 nM。该系列适合扩展 C6 侧链设计空间，但 C6 的 O-linked → C-linked 属于比末端基团替换更大的改动。
2. **Tricyclic oxazine/oxazepine-fused gefitinib analogues**，DOI `10.1016/j.bmcl.2016.08.007`，ChEMBL `CHEMBL4431348`：14 个分子、23 条活性记录，其中全部有 cellular pEGFR 数据，9 个有 biochemical EGFR 数据。该系列通过环融合刚性化 C6/C7 溶剂区，适合作为较高难度的 pose-retention 测试。

三套已结构化系列合计 **120 条 assay-level 活性记录、75 个独立分子**；排除 gefitinib 和 erlotinib 两个对照后，有 **73 个独立 analogue**。统一长表见：

- `gefitinib_related_reference_database.csv/json`
- `gefitinib_related_reference_database_summary.json`

单系列宽表见：

- `gefitinib_c6_carbon_linked_series_35.csv/json`
- `gefitinib_tricyclic_analogues_series_14.csv/json`

另有 pyrrolidino、piperazino、N-alkyl 等直接 gefitinib 衍生文献，已登记在 `gefitinib_analogue_literature_index.csv/json`。其中部分没有可直接复用的 ChEMBL 标准化直接-EGFR表，暂只作为待人工抽表或定性来源，不能混入严格定量 benchmark。

**重要：不同文献/assay 的 IC50 不能合并成一条数值标尺。** 新数据库中的 fold change 只相对于同一 assay 内测得的 gefitinib 计算。

### 4.3 主要改造方向与实验变化

| 改造方向 | 代表化合物 | IC50 | 相对 gefitinib |
|---|---|---:|---:|
| C6 侧链改为二胺/末端二甲氨基 | CHEMBL56936 | 2 nM | 11.5× 提升 |
| C6 侧链改为氨基醇 | CHEMBL56266 / CHEMBL56502 | 5–6 nM | 3.8–4.6× 提升 |
| C6 侧链改为带多羟基叔胺 | CHEMBL299893 | 5 nM | 4.6× 提升 |
| C6 侧链缩短为甲氧基 | CHEMBL301018 | 9 nM | 2.56× 提升 |
| morpholine linker 从三碳缩短到二碳 | CHEMBL14699 | 10 nM | 2.3× 提升 |
| morpholine 三碳链，即 gefitinib | CHEMBL939 | 23 nM | 参考 |
| 末端换为 pyrrolidine/piperidine | 多个 | 71–98 nM | 约 3–4× 下降 |
| 某些支化或较疏水的无环胺 | CHEMBL55794 | 310 nM | 13.5× 下降 |

总体 SAR：

1. 4-anilinoquinazoline hinge-binding core 可以保持不变，主要探索 C6 溶剂区侧链。
2. 小型、柔性、带极性或可质子化的末端基团总体更有利。
3. 单纯增大饱和环体积，如 pyrrolidine/piperidine，通常降低活性。
4. linker 长度非常敏感；同一 morpholine 末端，二碳 linker 优于 gefitinib 的三碳 linker。
5. 该系列既有明确提升，也有明确失败案例，适合测试 agent 是否能从 pocket/pose 证据学习，而不是只生成任意可成药片段。

## 5. 辅助验证集：PD153035 steep-SAR series

来源：ChEMBL `CHEMBL1129291`，DOI `10.1021/jm9503613`。

### 5.1 数据规模

- 64 个分子，即相对 PD153035 的 **63 个替代 analogues**
- 全部使用同一 EGFR assay：A431 cell vesicle EGFR 对 PLCγ1 片段的磷酸化抑制
- PD153035 IC50：**0.025 nM**
- 全系列范围：**0.006–12,000 nM**
- 动态范围约 **2,000,000 倍**
- 仅 1 个 analogue 优于 PD153035，62 个更弱，另 1 个为参考

逐分子表见：

- `pd153035_steep_sar_series_64.csv`
- `pd153035_steep_sar_series_64.json`

### 5.2 改造方向统计

| 改造方向 | 分子数 | 实验 IC50 范围 |
|---|---:|---:|
| 保留 3-Br aniline，改 quinazoline 6/7 取代或区域化学 | 32 | 0.006–2,000 nM |
| 同时修改 aniline 和 quinazoline 6/7 区域 | 19 | 0.25–12,000 nM |
| 保留精确 6,7-dimethoxy，扫描 aniline 取代/位置 | 7 | 0.24–128 nM |
| 保留参考主要 motif，进行其他局部/核心变化 | 6 | 0.025–463 nM |

代表变化：

- 6,7-dimethoxy → 6,7-diethoxy：0.025 → 0.006 nM，约 4.17× 提升。
- aniline 3-Br 改为 3-Cl、3-I、3-F、3-CF3 或位置异构体时，活性跨越亚 nM 到百 nM。
- 6/7 取代位置互换或引入 nitro 等变化可使活性下降至 μM 级。

该系列非常适合检验模型是否识别“很小的区域化学变化造成巨大活性变化”，但 PD153035 没有本次确认到的同配体高分辨率 PDB，因此建议作为外部 SAR 验证集，而不是主结构输入。

## 6. 推荐的实际测试设计

### 主任务：结构严格对应

- 使用 PDB `4WKQ`
- 使用 gefitinib 原始共晶 pose
- 实验 baseline：23 nM
- 只允许优先修改 C6 solvent-exposed side chain
- 共同 4-anilinoquinazoline core 保持
- 将同文献其余 27 个分子作为隐藏实验验证集

成功标准不只看 docking score，还应包括：

1. 是否恢复文献中已知的有利方向；
2. 是否避免 pyrrolidine/piperidine 等已知不利方向；
3. 是否生成与 2–10 nM 高活性 analogues 相同或相近的 transformation；
4. candidate common-core pose 是否保持 4WKQ 结合模式；
5. docking 排序与同 assay 实验 IC50 的 Spearman 相关性。

### 可选难度更高任务：大动态范围

把同系列弱化合物 CHEMBL55794（310 nM）通过共同核心对齐到 4WKQ，要求 agent 找到已知的侧链缩短、极性化和二胺方向。该任务具有更大的改善空间，但起始配体不是原始共晶配体，应标记为 aligned analogue，而非实验共晶 pose。

## 7. 限制

- ChEMBL 的跨文献 IC50/Ki/Kd 不能直接混合比较；因此详细活性变化只在同一文献、同一 assay 内计算。
- RCSB “含非聚合物条目数”仅作为结构丰富度代理，包含离子、缓冲剂和其他非配体成分，不等于纯配体复合物数。
- broad EGFR 记录混合 WT、突变体、可逆、共价、allosteric 和细胞/蛋白 assay；主测试集应固定 WT EGFR、可逆 ATP-site、同 assay。
- `4WKQ` 的 gefitinib 是 WT EGFR 高分辨率结构，最适合当前 scaffold-preserving 测试；不要把 T790M/C797S 系列直接并入同一个评价集合。

## 8. 数据文件

- `source_manifest.json`：数据来源、查询规则和推荐体系身份
- `target_survey.csv/json`：10 个候选靶点比较
- `egfr_all_chembl_documents.csv/json`：全部 1,093 个关联 ChEMBL 文档
- `egfr_publication_series_ge25.csv/json`：117 篇高密度 publication
- `gefitinib_discovery_series_28.csv/json`：主测试 SAR 表
- `pd153035_steep_sar_series_64.csv/json`：辅助陡峭 SAR 表
- `gefitinib_analogue_literature_index.csv/json`：直接或近直接 gefitinib 衍生文献的分级索引
- `gefitinib_c6_carbon_linked_series_35.csv/json`：C6 carbon-linked 扩展系列
- `gefitinib_tricyclic_analogues_series_14.csv/json`：三环刚性化 gefitinib analogue 系列
- `gefitinib_related_reference_database.csv/json`：三套 gefitinib 相关系列的标准化 assay-level 长表
- `gefitinib_related_reference_database_summary.json`：扩展数据库统计和 assay 隔离警告
