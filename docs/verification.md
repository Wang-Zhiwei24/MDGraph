# 验证记录

## 2026-11-19 专职 Agent 改版

- `python -m pytest -q` 通过 28 项测试。新增覆盖主 Agent 分派、准备后恢复提交、候选歧义与模型哈希异常。旧版运行仍按原图恢复。
- `runs/20261119_164711_cu_catalog_md_ea65ae85/` 使用新图构建了 256 原子的 FCC Cu 超胞，下载并核验了 LAMMPS 发布的 ACE 势，生成 `in.lammps` 后暂停在提交前。本机测试用的 LAMMPS wheel 缺少 ML-PACE，因此这次验证停在输入准备阶段。
- `runs/20261119_164507_argon_smoke_161284b8/` 使用新版 LangGraph 完成了 256 原子 Ar 的 2000 步 LAMMPS LJ 模拟，生成轨迹分析和报告。
- 远程生产集群与在线 LLM 服务仍待实际环境验收。

日期：2026-11-17。平台：Windows，Python 3.12.14。

## 自动化检查

- `python -m pytest -q`：**25 passed**。
- 覆盖 schema、单位转换、模型映射/哈希、输入防篡改、非完整运行、NaN、原子丢失、三斜盒展开、检查点恢复、提交不确定状态、控制器互斥、Slurm 终态分类与有限重试。
- 远程测试使用明确标注的内存 SSH/调度器契约夹具，没有提交真实集群作业。
- `python -m pip check`：No broken requirements found。
- LLM 客户端与 Pydantic 结构化输出构造成功；没有发送真实 API 请求，没有使用密钥。

## 真实 LAMMPS 冒烟计算

运行目录：`runs/20261117_163844_argon_smoke_7a463a1e/`。

| 指标 | 实际结果 |
|---|---:|
| 原子数 | 256 Ar |
| 势函数 | 明确标注的 LJ 演示势 |
| 系综 | NVT |
| 目标温度 | 50 K |
| 时间步 | 1 fs |
| 实际完成步数 | 2000 |
| 时长 | 2 ps |
| 后半段 thermo 平均温度 | 49.7037815 K |
| 轨迹帧数 | 11 |
| RDF 截断 | 8 Å |
| MSD 末值 | 0.06224958 Å² |
| 配置运行检查 | 通过，无 issues |

原始输入、LAMMPS 日志、最终结构/restart、轨迹、RDF/MSD CSV 和完整 JSON 报告均保存在上述目录。对已完成目录再次执行 `resume` 返回 completed，不重新提交或重跑模拟。

初次本地尝试因缺少 MS-MPI DLL 失败，失败日志保留在 `runs/20261117_163141_argon_smoke_88a76845/`。随后使用项目内运行库完成上述真实模拟，没有安装系统软件或修改全局 PATH。测试用 MPI 运行库来自 conda-forge 的 `win-64/msmpi-10.1.1-h10f94be_9.conda`，包 SHA256 为 `8beb4a5e31b4f235d5792d72c3a5329915886b9c84cef4df5e2b9ed1ee3701f1`，不打包分发该第三方库。

## 尚未完成的生产验收

- 实际目标集群的 SSH、Slurm/PBS 与模块环境。
- 用户真实 ML 势文件及其 LAMMPS 兼容性、独立参考精度。
- 真实 LLM 服务调用。

该 Ar 计算用于验证执行和分析链路，不是 ML 势基准，也不能验证生产材料的物理正确性。原始运行数据仍在项目目录，源代码压缩包不含环境、第三方运行库、原始轨迹或上游参考仓库。

## 2026-11-19 Cu EAM 静态晶格常数

使用 LAMMPS 22 Jul 2025 Update 4，通过实际运行的 LangGraph 主 Agent 顺序调度 `structure_agent → potential_agent → input_agent → operator → monitor → collector → analysis_agent → report`。势函数节点从 `examples/cu_eam_catalog.yaml` 选取并校验 Mishin EAM1 势；执行节点启动本地 LAMMPS。此次 `llm.enabled: false`，未调用外部 LLM 或集群。

| 独立运行 | 起始晶格常数（Å） | 平衡晶格常数（Å） | 最终压力（bar） |
| --- | ---: | ---: | ---: |
| `runs/20261119_212612_cu_eam_lattice_305039f5/` | 3.50 | 3.614925065888075 | −1.06×10⁻¹⁰ |
| `runs/20261119_212651_cu_eam_lattice_expanded_508eb777/` | 3.75 | 3.614925065888075 | −6.98×10⁻¹⁰ |

两个 256 原子 FCC 周期超胞在 0 K、零外压下的每原子势能均为 −3.540218330204 eV。晶格常数与能量均在 10⁻⁸ 的绝对容差内复现先前直接调用 LAMMPS 的基准 `runs/20261119_2120_cu_eam_lattice/`。另复跑原有 `runs/20261119_212658_argon_smoke_bdf2430e/`，确认 2000 步 NVT 和轨迹分析仍正常；`python -m pytest -q` 通过 33 项。

## 2026-11-20 角色决策与六体系复跑

- 本地及集群验证均为 36 项测试通过。测试覆盖选势改变实际势文件、作业 Agent 可阻止提交、恢复不重复选势、非法 ID 和不完整比较被拒绝。
- 集群批次 `runs/20261120_1032_decision_table3/20261120_1038_verified_handoff/`：6 个体系均调用 DeepSeek Flash 完成各角色决策并通过数值验收。每个体系有两份真实候选；均选择与复跑目标一致的作者归档势，所用文件与选择记录的哈希一致。
- 第一轮 Ti 因输入 Agent 没有收到势文件来源及 SHA256 而在执行前停止，补充交接字段后六体系统一重跑，原日志保留。
- 本地 `runs/20261120_1053_decision_table3/decision_report.md` 保留逐项比较和角色理由，结果包包含各角色的 JSON 决策以及实际 LAMMPS 输入、日志和验收结果。
