# MDGraph 2.1

面向 **LAMMPS + 机器学习势函数 + SSH 远程集群**。LangGraph 主 Agent 调度建模、势函数选取、输入编写、作业执行和轨迹分析五个专职 Agent。支持 Slurm 与 PBS，支持 DeePMD、ACE/PACE 和 `eam/alloy` 势函数。

本项目参考 ChatMat 的角色分工、MD-Simulation-Agent 的 Supervisor 与工具组织方式，以及 Vriza 等人的 LAMMPS 多智能体框架。角色、实现和参考来源见[设计说明](docs/design.md)。

## 已实现的范围

- 真正使用 `StateGraph` 和 SQLite checkpointer，不用聊天历史假装持久化工作流。
- 主 Agent 根据各节点返回的状态分派下一步。启用 `llm.enabled=true` 时，规划、结构审阅、势函数审阅、输入审阅、故障解释和报告解读可调用配置的模型。计算、提交和验收始终按代码执行。
- 加上 `llm.decisions: true` 可启用角色决策：主 Agent 决定是否开始，建模 Agent 决定保留坐标或包装进周期盒，选势 Agent 比较兼容候选，输入 Agent 选择受限模板，作业 Agent 决定是否提交，分析 Agent 在数值验收后决定报告要点。各节点通过共享状态回传决定及产物，脚本和验收标准由程序控制。
- ASE 生成指定参数的单元素 SC、BCC、FCC、diamond 或 HCP 超胞，也可读取已有三维周期结构。统一使用 `units metal` 和 `atom_style atomic`，支持 NVT/NPT/NVE，以及立方晶体的 0 K 静态晶胞松弛。
- 建模可用 `structure.builder: ase`（默认）、`atomsk`（强制）或 `auto`（决策模式下从可用工具选）。当前 Atomsk 接口支持单元素 BCC、FCC、金刚石和 HCP 建模及扩胞，由 ASE 读取和转换。已有文件由 ASE 导入；复杂合金与缺陷需先提供结构或另加工具接口。
- 势函数可显式指定，也可按元素顺序、用途、温度及 NPT 或静态松弛的压力从配置目录检索。目录条目可指向本地模型或可下载的 HTTPS 地址，所用文件均核对 SHA256。
- 本地 LAMMPS、SSH+Slurm、SSH+PBS；集群环境由用户提供，程序不猜账户或队列。
- 提交回执、远程提交锁、控制器互斥锁、冻结任务配置；恢复查询已有任务，防止普通重试导致重复提交。
- 仅针对确认的 Slurm NODE_FAIL/PREEMPTED/BOOT_FAIL 自动重新提交，次数有上限；保留每次尝试的全部结果。
- 退出状态、完成标记、实际步数、原子数、有限数值、物理时长、尾段平均温度、可选 NVE 能量漂移检查。
- 总 RDF、基于 image flags 的非折叠坐标 MSD、thermo CSV；输出 JSON/Markdown 报告。

势函数目录中的工况与适用范围是维护者声明。MACE、NEP 等模型需要相应适配器。当前范围是原子体系建模和 MD 执行，势函数训练、DFT 标注、主动学习、复杂结构生成及自由网页搜索属于扩展方向。

`completed` 只表示本次 MD 及所配置数值检查通过，**不是势函数精度认证、采样收敛证明或材料结论**。

## 1. 安装

### 一句话启动：元素立方晶体的弹性常数

主 Agent 从自然语言识别体系、性质和约束，从已注册的建模、选势、LAMMPS 协议和分析能力中选择方案。识别结果和方案依据写入 `manager_plan.json`。建模、选势、输入、执行和分析 Agent 通过任务状态接力。执行器校验已明确给出的工况。

- `mdgraph ask "Calculate the elastic constants of Ag"`。单元素 FCC/BCC 的 0 K、零外压静态计算通过 NIST 和 OpenKIM 在线选势。
- `mdgraph ask "Calculate the RDF of bulk FCC aluminum at 300 K"`。周期性原子体系可先平衡再采样，输出 RDF 和 MSD。仓库中的晶相适用证据和有限温度精度应分别评估。
- `mdgraph ask "Calculate the RDF of bulk water"`。生成带键角和分子 ID 的水盒，比较刚性 SPC/E 和 TIP3P (Ewald)，使用支持 MOLECULE 和 KSPACE 的 LAMMPS。RDF CSV 包含分子间 O–O、O–H 和 H–H；采样收敛需另行检查。
- `mdgraph ask "Simulate water between two graphene plates and calculate its wall-normal density profile"`。主 Agent 给出夹层几何、间距、堆叠、初始载水量和系综。建模 Agent 用 ASE 组装石墨烯，并生成显式水的坐标和键角拓扑。势函数 Agent 记录 SPC/E 和 C–O 参数来源。生产轨迹输出以碳平面为原点的分子质心质量密度，数据与图分别在 `density_profile.csv`、`density_profile.png` 和 `density_profile.pdf`。绘图依赖 Matplotlib，可用 `trajectory` 扩展安装。
- `mdgraph ask "Calculate the RDF of this argon system" --system examples/argon_mdanalysis.yaml`。给其他体系提供已验证的结构、势和运行协议时，Agent 检查任务能否复用该配置。自动生成体系时可用 `--lammps` 选择可执行程序。

自动规划的 RDF/MSD 默认采用正式生产模式。主 Agent 根据性质和体系解释平衡与生产时长。分析记录平衡后半段的分块温度和 NPT 体积，以及生产轨迹的三块一致性。水 MSD 从固定体积轨迹的分子质心计算，平均多个时间原点，检查扩散区间迹象。检查不足时状态为 `sampling_unverified`，原始曲线和诊断保留，后续采样需根据诊断制定。分块诊断用于检验采样稳定性，势函数和有限尺寸效应仍需独立评估。

如只验证流程，使用 `mdgraph ask "Calculate the MSD of bulk water" --smoke`。这时主 Agent 可以选择短采样预算，报告会标为试跑。提供 `--system` 时采用该 YAML 定义的科学协议。

### 在集群登录节点提交 PBS 作业

直接运行 `.venv-hpc/bin/mdgraph` 可进入终端对话。界面和答复默认使用英语，任务与追问可以用中文。主 Agent 接收用户原话、当前作业和最近几轮对话，一次决定是否规划新计算、查询进度、读取结果或回答问题。新计算的结构化方案经校验后保存为 `manager_plan.json`，再由 LangGraph 调度各专业 Agent，并开启后台监控。你可以接着问「算完了吗」「给我结果」，也可输入 `/status`、`/result` 或 `/exit`。用 `/use 运行目录` 或 `mdgraph chat --run 运行目录` 接入已有作业。需要试跑时，用 `mdgraph chat --smoke` 启动。

对话和自动规划默认使用 `deepseek-v4-pro`，思考强度为 `high`。显式任务 YAML 可通过 `llm.reasoning_effort` 或对应的 `llm.roles` 条目指定 DeepSeek 的思考强度。

`mdgraph chat --review` 先提出计算草案，再等待 `/confirm` 或「确认运行」。终端后台规划时可用 `/status` 查看阶段，或用 `/wait` 等待回复。可用自然语言修改草案，用 `/plan` 查看或用 `/discard` 清除。可用 `--draft` 载入之前保存的草案。对于 `1212CNT` 受限水，主 Agent 提出管长、载水量和 NVT 方案，ASE 构建扶手椅型单壁 CNT，分析 Agent 按圆柱壳体积计算 `radial_density_profile.csv`，并输出 PNG 和 PDF。

新的一元或二元蜂窝纳米管材料可以使用同一套 `honeycomb_tube` 组合建模。主 Agent 给出手性、亚晶格元素和文献查询词。Crossref 与 Europe PMC 检索原始研究；势函数 Agent 比较可读全文里与目标结构和水模型匹配的参数，按出处核对电荷、LJ 数值与单位。公开文献摘录或本地原始 PDF 可作为补充，通过 `MDGRAPH_LITERATURE_DOCUMENTS` 指向文献 JSON；它提供出处，不注册材料或力场。模型采用通用的蜂窝卷管、邻接图亚晶格分配、分子装填和 LAMMPS 输入操作。

`mdgraph plan "计算 (12,12) 氮化硼纳米管内受限水的径向密度分布" --output REVIEW_DIR` 在集群登录节点准备结构并调用 LAMMPS `run 0` 检查编译后的输入，然后保存 `pending_plan_validated.json` 和 `preview/`。用 `mdgraph chat --draft REVIEW_DIR/pending_plan_validated.json` 继续审核。生产计算经确认后才提交，运行输入会再次做零步检查。

遇到现有模拟流程无法执行的新体系时，主 Agent 可以进入 `research` 分支。它保留用户原话、后续澄清和已识别的工况，用材料、介质与几何组合查询 Crossref、Europe PMC、PMC 和 OSTI。全文证据不足时，Agent 根据缺口改写查询，最多检索三轮。`research_plan.json` 记录实际查询、已核对的原文、证据覆盖和停止原因。停止为证据充足时，体系与力场须由同一篇原始研究支持，并有依据支撑拟议的分析方法。

可选建模工具 CelluloseForge 帮助建立有出处的纤维素或几丁质晶体片层，ASE 组合周期夹层和显式水坐标。坐标检查通过后可在 `preview/initial.extxyz` 查看模型。OSTI 的 PDF 全文提取使用 `pdftotext`。

`mdgraph chat --draft REVIEW_DIR/research_plan.json` 可重新打开研究草案。后台检索时，`/plan` 可读取初步文献结果和已完成的证据核验。审阅会话可复用同一原话在 24 小时内保存的草案。若需要重新检索，通过 `mdgraph chat --review --refresh-research` 启动。研究草案须先补齐力场、拓扑和求解器验证，才具备提交条件。

主 Agent 的结构化回复校验失败时，终端打印 `MANAGER_VALIDATION_FILE`。对应的 `manager_validation.json` 记录失败字段的路径、错误类型、修复方式和最终动作。JSON 语法错误记录行列号。

集群安装把 `examples/mdgraph-execution.cpu-pbs.yaml` 配置为项目根目录下的 `mdgraph-execution.yaml`。`chat`、`ask`、`run` 和 `prepare` 会读取这个默认执行配置。任务 YAML 自带的集群或 SSH 后端优先，也可以用 `--execution` 指定执行配置。

HPC Agent 提交前用 `qstat -Qf` 和 `pbsnodes` 查询队列和候选节点，记录在 `attempt_00/queue_probe.json`。默认优先使用 `cpu`，当它没有可用候选资源时检查 `testcpu`；两个队列都忙时，在首个接受该资源申请的队列排队。PBS 根据实时资源和队列策略分配节点。`attempt_00/job.json` 记录作业 ID 和队列，`compute_host.txt` 记录实际启动 LAMMPS 的节点。可用 `qstat -f JOB_ID` 查看调度状态。

集群默认申请单节点 32 个 CPU 槽位，用与 LAMMPS 同一环境的 MPICH 启动 32 个 MPI 进程，每个进程使用一个线程。`mpi_launcher` 负责启动，进程数由 `tasks` 决定，`command` 继续指定 LAMMPS。可在 `job.sh` 查看资源申请，并在 `log.lammps` 核对实际 MPI 进程数。

弹性路径在零压平衡晶胞上，以正负两组幅度分别对六个 Voigt 应变方向施加变形；每次固定变形后的胞并松弛原子。程序从应力中心差分得到 6×6 矩阵，检查步长敏感性、立方对称性、残余力和力学稳定性，输出 `attempt_00/elastic.json` 与 `report.md`。不计算有限温度弹性。

各角色默认继承 `llm.model`、`base_url`、`api_key_env` 和 `direct_connection`，也可通过 `llm.roles.<角色>` 分别覆盖。可配置角色为 `manager`、`structure`、`potential`、`designer`、`operator`、`analyst`、`diagnostician`。角色各自构建客户端和提示词，不共享对话历史。

Atomsk 必须装在启动 MDGraph 的机器上，可通过 `structure.atomsk_executable` 显式指定程序。对于 `builder: auto`，只在支持该晶体且找到程序时才把 Atomsk 加入建模 Agent 的动作列表；关闭 LLM 时默认使用 ASE。工具参数来自已验证的 YAML。 `result.json` 的 `tools` 字段、`structure_build.json` 和 `atomsk.log` 可用于核查工具调用。

选势可设 `potential.kind: auto` 并给出 `allowed_kinds`。程序先按势类型、元素顺序、用途、温度和压力筛选目录；若有多个可用候选，选势 Agent 必须逐个比较并返回其中一个 ID。通过后下载或复制该文件并核对 SHA256。可给候选加 `fit_notes` 描述拟合和适用性。选择证据保存在 `attempt_00/potential_decision.json`、`potential_selection.json` 及最终报告。这是基于受控目录的决策，候选间的预测精度仍需独立验证。

### 新体系：按材料和研究目标匹配势函数

从 `examples/new_al_fcc.yaml` 开始。它从晶胞参数生成结构，不用论文算例的势文件、结构或用途标签。复制任务时修改 `goal`、`structure`、`potential`、`protocol` 和 `execution`。任务中的初始晶格参数只是松弛起点。

- 省略 `potential.application`，程序按相和目标性质匹配通用目录。使用内置晶体生成器时相默认取晶格类型；静态松弛的目标性质默认取 `lattice_constant`。导入自建合金结构时填写 `potential.phase`，例如 `fcc` 或更具体的结构标签。其他性质按实际任务填写 `target_property`，独立检查相应计算和分析协议是否已经实现。
- 通用目录（见 `examples/materials_potentials.yaml`）的条目声明 `phases`、`target_properties`、工况、元素类型顺序和来源。合金还必须声明各元素的 `atom_fraction_ranges`，程序以结构实际原子数算出的比例筛选。目录中的适用范围是维护者声明，应有原始资料支持。
- 目录模式没有匹配条目时会停止。可以添加核验过的条目、显式提供模型和元数据，或在支持的任务上改用下面的联网模式。旧的 `application` 仅用于显式选择已有专用目录条目。

### NIST 和 OpenKIM 实时选势

```bash
.venv-hpc/bin/mdgraph ask "Calculate the equilibrium lattice parameter of ordered B2 NiAl at 0 K and 0 bar."
```

自然语言晶体任务由主 Agent 生成结构、工况和观测量。势函数 Agent 每次新任务实时查询 NIST 与 OpenKIM，根据实际元素与相态比较候选，并决定下载哪个模型或说明证据不足。程序验证所选文件及元素映射，在已构建结构上执行 LAMMPS `run 0`。如果所选文件下载或检查失败，势函数 Agent 根据失败证据复选其余候选。用 `potential_discovery.json`、`potential_decision.json`、`potential_verification.json` 和 `potential_selection.json` 检查检索、决定、验证与文件哈希。运行中断后复用该次任务的检索记录和经过验证的参数。

默认没有预选势文件，也无需准备材料目录。自然语言建模支持单元素 FCC、BCC、SC、金刚石结构，以及有序二元 B2、岩盐和闪锌矿结构。静态弹性张量入口目前限于单元素 FCC/BCC。也可以在自备原子结构任务中用 `potential.kind: auto` 和 `discover: repositories`。集群需能访问两个仓库的 HTTPS 端点。NIST 接受兼容的原生 EAM、SW 和 Tersoff 文件，OpenKIM 适配器读取 Dynamo EAM 模型包中的原生参数表，能在没有 KIM 插件的 LAMMPS 上运行这些表。`discover: nist` 仍适用于旧的显式任务。

已登记的分子计算沿用兼容拓扑。目录外分子可进入势函数 Agent 的联网检索和证据审阅，缺少相应拓扑或可执行参数时报告所缺材料。`run 0` 验证当前 LAMMPS 能读取并评估结构与势函数，正式计算的物性精度还需按任务对照参考数据。

### MD 轨迹分析：ASE、MDAnalysis 和 Matplotlib

内置分析未覆盖的性质可以由分析 Agent 编写 Python 估计器。主 Agent 会检查现有体系和轨迹字段能否支持该性质。已有轨迹可用 `mdgraph analyze RUN_DIR "Compute the mean oxygen coordination number within 3.5 angstrom"` 直接重分析。生成的脚本、单位、方法、数据来源和 CSV 保存在独立的日期目录。Linux 执行节点需要 `libseccomp.so.2`，生成脚本使用预加载的 NumPy、SciPy 和 ASE 数值函数。若需要轨迹中未保存的数据，报告会指出缺失项。

`python -m pip install -e '.[llm,trajectory]'` 安装可选轨迹工具。在 NVT/NPT/NVE 的 `validation` 中设 `trajectory_backend: mdanalysis`，用 MDAnalysis 计算 RDF 的周期距离；默认用 ASE。设 `auto` 时，若启用了 LLM 决策并且两种工具都可用，分析 Agent 从中选用一种，选择会写入 `trajectory_tool_decision.json`。未启用 LLM 时，自动模式优先用可用的 MDAnalysis。

`species_resolved: true` 输出元素分辨的 RDF 和 MSD；`trajectory_plots: true` 用 Matplotlib 生成 `trajectory_summary.png`。结果在 `attempt_00/` 的 `rdf.csv`、`msd.csv` 和 `validation.json`；报告记录库版本和 RDF 所用的帧范围。MSD 以首帧为参考，用 LAMMPS image flags 展开坐标，不自动拟合扩散系数。0 K 静态松弛没有时间轨迹，仍走晶格参数分析。集群示例：`mdgraph run examples/argon_mdanalysis.yaml --watch`。

建议 Python 3.11/3.12。在项目根目录运行 `python -m pip install -r requirement.txt` 可以安装 MDGraph 及 DeepSeek 客户端依赖。控制器可以在本地运行，通过 SSH 提交任务。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e '.[dev,llm,smoke]'
```

Linux 控制器：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,llm]'
```

`smoke` 安装 LAMMPS Python wheel，仅用于本地冒烟验证。真实 ML 势运行需要集群上匹配版本的 LAMMPS 与 ML 插件。依赖范围见 `pyproject.toml`，本机验证环境清单见 `requirements-tested.txt`（生成后提供；包含平台相关软件包，不是跨平台锁文件）。

Windows 的 LAMMPS wheel 还需要 Microsoft MPI 与 VC++ 运行库。缺少 DLL 时不会自动安装系统软件或伪造模拟结果；请按 [LAMMPS 官方说明](https://www.lammps.org/download/windows/)配置。也可通过本进程环境变量 `MDGRAPH_DLL_DIR` 指向已有、已核验的项目内运行库目录。

## 2. 无 API Key 跑通真实 MD

```powershell
python -m mdgraph.cli run examples/argon_local.yaml --watch
```

在本次已准备的 Windows 工作目录中，也可运行 `powershell -File scripts/run_demo.ps1`。该脚本只对当前进程设置已有项目内 MPI 库路径，不安装系统软件、不修改系统 PATH；源码包不分发该第三方运行库。

这个示例运行 256 个 Ar 原子的真实 LAMMPS LJ 模拟，共 2000 步、2 ps，验证执行链路。它被明确限制为 `lj_demo`，**不是机器学习势函数结果**。

**计算 FCC Cu 的静态晶格常数：**

```powershell
python -m mdgraph.cli run examples/cu_eam_lattice.yaml --watch
```

在使用项目内便携 MPI 的 Windows 环境，先设置 `$env:MDGRAPH_DLL_DIR = (Resolve-Path .tools/msmpi/portable).Path`。此例通过本地目录选取 Mishin EAM1 势并校验 SHA256，使用 `ensemble: relax`、`temperature_k: 0` 做各向同性晶胞松弛。分析节点将最终盒长除以各轴的晶胞重复次数，并检查应力和最大原子力。结果见 `report.md`、`result.json`、`attempt_00/relaxation.csv` 和 `attempt_00/final.data`。本计算是静态能量最小化，无时间轨迹；使用 LAMMPS 的 MANYBODY 包即可。

命令会打印 `RUN_DIR=...`。输出目录以运行系统本地日期时间命名，不覆盖旧运行。

```text
runs/YYYYMMDD_HHMMSS_name_uniqueid/
├── task.json                  # 冻结配置，不含 API Key
├── run_manifest.json          # 配置哈希
├── checkpoints.sqlite         # LangGraph 状态
├── manager.json               # 角色输出，标明 LLM/规则模式
├── status.json
├── result.json
├── report.md
└── attempt_00/
    ├── manifest.json          # 文件哈希、原子数、单位与映射
    ├── in.lammps
    ├── structure.data
    ├── initial.extxyz
    ├── model.pb / model.yace   # 真实 ML 模型；demo 不需要
    ├── job.json               # 本地 PID 或集群 job ID
    ├── job.sh                 # 远程运行时生成
    ├── log.lammps
    ├── trajectory.lammpstrj
    ├── final.data
    ├── final.restart
    ├── thermo.csv
    ├── rdf.csv
    ├── msd.csv
    └── validation.json
```

## 3. 从势函数目录准备 Cu 算例

```powershell
python -m mdgraph.cli prepare examples/cu_catalog_md.yaml
```

主 Agent 调用三个准备角色，构建 256 原子的 FCC Cu 超胞，从 [示例目录](examples/cu_catalog.yaml) 匹配势函数、下载 LAMMPS 发布的 Cu ACE 文件并核对哈希，生成 LAMMPS 输入。流程暂停于作业提交前。准备记录见 `attempt_00/structure_build.json` 和 `attempt_00/potential_selection.json`。

目录条目需要声明 `id`、`kind`、`elements`（按 LAMMPS 类型顺序）、`units: metal`、`application`、`temperature_k: [最低, 最高]`、`source`、`model_sha256`，以及 `model` 或 `model_url`。NPT 与静态松弛匹配还要求 `pressure_bar: [最低, 最高]`。条目的相对模型路径以目录文件为准。多个条目匹配时，用任务配置中的 `potential.candidate_id` 指定其中一个。使用自己核验的来源和适用范围维护目录。

示例中的温度与用途仅用于演示匹配。实际 Cu 模拟需要支持 ML-PACE 的 LAMMPS。可以在准备前编辑 YAML 中的运行环境，生成独立运行目录。日常使用可直接提供模型和元数据，见下一节。

## 4. 配置真实 ML 势与远程集群

复制 `examples/deepmd_slurm.yaml` 或 `examples/pace_pbs.yaml` 到你的工作目录，再按实际情况填写：

1. 结构文件、元素顺序、已经训练好的模型。
2. 势函数元数据 JSON：参考 `examples/potential_metadata.example.json`。`elements` 必须与模型内部 type_map 一致；当前检查的是**用户声明**，不会解析模型内部映射。`model_sha256` 必须是该模型真实哈希。
3. 集群 hostname、username、绝对 remote_root、队列/分区、资源参数、模块环境、LAMMPS 启动 argv。
4. 科学合理的温压、时间步、系综和检查阈值；示例数值不是对你的体系的建议。

计算 SHA256：

```powershell
Get-FileHash -Algorithm SHA256 'D:\models\graph.pb'
```

```bash
sha256sum /path/to/graph.pb
```

SSH 使用本机 agent/密钥和已有 `known_hosts`。先用正常 SSH 验证主机指纹和连通性；程序不会自动信任未知主机。`host` 填实际 hostname/IP，当前不解析 `~/.ssh/config` 的别名或 ProxyJump。若集群必须经跳板机，需先建立本机可访问的 SSH 转发再配置端口。

所有文件路径相对于任务 YAML 所在目录解析（集群 `remote_root` 除外）。`command` 是 argv 数组；`setup` 是管理员提供的 Bash 初始化语句，属于受信任配置，不交给 LLM 编写。

仅准备并停在提交节点。采用 HTTPS 势函数条目时，此阶段会下载模型：

```bash
python -m mdgraph.cli prepare my_cluster_task.yaml
```

检查准备目录的 `in.lammps`、`manifest.json`、模型映射与物理参数后，可启动该目录：

```bash
python -m mdgraph.cli resume runs/实际运行目录 --watch
```

也可直接准备并提交：

```bash
python -m mdgraph.cli run my_cluster_task.yaml --watch
```

不加 `--watch` 时在作业等待处持久化暂停，稍后查询/继续：

```bash
python -m mdgraph.cli status runs/实际运行目录
python -m mdgraph.cli resume runs/实际运行目录 --watch
```

控制器退出或达到 `max_watch_seconds` 不会取消集群作业。再次 resume 会查询同一 job ID。取消请根据回执明确执行 `scancel JOBID` / `qdel JOBID`；首版不自动取消或清理远程结果。

提交期间断网且无确认回执时，程序会报 `AmbiguousSubmission` 并阻止重提。检查远程 `.job_id`、`.submit_lock`、`.submit_error` 和调度器记录。锁的存在不证明作业已经提交，也不证明未提交，**不要自动删除锁后重提**。确认调度器已有任务后可恢复 `.job_id`；确认从未提交则创建新的运行目录。远程 `.job_id` 格式为 `sbatch --parsable` / `qsub` 的原始回执。

## 5. 开启多角色 LLM

安装 `.[llm]`，在任务 YAML 中设置 `llm.enabled: true`、实际 `model`，需要时设置兼容服务的 `base_url`。把 API Key 放在本机环境变量 `MDGRAPH_API_KEY`，不要写入任务 YAML、日志或聊天。

模型生成结构化的规划和审阅意见。节点调度读取程序状态，势函数匹配读取目录与配置，LAMMPS 输入由模板生成。LLM 服务异常时可在服务恢复后用 `resume` 继续。

## 6. 测试与扩展

```bash
python -m pytest -q
python -m mdgraph.cli graph
```

测试覆盖输入校验、单位转换、模型映射、哈希防篡改、不完整运行拒绝、NaN/原子丢失、三斜盒展开、断点恢复不重提、提交不确定状态与控制器互斥。模拟后端的测试数据只验证流程，真实 MD 另由 argon 示例验证；集群提交需要你的实际环境做验收。

扩展入口：

- `src/mdgraph/config.py`：严格任务 schema。
- `src/mdgraph/agents.py`：角色与结构化 LLM 输出。
- `src/mdgraph/prepare.py`：结构与势函数适配器、模板。
- `src/mdgraph/backends.py`：本地/SSH、Slurm/PBS 作业执行。
- `src/mdgraph/graph.py`：节点、分支、有限重试、报告。
- `src/mdgraph/analysis.py`：真实日志、RDF/MSD 与数值检查。

后续添加主动学习时，在失败诊断或验证节点后连接“选样→DFT→数据质检→训练→重新采样”子图。模型分歧不能直接当成真实误差，需要独立 DFT 参照和经过校准的阈值。

## 科学与工程限制

- NPT 固定各向同性压力控制；不能代表所有材料所需的形变方式。
- MSD 使用 image flags 展开后的坐标，包含集体漂移；NPT 盒形变可贡献位移，不自动拟合扩散系数。
- RDF 默认统计总 RDF，并可通过 `species_resolved` 输出分物种 RDF。截断距离受最短盒高限制。
- 温度均值检查不是平衡或统计收敛判据；NVT/NPT 不检查“总能量守恒”。
- `max_watch_seconds` 限制本次控制器等待，不是总计算预算；计算资源由队列资源和 walltime 约束。
- Slurm 重试从相同初始输入重新开始，不声称从最后 restart 续算；首版保留 restart 供明确配置后使用。
- SQLite 用于单控制器；多用户服务应改成 PostgreSQL 与独立工作进程。
- 默认不下载中途 checkpoint 文件，只回收最终 restart、轨迹与日志；大轨迹下载仍可能耗时。
## 分子体系

可以直接输入 `calculate the surface tension of water`、`计算CO2的自扩散系数` 或 `计算乙醇的自扩散系数`。主 Agent 把选定的相态、温压、采样长度及依据写入 `manager_plan.json`。选势 Agent 选择匹配的已登记模型，建模 Agent 依据所选模型的拓扑建立分子盒。采样结束后，分析 Agent 选择分块或拟合窗口，并在报告中记录相应的敏感性和不确定度检查。

分子模型和适用相态登记在 `src/mdgraph/molecular_models.yaml`。本地管理员可以通过 `MDGRAPH_MOLECULAR_CATALOG` 指定同格式的模型目录，注册其他使用同类拓扑和相互作用的纯流体，无需修改主 Agent 或扩散分析代码。每次选中的模型参数和来源会保存到运行目录并纳入输入校验。

- 自扩散采用解包的分子质心轨迹，对三维 MSD 的扩散区间做拟合。结果以 m²/s 输出，并记录拟合窗口、分块误差和窗口敏感性。
- 表面张力采用三维周期液–气平板、长程静电与色散，以及法向和切向压力张量。结果以 mN/m 输出，同时保存分块数据和密度剖面。
- `sampling_unverified` 表示数值执行结束、当前采样未达到配置的诊断条件。单盒和单力场的分块误差不包含尺寸及模型偏差。

