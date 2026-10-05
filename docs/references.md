# References and attribution

SwarmBots builds on prior reinforcement-learning methods and scientific software. This guide identifies the sources of the included learners, the related work discussed in the thesis, and the main software dependencies. BibTeX entries for the works below are available in [`references.bib`](../references.bib); the citation keys are shown in backticks.

## What to cite

- **Using the benchmark:** cite the software version described by [`CITATION.cff`](../CITATION.cff) and Dominik Baron's [SwarmBots master's thesis](https://resolver.obvsg.at/urn:nbn:at:at-ubl:1-108602) (`swarmbots_thesis`). Record the package version and Git commit used in the experiment.
- **Using an included learner:** also cite the original algorithm papers listed below. For TMASAC, cite the thesis for the particular SwarmBots design and the SAC papers for the learning algorithm; cite MAT when describing its transformer policy lineage.
- **Discussing attention-based multi-agent SAC:** cite the relevant related papers directly when comparing their ideas or architectures with TMASAC. The related-work list provides context; it does not mean those algorithms are implemented here.
- **Describing the implementation or measuring runtime:** acknowledge the main scientific software below and report the installed versions. Select additional component citations according to the recurrent modules, action distributions, or auxiliary objectives actually used.

## Algorithm origins

| Included learner or design | Source and BibTeX key | Relationship to SwarmBots |
| --- | --- | --- |
| PPO (`ppo`, `ppo_small`, and the on-policy learners) | Schulman et al. (2017), [*Proximal Policy Optimization Algorithms*](https://arxiv.org/abs/1707.06347), `ppo` | PPO supplies the clipped on-policy optimization objective. |
| MAPPO (`mappo`, `mappo_small`, `mappo_mlp`, `mappo_mlp_small`) | Yu et al. (2022), [*The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games*](https://proceedings.neurips.cc/paper_files/paper/2022/hash/9c1535a02f0ce079433344e14d910597-Abstract-Datasets_and_Benchmarks.html), `mappo`; also `ppo` | Multi-agent PPO with shared per-agent actors and a centralized Deep Set or flattened MLP critic. |
| MAT (`mat_orig`) | Wen et al. (2022), [*Multi-Agent Reinforcement Learning Is a Sequence Modeling Problem*](https://proceedings.neurips.cc/paper_files/paper/2022/hash/69413f87e5a34897cd010ca698097d0a-Abstract-Conference.html), `mat`; also `ppo` | Source of the transformer encoder and autoregressive multi-agent action-generation formulation. |
| MAT adaptations (`mat_ind`, `mat_dec`, `mat_qcx`, and recurrent variants) | Wen et al. (2022), `mat`; Baron (2026), `swarmbots_thesis`; also `ppo` | MAT supplies the policy lineage. The thesis describes the QCX decoder, independent action heads, and recurrent extensions; `mat_dec` is the repository's decentralized-actor variant. |
| DDPG / MADDPG (`maddpg_mlp`, `maddpg_deepset`) | Lillicrap et al. (2016), [*Continuous Control with Deep Reinforcement Learning*](https://arxiv.org/abs/1509.02971), `lillicrap2016continuous`; Lowe et al. (2017), [*Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments*](https://arxiv.org/abs/1706.02275), `lowe2017multiagent` | Deterministic policy gradients, target actor/critic networks, and centralized multi-agent critics, adapted to a shared actor and scalar team reward. |
| TD3 / MATD3 (`matd3_mlp`, `matd3_deepset`) | Fujimoto et al. (2018), [*Addressing Function Approximation Error in Actor-Critic Methods*](https://proceedings.mlr.press/v80/fujimoto18a.html), `fujimoto2018addressing`; Ackermann et al. (2019), [*Reducing Overestimation Bias in Multi-Agent Domains Using Double Centralized Critics*](https://arxiv.org/abs/1910.01465), `ackermann2019reducing` | Twin critics, delayed actor/target updates, and target-action smoothing applied to centralized joint-action team values. |
| SAC (`masac_mlp`, `masac_deepset`, and TMASAC) | Haarnoja et al. (2018), [*Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor*](https://proceedings.mlr.press/v80/haarnoja18b.html), `sac`; [*Soft Actor-Critic Algorithms and Applications*](https://arxiv.org/abs/1812.05905), `sacapps` | Sources of the off-policy maximum-entropy actor-critic algorithm and automatic temperature tuning. MASAC uses decentralized shared actors and MLP or Deep Set team critics. |
| TMATD3 (`tmatd3`, `tmatd3_dec`) | Foundations `fujimoto2018addressing`, `ackermann2019reducing`, `mat`, `attiayn`; architectural components `swarmbots_thesis`, `deepsets` | Repository extension combining TD3 with TMASAC's transformer critic and a transformer or decentralized actor. TMATD3 is an added baseline, not a method attributed to the thesis. |
| TMASAC and its variants (`tmasac*`) | Baron (2026), [*SwarmBots: a GPU-accelerated multi-agent continuous control benchmark with transformer baselines*](https://resolver.obvsg.at/urn:nbn:at:at-ubl:1-108602), master's thesis, Johannes Kepler University Linz, `swarmbots_thesis`; foundations `sac`, `sacapps`, `mat` | Direct source for the SwarmBots combination of a transformer actor, action-conditioned transformer twin critics, active-agent masking, and set aggregation, together with its recurrent extensions. |
| Next-observation prediction (NOP) and Signed-Magnitude Beta (SMB) | Baron (2026), `swarmbots_thesis` | Source for these SwarmBots extensions; related reusable components are credited below. |

The presets are configurable SwarmBots adaptations, not fully faithful reproductions of the original multi-agent formulations. MAT (including `mat_orig`), MAPPO, and MADDPG/MATD3 use global shared team value functions here instead of the original per-agent value estimates. The deterministic baselines also optimize all current actor actions jointly through a team critic, rather than replacing only one agent's replay action per actor update. See [differences from the original multi-agent methods](learning.md#differences-from-the-original-multi-agent-methods) and report these distinctions when comparing results. Current settings can also differ from the papers and thesis experiments: the thesis's attention-disabled baseline labelled MAPPO is now exposed as `mat_ind_no_attention`, while the current `mappo` preset selects the separate MAPPO implementation. Use the [learning guide](learning.md) for current preset definitions and report the actual configuration used.

## Deterministic-baseline implementation sources

The implementation was checked against [OpenAI's MADDPG](https://github.com/openai/maddpg), [Johannes Ackermann's MATD3 implementation](https://github.com/JohannesAck/MATD3implementation), and [Stable-Baselines3's TD3/DDPG implementation](https://github.com/DLR-RM/stable-baselines3). Cite Raffin et al. (2021), [*Stable-Baselines3: Reliable Reinforcement Learning Implementations*](https://jmlr.org/papers/v22/20-1364.html), `stable-baselines3`, for the SB3 implementation reference, alongside the associated algorithm papers above for the methods. These repositories are not runtime dependencies of SwarmBots.

SwarmBots uses its existing PyTorch replay, masking, normalization, and checkpoint lifecycle. Its cooperative baselines share actor parameters and optimize a scalar team Q through all active joint actions. The original MADDPG/MATD3 code uses per-agent actors and critics and replaces only the current agent's replay action for that actor's update. Report this adaptation and the chosen critic architecture when comparing results. DDPG is the single-critic, undelayed, unsmoothed variant; TD3 uses two independent critics and target smoothing. Add `deepsets` when using a Deep Set critic, and report the hidden dimensions from the [learning guide](learning.md).

## TMASAC and related work

Chapter 3, “Methods,” of the [thesis](https://resolver.obvsg.at/urn:nbn:at:at-ubl:1-108602) places TMASAC in an existing line of attention-based multi-agent SAC research. Attention in multi-agent SAC actors and critics predates TMASAC. The following papers establish relevant precedents and comparisons:

| Work | Relevant prior contribution | Relationship to the thesis's TMASAC design |
| --- | --- | --- |
| **MAAC:** Iqbal and Sha (2019), [*Actor-Attention-Critic for Multi-Agent Reinforcement Learning*](https://proceedings.mlr.press/v97/iqbal19a.html), `maac` | SAC-based learning with attention in centralized per-agent critics and decentralized actors. | Early precedent for attending to other agents' observation-action information. TMASAC uses a transformer actor over joint public observations and scalar team critics. |
| **SACHA:** Lin and Ma (2023), [*SACHA: Soft Actor-Critic with Heuristic-Based Attention for Partially Observable Multi-Agent Path Finding*](https://arxiv.org/abs/2307.02691), `sacha` | Heuristic-based attention in SAC actors and agent-centered critics for partially observable path finding. | Precedent for attention in both policy and value estimation, with task-specific heuristic inputs. |
| **MDAC:** Xu, Zheng, and Wang (2025), [*MDAC: Multi-Agent Distributional Soft Actor-Critic with Attention for Warehouse Task Coordination*](https://doi.org/10.1145/3746709.3746860), `mdac` | Multi-head attention in actors and critics, a distributional critic, and action masking for warehouse coordination. | Precedent for attention in both SAC branches; its distributional value model differs from TMASAC's scalar team critics. |
| **MATRS:** Ge et al. (2025), [*Transformer-Based Soft Actor–Critic for UAV Path Planning in Precision Agriculture IoT Networks*](https://doi.org/10.3390/s25247463), `matrs` | A centralized transformer critic over per-agent observation-action tokens, paired with local MLP actors. | Close comparison for the critic architecture. The thesis contrasts its flattened critic output with TMASAC's masked set aggregation for variable active-agent counts. |

These references credit prior attention and transformer combinations with SAC. The thesis is the source for the specific TMASAC formulation used in SwarmBots; these related papers describe their own methods rather than serving as alternative citations for TMASAC.

For example, a methods section using the standard feed-forward TMASAC design could state:

> We use the SwarmBots TMASAC design described by Baron (2026), combining Soft Actor-Critic (Haarnoja et al., 2018) with transformer policies drawing on MAT (Wen et al., 2022). Related attention-based multi-agent SAC methods include MAAC, SACHA, MDAC, and MATRS.

Use `\cite{swarmbots_thesis,sac,sacapps,mat}` for the design and foundations, and `\cite{maac,sacha,mdac,matrs}` when discussing that related work. Add component citations below if they are part of the reported configuration.

## Reusable learning components

| Component | Source and BibTeX key |
| --- | --- |
| Transformer attention | Vaswani et al. (2017), [*Attention Is All You Need*](https://proceedings.neurips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html), `attiayn` |
| Permutation-invariant critic aggregation | Zaheer et al. (2017), [*Deep Sets*](https://proceedings.neurips.cc/paper/2017/hash/f22e4747da1aa27e363d86d40ff442fe-Abstract.html), `deepsets` |
| LSTM recurrent modules | Hochreiter and Schmidhuber (1997), [*Long Short-Term Memory*](https://www.bioinf.jku.at/publications/older/2604.pdf), `lstm` |
| sLSTM / mLSTM recurrent modules | Beck et al. (2024), [*xLSTM: Extended Long Short-Term Memory*](https://proceedings.neurips.cc/paper_files/paper/2024/hash/c2ce2f2701c10a2b2f2ea0bfa43cfaa3-Abstract-Conference.html), `xlstm` |
| Optional self-predictive representations (SPR) | Schwarzer et al. (2021), [*Data-Efficient Reinforcement Learning with Self-Predictive Representations*](https://openreview.net/forum?id=uCQfPZwRaUu), `spr` |
| Beta action policies, also a foundation for SMB | Chou et al. (2017), [*Improving Stochastic Policy Gradients in Continuous Control with Deep Reinforcement Learning Using the Beta Distribution*](https://proceedings.mlr.press/v70/chou17a.html), `betapolicy` |
| Straight-through Gumbel–Softmax used by Gumbel SMB | Jang et al. (2017), [*Categorical Reparameterization with Gumbel-Softmax*](https://openreview.net/forum?id=rkE3y85ee), `gumbelsoftmax` |
| Optional generalized state-dependent exploration (gSDE) | Raffin et al. (2022), [*Smooth Exploration for Robotic Reinforcement Learning*](https://proceedings.mlr.press/v164/raffin22a.html), `gsde` |

## Scientific software

| Software | Role in SwarmBots | Reference and BibTeX key |
| --- | --- | --- |
| **Stable-Baselines3 (SB3)** | Implementation reference for DDPG/TD3; not a runtime dependency | Raffin et al. (2021), [*Stable-Baselines3: Reliable Reinforcement Learning Implementations*](https://jmlr.org/papers/v22/20-1364.html), `stable-baselines3`; [official repository](https://github.com/DLR-RM/stable-baselines3). |
| **Gymnasium** | Environment spaces and reset/step conventions | Towers et al. (2024), [*Gymnasium: A Standard Interface for Reinforcement Learning Environments*](https://arxiv.org/abs/2407.17032), `gymnasium`, using the citation supplied by the [maintainers](https://github.com/Farama-Foundation/Gymnasium#citation). |
| **MuJoCo** | Physics models and simulation engine | Todorov, Erez, and Tassa (2012), [*MuJoCo: A Physics Engine for Model-Based Control*](https://doi.org/10.1109/IROS.2012.6386109), `todorov2012mujoco`. |
| **MuJoCo Warp (MJWarp)** | GPU-vectorized MuJoCo simulation backend | Google DeepMind and NVIDIA, [*MuJoCo Warp*](https://github.com/google-deepmind/mujoco_warp), `mujoco_warp`. This entry credits the software project through its official repository. |
| **NVIDIA Warp** | GPU kernels and simulation data operations | Macklin (2022), [*Warp: A High-performance Python Framework for GPU Simulation and Graphics*](https://github.com/NVIDIA/warp), `warp`, following the project's [preferred citation](https://github.com/NVIDIA/warp/blob/main/CITATION.cff). |
| **PyTorch** | Tensor operations, neural networks, automatic differentiation, and training | Paszke et al. (2019), [*PyTorch: An Imperative Style, High-Performance Deep Learning Library*](https://proceedings.neurips.cc/paper/2019/hash/bdbca288fee7f92f2bfa9f7012727740-Abstract.html), `pytorch`. |

The software references identify the projects and their foundations. Report the installed `gymnasium`, `mujoco`, `mujoco-warp`, `warp-lang`, and `torch` versions separately, especially for simulation-throughput comparisons. The [benchmark protocol](benchmark_protocol.md) specifies runtime reporting, and [`pyproject.toml`](../pyproject.toml) records dependency constraints.
