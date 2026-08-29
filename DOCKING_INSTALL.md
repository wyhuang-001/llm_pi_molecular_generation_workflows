# Docking-Only Environment

This document installs and checks the design plus docking dependencies only. It does **not** install or invoke RBFE/AsyncFEP packages.

## Scope

The docking-only environment contains:

- Python 3.11
- RDKit, Gemmi, NumPy 1.26.x, SciPy and the project package
- GNINA as the configured docking executable
- AutoDock Vina, Open Babel, Meeko and ProDy for alternative docking and structure preparation
- Optional pose/QC packages used by the project: MDAnalysis, ProLIF, spyrmsd and FreeSASA

It deliberately excludes OpenMM, AmberTools, OpenFF, OpenMMForceFields, PDBFixer, JAX, pymbar, AsyncFEP and `bloom_prepare`.

## Install

Run this on the docking/CPU or GPU node. The command only creates or updates the environment named `molecular-agent-docking`. The installer defaults to the Tsinghua TUNA conda-forge and PyPI mirrors:

```bash
cd /mnt/f/doctoral_period_huangwy/PhD_project/external_model/context_learn/test/simple_molecular_agent

./scripts/install_docking_env.sh --dry-run

DOCKING_CONDA_FORGE_URL=https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge \
DOCKING_PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
GNINA_URL=https://ghproxy.net/https://github.com/gnina/gnina/releases/download/v1.3.2/gnina.1.3.2 \
REQUIRE_GNINA=1 \
./scripts/install_docking_env.sh
```

The accelerated GNINA proxy is optional and may be replaced by an internal mirror. The installer always checks the downloaded bytes against the configured official SHA256 before installing them. If the proxy is unavailable, use the official URL while retaining the same checksum:

```bash
GNINA_URL=https://github.com/gnina/gnina/releases/download/v1.3.2/gnina.1.3.2 \
REQUIRE_GNINA=1 \
./scripts/install_docking_env.sh
```

The installer uses the first available `mamba`, `micromamba` or `conda`. It installs the project editable and then installs the small pip-only Meeko requirement with `--no-deps`, leaving the conda-managed scientific stack intact.

GNINA is downloaded from the configured Linux release URL and verified with `GNINA_SHA256`. For an internal mirror:

```bash
GNINA_URL=https://mirror.example/gnina \
GNINA_SHA256=<64-hex-sha256> \
REQUIRE_GNINA=1 \
./scripts/install_docking_env.sh
```

The existing NVIDIA driver and kernel CUDA support are left unchanged. GNINA will use the available GPU runtime when compatible; the installer does not install or modify NVIDIA drivers, kernel modules, OpenMM, or RBFE CUDA packages.

## Verify

```bash
nvidia-smi
mamba run -n molecular-agent-docking gnina --version
mamba run -n molecular-agent-docking \
  python scripts/check_docking_env.py --require-gnina
mamba run -n molecular-agent-docking pytest -q
```

The check requires GNINA, Vina, Open Babel and both Meeko preparation commands. A missing GNINA is only acceptable when the installer was run with `REQUIRE_GNINA=0`; it must be installed before enabling the configured GNINA command.

## Inputs And Preparation

The workflow writes these files after deterministic candidate validation:

- `candidate-XX.sdf`: candidate in the constrained parent coordinate frame
- `reference-ligand.sdf`: co-crystal reference ligand
- `receptor-protein-only.pdb`: protein-only receptor

The configured GNINA command uses the co-crystal ligand as the search-box reference:

```text
gnina -r {receptor} -l {candidate} \
  --autobox_ligand {reference} --autobox_add 4 \
  --num_modes 20 -o {output_dir}/docked.sdf \
  --log {output_dir}/gnina.log
```

GNINA accepts the SDF inputs used by this project. If switching to Vina, add an explicit PDBQT preparation workflow with Meeko and change both the command and output contract; Vina is not a drop-in replacement for the GNINA command above.

## What Is Checked Before Docking

`candidate_geometry_accepted` is only a deterministic pre-screen. Before launching docking, the adapter also checks that:

1. Candidate and reference SDF files exist, are readable by RDKit, contain one molecule, and have 3D conformers.
2. Receptor PDB exists, contains `ATOM` records, and contains no ligand `HETATM` records.
3. The configured output directory is writable.
4. The docking executable is present and the command can be audited.

Before evaluating a candidate, the adapter independently docks `reference-ligand.sdf` into the same protein-only receptor using the same autobox and GNINA command. By default it uses the paired seeds `[17, 29, 43]`; each seed runs reference and candidate with the same explicit GNINA `--seed`, and each output is stored in its own directory. After each command exits successfully, the adapter requires `docked.sdf` and at least one readable pose. It records up to 20 pose property dictionaries, including docking score fields emitted by GNINA, plus stdout/stderr and return code. A command that exits zero without readable poses is `failed`, not `complete`.

The result includes `reference_baseline`, per-seed candidate summaries, and a multi-seed `comparison`. For each seed and score field, `delta_candidate_minus_reference` is reported: more negative is better for `minimizedAffinity`, while more positive is better for `CNNscore`, `CNNaffinity` and `CNN_VS`. The aggregate reports mean, sample standard deviation, min/max, candidate-better seed count and candidate-better seed fraction. The full reference audits are stored under `docking-reference-baseline/seed-*/`; candidate audits remain under `docking-attempt-XX/seed-*/`. These are protocol-relative docking observations, not experimental affinity or activity predictions.

Only a `complete` docking result is sent to the next LLM edit-retry round. `not_configured` and `failed` are recorded and do not trigger blind redesign.

## Workflow Boundary

The current loop is:

```text
LLM proposal
-> RDKit/UFF/rigid-receptor candidate pre-screen
   -> rejected: return structured failure/clash evidence to the LLM; docking is not run
   -> accepted: continue to docking input preflight
-> docking input preflight
-> GNINA docking
-> readable pose/score audit
-> optional LLM revision
```

A geometry rejection observation includes `failure_stage=deterministic_geometry_prescreen`, the structured failure class/clash details, recommended next queries, and `docking.status=not_run_geometry_rejected`. The next LLM decision may query more site evidence or revise the atom/fragment. The attempt limit remains `max_edit_attempts`.

RBFE/AsyncFEP is outside this loop. Its configuration files may remain in the repository for a future phase, but this installer does not install it and the current workflow records RBFE as `deferred`.

## Enable Docking

Keep docking disabled until the environment check and a small smoke run pass. Then copy the docking-only settings into a runtime config and change only:

```json
{
  "docking": {
    "enabled": true
  }
}
```

Do not enable RBFE for this phase. The repository examples keep both stages disabled by default.

The docking score is an external pose-ranking observation, not an experimental affinity or activity result. Candidate scores should be interpreted relative to the independently redocked reference baseline and only under the same receptor preparation, search-box, exhaustiveness, number-of-modes and scoring protocol. A reference score from a different GNINA run, a crystal pose score, or a different receptor/search box is not a valid baseline.
