# LLM-Raft

Enhancing Urban Traffic Efficiency and Safety through Decentralized Coordination of Autonomous Vehicles


## Installation

Requires Python ≥ 3.10 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/shxingch/llm-raft.git
cd llm-raft

# Install core package
uv sync

# MetaUrban environment (assets download on first run)
uv sync --extra metaurban

# OpenAI-compatible LLM client
uv sync --extra llm

# Or install everything at once
uv sync --all-extras
```

## Configuration

### LLM API

An OpenAI-compatible API endpoint is required. Set via environment variables:

```bash
export OPENAI_API_KEY="your-key"
export OPENAI_BASE_URL="https://api.openai.com/v1"   # or custom endpoint
export OPENAI_MODEL="gpt-4.1"                         # optional override
```

### Scenario Configs

Each scenario YAML defines environment parameters and driving rules that are injected into the LLM prompt:

```
configs/
├── algorithms/llm_raft.yaml                # grouping, consensus, controller
├── metaurban/{sparse,normal,dense,emergency}.yaml
└── limsim/{normal_road,crowded_road,highway,intersection}.yaml
```

Key config fields per scenario:
- `speed_limit_kmh` — road speed limit (passed to LLM)
- `following_distance_m` — minimum following gap (passed to LLM)
- `action_noise` — control noise simulating actuator imprecision
- `scene_rules` — natural-language driving rules for the scenario

## Usage

### MetaUrban

```bash
cd runs/metaurban

# Single scenario
python run.py --scenario dynamic_sparse --mode raft --trials 5
```

| Arg | Default | Options |
|-----|---------|---------|
| `--scenario` | `dynamic_sparse` | `dynamic_sparse`, `dynamic_normal`, `dynamic_dense`, `emergency_pedestrian`, `all` |
| `--mode` | `raft` | `raft`, `no-consensus`, `zero-shot` |
| `--trials` | `5` | any integer |
| `--gui` | off | enable 3D rendering |
| `--decision-interval` | `10` | simulation steps between LLM calls |

### LimSim

```bash
cd runs/limsim

# With GUI
python run.py --scenario normal_road

# Headless batch
python run.py --scenario highway --trials 30 --no-gui

# Ablation: without consensus
python run.py --scenario intersection --no-consensus --no-gui
```

| Arg | Default | Options |
|-----|---------|---------|
| `--scenario` | required | `normal_road`, `crowded_road`, `highway`, `intersection`, `all` |
| `--trials` | `1` | any integer |
| `--no-gui` | off | headless mode |
| `--no-consensus` | off | disable consensus for ablation |
| `--log-dir` | `results/logs` | output path |

## Project Structure

```
llm_raft/                  Core Python package
├── core/                  Algorithm: engine, grouping, consensus, controller, narrative
├── envs/                  MetaUrban environment builder
├── data_types.py          VehicleState, NarrativeProposal, GroupPlan, ActionCommand
├── helpers.py             Utility functions
└── llm_runtime.py         LLM API client, prompt builder, proposal/consensus generators

runs/                      Runner scripts for each simulator
configs/                   Per-scenario YAML configs with driving rules
third_party/               LimSim and MetaUrban simulator source
```

## License

- `third_party/limsim/` — GPL v3 (see `third_party/limsim/LICENSE`)
- `third_party/metaurban/` — Apache 2.0 (see `third_party/metaurban/LICENSE.txt`)
