# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Digital Red Queen (DRQ) is a research project implementing adversarial program evolution using LLMs and Core War. The system evolves assembly-like programs ("warriors") that compete for control of a virtual machine, using Red Queen dynamics (continuous adaptation against previous champions) and Map-Elites for behavioral diversity.

## Commands

### Installation
```bash
uv sync
```

### Run Tests
```bash
uv run python -m unittest discover corewar/tests/
# Single test:
uv run python -m unittest corewar.tests.mars_test.TestMars.test_dwarf_versus_sitting_duck
```

### Run DRQ Algorithm
```bash
uv run python src/drq.py --seed=0 --save_dir="./drq_run_0/" --n_processes=20 \
  --simargs.rounds=20 --simargs.size=8000 --initial_opps="human_warriors/imp.red" \
  --n_rounds=20 --n_iters=250 --gpt_model="gpt-4.1-mini-2025-04-14"

# With OpenRouter or other OpenAI-compatible API:
uv run python src/drq.py --api_base_url="https://openrouter.ai/api/v1" --gpt_model="openai/gpt-4o-mini" ...
```

### Evaluate Warriors
```bash
uv run python src/eval_warriors.py --warrior_path="./warrior.red" \
  --opponents_path_glob="human_warriors/*.red"
```

### Visualize Battle
```bash
uv run python -m corewar.graphics --warriors path/to/warrior1.red path/to/warrior2.red
```

## Architecture

### Core Components

**`src/drq.py`** - Main orchestrator
- `Args`: Dataclass with all configuration (~25 parameters)
- `MapElites`: Archive mapping behavior characteristics (BC) to best phenotypes
- `Main`: Runs evolution loop with checkpoint/resume support

**`src/corewar_util.py`** - Simulation interface
- `SimulationArgs`: Core War configuration (core size, cycles, processes)
- `MyMARS`: Extended MARS simulator tracking memory coverage
- `run_multiple_rounds()`: Parallel battle execution via multiprocessing

**`src/llm_corewar.py`** - LLM warrior generation
- `CorewarGPT`: Async OpenAI client for warrior generation/mutation
- `GPTWarrior`: Data structure with parsed Redcode + metadata

**`corewar/corewar/`** - Core War engine (from [rodrigosetti/corewar](https://github.com/rodrigosetti/corewar))
- `mars.py`: MARS virtual machine
- `redcode.py`: Two-pass Redcode parser
- `core.py`: Circular memory arena

### Data Flow

1. LLM generates Redcode source via prompts in `src/prompts/`
2. `redcode.py` parses into `Warrior` objects
3. `mars.py` executes battles, returns fitness metrics (score, memory coverage, spawned processes)
4. `MapElites` stores best warriors per behavior bin
5. Previous champions become opponents for next round (Red Queen dynamics)

### Key Abstractions

- **Fitness**: Average score across battle rounds (time alive / total time / num warriors)
- **Behavior Characterization (BC)**: 2D bins from TSP (total spawned processes) and MC (memory coverage)
- **Evolution loop**: 90% mutate from archive, 10% generate new; stop round at fitness threshold

## File Types

- `.red` - Redcode source (warrior programs)
- Checkpoints: pickle files containing args, map-elites archive, generation history
