# software-control

creating new models capable of taking action on software

## Setup

```
git clone git@github.com:ManifoldRG/software-control.git
cd software-control

conda create --env software-control -f environment.yml
conda activate software-control

uv sync --group dev

uv run pre-commit install
```

## Roadmap

Phase 1 - Offline Perturbation (with a focus on GUI)

1. Establish baseline
   - [ ] Curate a small set (20-200) of ideal successful & high value trajectories from Mind2Web, OSWorld, GUI-Robust, STEVE, WorkArena++, MinWoB++ as the Baseline Trajectories
   - [ ] Scope test cost for SoTA models UI-TARS 1.5 7B, GTA1 7B w/ o3, Jedi 7B, InfantAgent and decide on the number of SoTA models to evaluate with.
   - [ ] Set up eval scripts for SoTA models
   - [ ] Set up data ingestion for SoTAs
   - [ ] Test SoTAs on the curated baseline trajectories to verify consistent success if not, re-select the trajectories
2. GUI Perturbation Design
   - [ ] Research to see what variables have the highest value
   - [ ] Visual (easy → hard, measured by information density, structural complexity & number of swaps/changes, size difference, color difference, etc)
   1. HTML
      1. Resize, relocate components
      2. CSS-style perturbation
      3. Adding visual distractors (e.g., popups, fake search bars, etc)
      4. Reconstruct object hierarchy
      5. Menu structure reorganization
      6. Sub-menu swapping
      7. Text content shuffling / substitution
      8. Element ID / class renaming
      9. Position jitter / relative swapping
      10. Element visibility / presence randomization
      11. Text length / word randomization
      12. Icon / image randomization
   - [ ] Task Trajectory
     1. Randomly sample a subset as new trajectory given a trajectory
     2. Cross-task substitution
     3. Redundant continuation (add irrelevant steps after goal state)
     4. Simulate unresponsive state and add duplicated observation timestep
3. Set up HTML perturbation renderer script using Playwright APIs or equivalent (e.g., browserbase)
   - [ ] set up screenshot taking, component bounding box returning functionality
   - [ ] refer to OSWorld-G / jedi codebase for visual component generation for adding visual distractors
   - [ ] set up perturbation for each variable
   - [ ] [maybe] set up solvability and plausibility verification for perturbation
   - [ ] set up task success verification (OSWorld has many examples)
   - [ ] set up automated perturbation adjustment based on task completion status
   - [ ] set up data collection logic
4. Evaluate SoTA models with the perturbation script
5. Analyze the results
   - [ ] cluster failure modes
   - [ ] identify data bottlenecks
   - [ ] identify architectural bottlenecks
   - [ ] assess perturbation experiment & update the perturbation design & repeat the evaluation with the script
6. Release 0: Release code, data, blog
7. Augment data and curate more data (around 2k trajectories) based on the identified weakness for each model
8. Set up fine-tune scripts for each SoTA models
9. Fine-tune the SoTAs on new data targeting their data bottlenecks
10. Evaluate the SoTAs on baselines and the test sets of their benchmark / training datasets using existing eval scripts
11. Analyze the results
12. Release 1: fine-tuned models, datasets, updated toolkit code, blog, paper
