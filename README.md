# software-control

creating new models capable of taking action on software

## Setup

```
git clone git@github.com:ManifoldRG/software-control.git
cd software-control

conda create --env software-control -f environment.yml
conda activate software-control

uv sync --group dev

# set up pre-commit linter & formatter
uv run pre-commit install

# install package (editable) for proper imports
uv pip install -e .

# verify
uv run python -m perturbation_engine.tests.test_scene_analyzers
```

## Roadmap

This is a rough roadmap for Phase 1 - Offline Perturbation. For most up-to-date project status, refer to the [project board](https://github.com/orgs/ManifoldRG/projects/30/views/1).

1. Perturbation engine

- environment loading & augmented data saving setup
  - mind2web data loading
- randomization constraints & verification
  - identify randomization variables targeting known failure modes
  - Set up perturbation engine basic structure & io logic
  - implement rolling statistics script to extract stats from seed trajectories
  - experiment & implement scene analysis for downstream perturbation
    - experiment with element identification & selection (e.g. what elements can be identified & selected accurately for downstream manipulation)
  - experiment with html and css injection with playwright
    - randomization
    - layout change
    - element addition/removal
  - experiment with stylebot APIs for more constrained randomization with css injection
  - experiment with rule-based and VLM-based verification and filtering algorithm
    - metrics regarding augmented data quality (for iterating the perturbation script)

2. Establish baseline

- data
  - source seed trajectories (mind2web, and more if needed)
  - set up data ingestion for mind2web
- model
  - adapt model and set up eval script (UI-TARS1.5, GTA1, JEDI for grounding tasks (screenspot-pro)
- infra
  - set up cloud instance
- baseline evaluation
  - verify model performance on seed trajectories

3. Iterative evaluation & perturbation script tuning

- data
  - data ingestion for new augmented data
  - data mixture sampling
  - data formatting
- evaluation
  - use the same eval setup from step 1. establish baseline to evaluate SOTAs on augmented data
- analysis
  - analyze augmented data quality
  - analyze more fine-grained failure modes with SOTAs
- perturbation script tuning
  - adjust randomization constraints
  - adjust perturbation variables
  - adjust verification mechanisms
- RELEASE generated datasets and code

4. Augment data & finetune

- training
  - determine what training methods and configs
    - implement new SOTA rewards if needed (such as the reward used in GUI-G^2 instead of IoU with bbox)
  - set up training methods & run a small scale experiment
    - set up training logging
  - finetune the model checkpoint before a finetune experiment on augmented data and compare results
- data
  - determine data distribution and data sampling methods
  - implement data sampling / mixture methods
- evaluation
  - set up screenspot-pro evaluation
    - set up eval script
    - download data
    - set up data ingestion
  - set up screenspot-v2 evaluation
    - ... (same as above)
  - set up eval on other benchmarks if needed or have time (such as WebArena, VisualWebArena, OSUniverse, WebVoyager, WebBench)
  - benchmark SOTAs
- RELEASE fine-tuned models, updated datasets, evaluation results

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

This project incorporates code from various third-party projects with different licenses (Apache 2.0 and MIT). Please see the [NOTICE](NOTICE) file for complete attribution details.
