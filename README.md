# software-control

### Problems

1. SOTA Models fail to perform real world tasks reliably to be production ready in dynamic computer environments suffering from failure modes such as task deviation, UI grounding error, missing key goal requirements, and more.
2. There are no systematic failure-driven “demo2production gaps” assessment for computer use agents
3. Current benchmarks either only focus on one or few capability, are static, doesn’t target failure modes, can be easily overfitted, or are not reproducible.

### Our approach

We are building a platform that consists of

- a failure-driven scenario simulation engine with perturbation injections
- a data curation & augmentation pipeline
- benchmarks measure agent reliability, generalization, & robustness towards the realistic, diverse, and complex environments & tasks

Our release v1.0 focuses on the simulation engine & data generation process to validate this systematic development process for software control agents. We aim to address the gaps in software control agent development with

- a simulation engine that identifies and generate failure scenarios
- a generated training dataset targeting the failure scenarios
- fine-tuning studies with the generated training dataset

## Setup

```
git clone --recurse-submodules git@github.com:ManifoldRG/software-control.git
cd software-control

conda env create -f environment.yml -n software-control
conda activate software-control

uv sync --all-groups

# set up pre-commit linter & formatter
uv run pre-commit install

# install package (editable) for proper imports
pip install -e .

cd src/OSWorld
pip install -e .

cd ../../
```

Make sure when you are using the correct conda env `software-control` & python created by uv in `.venv/bin/python`

## Run data generation

```
uv run src/perturbation_engine/generate_trajectories.py
```

or use `launch.json` PythonDebugger: Current File config to run `generate_trajectories.py`

## Working with AWS

```
# Checking for activate EC2 instances
aws ec2 describe-instances --region us-east-1 --query 'Reservations[*].Instances[*].[InstanceId,State.Name,LaunchTime,PublicIpAddress]' --output table

# Start instance
aws ec2 start-instances --region us-east-1 --instance-ids <replace-with-the-target-i-xxxxxxxxx>

# Stop instance
aws ec2 stop-instances --region us-east-1 --instance-ids <replace-with-the-target-i-xxxxxxxxx>

# Terminate them using awscli if ctrl+c didn't shut them down gracefully
aws ec2 terminate-instances --region us-east-1 --instance-ids <replace-with-the-target-i-xxxxxxxxx>

```

## Roadmap

This is a rough roadmap for Phase 1. For most up-to-date project status, refer to the [project board](https://github.com/orgs/ManifoldRG/projects/30/views/1).

Preliminary Design & Env setup

- MVP perturbable trajectory schema design
- OSworld & TheAgentCompany env setup
- Stats & sampling initial design

Core Pipeline Design & Scaffolding & Seed Dataset Collection

- Perturbation design
- Scaffold core MVP pipeline (including the stats, sampling)
- Seed trajectory collection

Pipeline Tuning & Data Scaling

- Seed trace perturbation
- pipeline tuning
  - stats sampler
  - perturbation
- seed trajectory scaling

Evaluation & Finetuning

- prompt based validation setup
- OSWorld evaluation setup
- agent finetuning script setup

Analysis & Finetuning Iteration

- establish baseline evaluation results
- iterative finetuning & monitoring & evaluation with variants of perturbations

Release Preparation

- analyze the results & paper writing & release code, data, evaluation results (paper), & finetuned model

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

This project incorporates code from various third-party projects with different licenses (Apache 2.0 and MIT). Please see the [NOTICE](NOTICE) file for complete attribution details.
