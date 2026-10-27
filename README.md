<img src="docs/assets/readme-banner-cifar10.png" alt="PRISM: Uncond CIFAR10" width="250"><img src="docs/assets/readme-banner-mnist.png" alt="PRISM: Uncond MNIST" width="250">

# PRISM

### _Probabilistic Reverse-diffusion for Image Synthesis with Multimodal guidance_
_Research project based on DDPM, DDIM and CFG papers._

PRISM is a research initiative aimed at understanding unconditional and guided diffusion models for image generation. The project replicates three foundational papers: DDPM for the core denoising framework, DDIM for faster deterministic sampling, and classifier-free guidance (CFG) for conditional control. The replication is done on MNIST and CIFAR-10, following professional research practices: planning and tracking work in the open, fixed seeds and pinned dependencies for reproducibility, experiment tracking, and an honest comparison against the published results.

The conditional configurations use the dataset class label as the U-Net conditioning. During
training, `entrenamiento.p_uncond` randomly replaces a label with the reserved null class, so
the same model learns both conditional and unconditional predictions. During sampling,
`entrenamiento.guidance_scale` combines those predictions; a value of `1` disables guidance.

The developed models will be exposed in a web app that showcases the generation process: the initial full-noise state, the final denoised image, and the intermediate states in between. This makes it possible to see how the reverse-diffusion process gradually turns noise into structure, and to compare DDPM and DDIM sampling side by side.

Because the training hardware and budget are private and limited, training time has been cut relative to the original papers. As a result, absolute quality metrics (FID, IS) are expected to fall short of the published figures. Every deviation from the papers is documented in the sections below, along with its likely consequences, so that results can be interpreted fairly rather than compared naively.

[![License](https://img.shields.io/github/license/josuviteri/prism)](LICENSE)
[![CI](https://img.shields.io/github/actions/workflow/status/josuviteri/prism/ci.yml?branch=main&label=CI)](.github/workflows/ci.yml)
[![Latest release](https://img.shields.io/github/v/release/josuviteri/prism)](https://github.com/josuviteri/prism/releases)
[![Contributors](https://img.shields.io/github/contributors/josuviteri/prism)](https://github.com/josuviteri/prism/graphs/contributors)
[![Open Issues](https://img.shields.io/github/issues/josuviteri/prism)](https://github.com/josuviteri/prism/issues)

[Roadmap](#roadmap) | [Installation](#installation) | [Contributing](CONTRIBUTING.md) | [License](#license)


# References
Replication of **Jonathan Ho, Ajay Jain & Pieter Abbeel, 2020 ·  DDPM**, **Jiaming Song, Chenlin Meng & Stefano Ermon, 2021 · (DDIM)**, **Jonathan Ho & Tim Salimans, 2022 · (CFG)**, over **MNIST** and **CIFAR-10** datasets,   
with a web UI for experimental guided generation on top.

- **[DDPM Paper](https://arxiv.org/pdf/2006.11239)**
- **[DDIM Paper](https://arxiv.org/pdf/2010.02502)**
- **[CFG Paper](https://arxiv.org/pdf/2207.12598)**
- **[Reference of DDPM Implementation](https://github.com/hojonathanho/diffusion)**
- **[Experiment Tracking: W&B](https://forge.coreweave.com/wandb/Viteritarrak/PRISM)**
- **Authors:** **[Josu Viteri](https://www.linkedin.com/in/josu-viteri/)** & **[Gotzon Viteri](https://www.linkedin.com/in/gotzonviteri/)**
## Decisions Papers Didn't Specify

NA

## Results

| Best Model ID | Denoising Method | Dataset | FID (lower is better) | IS (higher is better) | 
|---|---|---|---|---|
| Original DDPM | DDPM |Complete CIFAR-10 Test Set | 3.17 | 9.46 |
| cifar_ddpm_v0.3 | DDPM | Complete CIFAR-10 Test Set | NA | NA |
| cifar_ddpm_v0.3 | DDIM | Complete CIFAR-10 Test Set | NA | NA |
| cifar_ddpm_v0.3 | DDIM | 2k sample subset from CIFAR10 Test Set | 59.09 | 6.49 |
| mnist_ddpm_v0.2 | DDPM | Complete MNIST Test Set | NA | NA |
| mnist_ddpm_v0.2 | DDIM | Complete MNIST Test Set | NA | NA |
| mnist_ddpm_v0.2 | DDIM | 2k sample subset from MNIST Test Set | 31.93 | 2.13 |

<Honest explanation of the results, deviations, ...>

## Architecture

U-Net


## Repository Map

```
configs/      Hyperparameters and configurations.
src/data/     Load, splits and transformations.
src/models/   Replicated architecture.
src/training/ Train loop and evaluation.
src/app/      Service layer: Loads checkpoints and answers.
scripts/      Data download and utilities.
tests/        Unit tests, invariance, linting and reproducibility.
run.py        Execution.
```

## Installation

```bash
git clone https://github.com/josuviteri/PRISM
cd PRISM
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### Local Development Environment

NVIDIA RTX 5070 12GB

### Dataset Retrieval

```bash
python scripts/get_data.py # include --unzip argument to decompress after download
```

Downloads CIFAR10 and MNIST datasets, previously retrieved from official sources at <https://cave.cs.toronto.edu/kriz/cifar.html> and <https://www.kaggle.com/datasets/hojjatk/mnist-dataset> respectively. Verifies the SHA-256. 142,6MB compressed
CIFAR10 License: MIT
MNIST LICENSE: CC BY-SA 3.0


### Execution

```bash
# Smoke test.
python run.py --smoke

# UncoUnconditionaldnitional DDPM training over local CIFAR10 IDX.
python run.py --config configs/cifar10.yaml

# Unconditional DDPM training over local MNIST IDX.
python run.py --config configs/mnist.yaml

# Class-conditional DDPM + classifier-free guidance (CIFAR-10).
python run.py --config configs/guided_cifar10.yaml

# Class-conditional DDPM + classifier-free guidance (MNIST).
python run.py --config configs/guided_mnist.yaml

# Improved LeNet CNN for MNIST dataset IS and FID metric evaluations.
python run.py --config configs/mnist_evaluator.yaml

# Start the web app through the project entry point.
python run.py --serve

# Calculate metrics and log FID/IS to W&B. Paths and sampling options live in the config.
python run.py --solo-metricas --config configs/cifar10.yaml
python run.py --solo-metricas --config configs/mnist.yaml

# Generate images from gaussian noise using EMA weights.
python scripts/inference_diffusion_uncond.py --checkpoint checkpoints/model.pt \
	--output-dir generated --num-samples 64 --DDPM 
    # use --DDPM or --DDIM + --denoising-steps --num-samples depending on desired reversing technique out of the web UI
```

MNIST expects four uncompressed IDX files at `data/data/mnist`: `train-images.idx3-ubyte`, `train-labels.idx1-ubyte`, `t10k-images.idx3-ubyte` and `t10k-labels.idx1-ubyte`. Images are maintained in `28x28` with a single channel; normalization is calculated only with the train split.

Evaluation downloads weights from Inception-v3 first time and requires `torchvision`,
included at `requirements.txt`. For a fast check, try to reduce the cost of evaluation by using a subset from the test set; subsets smaller than the default 10,000 samples are selected with balanced class stratification.

### Reproducibility

- Seed at `src/utils/seed.py`, value at `configs/`.
- Dependencies with exact versions at `requirements.txt`.
- Compute budget: **NVIDIA RTX 5070 12 GB**.
- What the paper stated but was cut and its consequences: <explanation>.

### Tests

```bash
pytest tests/ -q
```

## Contributing

## Roadmap
Long-range feature roadmap is susceptible to changes. What is planned and in progress is tracked in the open:

- **[Open issues](https://github.com/josuviteri/prism/issues)** - Planned work, accepted feature requests, and known gaps. Issues are triaged and labelled by maintainers.
- **[Releases](https://github.com/josuviteri/prism/releases)** - What actually shipped in each version, with notes generated from the merged pull requests on that release branch.
- **[Kanban Board](https://github.com/users/josuviteri/projects/4)** - Interactive view of current tasks, active development, and project milestones across different workflow stages.

## License
Licensed under the Apache License, Version 2.0; see [`LICENSE`](LICENSE). Third-party attributions are listed in [`NOTICE`](NOTICE) and in the per-component `NOTICE` files.

### Nested Third-Party Licenses
Some subdirectories contain vendored third-party code or bundled assets under their own license. Each is listed below with its license and the path to that license file:

| Subdirectory | License | License file |
|--------------|---------|--------------|
| `data/` (downloaded datasets) | CIFAR-10: MIT; MNIST: CC BY-SA 3.0 | Dataset terms are referenced in [`NOTICE.md`](NOTICE.md); no dataset license file is bundled |
| Python dependencies in [`requirements.txt`](requirements.txt) | Individual licenses listed in [`NOTICE.md`](NOTICE.md) | [`NOTICE.md`](NOTICE.md) |


## AI Usage

Documented at [`AI_USAGE.md`](AI_USAGE.md).
