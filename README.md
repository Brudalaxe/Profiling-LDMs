# Comparative Profiling of Latent Diffusion Model Training

This repository accompanies the paper "Comparative Profiling: Insights Into Latent Diffusion Model Training" by Bradley Aldous and Ahmed M. Abdelmoniem, presented at the 4th Workshop on Machine Learning and Systems (EuroMLSys '24). The repository includes the code and configurations used for training and profiling the AudioLDM and Stable Diffusion models.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
  - [Training Commands](#training-commands)
  - [Profiling with PyTorch Profiler](#profiling-with-pytorch-profiler)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Generative AI models, particularly latent diffusion models (LDMs), have shown remarkable capabilities in generating high-fidelity audio and images. This repository provides the implementation details, training commands, and profiling setup for AudioLDM and Stable Diffusion models as discussed in our paper.

The training environments for each model were set up according to the requirements specified in their respective repositories:
- AudioLDM: [AudioLDM Training and Finetuning](https://github.com/haoheliu/AudioLDM-training-finetuning/tree/main)
- Stable Diffusion: [Stable Diffusion Training](https://github.com/CompVis/latent-diffusion/tree/main)

## Installation

Clone this repository and navigate to the project directory:

```bash
git clone https://github.com/yourusername/comparative-profiling-ldm.git
cd comparative-profiling-ldm
```

Install the dependencies for each model as per their respective repositories.

### AudioLDM Environment Setup

Follow the instructions in the [AudioLDM repository](https://github.com/haoheliu/AudioLDM-training-finetuning/tree/main) to set up the environment and install dependencies.

### Stable Diffusion Environment Setup

Follow the instructions in the [Stable Diffusion repository](https://github.com/CompVis/latent-diffusion/tree/main) to set up the environment and install dependencies.

## Usage

### Training Commands

The training commands for each model follow the instructions outlined in their respective repositories. Below are the example commands used to train the models:

#### AudioLDM

```bash
python train.py --config configs/audioldm.yaml --gpus 0,1 --batch_size 16
```

#### Stable Diffusion

```bash
python main.py --base configs/stable-diffusion.yaml --gpus 0,1 --batch_size 16
```

### Profiling with PyTorch Profiler

Both models were profiled using Weights & Biases and PyTorch Profiler to monitor GPU utilization, execution time, and memory usage. The PyTorch Profiler can be integrated into the training script as follows:

```python
from pytorch_lightning import Trainer
from pytorch_lightning.profiler import PyTorchProfiler

profiler = PyTorchProfiler(
    dirpath="profiler_logs",
    filename="profiler_output",
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('log_dir'),
    record_shapes=True,
    profile_memory=True,
    with_stack=True
)

trainer = Trainer(
    profiler=profiler,
    ...
)
```

In the provided training script, the `trainer_kwargs` dictionary is used to configure the trainer and integrate profiling tools:

```python
# Trainer and callbacks
trainer_kwargs = dict()

# Add PyTorch profiler to trainer_kwargs
profiler = PyTorchProfiler(
    dirpath="profiler_logs",
    filename="profiler_output",
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('log_dir'),
    record_shapes=True,
    profile_memory=True,
    with_stack=True
)
trainer_kwargs["profiler"] = profiler

trainer = Trainer(**trainer_kwargs)
```

## Contributing

We welcome contributions to this project. Please follow these steps to contribute:

1. Fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature-branch`)
5. Create a new Pull Request

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
