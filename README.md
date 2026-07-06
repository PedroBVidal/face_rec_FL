---
tags: [quickstart, vision, fds]
dataset: [CIFAR-10]
framework: [torch, torchvision]
---

# Federated Facial Recognition
This research explores the application of Federated Learning (FL) as a privacy-preserving paradigm for 2D facial recognition systems, directly addressing the growing concerns surrounding biometric data collection and centralization. We investigate the efficacy of integrating state-of-the-art FL techniques with existing facial recognition methods, with a particular focus on the potential benefits of incorporating synthetically generated facial data to enhance model robustness and privacy. Despite its promise, significant challenges persist, including managing non-IID (non-independent and identically distributed) data heterogeneity, overcoming hardware constraints on client devices, and ensuring sufficient data availability for effective federated training.

## Set up the project

### Fetch the app

Install Flower:

```shell
pip install flwr
```

Fetch the app:

```shell
flwr new @flwrlabs/quickstart-pytorch
```

This will create a new directory called `quickstart-pytorch` with the following structure:

```shell
quickstart-pytorch
├── pytorchexample
│   ├── __init__.py
│   ├── client_app.py   # Defines your ClientApp
│   ├── server_app.py   # Defines your ServerApp
│   └── task.py         # Defines your model, training and data loading
├── pyproject.toml      # Project metadata like dependencies and configs
└── README.md
```

### Install dependencies and project

Install the dependencies defined in `pyproject.toml` as well as the `pytorchexample` package.

```bash
pip install -e .
```

