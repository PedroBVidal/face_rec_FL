# Federated Facial REcogntion

This repository provides a comprehensive framework for Federated Learning applied to Facial Recognition, built on top of [Flower (flwr)](https://flower.ai/) and [PyTorch](https://pytorch.org/), integrating [ArcFace](https://github.com/deepinsight/insightface) backbones.

---

## 1. Environment Setup

First, clone the repository and navigate to the project directory:

```bash
git clone https://github.com/PedroBVidal/face_rec_FL.git
cd face_rec_FL
```

Install the required dependencies using the provided `requirements.txt` file. We recommend using a virtual environment or conda:

```bash
pip install -r requirements.txt
```

Additionally, install the current project in editable mode so that imports work correctly:

```bash
pip install -e .
```

---

## 2. Dataset Preparation (Aligned and Cropped)

For facial recognition tasks, it is critical that your training and evaluation images are **aligned and cropped** (typically to 112x112 pixels) using facial landmarks before feeding them into the network. 

1. **Obtain a dataset**: You can use datasets like MS-Celeb-1M, WebFace42M, CASIA-WebFace, or your own custom dataset.
2. **Align and crop**: If your dataset is not already aligned, use tools like MTCNN or RetinaFace to detect landmarks and crop the faces to 112x112.
3. **Structure**: Organize the dataset such that each identity has its own folder, e.g.:
   ```text
   /path/to/dataset/
   ├── identity_1/
   │   ├── image_0001.jpg
   │   └── image_0002.jpg
   ├── identity_2/
   │   └── image_0001.jpg
   └── ...
   ```

### Configuring the Dataset Path

Once your dataset is ready, you must configure the project to point to it. 
Open `pyproject.toml` in the root of the project and update the `data-path` value under `[tool.flwr.app.config]`:

```toml
[tool.flwr.app.config]
# Other configurations...
data-path = "/path/to/your/aligned/and/cropped/dataset/"
```

You can also adjust other hyperparameters here, such as `num-server-rounds`, `batch-size`, and `learning-rate`.

---

## 3. Running a Simulation with GPU

Flower's Simulation Engine allows you to simulate a large number of clients on a single machine or a cluster, automatically managing GPU resources.

To run a simulation utilizing GPU resources, use the `flwr run` command. By default, Flower will allocate available GPU resources to the client applications:

```bash
flwr run . 
```

### Overriding Configuration on the Fly

You can easily override the configuration defined in `pyproject.toml` directly from the command line. For example, to run for 50 rounds with a batch size of 128:

```bash
flwr run . --run-config "num-server-rounds=50 batch-size=128 learning-rate=0.05"
```

### Using Custom Schedulers (Advanced)

If you are running specific benchmarks or need more granular control over how clients are scheduled, you can use the provided bash scripts that leverage custom schedulers:

```bash
# Example: Run Emore benchmarks
nohup bash scripts/run_emore_benchmarks.sh > benchmark_run_emore.log 2>&1 &
```
*(Make sure to update the conda environment name inside the bash script if necessary)*

---

## 4. Outputs, Logs, and Metrics

As the federated training progresses, several artifacts will be generated:

- **Checkpoints**: Stored in the `checkpoints/` directory. The best global models and individual round checkpoints are saved here.
- **Metrics**: A `metrics.json` file will be updated with evaluation metrics (e.g., accuracy, loss) across rounds.
- **Logs**: Detailed execution logs will be available in the `logs/` directory.
- **TensorBoard**: You can visualize the training progress in real-time by pointing TensorBoard to the logs directory:
  
  ```bash
  tensorboard --logdir logs/tensorboard/
  ```

---

## 5. Repository Structure Overview

- `client_app.py`: Defines the Flower Client behavior (local training and evaluation).
- `server_app.py`: Defines the Flower Server behavior (aggregation strategies).
- `task.py`: Contains the model definitions (ArcFace backbones), data loading logic, and training loop.
- `custom_strategy.py`: Implements any custom aggregation strategies (like FedAvg with specific metric aggregation).
- `pyproject.toml`: The main configuration file for Flower and Python dependencies.
- `arcface_torch/`: A submodule/directory containing the core ArcFace model implementations.
