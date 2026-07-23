# Federated Facial Recogntion

This repo explores the application of Federated Learning (FL) as a privacy-preserving paradigm for 2D facial recognition systems, directly addressing the growing concerns surrounding biometric data collection and centralization. We investigate the efficacy of integrating state-of-the-art FL techniques with existing facial recognition methods, with a particular focus on the potential benefits of incorporating synthetically generated facial data to enhance model robustness and privacy. Despite its promise, significant challenges persist, including managing non-IID (non-independent and identically distributed) data heterogeneity, overcoming hardware constraints on client devices, and ensuring sufficient data availability for effective federated training.

---

## 1. Environment Setup

First, clone the repository and navigate to the project directory:

```bash
git clone https://github.com/PedroBVidal/face_rec_FL.git
cd face_rec_FL
```

Install the required dependencies

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

To override the configuration defined in `pyproject.toml` directly from the command line. For example, to run for 50 rounds with a batch size of 128:

```bash
flwr run . --run-config "num-server-rounds=50 batch-size=128 learning-rate=0.05"
```

