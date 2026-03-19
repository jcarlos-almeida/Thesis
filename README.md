# BLSTM‑VAE-R: Bidirectional LSTM Variational Autoencoder with Regression Layer for Prognostics

This repository contains the implementation of a **Bidirectional LSTM Variational Autoencoder (BLSTM‑VAE-R)** developed as part of a research project evaluating prognostic performance.  
The project explores deep generative architectures capable of learning latent representations from sequential data and reconstructing temporal patterns with high fidelity.
---

## 📂 Project Structure
BLSTM-VAE/ │ ├── data/                 # Raw and processed datasets (not tracked by Git) ├── dicts/                # Dictionaries or lookup tables used by the model ├── images/               # Figures, plots, and visualizations ├── lists/                # Auxiliary lists or metadata ├── models/               # Saved model weights and checkpoints ├── BLSTM-VAE-R.ipynb     # Main research notebook ├── ProgPerfMetrics.py    # Performance metrics and evaluation utilities ├── utils.py              # Helper functions └── environment.yml       # Conda environment specification

---

## 🧠 Overview

The BLSTM‑VAE architecture combines:

- **Bidirectional LSTMs** for capturing forward and backward temporal dependencies  
- **Variational Autoencoders** for learning smooth latent spaces  
- **Reconstruction‑based anomaly detection** for identifying deviations in time‑series patterns  

This approach is suitable for:

- Predictive maintenance  
- Sensor data modeling  
- Fault detection  
- Sequence compression  
- Representation learning  

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/jcarlos-almeida/BLSTM-VAE.git
cd BLSTM-VAE


2. Create the Conda environment
conda env create -f environment.yml
conda activate phm_env



▶️ Usage
Run the main notebook
Open the Jupyter notebook:
jupyter notebook BLSTM-VAE-R.ipynb


Or run scripts directly
python ProgPerfMetrics.py



📊 Results & Visualizations
Plots and figures generated during training and evaluation are stored in the images/ directory.
These include:
- Reconstruction curves
- Latent space projections
- Anomaly scores
- Training loss evolution

📦 Reproducibility
The full Conda environment is provided in:
environment.yml


To recreate the environment:
conda env create -f environment.yml


🤝 Contributions
Contributions, suggestions, and improvements are welcome.
Feel free to open issues or submit pull requests.

---

If you want, I can also help you:

- refine the README with diagrams or equations  
- add badges (Python version, license, environment, etc.)  
- write a proper citation in BibTeX format  
- generate a `.gitignore` tailored to your project  
- reorganize your repo structure for publication  

Just tell me what direction you want to take next.




---

## 🧠 Overview

The BLSTM‑VAE-R architecture combines:

- **Bidirectional LSTMs** for capturing forward and backward temporal dependencies  
- **Variational Autoencoders** for learning smooth latent spaces  
- **Regression model, such as ANN, CNN, LSTM, for RUL prediction** for generating RUL prediction.  

This approach is suitable for:

- Predictive maintenance  
- Sensor data processing  
- RUL prediction


---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/jcarlos-almeida/BLSTM-VAE.git
cd BLSTM-VAE

### 2. Create the Conda environment

conda env create -f environment.yml
conda activate phm_env

### 3. Usage

jupyter notebook BLSTM-VAE-R.ipynb
