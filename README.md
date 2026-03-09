# CoreCharge-LLM

**CoreCharge-LLM** is a lightweight **25M parameter Large Language Model (LLM)** designed for research and experimentation with transformer architectures and attention mechanisms.

> ⚠️ **Note:** Data pipelines and training pipelines are currently not included due to experimental and unstable code. This repository focuses on the **model architecture and core components**.

---

## 🚀 Features

* **25M parameter transformer-based LLM**
* **Modular architecture**: attention, feedforward, embeddings, and more
* **Research-oriented** and easy to extend
* Ideal for **custom attention mechanism experiments**

---

## 📦 Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/your-username/CoreCharge-LLM.git
cd CoreCharge-LLM
pip install -r requirements.txt
```

---

## ▶️ Usage

### Training

```bash
python -m script --mode train
```

### Testing

```bash
python -m script --mode test
```

### Inference

Run inference with a prompt:

```bash
python -m script --mode infer --prompt "Your prompt here"
```

Run inference using a specific checkpoint:

```bash
python -m script --mode infer --prompt "Your prompt here" --checkpoint checkpoints/ckpt.pt
```

---

## 🛠️ Notes

* The repository is **research-focused**; certain features like the full training/data pipeline are not included.
* Designed for **extending and experimenting** with transformer and attention modules.
* Contributions and experiments are welcome!
