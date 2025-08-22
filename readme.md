# 🐍 Toy Mamba Implementation

This repo contains a **from-scratch educational implementation** of the [Mamba architecture](https://arxiv.org/abs/2312.00752), a modern **state space model (SSM)** for sequence modeling.  
It’s built in PyTorch and kept small so you can study, train, and chat with it.

---

## 📂 Files
- **`model.py`** – Core implementation:
  - `Embedder`: token embeddings + vocab projection  
  - `SSM`: simplified selective state-space layer  
  - `MambaBlock`: normalization + gating + convolution + SSM  
  - `BasicMambaModel`: stacks multiple blocks into a full model  

- **`test.py`** – Script to **train** or **run** a Mamba model interactively.  
  - Includes a small toy **conversation dataset** for training.  
  - Provides a simple **chat interface** once the model is trained or loaded.  

---

## ⚡ Training
To train your own toy model:

```bash
python test.py
````

If no saved model is found, the script will ask if you want to train one.
The training loop:

* Tokenizes the dataset with GPT-2 tokenizer
* Trains `BasicMambaModel` for a small number of steps
* Saves weights as `mamba_conv_<timestamp>.pth`

---

## 💾 Using a Pretrained Model

Pretrained `.pth` weights (included in the repo) can be loaded directly:

```bash
python test.py
```

When prompted, choose the model index from the detected `.pth` files.
The script will:

1. Load the weights into `BasicMambaModel`
2. Start an **interactive chat session**

---

## 💬 Chatting

Once running, you can talk with the model:

```
Guy: Hey, how was your day?
Waifu: Calm, actually. I spent a few hours in the garden...
```

Type `quit` or `exit` to end the session.

---

## ⚙️ Requirements

* Python 3.9+
* [PyTorch](https://pytorch.org/)
* [tiktoken](https://github.com/openai/tiktoken)

Install deps:

```bash
pip install torch tiktoken
```

---

## 📌 Notes

* This is a **toy demo**, not optimized for speed or scale.
* Intended for **learning** and **experimentation** with Mamba-style architectures.

---
