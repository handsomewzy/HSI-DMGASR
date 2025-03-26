# HSI-DMGASR (AAAI 2024) – Official Documentation  

**Authors**: Zhaoyang Wang, Dongyang Li, Mingyang Zhang†, Hao Luo, Maoguo Gong  

This repository provides the official implementation of the AAAI 2024 paper:  
**"Enhancing Hyperspectral Images via Diffusion Model and Group-Autoencoder Super-Resolution Network"**

The proposed model, **HSI-DMGASR**, integrates a **Group-Autoencoder (GAE)** with a **Diffusion Model** to tackle the challenges of hyperspectral image super-resolution (HSI-SR).  

> 🔧 The code is based on prior works including **SR3**, **SSPSR**, **GELIN**, and **MCNet**, and is structured in two stages:
1. Training the Group-Autoencoder (GAE)  
2. Joint training with the Diffusion Model for HSI Super-Resolution

---

## 🧱 Network Architecture  
![stage.png](image/stage.png)

---

## 🧠 Motivation

> *Existing hyperspectral image (HSI) super-resolution (SR) methods struggle to effectively capture the complex spectral–spatial relationships and fine-grained visual details. In contrast, diffusion models have demonstrated powerful generative capabilities in modeling both high-level semantics and low-level structures. However, their direct application to HSI-SR faces challenges including convergence difficulty and high inference time.*

To overcome these issues, we propose a **Group-Autoencoder (GAE)** that:
- Encodes high-dimensional HSI data into a compact latent space
- Facilitates stable diffusion model training
- Preserves spectral correlation
- Significantly accelerates inference

This synergy forms the **DMGASR** framework, achieving superior performance on both natural and remote sensing HSI datasets—**both visually and quantitatively**.

---

## 📊 Experimental Results  
![ex1.png](image/ex1.png)  
![ex2.png](image/ex2.png)  
![vis.png](image/vis.png)

---

## ⚙️ Installation

Install required dependencies using:

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage Instructions

### Step 1️⃣: Train the Group-Autoencoder (GAE)

After setting the dataset paths, run:

```bash
python AE.py
```

---

### Step 2️⃣: Train the Diffusion Model

Once the GAE is trained, load its weights and run:

```bash
python sr_gae.py
```

> 🔄 Note: You can toggle between training and inference modes in `sr_gae.py`.

---

## ⚙️ Configuration

The training configuration file is located at:  
```bash
EHSI-DMGESR/config/sr_sr3_16_128.json
```

- Use this file to set:
  - Dataset paths
  - Model parameters
  - Training/inference settings

### 📁 Data Loading Methods

We support two ways of loading HSI datasets:

#### 1. `TrainsetFromFolder`
- Follows the **MCNet**-style data pipeline
- Data is pre-processed using **MATLAB** and loaded from local folders

#### 2. `HSTrainingData` / `HSTestData`
- Handles **online preprocessing**
- Offers greater **flexibility** and is suitable for end-to-end pipelines  
- Check the function definitions for more usage details

---

## 📌 Notes

- The codebase is primarily built upon **SR3**, **SSPSR**, and **MCNet** frameworks.
- We thank the authors of these works for their valuable contributions to the community.

---
