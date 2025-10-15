# Audio Anomaly Detection for Predictive Maintenance

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/wasaeee/predicting-maintainence-using-anomalous-machine-sound-detection)

A deep learning project that uses a Convolutional Autoencoder (CAE) to detect anomalies in machine sounds, trained and deployed as an interactive web application.

---

## 📝 Project Overview

This project addresses the problem of predictive maintenance by analyzing the sounds produced by industrial machinery (specifically, a fan). The core idea is to build a model that understands what a "normal" or "healthy" machine sounds like. When the model encounters a sound that deviates from this learned norm, it flags it as an anomaly, potentially indicating a fault or impending failure.

The workflow is as follows:
1.  **Data Preprocessing:** Audio signals are converted into Mel spectrograms, which are 2D visual representations of the sound.
2.  **Model Training:** A Convolutional Autoencoder (CAE) is trained exclusively on spectrograms of **normal** machine sounds.
3.  **Anomaly Detection:** The model learns to reconstruct normal sounds with a very low error. When it is fed an **abnormal** sound, the reconstruction is poor, resulting in a high error.
4.  **Thresholding:** A statistical threshold is calculated. Any sound with a reconstruction error above this threshold is classified as an anomaly.

## 🛠️ Technology Stack

- **Python**: Core programming language.
- **PyTorch**: For building and training the Convolutional Autoencoder.
- **Librosa**: For audio processing and feature extraction (Mel spectrograms).
- **Gradio**: To create and power the interactive web UI.
- **Hugging Face Spaces**: For deploying and hosting the live application.
- **Jupyter Notebook**: For model development, training, and evaluation.

## 📁 Repository Files

This repository contains the key artifacts of the project:

- **`notebook.ipynb`**: A Jupyter Notebook detailing the complete end-to-end process, from data loading and preprocessing to model training and evaluation.
- **`cae_autoencoder.pth`**: The saved weights of the final trained PyTorch model.
- **`app.py`**: The Python script for the Gradio web application.
- **`model.py`**: Contains the `ConvAutoencoder` class definition.
- **`requirements.txt`**: A list of Python libraries required to run the project.
- **`normal.wav` / `abnormal.wav`**: Example audio files used for the live demo.
