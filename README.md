# Audio Anomaly Detection for Predictive Maintenance

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/wasaeee/predicting-maintainence-using-anomalous-machine-sound-detection)

A deep learning project that uses a Convolutional Autoencoder (CAE) to detect anomalies in machine sounds, trained and deployed as an interactive web application.

---

## 🚀 Live Demo

An interactive demo of this project is hosted on Hugging Face Spaces. You can upload your own audio files or use the provided examples to see the model in action.

**[>> Click here to access the live demo <<](https://huggingface.co/spaces/wasaeee/predicting-maintainence-using-anomalous-machine-sound-detection)**

*(Note: Please replace the link above with the actual URL to your Hugging Face Space.)*

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

## ⚙️ How to Run Locally

To explore the code or run the application on your own machine:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git)
    cd YOUR_REPOSITORY_NAME
    ```

2.  **Set up a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Explore the notebook:**
    Launch Jupyter and open `notebook.ipynb` to see the model training and evaluation process.

5.  **Run the Gradio app:**
    ```bash
    python app.py
    ```
    The application will be available at a local URL (e.g., `http://127.0.0.1:7860`).
