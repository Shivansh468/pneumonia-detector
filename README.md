# 🩺 Deep Learning Pneumonia Detector
**🔴 [Live Demo: Try the App Here](https://pneumonia-detector-ytodkskzvoksqvqorxlsnp.streamlit.app/)**

<p align="center">
  <figure>
    <img width="1440" height="808" alt="App interface" src="https://github.com/user-attachments/assets/d1231e8e-d5c6-4ff3-a77b-b85005166625" />
    <figcaption align="center"><i>Figure 1: The clean Streamlit user interface upon launch, ready for image upload.</i></figcaption>
  </figure>
</p>

A full-stack deep learning web application that classifies chest X-rays to detect Pneumonia using Transfer Learning (ResNet18) and a Streamlit frontend. Built with PyTorch and optimized for Apple Metal Performance Shaders (MPS).

## Usage & Live Demo

<p align="center">
  <figure>
    <img width="1440" height="807" alt="App with prediction" src="https://github.com/user-attachments/assets/d7f67854-6198-43cf-afb1-c59ea7a8cdea" />
    <figcaption align="center"><i>Figure 2: The pipeline successfully classifying an uploaded X-ray with high confidence.</i></figcaption>
  </figure>
</p>

## The Dataset
The model was trained on the "Chest X-Ray Images (Pneumonia)" dataset from Kaggle (Paul Mooney), containing 5,863 JPEG images split into Normal and Pneumonia categories.

<p align="center">
  <figure>
    <img width="1440" height="807" alt="Different X-Rays" src="https://github.com/user-attachments/assets/b24ac1e9-81c1-410d-ae74-6fe2900cbf21" />
    <figcaption align="center"><i>Figure 3: Visual comparison of Normal, Bacterial Pneumonia, and Viral Pneumonia chest X-rays.</i></figcaption>
  </figure>
</p>

## Architecture & Pipeline
The core engine relies on a ResNet18 Convolutional Neural Network. We utilized Transfer Learning by freezing the pre-trained core layers and replacing the final fully connected layer to output two classes. 

```mermaid
flowchart LR
    subgraph Frontend [🖥️ Streamlit UI]
        direction TB
        A[Upload X-Ray Image] --> B[Web Dashboard]
    end

    subgraph Engine [⚙️ PyTorch Inference]
        direction TB
        C[Image Preprocessing<br>Resize 224x224 & Normalize] 
        D[ResNet18 Architecture<br>Transfer Learning]
        E[Modified Linear Layer<br>Outputs: Normal / Pneumonia]
    end

    subgraph Assets [📦 Storage & Hardware]
        direction TB
        F[(best_model.pth)]
        G[Apple MPS / GPU Acceleration]
    end

    A --> C
    F -.-|Loads Saved Weights| D
    G -.-|Powers| Engine
    C --> D
    D --> E
    E -->|Confidence Score| B
```

- **Environment:** Google Colab (T4 GPU)
- **Epochs:** 5
- **Optimizer:** Adam
- **Loss Function:** CrossEntropyLoss

## Installation & Local Execution

To set up and run the Pneumonia Detector locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/pneumonia-detector.git
    cd pneumonia-detector
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```

## Usage

Once the Streamlit application is running, you can:

1.  Open your web browser and navigate to the local URL provided by Streamlit (typically `http://localhost:8501`).
2.  Use the file uploader interface to select and upload a chest X-ray image (in JPEG or PNG format).
3.  The application will process the image and display the prediction, indicating whether the X-ray is classified as "Normal" or "Pneumonia."

**Tip**: Test images are provided in the `/sample_images` directory so you can test the app immediately.
