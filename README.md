# ReflectoCam

ReflectoCam is a deep learning-based project that detects and segments mirror and glass regions in images. The project uses a U-Net architecture to distinguish between transparent and reflective surfaces, improving scene understanding for applications such as robotics, autonomous systems, and augmented reality.

---

## Overview

The goal of this project is to accurately identify mirror and glass areas in visual data — regions that often confuse traditional recognition systems due to their reflective or transparent nature.  
By training a convolutional neural network (CNN) on a specialized dataset, the model learns to differentiate reflective materials from regular objects in real-world environments.

ReflectoCam performs:
- End-to-end segmentation of mirror and glass surfaces  
- Preprocessing and augmentation of image–mask pairs  
- Model training, validation, and evaluation with visualization  

---

## Dataset

The dataset used for this project is sourced from the **Mirror and Glass Surface Detection Dataset**:  
[Mirror-Glass-Detection Dataset](https://github.com/Charmve/Mirror-Glass-Detection)

It contains:
- RGB images of indoor and outdoor scenes  
- Corresponding binary masks marking mirror and glass regions


---

## Methodology

### 1. Data Preprocessing
- Loaded image–mask pairs and resized them for uniformity.  
- Normalized pixel values and split the dataset into training and validation sets.  
- Applied augmentation to improve generalization.

### 2. Model Architecture
- Implemented **U-Net** using **TensorFlow/Keras**.  
- Encoder extracts spatial features through convolution and pooling.  
- Decoder performs upsampling to reconstruct the segmented mask.  
- Output is a binary mask indicating mirror/glass regions.

### 3. Model Training
- Loss Function: Binary Cross-Entropy  
- Optimizer: Adam  
- Evaluation Metrics: Accuracy and IoU (Intersection over Union)  
- Real-time visualization of loss and accuracy curves.

### 4. Evaluation and Visualization
- Tested the model on unseen data for performance validation.  
- Compared ground truth masks with predicted masks.  
- Visualized segmentation outputs using **Matplotlib**.

![ReflectoCam Methodology](https://github.com/architakulkarni10/ReflectoCam/blob/main/ReflectoCam_Methodology.png)

---

## Technologies Used

- **Programming Language:** Python  
- **Frameworks:** TensorFlow, Keras  
- **Libraries:** NumPy, Pandas, OpenCV, Matplotlib, Seaborn  
- **Environment:** Google Colab  

---

## Results

- The U-Net model successfully segments mirror and glass regions.  
- Achieved high visual accuracy and reliable performance on test data.  
- Generated precise binary masks for reflective and transparent surfaces.

---

## Future Enhancements

- Extend model for real-time mirror/glass detection using webcam input.  
- Optimize for faster inference and reduced computational load.  
- Explore multi-class segmentation for diverse material surfaces.
