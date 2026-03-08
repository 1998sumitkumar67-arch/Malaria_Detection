# Malaria Detection Using Deep Learning

## Overview

This project implements a deep learning model to detect malaria parasites in microscopic blood smear images. The system classifies cell images into two categories:

- Parasitized
- Uninfected

The model uses **Transfer Learning with MobileNetV2** to achieve high accuracy while maintaining efficient training time.

---

## Dataset

The dataset consists of microscopic images of red blood cells.

Classes:

- Parasitized (Malaria Infected Cells)
- Uninfected (Healthy Cells)

Dataset Structure:



Dataset split:

- Training: 70%
- Validation: 15%
- Testing: 15%

---

## Data Preprocessing

The following preprocessing techniques are applied:

- Image resizing to **224x224**
- Pixel normalization (0–1 range)
- Data augmentation

### Data Augmentation

To improve generalization, the following augmentations are used:

- Rotation
- Horizontal Flip
- Zoom
- Shear Transformations

---

## Model Architecture

The project uses **MobileNetV2 (Transfer Learning)**.

Pipeline:
