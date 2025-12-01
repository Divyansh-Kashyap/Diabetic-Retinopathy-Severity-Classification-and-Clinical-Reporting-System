# Diabetic-Retinopathy-Severity-Classification-and-Clinical-Reporting-System
## Overview 

This project presents a deep learning–based system designed to classify five stages of Diabetic Retinopathy (DR) using retinal fundus images. The model also generates clinical-style interpretive reports, leveraging Grad-CAM to visually highlight regions that influence the prediction.
This system is intended for academic research, clinical screening support.

## Problem Statement

Diabetic Retinopathy is one of the leading causes of preventable blindness worldwide. Manual grading of retinal images is time-consuming, subjective, and requires trained ophthalmologists. An automated, interpretable screening tool can significantly aid early diagnosis and reduce the clinical workload.

This project aims to build an AI-driven, explainable classifier that accurately predicts DR severity and presents insights in a structured, clinician-friendly format. 

## Goal

To develop an automated and interpretable system that classifies Diabetic Retinopathy into five severity stages and generates detailed clinical reports based on model predictions.

## Objectives

Build a PyTorch-based model to classify DR into five stages:
0 – No DR, 1 – Mild, 2 – Moderate, 3 – Severe, 4 – Proliferative DR

Preprocess and augment retinal fundus images.

Implement Grad-CAM for visual explainability.

Generate clinical-style reports for each prediction.

Evaluate the model using classification metrics and visual outputs.

## Dataset

APTOS 2019 Blindness Detection Dataset

Features:

3,662 labeled retinal images

Labels categorized into DR severity levels 0–4

Real-world variability such as blur, lighting differences, and noise

Dataset Source: Kaggle (APTOS 2019)

## Directory Structure

```
├── data/
├── models/
├── notebooks/
│   └── Diabetic_Retinopathy_Classification.ipynb
├── reports/
│   └── clinical_outputs/
├── README.md
└── requirements.txt
```
## Results

Successful classification of DR images across five severity levels

High-quality Grad-CAM explanations to support interpretability

Structured clinical-style reports generated for test samples

Visualization of label distribution and sample predictions

## Tech Stack

Python

PyTorch

Albumentations

OpenCV

NumPy

## Installation
1. Clone the Repository
```

git clone <repository-url>
cd <repository-folder>
```
2. Install Dependencies
```
pip install -r requirements.txt
```

Author

Divyansh Kashyap, Diksha 
B.Tech AIML, Semester 5
Jagannath University
