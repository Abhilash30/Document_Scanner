
# Classical OCR using Computer Vision and Machine Learning

A learning-oriented Optical Character Recognition (OCR) project exploring how classical computer vision and machine learning can be used to recognize printed English characters.

The project currently uses a Support Vector Machine (SVM) trained on raw pixel features from the Kaggle Typed English Characters dataset. Although it achieves **90.14% accuracy on the dataset**, predictions on real-world document images are currently incorrect and require further investigation.

## Project Overview

The goal is to build an OCR pipeline incrementally, starting with isolated-character classification and progressing toward character segmentation and full text recognition.

The current focus is understanding the relationship between image preprocessing, feature extraction, classification, and real-world generalization.

## Current Implementation

The current pipeline is:

```text
Typed Character Dataset (CSV)
             |
             v
      CharacterDataset
             |
             v
       ImageProcessor
             |
             v
     RawPixelExtractor
             |
             v
        SVMClassifier
             |
             v
          Evaluator
```

### Components

- **CharacterDataset:** Loads the CSV dataset and separates character images from their labels.
- **ImageProcessor:** Handles image reshaping and preprocessing.
- **RawPixelExtractor:** Represents each 28 × 28 grayscale character as a vector of 784 pixel values.
- **SVMClassifier:** Uses scikit-learn's Support Vector Classifier to learn and predict character classes.
- **Evaluator:** Measures classification performance on held-out data.

The project follows an object-oriented structure to keep dataset handling, preprocessing, feature extraction, classification, and evaluation separate.

## Dataset

The current training and evaluation dataset is the Kaggle Typed English Characters dataset.

- Character classes: 62 (digits 0–9, uppercase A–Z, and lowercase a–z)
- Image dimensions: 28 × 28 grayscale
- Raw pixel features: 784 per character
- Number of samples: approximately 63,000

Each sample represents an isolated character rather than a complete word, sentence, or document.

## Current Results

| Evaluation | Result |
|---|---|
| Dataset | Kaggle Typed English Characters |
| Feature representation | Raw pixel intensities |
| Classifier | SVM |
| Dataset test accuracy | **90.14%** |
| Real-world document predictions | Incorrect; investigation ongoing |

The 90.14% accuracy measures classification performance on the held-out dataset test split. It does not establish equivalent performance on real-world documents.

## Current Limitation: Real-World Generalization

When the model is applied to real-world document images, its predictions are currently incorrect. The cause has not yet been established.

Potential issues to investigate include:

- **Domain mismatch:** Real document characters may differ from the training samples in font, scale, contrast, noise, and appearance.
- **Character normalization:** Real character crops may not match the dataset's 28 × 28 representation.
- **Preprocessing differences:** Pixel intensity ranges, foreground/background polarity, thresholding, and resizing may differ between training and inference.
- **Segmentation errors:** Incorrect bounding boxes or incomplete character crops can distort the input.
- **Feature representation:** Raw pixels may be sensitive to shifts and variations in character appearance.

These are hypotheses to investigate, not confirmed causes of the current failure.

## Planned Improvements

1. Investigate real-world predictions and inspect the preprocessed character images.
2. Verify that training and inference use consistent preprocessing, normalization, and pixel polarity.
3. Build a small labelled dataset of characters extracted from printed document images.
4. Implement character segmentation using thresholding, morphology, contours, and bounding boxes.
5. Experiment with Histogram of Oriented Gradients (HOG) as an alternative feature representation.
6. Compare raw pixels and HOG using consistent evaluation data.
7. Integrate segmentation and character classification into an end-to-end classical OCR pipeline.
8. Explore CNN-based character recognition after establishing a reliable classical baseline.

## Learning Objectives

- Understand the fundamentals of supervised classification.
- Explore how SVMs classify high-dimensional image data.
- Compare raw pixel features with handcrafted feature representations such as HOG.
- Apply classical computer vision techniques to character segmentation.
- Investigate the difference between dataset accuracy and real-world performance.
- Build an OCR system incrementally with a modular, object-oriented design.

## Tech Stack

- Python
- NumPy
- Pandas
- scikit-learn
- Pillow / OpenCV, where applicable

## Project Status

**In progress — isolated-character classification implemented; real-world generalization and end-to-end OCR remain under investigation.**

The main objective is not just to maximize accuracy on a single dataset, but to understand why a model that performs reasonably well on controlled data may fail when exposed to real-world inputs.