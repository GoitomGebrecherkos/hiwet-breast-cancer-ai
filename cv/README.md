# Hiwet computer-vision module

This module adds three-class breast-ultrasound classification to the existing SVM application.

## BUSI dataset handling

Use the extracted BUSI folder containing `benign/`, `malignant/`, and `normal/`. The training script uses only original images and ignores every file containing `_mask` in its filename. Segmentation masks are therefore not accidentally treated as classification images.

## Train

From the project root:

```bash
python cv/train_cv.py --data "C:/path/to/Dataset_BUSI_with_GT" --epochs 12
```

The script creates:
- `cv/cv_model.pt` — trained CNN weights
- `cv/cv_metrics.json` — split information and test metrics
- `cv/train_files.json`, `cv/val_files.json`, `cv/test_files.json` — exact split manifests

The baseline CNN is intentionally compact so it can train on CPU. For a stronger research result, replace it with transfer learning (e.g. ResNet/EfficientNet) after validating the data split.

## Run app

```bash
streamlit run app.py
```

The app keeps the original 30-feature SVM workflow and adds an ultrasound image workflow under section 7.

## Recommended training

For the final project, use `train_transfer.py`. It uses ImageNet-pretrained ResNet18 and saves the exact train/validation/test manifests. This is preferable to the compact CPU baseline when you have enough compute. The current environment could not download ImageNet weights, so no unvalidated/low-performing CV model is shipped with this project.


## Fast setup for the Hiwet presentation

The main `app.py` now has an **Install / Train Ultrasound Computer Vision Model**
button. After installing requirements, run:

```bash
streamlit run app.py
```

Then click the Section 7 button. Hiwet downloads the BUSI dataset through
KaggleHub, finds `benign/`, `malignant/`, and `normal/`, trains the compact CNN
for 8 epochs, and creates `cv/cv_model.pt` and `cv/cv_metrics.json`.

If you already have an extracted BUSI folder, you can still train manually:

```bash
python cv/train_cv.py --data "C:/path/to/Dataset_BUSI_with_GT" --epochs 12
```
