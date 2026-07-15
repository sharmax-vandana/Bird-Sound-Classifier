# Bird Species Classification via Transfer Learning

A systematic side-by-side comparison of two transfer learning paradigms for imbalanced bird sound classification on a 100+ species Xeno-canto derived dataset.

- **Audio-domain approach** — Google's YAMNet (pretrained on AudioSet) as a fixed feature extractor, paired with three classical ML classifiers (SVM, MLP, Random Forest).
- **Image-domain approach** — MobileNetV2 (pretrained on ImageNet) fine-tuned on mel-spectrograms treated as images.

Both pipelines are evaluated on the same dataset with macro-averaged and per-species metrics, transparently characterizing model behavior on rare (long-tail) species.

> This project was developed as part of the 6-Week Summer Internship on Machine Learning & Generative AI at IGDTUW. A research paper based on this work is in preparation.

---

## Results Summary

| Pipeline / Model              | Val Acc | Test Acc | Macro Acc | Macro-F1 | Weighted F1 |
|-------------------------------|:-------:|:--------:|:---------:|:--------:|:-----------:|
| YAMNet + SVM (RBF)            | 67.08%  | 68.12%   | —         | 0.646    | —           |
| YAMNet + MLP (512, 256)       | 64.89%  | 68.44%   | —         | 0.636    | —           |
| **YAMNet + Random Forest**  | 69.91%  | **70.00%** | 67.34%  | **0.654** | 0.696      |
| MobileNetV2 (fine-tuned)      | 53.74%  | 57.01%   | 50.80%    | 0.451    | 0.563       |

**Key finding:** Audio-domain transfer (YAMNet + classical ML) outperformed image-domain transfer (MobileNetV2 on mel-spectrograms) by approximately 13 percentage points in test accuracy and 20 points in macro-F1 on this dataset.

---

## Dataset

**Sound of 114 Species of Birds till 2022** — a Kaggle corpus derived from Xeno-canto.

- **Source:** [Kaggle dataset link](https://www.kaggle.com/datasets/soumendraprasad/sound-of-114-species-of-birds-till-2022)
- **Advertised size:** 3,336 recordings across 114 species
- **Working sample count after ingestion:** 2,161 recordings
- **Effective species after rare-class filtering:** 100 (YAMNet pipeline, ≥5 samples), 102 (CNN pipeline, ≥4 samples)

---

## Repository Structure

```
.
├── BirdSound_YAMNet_ML_v2.ipynb           # Pipeline A: YAMNet + SVM/MLP/RF
├── BirdSound_Spectrogram_CNN_v2.ipynb     # Pipeline B: MobileNetV2 on mel-spectrograms
├── README.md
└── docs/
    ├── literature_review.pdf              # 15-paper LR matrix
    └── progress_report.pdf                # Internship progress report
```

---

## Pipeline A — YAMNet + Classical ML

Located in `BirdSound_YAMNet_ML_v2.ipynb`.

### Approach
1. Audio resampled to 16 kHz mono (YAMNet requirement)
2. YAMNet frame scores mean-pooled to produce a single **1024-dim embedding per recording**
3. Embeddings cached to disk to avoid re-extraction
4. StandardScaler feature normalization (fit on training set only)
5. Three classifiers trained and compared:
   - **SVM** — RBF kernel, C=10, gamma=scale
   - **MLP** — hidden layers (512, 256), early stopping
   - **Random Forest** — grid-searched over n_estimators, max_depth, max_features

### Key output artifacts
- `yamnet_embeddings.csv` — cached 1024-dim embeddings
- `final_metrics.json` — consolidated results for paper's Table 1
- `per_species_accuracy.csv` — per-species accuracy, precision, recall, F1
- Confusion matrix + top confusion pairs for error analysis

---

## Pipeline B — MobileNetV2 on Mel-Spectrograms

Located in `BirdSound_Spectrogram_CNN_v2.ipynb`.

### Approach
1. Audio resampled to 16 kHz mono
2. Mel-spectrograms computed (128 mel bins, n_fft=2048, hop_length=512) and rendered as **224×224 RGB images**
3. Class weights computed from training-set frequency (range: 0.598 to 5.582) to counter long-tail imbalance
4. Two-stage training:
   - **Head-only training** — 20 epochs, lr=1e-3, MobileNetV2 base frozen
   - **Fine-tuning** — 15 epochs, lr=1e-5, deeper layers unfrozen
5. Checkpointing on best validation accuracy

### Key output artifacts
- `bird_finetuned_best.h5` — best fine-tuned model checkpoint
- `final_bird_classifier.keras` — final saved model
- `class_indices.json` — species → index mapping
- Per-species metrics CSV + confusion matrix

---

## Getting Started

### Requirements

```bash
tensorflow>=2.20
tensorflow-hub
librosa
scikit-learn
pandas
numpy
matplotlib
seaborn
joblib
```

Recommended: run in **Google Colab with GPU** — MobileNetV2 fine-tuning is impractical on CPU. YAMNet embedding extraction works on CPU but is slow.

### Running the notebooks

Both notebooks are self-contained and use Google Drive for caching intermediate outputs (embeddings, model checkpoints, results). Update the `DRIVE_ROOT` variable at the top of each notebook to point to your own Drive folder.

The notebooks are guarded — if cached outputs already exist, expensive steps (dataset download, embedding extraction, model training) are skipped automatically.

### Prediction on new audio

Both notebooks include a prediction pipeline. Point the `test_audio` variable at any WAV/MP3 file:

```python
# Pipeline A — YAMNet + Random Forest
predictions = predict_topk(audio_path, k=5)

# Pipeline B — MobileNetV2
predictions = predict_from_audio(audio_path, top_k=5)
```

---

## Reproducibility

- **Random seed:** 42 (fixed across NumPy, TensorFlow, and sklearn)
- **Stratified train/val/test split** with safe fallback for very small classes
- All models, scalers, encoders, and metrics saved to disk for exact reproduction

---

## Research Context

A literature review of 15 recent bird bioacoustic classification papers identified the following gaps this project addresses:

1. Most studies commit to a single transfer learning paradigm (either audio-domain OR image-domain) — few compare the two on the same dataset.
2. When pretrained embeddings are used, typically only one downstream classifier is evaluated (commonly logistic regression).
3. Prior work commonly reports only overall accuracy, hiding poor performance on rare long-tail species.
4. Studies at high accuracy typically use very small species pools (10-20 species) or bird-specific pretrained models (BirdNET, Perch) that are not available for many taxa.

Full literature review matrix available in `docs/literature_review.pdf`.

---

## Limitations and Next Steps

**Known limitations at this stage:**
- Two pipelines currently use slightly different filtering thresholds (≥5 vs ≥4 samples per class) and split ratios (70/15/15 vs 80/10/10). Standardizing to identical splits is a planned next step.
- 12–14 rare species are dropped due to insufficient samples for stratified splitting.
- Only single-seed runs completed — multi-seed statistical validation planned.

**Planned improvements:**
- Align both pipelines to identical species filtering and identical train/val/test splits
- Multi-seed runs (3–5) to report accuracy ± std
- BirdNET-embedding comparison as a bird-specific upper bound
- Ensembling of the two paradigms (logit averaging)
- Streamlit demo for interactive top-K prediction

---

## Acknowledgments

- **YAMNet** — Google Research, TensorFlow Hub
- **MobileNetV2** — Sandler et al., 2018 (Keras Applications)
- **Dataset** — Sound of 114 Species of Birds till 2022, Kaggle (derived from [Xeno-canto](https://xeno-canto.org))

Developed as part of the 6-Week Summer Internship on Machine Learning & Generative AI, IGDTUW, under faculty guidance.
