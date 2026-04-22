# Model Card — Drunk Detection System

## Model Details

| Property | Value |
|----------|-------|
| **Model Name** | Drunk Detection Classifier |
| **Architecture** | MobileNetV3Small + Custom Head |
| **Framework** | TensorFlow / Keras |
| **Input Size** | 224 × 224 × 3 (RGB) |
| **Output** | 2 classes (Drunk, Not Drunk) |
| **Parameters** | ~2.5M total (~1.5M trainable after fine-tune) |
| **Model Size** | ~10MB (Keras), ~3MB (TFLite quantized) |
| **License** | MIT |
| **Version** | 1.0.0 |

## Intended Use

### Primary Use Case
Real-time detection of drunk/intoxicated drivers using facial images captured from a dashboard camera on a Raspberry Pi 5, integrated with an MQ3 alcohol sensor for multi-modal decision fusion.

### Intended Users
- Fleet management companies
- Transportation safety systems
- Research in driver monitoring systems

### Out-of-Scope Uses
- ❌ Law enforcement evidence (not forensically validated)
- ❌ Medical diagnosis of intoxication levels
- ❌ Stand-alone drunk detection without human oversight
- ❌ Use on demographics not represented in training data

## Training

### Architecture Design

```
MobileNetV3Small (ImageNet pretrained)
    ↓ (frozen in Phase 1, top layers unfrozen in Phase 2)
GlobalAveragePooling2D
    ↓
Dense(128, activation='swish')
    ↓
BatchNormalization
    ↓
Dense(64, activation='swish')
    ↓
Dropout(0.3)
    ↓
Dense(2, activation='softmax')
```

### Why MobileNetV3Small?
- **Designed for mobile/edge**: Uses inverted residuals, squeeze-excitation blocks, and h-swish activation
- **Compact**: ~2.5M params vs ResNet50's 25M — critical for Raspberry Pi deployment
- **Efficient**: Achieves competitive accuracy with 10x fewer FLOPs than larger models

### Training Strategy
**Two-phase transfer learning:**

1. **Phase 1 — Head Training** (frozen backbone):
   - Train only the classification head
   - LR: 1e-4, Epochs: up to 100 (early stopping)
   - Prevents catastrophic forgetting of ImageNet features

2. **Phase 2 — Fine-tuning** (partial backbone unfreeze):
   - Unfreeze top layers of MobileNetV3 (layer 100+)
   - LR: 1e-5 (10x lower to preserve features)
   - Epochs: 20
   - Adapts backbone features to the drunk detection domain

### Data Augmentation (Training Only)
| Augmentation | Value |
|-------------|-------|
| Brightness | ±20% |
| Zoom | ±10% |
| Width Shift | ±5% |
| Height Shift | ±5% |

> **Critical Fix**: Validation and test sets use only rescaling (1/255) — NO augmentation. This ensures unbiased evaluation metrics.

### Preprocessing Pipeline
1. Face detection using MediaPipe FaceMesh
2. Face region extraction using FACE_OVAL polygon mask
3. Cropping to bounding box
4. Resize to 224×224
5. Normalize pixel values to [0, 1]

## Evaluation

### Metrics Computed
| Metric | Description |
|--------|-------------|
| Accuracy | Overall classification accuracy |
| Precision (per-class) | Correct positive predictions / total positive predictions |
| Recall (per-class) | Correct positive predictions / total actual positives |
| F1-Score (weighted) | Harmonic mean of precision and recall |
| ROC-AUC | Area under the ROC curve |
| False Negative Rate | **Critical** — drunk drivers missed as sober |
| Confusion Matrix | Full TP/TN/FP/FN breakdown |

### Safety Consideration

> **In a safety-critical system, False Negative Rate matters more than overall accuracy.** A drunk driver classified as sober is dangerous. The threshold is optimized for high recall on the 'Drunk' class.

### Threshold Analysis
The system includes threshold analysis to find the optimal operating point that balances:
- High recall on "Drunk" class (minimize missed drunk drivers)
- Acceptable precision (minimize false alarms)
- The optimal threshold is determined per deployment scenario

## Ethical Considerations

### Privacy
- Face images are processed locally on the edge device
- No images are sent to cloud servers
- Evidence photos are stored locally and sent only via Telegram to authorized personnel

### Bias & Fairness
- Model performance should be evaluated across demographic groups
- Training data distribution should be documented
- Regular bias audits are recommended

### Consent
- Drivers should be informed about the monitoring system
- System is designed for fleet/company use with driver consent

## Deployment

### Hardware Requirements
| Component | Specification |
|-----------|-------------- |
| **Compute** | Raspberry Pi 5 (4GB+) |
| **Camera** | Raspberry Pi Camera Module / USB webcam |
| **Sensor** | MQ3 alcohol sensor + Arduino |
| **Storage** | 16GB+ microSD |

### Inference Performance
| Metric | Target |
|--------|--------|
| Model Size (TFLite) | ~3 MB |
| Inference Latency | <100ms per frame |
| FPS | ~10-15 FPS |
| Memory Usage | <500 MB |

### Decision Fusion Logic
The final detection decision combines two signals:
```
IF (camera_prediction == "Drunk") OR (MQ3_value > threshold):
    status = "Drunk"
    IF continuous_drunk_frames >= required_frames:
        → Trigger alarm (speaker + warning light)
        → Send Telegram alert with photo evidence
        → Log violation to CSV
```

## Limitations

1. **Lighting conditions**: Performance may degrade in very low light
2. **Occlusion**: Sunglasses, masks, or hand-over-face may affect accuracy
3. **Drowsiness vs. Drunkenness**: The model may confuse drowsy appearance with intoxication
4. **Single face**: Only processes one face per frame
5. **Static threshold**: MQ3 threshold is fixed; individual alcohol tolerance varies

## Model Maintenance

- Retrain periodically as deployment data grows
- Monitor for distribution shift (seasonal lighting changes, new driver demographics)
- A/B test threshold changes before deploying
- Track model performance via MLflow experiments
