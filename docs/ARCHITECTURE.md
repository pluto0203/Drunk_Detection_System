# Architecture Documentation — Drunk Detection System

## System Overview

The Drunk Detection System is an end-to-end AI solution for real-time intoxication detection, deployed on edge hardware (Raspberry Pi 5). It combines computer vision (facial analysis) with IoT sensor data (MQ3 alcohol sensor) for multi-modal decision fusion.

## System Architecture

```mermaid
graph TB
    subgraph Edge Device - Raspberry Pi 5
        CAM[📷 Camera Module] --> FE[Face Extraction<br/>MediaPipe FaceMesh]
        FE --> INF[TFLite Inference<br/>MobileNetV3Small]
        MQ3[🔬 MQ3 Sensor] --> ARD[Arduino<br/>Serial Comm]
        ARD --> FUSION
        INF --> FUSION[Decision Fusion]
        FUSION -->|Drunk Detected| ALERT[Alert System]
        ALERT --> SPK[🔊 Speaker]
        ALERT --> LED[💡 Warning Light]
        ALERT --> TG[📱 Telegram Bot]
        ALERT --> LOG[📝 CSV Logger]
    end

    subgraph Monitoring
        LOG --> DASH[📊 Flask Dashboard]
        TG --> USER[👤 Fleet Manager]
    end
```

## Data Flow Pipeline

```mermaid
flowchart LR
    subgraph Training Pipeline
        RAW[Raw Images] --> FACE[Face Extraction<br/>MediaPipe] --> AUG[Augmentation<br/>Train Only] --> TRAIN[Two-Phase<br/>Training]
        TRAIN --> EVAL[Comprehensive<br/>Evaluation]
        TRAIN --> EXPORT[TFLite Export<br/>+ Quantization]
        EVAL --> MLFLOW[(MLflow<br/>Tracking)]
        TRAIN --> MLFLOW
    end

    subgraph Deployment Pipeline
        EXPORT --> DEPLOY[Raspberry Pi<br/>Deployment]
    end
```

## Model Architecture

```mermaid
graph TB
    INPUT[Input Image<br/>224×224×3] --> BACKBONE

    subgraph BACKBONE [MobileNetV3Small - Backbone]
        direction TB
        CONV[Initial Conv2D] --> IR1[Inverted Residual Blocks<br/>with Squeeze-Excitation]
        IR1 --> IR2[Hard-Swish Activation]
        IR2 --> LAST[Last Conv Layer]
    end

    BACKBONE --> GAP[GlobalAveragePooling2D]
    GAP --> D1[Dense 128 - Swish]
    D1 --> BN[BatchNormalization]
    BN --> D2[Dense 64 - Swish]
    D2 --> DROP[Dropout 0.3]
    DROP --> OUT[Dense 2 - Softmax<br/>Drunk / Not Drunk]
```

## Two-Phase Training Strategy

```mermaid
sequenceDiagram
    participant Data as Training Data
    participant Head as Classification Head
    participant Backbone as MobileNetV3 Backbone

    Note over Backbone: Phase 1: Backbone FROZEN
    Note over Head: LR = 1e-4

    Data->>Head: Train head layers
    Head->>Head: Dense(128) → BN → Dense(64) → Dropout → Softmax
    Note over Head: Until convergence (early stopping)

    Note over Backbone: Phase 2: Top layers UNFROZEN
    Note over Head: LR = 1e-5 (10x lower)

    Data->>Backbone: Fine-tune layers 100+
    Data->>Head: Continue training head
    Note over Backbone,Head: 20 additional epochs
```

## Decision Fusion Logic

```mermaid
flowchart TD
    FRAME[Camera Frame] --> FD{Face Detected?}
    FD -->|No| SAFE[Not Drunk]
    FD -->|Yes| PRED[Model Prediction]

    MQ3[MQ3 Sensor Reading] --> THR{Value > 400?}

    PRED --> FUSION{Fusion Decision}
    THR --> FUSION

    FUSION -->|Image=Drunk OR<br/>MQ3>Threshold| DRUNK[Status: DRUNK]
    FUSION -->|Image=NotDrunk AND<br/>MQ3≤Threshold| SAFE

    DRUNK --> CNT{Continuous<br/>Drunk Frames ≥ N?}
    CNT -->|No| ALARM[Activate Alarm<br/>Speaker + LED]
    CNT -->|Yes| FULL[Full Alert]
    FULL --> PHOTO[Save Evidence Photo]
    FULL --> TELEGRAM[Send Telegram Alert]
    FULL --> CSVLOG[Log to CSV]
    FULL --> ALARM
```

## Project Structure

```
Drunk_Detection_System/
├── configs/
│   └── default.yaml              # Centralized configuration
├── src/                           # Core ML package
│   ├── data/
│   │   ├── face_extraction.py    # MediaPipe face extraction
│   │   ├── preprocessing.py      # Image preprocessing
│   │   ├── dataset.py            # Data generators (augmentation fix)
│   │   └── augmentation.py       # Augmentation configs
│   ├── models/
│   │   ├── mobilenet_v3.py       # Model architecture
│   │   └── export.py             # TFLite/ONNX export
│   ├── training/
│   │   ├── trainer.py            # Two-phase training orchestrator
│   │   └── callbacks.py          # Training callbacks
│   ├── evaluation/
│   │   └── evaluator.py          # Comprehensive eval + Grad-CAM
│   └── utils/
│       ├── logger.py             # Structured logging
│       └── config.py             # YAML + env config loader
├── scripts/                       # CLI entry points
│   ├── train.py                  # Training pipeline
│   ├── evaluate.py               # Evaluation pipeline
│   └── export_tflite.py          # Model export
├── deployment/
│   ├── raspi/                    # Raspberry Pi deployment
│   │   ├── main.py               # Main loop + health check + benchmark
│   │   ├── config.py             # Env-based config (no hardcoded secrets)
│   │   └── modules/
│   │       ├── camera.py         # Camera with fallback
│   │       ├── image_processing.py # Edge inference engine
│   │       ├── mq3_sensor.py     # MQ3 sensor interface
│   │       ├── telegram_bot.py   # Async notifications
│   │       └── logger.py         # CSV warning logger
│   └── dashboard/
│       ├── app.py                # Flask monitoring dashboard
│       └── templates/
│           └── index.html        # Dashboard UI
├── docs/
│   ├── ARCHITECTURE.md           # This file
│   └── MODEL_CARD.md             # Model documentation
├── .env.example                   # Environment variable template
├── .gitignore                     # Git ignore rules
├── requirements.txt               # Python dependencies
├── setup.py                       # Package installation
└── README.md                      # Project overview
```

## Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **ML Framework** | TensorFlow/Keras | Model training & evaluation |
| **Backbone** | MobileNetV3Small | Feature extraction (transfer learning) |
| **Face Detection** | MediaPipe FaceMesh | Face region extraction |
| **Edge Runtime** | TFLite | Optimized inference on Raspberry Pi |
| **Experiment Tracking** | MLflow | Hyperparameter & metric logging |
| **Sensor Interface** | PySerial | MQ3 sensor via Arduino |
| **Notifications** | python-telegram-bot | Real-time alerts |
| **Dashboard** | Flask + Bootstrap 5 | Monitoring UI |
| **Config** | YAML + python-dotenv | Centralized configuration |
| **Logging** | Python logging | Structured log output |

## Key Design Decisions

### 1. Why MobileNetV3 over other architectures?
MobileNetV3Small was chosen for its optimal balance of accuracy and efficiency on edge devices. With ~2.5M parameters (vs ResNet50's 25M), it runs at 10-15 FPS on Raspberry Pi while maintaining competitive accuracy.

### 2. Why two-phase training?
Phase 1 (frozen backbone) prevents catastrophic forgetting of ImageNet features while training the classification head. Phase 2 (fine-tuning) adapts upper backbone features to the drunk detection domain with a lower learning rate.

### 3. Why sensor fusion?
Camera-only detection has limitations (lighting, occlusion). The MQ3 alcohol sensor provides a complementary signal. The OR-based fusion strategy prioritizes safety (detecting all drunk cases) over specificity.

### 4. Why edge deployment?
- **Privacy**: Face images stay on-device
- **Latency**: No network round-trip for real-time response
- **Reliability**: Works without internet connectivity
- **Cost**: No cloud inference costs
