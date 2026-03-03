# Face Parsing and Semantic Segmentation on CelebA-HQ

이 프로젝트는 사전학습된 **DeepLabV3-ResNet50** 모델을 활용하여 **CelebAMask-HQ** 데이터셋에 대해 얼굴 영역 세그멘테이션(Face Parsing)을 수행한 실습 기록입니다. 
하이퍼파라미터 튜닝과 손실 함수 개선(Weighted Loss)을 통해 클래스 불균형 문제를 해결하고 정량적 성능을 분석하는 과정을 포함합니다.

## 🚀 Key Features
* **Transfer Learning:** MS COCO로 사전학습된 DeepLabV3 모델을 19개의 얼굴 파트 클래스에 맞춰 재구성 및 Fine-tuning 하였습니다. 
* **Optimization:** 학습 안정성을 위해 Learning Rate 스케줄링, Weight Decay 강화, Backbone Freeze 전략을 적용했습니다. 
* **Imbalance Handling:** 데이터가 극히 적은 희소 클래스(`neck_l`, `ear_r`)의 탐지율을 높이기 위해 Inverse Frequency 기반의 **Weighted CrossEntropyLoss**를 실험했습니다.
* **Performance Analysis:** mIoU, Dice Coefficient, Pixel Accuracy 등 다양한 지표를 통한 다각도 성능 분석을 수행했습니다.

---

## 🏗️ Model Architecture
* **Model:** DeepLabV3 
* **Backbone:** ResNet-50 
* **Pretrained Weights:** `DeepLabV3_ResNet50_Weights.DEFAULT` (COCO)
* **Special Tech:** Atrous Spatial Pyramid Pooling (ASPP)를 통한 멀티 스케일 특징 추출

---

## 📊 Experiments & Results

총 3가지 핵심 실험을 통해 최적의 설정을 도출했습니다.

| Experiment | Epochs | LR | Loss Function | mIoU | mPA |
| :--- | :---: | :---: | :--- | :---: | :---: |
| **Exp 1: Baseline** | 20 | $3e-4$ | CrossEntropy | 0.7312 | 0.8221 |
| **Exp 2: Tuned** | 40 | $1e-4$ | CrossEntropy | **0.7324** | 0.8292 |
| **Exp 3: Weighted** | 40 | $1e-4$ | Weighted CE | 0.6883 | **0.8902** |

> *실험 결과 지표는 Best Checkpoint 기준입니다.*

### Analysis
* **HyperParam Tuning:** LR 감소와 Weight Decay 증가를 통해 학습 안정성을 확보하여 전반적인 지표를 향상시켰습니다.
* **Weighted Loss:** 희소 클래스(`neck_l`)의 재현율(PA)을 $17.76\%$에서 $46.75\%$까지 대폭 끌어올렸으나, False Positive 증가로 인해 전체 mIoU는 하락하는 Trade-off가 관찰되었습니다.

---

## 📂 Project Structure
```text
.
├── CelebAMask-HQ                  # 데이터셋
├── analysis.ipynb                 # 결과 분석 및 시각화 코드
├── baseline_+_hyperparam_tuned.ipynb # 실험 1, 2 학습 코드
├── weightedLoss.ipynb             # 실험 3 (Weighted Loss) 학습 코드
├── runs_celebamaskhq/             # 실험 1 결과 (로그, 리포트, 시각화)
├── runs_celebamaskhq_hyperparam_tuning/ # 실험 2 결과
├── runs_celebamaskhq_hyperparam_tuning_V2/ # 실험 2-V2 결과
└── .gitignore                     # .pt 등 대용량 파일 제외 설정
```

---

## 🛠️ Dataset Setup

본 리포지토리에는 용량 제한으로 인해 실제 데이터셋이 포함되어 있지 않습니다. 원활한 코드 실행을 위해 아래 가이드에 따라 데이터를 준비해 주세요.

### 1. 데이터 다운로드
* CelebAMask-HQ 공식 저장소에서 데이터를 다운로드합니다.
### 2. 데이터 전처리 (Preprocessing)
* 각 클래스별 바이너리 마스크를 단일 정수 라벨맵(512 x 512)으로 병합하는 전처리 과정을 거쳐야 합니다.
### 3. 디렉토리 구조
* 데이터는 CelebAMask-HQ/ 경로에 위치시켜야 정상적으로 작동합니다.

---

## 💻 Environment
* **Framework: PyTorch 2.5.1 / torchvision 0.20.1**
* **Hardware: NVIDIA RTX A6000 **
* **OS: Ubuntu (via Python 3.10+)**
