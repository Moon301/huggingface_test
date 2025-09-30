# Gemma-2B FSDP 파인튜닝 가이드

RTX A4000 8장 GPU 환경에서 `google/gemma-2b-it` 모델을 FSDP(Fully Sharded Data Parallel) 기술을 사용하여 파인튜닝하는 코드입니다.

## 환경 요구사항

- **GPU**: RTX A4000 8장
- **라이브러리 버전**:
  - `transformers==4.56.2`
  - `accelerate==1.10.1`
  - `peft==0.17.1`
  - `trl==0.23.0`
  - `torch>=2.0.0`
  - `datasets`
  - `bitsandbytes`


## 주요 특징

### 1. FSDP 최적화
- **FULL_SHARD**: 모델 파라미터를 모든 GPU에 분산
- **TRANSFORMER_BASED_WRAP**: 트랜스포머 레이어별로 자동 래핑

### 2. LoRA 설정
- **Rank**: 8
- **Alpha**: 16
- **Dropout**: 0.05
- **Target Modules**: 모든 attention과 MLP 레이어


## 실행 방법

### 1. 환경 설정
```bash
# 필요한 라이브러리 설치
pip install transformers==4.56.2 accelerate==1.10.1 peft==0.17.1 trl==0.23.0
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install datasets bitsandbytes
```

### 2. Accelerate 설정 확인
```bash
cd finetuning
cd gemma_test
cd use_accelerate
accelerate config --config_file accelerate_config.yaml
```

### 3. 파인튜닝 실행
```bash
cd finetuning
cd gemma_test
cd use_accelerate
accelerate launch --config_file accelerate_config.yaml fsdp_train.py
```


## 결과 확인
학습 완료 후:
- **체크포인트**: `./outputs/` 디렉토리
