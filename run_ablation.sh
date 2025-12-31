#!/bin/bash
# Ablation Study: 각 분석 제거별 Demonstrations 생성 및 예측 실행 스크립트
# Usage: bash run_ablation.sh

cd "$(dirname "$0")"

IR=10
DATASET="kdd"
BATCH_SIZE=500

echo "=========================================="
echo "Ablation Study: Demonstrations 생성 및 예측 실행"
echo "IR: $IR, Dataset: $DATASET"
echo "=========================================="
echo ""

# 함수: Mode 1 실행 (예측)
run_mode1() {
    local SETTING=$1
    local API_KEY_NAME=$2
    local ABLATION_ARGS=$3
    
    echo "[$SETTING] Mode 1: 예측 실행 시작 (API Key: $API_KEY_NAME, $ABLATION_ARGS)"
    python3 run_prediction.py \
        --mode 1 \
        --setting $SETTING \
        --ir $IR \
        --dataset $DATASET \
        --batch_size $BATCH_SIZE \
        --api_key_name $API_KEY_NAME \
        $ABLATION_ARGS
    
    if [ $? -eq 0 ]; then
        echo "[$SETTING] Mode 1 완료"
        return 0
    else
        echo "[$SETTING] Mode 1 실패"
        return 1
    fi
}

# Ablation study 실행 함수
run_ablation() {
    local ABLATION_NAME=$1
    local ABLATION_ARGS=$2
    
    echo ""
    echo "=========================================="
    echo "Ablation: $ABLATION_NAME"
    echo "=========================================="
    echo ""
    
    # Step 1: Demonstrations 생성 (한 번만 실행, transductive 사용)
    echo "Step 1: Demonstrations 생성 ($ABLATION_NAME)"
    echo "Transductive 설정으로 생성 (API Key: 1)"
    echo ""
    python3 run_prediction.py \
        --mode 0 \
        --setting transductive \
        --ir $IR \
        --dataset $DATASET \
        --batch_size $BATCH_SIZE \
        --api_key_name 1 \
        $ABLATION_ARGS
    
    if [ $? -ne 0 ]; then
        echo "Demonstrations 생성 실패!"
        return 1
    fi
    
    echo ""
    echo "Demonstrations 생성 완료"
    echo ""
    
    # Step 2: Transductive와 Inductive의 Mode 1을 병렬로 실행
    echo "Step 2: Transductive와 Inductive 예측을 병렬로 실행합니다..."
    echo "Transductive: OPENAI_API_KEY_1 사용"
    echo "Inductive: OPENAI_API_KEY_2 사용"
    echo ""
    
    # 백그라운드로 실행
    run_mode1 "transductive" "1" "$ABLATION_ARGS" &
    PID_TRANSDUCTIVE=$!
    
    run_mode1 "inductive" "2" "$ABLATION_ARGS" &
    PID_INDUCTIVE=$!
    
    # 두 프로세스가 모두 완료될 때까지 대기
    wait $PID_TRANSDUCTIVE
    RESULT_TRANSDUCTIVE=$?
    
    wait $PID_INDUCTIVE
    RESULT_INDUCTIVE=$?
    
    echo ""
    echo "[$ABLATION_NAME] 실행 완료"
    echo "Transductive 결과: $([ $RESULT_TRANSDUCTIVE -eq 0 ] && echo '성공' || echo '실패')"
    echo "Inductive 결과: $([ $RESULT_INDUCTIVE -eq 0 ] && echo '성공' || echo '실패')"
    echo ""
    
    if [ $RESULT_TRANSDUCTIVE -eq 0 ] && [ $RESULT_INDUCTIVE -eq 0 ]; then
        return 0
    else
        return 1
    fi
}

# 각 Ablation Study 실행
TOTAL_SUCCESS=0
TOTAL_FAILED=0

# 1. No Structure
run_ablation "No Structure" "--no_structure"
if [ $? -eq 0 ]; then
    TOTAL_SUCCESS=$((TOTAL_SUCCESS + 1))
else
    TOTAL_FAILED=$((TOTAL_FAILED + 1))
fi

# 2. No Semantic
run_ablation "No Semantic" "--no_semantic"
if [ $? -eq 0 ]; then
    TOTAL_SUCCESS=$((TOTAL_SUCCESS + 1))
else
    TOTAL_FAILED=$((TOTAL_FAILED + 1))
fi

# 3. No Trend
run_ablation "No Trend" "--no_trend"
if [ $? -eq 0 ]; then
    TOTAL_SUCCESS=$((TOTAL_SUCCESS + 1))
else
    TOTAL_FAILED=$((TOTAL_FAILED + 1))
fi

echo ""
echo "=========================================="
echo "모든 Ablation Study 완료"
echo "=========================================="
echo "성공: $TOTAL_SUCCESS"
echo "실패: $TOTAL_FAILED"
echo ""

if [ $TOTAL_FAILED -eq 0 ]; then
    echo "모든 Ablation Study가 성공적으로 완료되었습니다!"
    exit 0
else
    echo "일부 Ablation Study가 실패했습니다."
    exit 1
fi

