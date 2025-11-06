import pandas as pd
import re

def preprocess_sentence(sentence):
    """
    간단한 전처리 함수
    """
    if pd.isna(sentence) or sentence is None:
        return ""
    
    sentence = str(sentence)
    sentence = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣a-zA-Z0-9\s.,!?~ㅠㅜ]', ' ', sentence)
    sentence = re.sub(r'\s+', ' ', sentence)
    sentence = sentence.strip()
    sentence = re.sub(r'([!?.])\1+', r'\1', sentence)

    return sentence


def load_and_preprocess_data(train_path, test_path):
    """
    .csv 파일에서 데이터 로드하고 간단한 전처리를 진행하는 함수
    """
    print("=" * 50)
    print("데이터 로드 및 전처리 중...")
    print("=" * 50)

    # pandas로 CSV 파일 읽기
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    print(f"Train 데이터: {len(train_df)} 개의 conversation")
    print(f"Test 데이터: {len(test_df)} 개의 conversation")

    # 레이블 매핑
    class_to_idx = {
        '협박 대화': 0,
        '갈취 대화': 1,
        '직장 내 괴롭힘 대화': 2,
        '기타 괴롭힘 대화': 3,
        '일반 대화': 4
    }

    # Train 데이터 전처리
    train_conversations = []
    train_labels = []

    for i, row in train_df.iterrows():
        conv = preprocess_sentence(row['conversation'])
        label = class_to_idx[row['class']]

        if conv:
            train_conversations.append(conv)
            train_labels.append(label)

    # Test 데이터 전처리
    test_conversations = []
    test_ids = []

    for i, row in test_df.iterrows():
        conv = preprocess_sentence(row['conversation'])
        test_id = row['idx']

        if conv:
            test_conversations.append(conv)
            test_ids.append(test_id)

    # 샘플 데이터 출력
    print("\n샘플 데이터:")
    for i in range(min(3, len(train_conversations))):
        print(f"Conversation: {train_conversations[i]}")
        print(f"Label: {train_labels[i]}\n")

    return train_conversations, train_labels, test_conversations, test_ids, class_to_idx