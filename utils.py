import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # Add seaborn import
import tempfile
import shutil

import re
import random
from typing import List, Optional, Union
import pandas as pd
from tqdm import tqdm

# matplotlib 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic' # Windows
plt.rcParams['axes.unicode_minus'] = False # 마이너스 기호 깨짐 방지

from preprocessing import load_and_preprocess_data
from tokenization import train_sentencepiece_model, SentencePieceVocab


def hist_conversations_length(train_path, test_path, vocab_size=1320):
    """
    학습 데이터에 대해 각 레이블별 토큰화 후 conversations의 length 분포를
    시각화하고, 시각화 이미지를 Images 디렉토리에 저장

    Args:
        train_path: train data csv path
        test_path: test data csv path
        vocab_size
    """
    print("데이터 로드 및 전처리 중...")
    train_conversations, train_labels, _, _, class_to_idx = \
        load_and_preprocess_data(train_path, test_path)

    # 임시 디렉토리 생성
    temp_dir = tempfile.mkdtemp()
    model_prefix = os.path.join(temp_dir, 'temp_spm_for_hist')
    
    print("SentencePiece 모델 학습 중 (학습 데이터 기준)...")
    try:
        # 학습 데이터 기준으로 SentencePiece 모델 학습
        sp_model_path = train_sentencepiece_model(
            train_conversations, model_prefix=model_prefix, vocab_size=vocab_size
        )
        vocab = SentencePieceVocab(sp_model_path)

        print("토큰화 및 길이 계산 중...")
        # 학습 데이터의 길이 계산 ([CLS], [EOS] 포함)
        # [CLS], [EOS] 토큰은 encode함수에서 생성하지 않고 Dataset 클래스에서 추가하므로 +2 해야함
        all_train_lengths = [len(vocab.encode(conv)) + 2 for conv in train_conversations]

        # 레이블별 길이 분포 시각화
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        unique_labels = sorted(list(set(train_labels)))
        num_labels = len(unique_labels)

        # 서브플롯 그리드 크기 조정 (예: 2x3 또는 3x2)
        nrows = (num_labels + 1) // 2 if num_labels > 1 else 1
        ncols = 2 if num_labels > 0 else 1
        if num_labels == 0: # Handle case with no labels
            print("No labels found in training data to plot.")
            return

        fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5 * nrows))
        fig.suptitle('Distribution of Conversation Lengths by Label (Train Data)', fontsize=16)
        
        # axes가 1차원 배열일 경우를 대비하여 평탄화
        if nrows * ncols == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for i, label_idx in enumerate(unique_labels):
            ax = axes[i]
            label_name = idx_to_class[label_idx]
            
            # 해당 레이블에 속하는 대화 길이 필터링
            label_lengths = [all_train_lengths[j] for j, lbl in enumerate(train_labels) if lbl == label_idx]

            if not label_lengths:
                ax.set_title(f'{label_name} (No data)')
                ax.set_xlabel('Length of Conversation (tokens)')
                ax.set_ylabel('Frequency')
                continue

            sns.histplot(label_lengths, bins=50, kde=True, ax=ax, color=sns.color_palette("tab10")[i % 10])
            
            mean_len = np.mean(label_lengths)
            median_len = np.median(label_lengths)
            percentile_95 = np.percentile(label_lengths, 95)

            ax.axvline(mean_len, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_len:.2f}')
            ax.axvline(median_len, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median_len:.2f}')
            ax.axvline(percentile_95, color='purple', linestyle='dotted', linewidth=2, label=f'95th: {percentile_95:.2f}')

            ax.set_title(f'{label_name} (count: {len(label_lengths)})')
            ax.set_xlabel('Length of Conversation (tokens)')
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.grid(True)
        
        # 사용하지 않는 서브플롯 숨기기
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout() # 전체 타이틀을 위한 여백 확보
        plt.show()

        # --- 그래프 이미지 저장 --- #
        images_dir = './Images'
        if not os.path.exists(images_dir):
            os.makedirs(images_dir)
        
        save_path = os.path.join(images_dir, 'conversation_length_distributions_by_label.png')
        fig.savefig(save_path)
        print(f"그래프가 {save_path} 에 저장되었습니다.")

    finally:
        # 임시 디렉토리 및 파일 삭제
        print(f"임시 디렉토리({temp_dir}) 및 파일을 삭제합니다.")
        shutil.rmtree(temp_dir)


class TextAugmenter:
    """텍스트 데이터 증강 클래스"""
    
    def __init__(self, dropout_rate=0.15, exclude_labels=None):
        """
        Args:
            dropout_rate: 단어 삭제 비율
            exclude_labels: 증강하지 않을 라벨 리스트 (예: [4] for 일반대화)
        """
        self.dropout_rate = dropout_rate
        self.exclude_labels = set(exclude_labels) if exclude_labels else set()
    
    def apply_word_dropout(self, text):
        """랜덤 단어 삭제"""
        if pd.isna(text) or not isinstance(text, str):
            return text
        
        words = text.split()
        if len(words) <= 2:  # 너무 짧은 텍스트는 증강하지 않음
            return text
        
        new_words = []
        for word in words:
            if random.random() > self.dropout_rate:
                new_words.append(word)
        
        # 최소 1개 단어는 유지
        return ' '.join(new_words) if new_words else words[0]
    
    def augment_row(self, row, text_columns):
        """데이터 행 증강"""
        augmented_row = row.copy()
        for col in text_columns:
            if col in augmented_row:
                augmented_row[col] = self.apply_word_dropout(augmented_row[col])
        return augmented_row


def augment_csv(
    input_csv_path,
    output_csv_path,
    text_columns,
    label_column='label',
    augment_ratio=2,
    dropout_rate=0.15,
    exclude_labels=None
):
    """
    CSV 파일 데이터 증강
    
    Args:
        input_csv_path: 입력 CSV 파일 경로
        output_csv_path: 출력 CSV 파일 경로
        text_columns: 증강할 텍스트 컬럼 리스트 (예: ['input_text', 'target_text'])
        label_column: 라벨 컬럼명 (기본값: 'label')
        augment_ratio: 증강 배수 (2 = 원본 + 2배 증강 = 3배 데이터)
        dropout_rate: 단어 삭제 비율
        exclude_labels: 증강하지 않을 라벨 리스트
    
    Returns:
        증강된 데이터프레임
    """
    print(f"📂 Reading CSV: {input_csv_path}")
    df = pd.read_csv(input_csv_path)
    
    print(f"📊 Original data size: {len(df)}")
    print(f"📋 Columns: {df.columns.tolist()}")
    
    # 라벨별 통계
    if label_column in df.columns:
        print(f"\n📈 Label distribution:")
        print(df[label_column].value_counts().sort_index())
    
    # Augmenter 생성
    augmenter = TextAugmenter(dropout_rate=dropout_rate, exclude_labels=exclude_labels)
    exclude_labels_set = set(exclude_labels) if exclude_labels else set()
    
    # 증강된 데이터 저장 리스트
    augmented_data = []
    
    # 원본 데이터 추가
    augmented_data.append(df)
    
    # 라벨별로 증강
    if label_column in df.columns:
        for label in df[label_column].unique():
            # 제외 라벨은 증강하지 않음
            if label in exclude_labels_set:
                print(f"\n⏭️  Skipping label {label} (excluded)")
                continue
            
            label_df = df[df[label_column] == label]
            print(f"\n🔄 Augmenting label {label}: {len(label_df)} samples × {augment_ratio}")
            
            # augment_ratio만큼 증강
            for i in range(augment_ratio):
                augmented_rows = []
                for _, row in tqdm(label_df.iterrows(), 
                                  total=len(label_df), 
                                  desc=f"  Round {i+1}/{augment_ratio}"):
                    augmented_row = augmenter.augment_row(row, text_columns)
                    augmented_rows.append(augmented_row)
                
                augmented_data.append(pd.DataFrame(augmented_rows))
    else:
        # 라벨 컬럼이 없는 경우 전체 데이터 증강
        print(f"\n🔄 Augmenting all data × {augment_ratio}")
        for i in range(augment_ratio):
            augmented_rows = []
            for _, row in tqdm(df.iterrows(), 
                              total=len(df), 
                              desc=f"  Round {i+1}/{augment_ratio}"):
                augmented_row = augmenter.augment_row(row, text_columns)
                augmented_rows.append(augmented_row)
            
            augmented_data.append(pd.DataFrame(augmented_rows))
    
    # 모든 데이터 합치기
    final_df = pd.concat(augmented_data, ignore_index=True)
    
    # 셔플
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\n✅ Final augmented data size: {len(final_df)}")
    
    if label_column in final_df.columns:
        print(f"\n📈 Final label distribution:")
        print(final_df[label_column].value_counts().sort_index())
    
    # CSV 저장
    print(f"\n💾 Saving to: {output_csv_path}")
    final_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    
    print("✅ Done!")
    return final_df