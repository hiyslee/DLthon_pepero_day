'''
데이터 분석 및 시각화를 위한 유틸리티 함수 모음
'''

import os
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import shutil

from preprocessing import load_and_preprocess_data
from tokenization import train_sentencepiece_model, SentencePieceVocab


def hist_conversations_length(train_path, test_path, vocab_size=1200):
    """
    학습 데이터와 테스트 데이터의 토큰화 후 conversations의 length 분포를
    시각화하고, 시각화 이미지를 Images 디렉토리에 저장

    Args:
        train_path: train data csv path
        test_path: test data csv path
        vocab_size
    """
    print("데이터 로드 및 전처리 중...")
    train_conversations, _, test_conversations, _, _ = \
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
        # 학습 데이터와 테스트 데이터의 길이 계산 ([CLS], [EOS] 포함)
        # [CLS], [EOS] 토큰은 encode함수에서 생성하지 않고 Dataset 클래스에서 추가하므로 +2 해야함
        train_lengths = [len(vocab.encode(conv)) + 2 for conv in train_conversations]
        test_lengths = [len(vocab.encode(conv)) + 2 for conv in test_conversations]

        fig, axes = plt.subplots(1, 2, figsize=(20, 7))
        fig.suptitle('Distribution of Conversation Lengths (Train vs. Test)', fontsize=16)

        # train data histogram
        axes[0].hist(train_lengths, bins=50, alpha=0.7, color='blue', label=f'Train Data (count: {len(train_lengths)})')
        
        mean_len_train = np.mean(train_lengths)
        median_len_train = np.median(train_lengths)
        percentile_95_train = np.percentile(train_lengths, 95)
        percentile_99_train = np.percentile(train_lengths, 99)

        axes[0].axvline(mean_len_train, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_len_train:.2f}')
        axes[0].axvline(median_len_train, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median_len_train:.2f}')
        axes[0].axvline(percentile_95_train, color='purple', linestyle='dotted', linewidth=2, label=f'95th: {percentile_95_train:.2f}')
        axes[0].axvline(percentile_99_train, color='black', linestyle='dotted', linewidth=2, label=f'99th: {percentile_99_train:.2f}')

        axes[0].set_title('Train Data')
        axes[0].set_xlabel('Length of Conversation (tokens)')
        axes[0].set_ylabel('Frequency')
        axes[0].legend()
        axes[0].grid(True)

        # test data histogram
        axes[1].hist(test_lengths, bins=50, alpha=0.7, color='orange', label=f'Test Data (count: {len(test_lengths)})')

        mean_len_test = np.mean(test_lengths)
        median_len_test = np.median(test_lengths)
        percentile_95_test = np.percentile(test_lengths, 95)
        percentile_99_test = np.percentile(test_lengths, 99)

        axes[1].axvline(mean_len_test, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_len_test:.2f}')
        axes[1].axvline(median_len_test, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median_len_test:.2f}')
        axes[1].axvline(percentile_95_test, color='purple', linestyle='dotted', linewidth=2, label=f'95th: {percentile_95_test:.2f}')
        axes[1].axvline(percentile_99_test, color='black', linestyle='dotted', linewidth=2, label=f'99th: {percentile_99_test:.2f}')

        axes[1].set_title('Test Data')
        axes[1].set_xlabel('Length of Conversation (tokens)')
        axes[1].set_ylabel('Frequency')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # 전체 타이틀을 위한 여백 확보
        plt.show()

        # --- 그래프 이미지 저장 --- #
        images_dir = './Images'
        if not os.path.exists(images_dir):
            os.makedirs(images_dir)
        
        save_path = os.path.join(images_dir, 'conversation_length_distributions.png')
        fig.savefig(save_path)
        print(f"그래프가 {save_path} 에 저장되었습니다.")

    finally:
        # 임시 디렉토리 및 파일 삭제
        print(f"임시 디렉토리({temp_dir}) 및 파일을 삭제합니다.")
        shutil.rmtree(temp_dir)