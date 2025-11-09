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


def augment_csv_to_file(
    input_path: str,
    output_path: str,
    augment_ratio: int = 2,
    dropout_rate: float = 0.15,
    exclude_labels: Optional[Union[str, List[str]]] = None,
    min_tokens_after: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """
    - input CSV: 'idx','class','conversation' (conversation may contain internal '\n')
    - output CSV: 원본 + 증강본, conversation의 기존 줄바꿈 보존
    - 각 줄(line) 별로 word-dropout 적용 (줄바꿈 단위 보존)
    - 최종적으로 섞고 idx를 0..N-1 순차 재부여
    """

    # 초기화: 기존 output 파일 비우기(있으면 덮어씀)
    if os.path.exists(output_path):
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("")

    # 토크/유틸 정의 (간단한 공백/문장부호 기반)
    _PUNCT = r'([.,!?;:()\[\]{}\“\”\‘\’\"\'~…])'
    _WS = re.compile(r'\s+')

    def simple_tokenize(text: str) -> List[str]:
        text = re.sub(_PUNCT, r' \1 ', str(text))
        return [t for t in _WS.split(text) if t]

    def simple_detokenize(tokens: List[str]) -> str:
        s = " ".join(tokens)
        s = re.sub(r'\s+([.,!?;:)\]\}])', r'\1', s)
        s = re.sub(r'([(\[\{])\s+', r'\1', s)
        return re.sub(r'\s+', ' ', s).strip()

    def is_punct(tok: str) -> bool:
        return re.fullmatch(r'[.,!?;:()\[\]{}\“\”\‘\’\"\'~…]', tok) is not None

    # 핵심: 한 줄(line) 단위로 dropout 적용하고 다시 join해서 반환
    def word_dropout_preserve_lines(conv: str) -> str:
        """
        conv: 멀티라인 문자열(기존 '\n' 포함 가능)
        반환: 각 라인에 dropout 적용 후 '\n'으로 재조립 (빈 라인은 그대로 유지)
        """
        if conv is None:
            return conv
        lines = conv.splitlines()  # 기존 줄바꿈 보존
        out_lines = []
        for line in lines:
            line = line.strip()
            if line == "":
                out_lines.append(line)
                continue
            toks = simple_tokenize(line)
            # 짧은 라인은 변화시키지 않음(안정성)
            if len(toks) <= min_tokens_after:
                out_lines.append(line)
                continue
            kept = []
            for t in toks:
                if is_punct(t) or random.random() > dropout_rate:
                    kept.append(t)
            # 내용 토큰이 너무 적어지면 원본 라인 유지
            if sum(1 for x in kept if not is_punct(x)) < min_tokens_after:
                out_lines.append(line)
            else:
                out_lines.append(simple_detokenize(kept))
        # join with newline to preserve multiline cell
        return "\n".join(out_lines)

    # 준비 및 로드
    random.seed(seed)
    if isinstance(exclude_labels, str):
        exclude_labels = [x.strip() for x in exclude_labels.split(',') if x.strip()]
    exclude_set = set(exclude_labels or [])

    df = pd.read_csv(input_path, dtype={'idx': object})  # idx may be overwritten later
    required = {'idx', 'class', 'conversation'}
    if not required.issubset(df.columns):
        raise ValueError(f"입력 CSV는 {required} 컬럼을 포함해야 합니다. 현재: {list(df.columns)}")

    # 원본은 class,conversation만 보관 (idx는 나중에 재부여)
    all_parts = [df[['class', 'conversation']].copy()]

    # 클래스별 증강 — 증강본에는 idx 필드 만들지 않음
    aug_parts = []
    for cls, group in df.groupby('class', sort=False):
        if cls in exclude_set:
            continue
        n = len(group)
        if n == 0 or augment_ratio <= 0:
            continue
        need = n * augment_ratio
        sampled = group.sample(n=need, replace=True, random_state=seed)

        rows = []
        for _, row in sampled.iterrows():
            orig_conv = str(row['conversation'])
            aug_conv = word_dropout_preserve_lines(orig_conv)
            rows.append({'class': row['class'], 'conversation': aug_conv})
        if rows:
            aug_parts.append(pd.DataFrame(rows))

    if aug_parts:
        all_parts.append(pd.concat(aug_parts, ignore_index=True))

    out_df = pd.concat(all_parts, ignore_index=True)

    # 셔플 후 idx 재부여 (0..N-1)
    out_df = out_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    # 기존에 'idx' 컬럼이 있으면 제거한 뒤 새로 생성
    if 'idx' in out_df.columns:
        out_df = out_df.drop(columns=['idx'])
    out_df.insert(0, 'idx', range(len(out_df)))

    # 저장: pandas는 멀티라인 셀을 자동으로 큰따옴표로 감쌈
    out_df.to_csv(output_path, index=False, encoding='utf-8-sig')

    return out_df