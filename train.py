from dataset import create_dataloaders
from utils import hist_conversations_length

# 파일 경로 설정
train_file_path = "./Data/aiffel-dl-thon-dktc-online-15/train.csv"
test_file_path = "./Data/aiffel-dl-thon-dktc-online-15/test.csv"

# max_length 확인하기
hist_conversations_length(train_file_path, test_file_path)

# 데이터로더와 vocab 준비 (작은 배치 사이즈로 테스트)
train_loader, test_loader, vocab = create_dataloaders(train_file_path, test_file_path, batch_size=4)

# 샘플 배치 확인
print("\n--- Train DataLoader 샘플 배치 확인 ---")
try:
    sample_batch = next(iter(train_loader))
    print("Input IDs Shape:", sample_batch['input_ids'].shape)
    print("Target IDs Shape:", sample_batch['target_ids'].shape)
    print("Labels Shape:", sample_batch['labels'].shape)
    print("\nSample 1 Input (Token IDs):", sample_batch['input_ids'][0])
    print("Sample 1 Decoded:", vocab.decode(sample_batch['input_ids'][0].tolist()))
    print("\nSample 1 Target (Token IDs):", sample_batch['target_ids'][0])
    print("Sample 1 Decoded:", vocab.decode(sample_batch['target_ids'][0].tolist()))
    print("Sample 1 Label:", sample_batch['labels'][0])
    print("=" * 25)
except StopIteration:
    print("Train 데이터로더가 비어있습니다. 데이터셋 크기를 확인해주세요.")

print("\n--- Test DataLoader 샘플 배치 확인 ---")
try:
    sample_batch = next(iter(test_loader))
    print("Input IDs Shape:", sample_batch['input_ids'].shape)
    print("Target IDs Shape:", sample_batch['target_ids'].shape)
    # 테스트 데이터로더는 'labels' 키가 없어야 합니다.
    print("'labels' in test batch:", 'labels' in sample_batch)
    print("\nSample 1 Input (Token IDs):", sample_batch['input_ids'][0])
    print("Sample 1 Decoded:", vocab.decode(sample_batch['input_ids'][0].tolist()))
    print("=" * 25)
except StopIteration:
    print("Test 데이터로더가 비어있습니다. 데이터셋 크기를 확인해주세요.")
