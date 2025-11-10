"""
일반 대화 생성기 (Gemini API)

Google Gemini API를 사용하여 일상 대화 데이터 생성.
"""

import pandas as pd
import re
import os 
import random
import time


def generate_normal_conversations(count=1000, use_cache=True, output_csv='normal_conversations.csv', start_idx=100000):
    """
    Google Gemini API로 일반 대화 생성 후 train.csv 형식으로 저장
    
    사전 준비:
        1. Google AI Studio에서 API 키 발급: https://makersuite.google.com/app/apikey
        2. 설치: pip install google-generativeai --break-system-packages
        3. 환경변수 설정:
           - Linux/Mac: export GEMINI_API_KEY="your-key"
           - Windows: set GEMINI_API_KEY=your-key
           - 또는 코드에서 직접 설정 (아래 참고)
    
    무료 할당량:
        - 분당 15회 호출
        - 일일 1,500회 호출
    
    
    Args:
        count (int): 생성할 대화 개수 (기본값 1000)
        use_cache (bool): 캐시 사용 (한 번 생성 후 재사용)
        output_csv (str): 출력 CSV 파일명 (train.csv 형식)
        start_idx (int): 시작 인덱스 (train.csv와 겹치지 않도록, 기본값 100000)
    
    Returns:
        pd.DataFrame: 생성된 대화 데이터프레임 (train.csv 형식)
    """
    # ========================================
    # 캐시 확인 (이미 생성한 데이터 재사용)
    # ========================================
    cache_file = f'normal_conversations_cache_{count}.csv'
    if use_cache and os.path.exists(cache_file):
        print(f"✓ 캐시에서 일반 대화 로드: {cache_file}")
        df = pd.read_csv(cache_file)
        print(f"  → {len(df)}개 대화 로드 완료")
        
        # output_csv가 다르면 복사
        if output_csv != cache_file:
            df.to_csv(output_csv, index=False, encoding='utf-8-sig')
            print(f"✓ 파일 저장: {output_csv}")
        
        return df
    
    # ========================================
    # Gemini API 설정
    # ========================================
    try:
        import google.generativeai as genai
    except ImportError:
        print("google-generativeai 패키지가 설치되지 않았습니다.")
        print("설치: pip install google-generativeai --break-system-packages")
        return _create_fallback_csv(output_csv, count, start_idx)
    
    # API 키 확인
    api_key = os.getenv("GEMINI_API_KEY")
    
    # 환경변수에 없으면 여기서 직접 설정 가능 (보안 주의!)
    if not api_key:
        # api_key = "your-gemini-api-key-here"  # 직접 입력 (비추천)
        print("⚠️ GEMINI_API_KEY 환경변수가 설정되지 않았습니다.")
        print("   설정 방법:")
        print("   - Linux/Mac: export GEMINI_API_KEY='your-key'")
        print("   - Windows: set GEMINI_API_KEY=your-key")
        print("   - API 키 발급: https://makersuite.google.com/app/apikey")
        print("   → 기본 템플릿 사용")
        return _create_fallback_csv(output_csv, count, start_idx)
    
    # Gemini 설정 (API 키 설정 후에 모델 생성)
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        print(f"✓ Gemini API 연결 성공")
    except Exception as e:
        print(f"Gemini API 초기화 실패: {e}")
        print("   → 기본 템플릿 사용")
        return _create_fallback_csv(output_csv, count, start_idx)
    
    # 대화 생성
    print(f"Google Gemini로 일반 대화 {count}개 생성 중...")    
    topics = [
        "카페에서 친구와 수다", "회사 동료들 점심 메뉴",
        "주말 영화 이야기", "요즘 듣는 음악", "반려동물 이야기",
        "운동 습관", "최근 읽은 책", "여행 계획", "날씨 이야기",
        "쇼핑과 옷 고르기", "요리 레시피", "게임 이야기",
        "드라마나 예능", "취미 생활", "건강 관리",
        "주말 계획", "가족과의 시간", "맛집 추천", "영화 추천"
    ]
    
    conversations = []
    batch_size = 15  # 한 번에 15개씩 (분당 제한 고려)
    
    for i in range(0, count, batch_size):
        remaining = min(batch_size, count - i)
        selected_topics = random.sample(topics, min(remaining, len(topics)))
        
        prompt = f"""당신은 한국어 일상 대화 생성 전문가입니다.

**임무:** train.csv 파일에 있는 협박 대화와 비슷한 길이와 형식의 하지만 자연스러운 일반 대화를 생성하세요. 이때, 일반 대화에는 친구들끼리 비속어 쓰는 장난스러운 대화도 포함될 수 있습니다.

**대화 형식 (매우 중요):**
- 줄바꿈으로 구분된 2-4명의 대화
- 각 발화는 짧고 자연스러워야 함
- 총 8-15개 정도의 발화로 구성
- 화자 표시 없이 대화 내용만 작성
- 줄바꿈으로 각 발화를 구분

**좋은 예시:**
오늘 날씨 진짜 좋다
맞아 산책하기 딱 좋아
이따가 같이 한강 가실?
좋지? 메뉴는?
치맥ㄱㄱㄱ
아니 등신아 ㅋㅋㅋ 뭔 치맥이야 어제 쳐먹고 또 처먹고 싶냐 어휴...
그럼 뭐 먹고 싶은데 병신아
엉아가 피자 사줄께.
넵, 형님. 바로 모시겠습니다
태세전환 보소 ㅋㅋㅋ
사주는 사람이 와따지
그건 그럼. 가자

**나쁜 예시 (절대 금지):**
A: 오늘 날씨 좋네요 (화자 표시 금지)
"오늘 날씨 좋네요" 하고 말했다 (서술 금지)
오늘 날씨 좋네요. 그래서 산책을 가기로 했다 (설명 금지)

**필수 조건:**
1. 협박, 갈취, 괴롭힘 등 부정적 내용 절대 금지
2. 긍정적이거나 중립적인 일상 대화를 만들되 위의 좋은 예시와 같이 친구사이에 있는 자연스러운 비속어 욕설 사용도 반드시 적절히 섞을것. ㅅㅂ ㅂㅅ ㄱㅅㄲ 등등 꼭 좀 섞을것.
3. 자연스러운 구어체 사용
4. 각 대화는 하나의 주제로 완결
5. 줄바꿈으로 발화 구분 (화자 표시 없이)

**주제 (각 주제당 1개씩):**
{chr(10).join([f"{idx+1}. {topic}" for idx, topic in enumerate(selected_topics)])}

위 {remaining}개 주제로 각각 독립적인 대화를 생성하세요.
각 대화는 빈 줄로 구분하고, 번호나 제목 없이 대화 내용만 출력하세요."""

        try:
            print(f"  API 호출 {i//batch_size + 1}... ({i+1}~{min(i+remaining, count)}/{count})", end='', flush=True)
            
            # Gemini API 호출
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.9,  # 다양성
                    max_output_tokens=2000,
                )
            )
            
            content = response.text.strip()
            
            # 응답 파싱: 빈 줄로 구분된 대화 블록 추출
            conversation_blocks = content.split('\n\n')
            
            for block in conversation_blocks:
                block = block.strip()
                if not block:
                    continue
                
                # 번호 제거 (1. 또는 1) 형식)
                block = re.sub(r'^\d+[\.\)]\s*', '', block, flags=re.MULTILINE)
                # 마크다운 리스트 제거
                block = re.sub(r'^[\*\-]\s*', '', block, flags=re.MULTILINE)
                # 화자 표시 제거 (A:, B:, 사람1: 등)
                block = re.sub(r'^[A-Z가-힣]+\d*\s*[:：]\s*', '', block, flags=re.MULTILINE)
                # 따옴표 제거
                block = block.strip('"\'')
                
                # 최소 길이 체크 (너무 짧은 대화 제외)
                lines = [l.strip() for l in block.split('\n') if l.strip()]
                if len(lines) >= 5 and len(block) >= 50:
                    conversations.append(block)
            
            print(f" ✓ {len(conversation_blocks)}개 생성")
            
            # API 호출 제한 대비 (분당 15회)
            if i + batch_size < count:
                time.sleep(4.5)  # 4.5초 대기 (안전하게)
            
        except Exception as e:
            print(f" ✗ 실패: {str(e)[:100]}")
            
            # Rate limit 오류 처리
            if "429" in str(e) or "quota" in str(e).lower() or "RESOURCE_EXHAUSTED" in str(e):
                print("API 호출 제한 도달, 60초 대기")
                time.sleep(60)
                # 재시도
                try:
                    print("재시도 중", end='', flush=True)
                    response = model.generate_content(
                        prompt,
                        generation_config=genai.types.GenerationConfig(
                            temperature=0.9,
                            max_output_tokens=3000,
                        )
                    )
                    content = response.text.strip()
                    
                    # 블록 파싱
                    conversation_blocks = content.split('\n\n')
                    for block in conversation_blocks:
                        block = block.strip()
                        if not block:
                            continue
                        
                        block = re.sub(r'^\d+[\.\)]\s*', '', block, flags=re.MULTILINE)
                        block = re.sub(r'^[\*\-]\s*', '', block, flags=re.MULTILINE)
                        block = re.sub(r'^[A-Z가-힣]+\d*\s*[:：]\s*', '', block, flags=re.MULTILINE)
                        block = block.strip('"\'')
                        
                        lines = [l.strip() for l in block.split('\n') if l.strip()]
                        if len(lines) >= 5 and len(block) >= 50:
                            conversations.append(block)
                    
                    print(f" ✓ {len(conversation_blocks)}개 생성")
                except Exception as e2:
                    print(f" ✗ 재시도 실패: {str(e2)[:50]}")
                    print("대체 템플릿 사용")
                    fallback_convs = _get_fallback_conversations()
                    conversations.extend(fallback_convs[:remaining])
            else:
                print(" 대체 템플릿 사용")
                fallback_convs = _get_fallback_conversations()
                conversations.extend(fallback_convs[:remaining])
    
    # ========================================
    # DataFrame 생성 (train.csv 형식)
    # ========================================
    print(f"\n✓ 총 {len(conversations)}개 대화 생성 완료")
    print(f"✓ train.csv 형식으로 변환 중...\n")
    
    # 필요한 개수만큼 자르기
    conversations = conversations[:count]
    
    # DataFrame 생성
    df = pd.DataFrame({
        'idx': range(start_idx, start_idx + len(conversations)),
        'class': ['일반 대화'] * len(conversations),
        'conversation': conversations
    })
    
    # CSV 저장 (UTF-8 with BOM for Excel compatibility)
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"✓ 파일 저장: {output_csv}")
    print(f"  - 총 {len(df)}개 대화")
    print(f"  - 인덱스 범위: {start_idx} ~ {start_idx + len(df) - 1}")
    
    # 캐시 저장
    if use_cache:
        cache_file = f'normal_conversations_cache_{count}.csv'
        df.to_csv(cache_file, index=False, encoding='utf-8-sig')
        print(f"✓ 캐시 저장: {cache_file}")
    
    # 샘플 출력
    print("\n[생성된 대화 샘플]")
    print("=" * 70)
    for i in range(min(3, len(df))):
        print(f"\n{i+1}. 인덱스 {df.iloc[i]['idx']}, 클래스: {df.iloc[i]['class']}")
        conv_preview = df.iloc[i]['conversation'][:200] + "..." if len(df.iloc[i]['conversation']) > 200 else df.iloc[i]['conversation']
        print(f"   대화:\n{conv_preview}")
    print("=" * 70)
    
    return df


def _get_fallback_conversations():
    """
    API 실패 시 사용할 기본 템플릿
    train.csv의 대화 형식과 유사하게 작성
    """
    return [
        """오늘 날씨 진짜 좋다
산책하기 딱 좋은데
한강 갈래?
좋지 뭐 먹고 갈까?
치킨이랑 맥주 어때?
완전 좋아 빨리 가자""",
        
        """점심 뭐 먹을까?
나 파스타 땡기는데
이탈리안 레스토랑 갈까?
좋아 거기 음식 맛있어
몇 시에 갈래?
12시 반쯤?
알았어 그때 보자""",
        
        """주말에 뭐 했어?
친구들이랑 영화 봤어
재밌었어?
응 진짜 재밌더라
나도 보고 싶었는데
같이 볼걸 그랬다
다음에 같이 보자""",
        
        """요즘 무슨 노래 들어?
나 요즘 팝송 많이 듣는데
추천 좀 해줘
이 노래 들어봐
오 좋은데?
그치? 나도 요즘 맨날 들어
나도 플레이리스트에 추가할게""",
        
        """강아지 키우고 싶다
무슨 견종?
골든 리트리버
완전 귀엽지
근데 산책 많이 시켜야 해
괜찮아 운동 삼아서
그럼 진지하게 알아봐""",
        
        """요즘 운동해?
응 헬스장 다녀
어디 다니는데?
집 근처 헬스장
나도 시작하고 싶은데
같이 다닐래?
좋지 언제부터?
다음 주부터""",
        
        """최근에 책 읽었어?
응 소설 한 권 읽었어
재밌었어?
완전 몰입해서 읽었어
제목이 뭐야?
이거야 한번 읽어봐
나도 읽어볼게""",
        
        """겨울 휴가 어디 갈까?
제주도 어때?
겨울에도 괜찮아?
겨울 제주도도 좋아
그럼 언제 갈까?
1월 중순?
좋아 계획 세워보자""",
        
        """오늘 비 온대
우산 챙겼어?
아 깜빡했다
내 거 같이 써
고마워
천만에 빨리 가자""",
        
        """쇼핑 가고 싶다
뭐 살 건데?
겨울 코트 하나
같이 갈래?
좋지 언제?
이번 주말?
알았어""",
    ]


def _create_fallback_csv(output_csv, count, start_idx):
    """
    API 사용 불가 시 fallback 대화로 CSV 생성
    """
    print("\n기본 템플릿으로 CSV 생성")
    
    fallback_convs = _get_fallback_conversations()
    
    # 필요한 개수만큼 반복하여 생성
    conversations = []
    while len(conversations) < count:
        conversations.extend(fallback_convs)
    conversations = conversations[:count]
    
    # DataFrame 생성
    df = pd.DataFrame({
        'idx': range(start_idx, start_idx + len(conversations)),
        'class': ['일반 대화'] * len(conversations),
        'conversation': conversations
    })
    
    # CSV 저장
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"✓ 파일 저장: {output_csv}")
    print(f"  - 총 {len(df)}개 대화 (템플릿 반복)")
    
    return df


if __name__ == "__main__":
    """
    파일을 직접 실행할 때만 이 부분이 작동.
    (API 테스트용)
    """
    
    print("=" * 70)
    print("일반 대화 생성기 테스트 (train.csv 형식)")
    print("=" * 70)
    
    # API 키가 설정되었는지 확인
    if not os.getenv("GEMINI_API_KEY"):
        print("\n GEMINI_API_KEY 환경 변수가 설정되지 않았습니다.")
        print("\n설정 방법:")
        print("  터미널에서 실행: export GEMINI_API_KEY='your-api-key'")
        print("  또는 코드에서 직접 입력\n")
        print("→ 기본 템플릿으로 진행합니다...\n")
    else:
        print("\n✓ GEMINI_API_KEY 확인됨\n")
    
    # 테스트: 20개 생성, 캐시 사용 안함
    print("=" * 70)
    print("테스트: 20개 일반 대화 생성")
    print("=" * 70)
    
    df = generate_normal_conversations(
        count=20,
        use_cache=False,
        output_csv='normal_conversations_test.csv',
        start_idx=100000
    )
    
    print("\n" + "=" * 70)
    print("테스트 완료!")
    print("=" * 70)
    print("\n생성된 파일:")
    print("  - normal_conversations_test.csv")
    print("\n다음 단계:")
    print("  1. CSV 파일 확인")
    print("  2. 만족스러우면 count를 늘려서 재실행 (예: count=1000)")
    print("  3. preprocessing.py에 바로 사용 가능")