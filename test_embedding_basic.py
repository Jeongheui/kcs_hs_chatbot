"""
임베딩 시스템 기본 기능 단위 테스트
- 벡터 차원 검증
- 코사인 유사도 계산 검증
- 임베딩 캐시 로드 확인
"""

import os
import sys
import numpy as np
from utils.data_loader import HSDataManager

# 컬러 출력 지원
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_test_header(test_name):
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*70}")
    print(f"TEST: {test_name}")
    print(f"{'='*70}{Colors.RESET}\n")

def print_success(message):
    print(f"{Colors.GREEN}✓ PASS{Colors.RESET}: {message}")

def print_fail(message):
    print(f"{Colors.RED}✗ FAIL{Colors.RESET}: {message}")

def print_info(message):
    print(f"{Colors.BLUE}ℹ INFO{Colors.RESET}: {message}")

def print_warning(message):
    print(f"{Colors.YELLOW}⚠ WARNING{Colors.RESET}: {message}")

# ========== Test 1: HSDataManager 초기화 ==========
def test_initialization():
    print_test_header("HSDataManager 초기화 테스트")

    try:
        manager = HSDataManager()
        print_success("HSDataManager 객체 생성 성공")

        # 데이터 로드 확인
        if len(manager.data) > 0:
            print_success(f"데이터 로드 완료: {len(manager.data)}개 소스")
            for source in manager.data.keys():
                print_info(f"  - {source}: {len(manager.data[source])} items")
        else:
            print_fail("데이터가 로드되지 않음")

        # 아이템 인덱스 확인
        if len(manager.items_by_id) > 0:
            print_success(f"아이템 인덱스 구축 완료: {len(manager.items_by_id)}개 항목")
        else:
            print_fail("아이템 인덱스가 비어있음")

        # 임베딩 캐시 확인
        if len(manager.embeddings_cache) > 0:
            print_success(f"임베딩 캐시 로드 완료: {len(manager.embeddings_cache)}개 항목")
        else:
            print_warning("임베딩 캐시가 비어있음 (아직 생성되지 않음)")
            print_info("Streamlit 앱에서 '임베딩 생성' 버튼을 클릭하세요")

        # 캐시 디렉토리 확인
        if os.path.exists(manager.embedding_cache_dir):
            print_success(f"캐시 디렉토리 존재: {manager.embedding_cache_dir}")
            cache_files = os.listdir(manager.embedding_cache_dir)
            if cache_files:
                print_info(f"  캐시 파일: {', '.join(cache_files)}")
            else:
                print_warning("  캐시 디렉토리가 비어있음")
        else:
            print_warning(f"캐시 디렉토리 없음: {manager.embedding_cache_dir}")

        return manager

    except Exception as e:
        print_fail(f"초기화 실패: {e}")
        return None

# ========== Test 2: 임베딩 벡터 검증 ==========
def test_embedding_vectors(manager):
    print_test_header("임베딩 벡터 검증")

    if len(manager.embeddings_cache) == 0:
        print_warning("임베딩 캐시가 비어있어 테스트를 건너뜁니다")
        return

    # 샘플 벡터 가져오기
    sample_items = list(manager.embeddings_cache.items())[:5]

    all_passed = True

    for item_id, embedding in sample_items:
        print_info(f"\n항목 ID: {item_id}")

        # 벡터 타입 확인
        if isinstance(embedding, np.ndarray):
            print_success(f"  타입: numpy.ndarray")
        else:
            print_fail(f"  타입 오류: {type(embedding)} (예상: numpy.ndarray)")
            all_passed = False

        # 벡터 차원 확인
        if embedding.shape == (768,):
            print_success(f"  차원: {embedding.shape} (768차원 벡터)")
        else:
            print_fail(f"  차원 오류: {embedding.shape} (예상: (768,))")
            all_passed = False

        # 벡터가 비어있지 않은지 확인
        if np.sum(np.abs(embedding)) > 0:
            print_success(f"  값 존재: L1 norm = {np.sum(np.abs(embedding)):.2f}")
        else:
            print_fail(f"  제로 벡터 (임베딩 생성 실패)")
            all_passed = False

        # 벡터 노름 확인
        norm = np.linalg.norm(embedding)
        print_info(f"  L2 norm: {norm:.4f}")

    if all_passed:
        print(f"\n{Colors.GREEN}{Colors.BOLD}모든 벡터 검증 통과{Colors.RESET}")
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}일부 벡터 검증 실패{Colors.RESET}")

# ========== Test 3: 코사인 유사도 계산 검증 ==========
def test_cosine_similarity(manager):
    print_test_header("코사인 유사도 계산 검증")

    if len(manager.embeddings_cache) == 0:
        print_warning("임베딩 캐시가 비어있어 테스트를 건너뜁니다")
        return

    # 테스트 벡터 준비
    sample_embeddings = list(manager.embeddings_cache.values())[:3]

    if len(sample_embeddings) < 2:
        print_warning("테스트할 벡터가 부족합니다")
        return

    vec1 = sample_embeddings[0]
    vec2 = sample_embeddings[1]

    # 1. 자기 자신과의 유사도 (1.0이어야 함)
    sim_self = manager.cosine_similarity(vec1, vec1)
    print_info(f"자기 자신과의 유사도: {sim_self:.6f}")
    if abs(sim_self - 1.0) < 0.0001:
        print_success("자기 자신과의 유사도 = 1.0 (정확)")
    else:
        print_fail(f"자기 자신과의 유사도 오류 (예상: 1.0, 실제: {sim_self:.6f})")

    # 2. 서로 다른 벡터 간 유사도 (0~1 범위)
    sim_diff = manager.cosine_similarity(vec1, vec2)
    print_info(f"다른 벡터 간 유사도: {sim_diff:.6f}")
    if 0.0 <= sim_diff <= 1.0:
        print_success(f"유사도가 정상 범위 내 (0~1)")
    else:
        print_fail(f"유사도가 범위 밖 (0~1 예상, 실제: {sim_diff:.6f})")

    # 3. 제로 벡터 처리
    zero_vec = np.zeros(768)
    sim_zero = manager.cosine_similarity(vec1, zero_vec)
    print_info(f"제로 벡터와의 유사도: {sim_zero:.6f}")
    if sim_zero == 0.0:
        print_success("제로 벡터 처리 정상 (유사도 = 0.0)")
    else:
        print_warning(f"제로 벡터 유사도 예상과 다름: {sim_zero:.6f}")

# ========== Test 4: 쿼리 임베딩 생성 테스트 ==========
def test_query_embedding(manager):
    print_test_header("쿼리 임베딩 생성 테스트")

    if not manager.client:
        print_warning("Gemini client가 초기화되지 않음 (GOOGLE_API_KEY 확인)")
        return

    test_queries = [
        "플라스틱으로 만든 병",
        "전자제품 배터리",
        "면 티셔츠"
    ]

    for query in test_queries:
        print_info(f"\n쿼리: '{query}'")

        try:
            embedding = manager.get_query_embedding(query)

            # 벡터 타입 확인
            if isinstance(embedding, np.ndarray):
                print_success(f"  임베딩 생성 성공")
            else:
                print_fail(f"  타입 오류: {type(embedding)}")
                continue

            # 차원 확인
            if embedding.shape == (768,):
                print_success(f"  차원: {embedding.shape}")
            else:
                print_fail(f"  차원 오류: {embedding.shape}")

            # 벡터 내용 확인
            if np.sum(np.abs(embedding)) > 0:
                print_success(f"  L1 norm: {np.sum(np.abs(embedding)):.2f}")
            else:
                print_fail(f"  제로 벡터 (API 호출 실패)")

        except Exception as e:
            print_fail(f"  임베딩 생성 실패: {e}")

# ========== 메인 실행 ==========
def main():
    print(f"\n{Colors.BOLD}{Colors.BLUE}")
    print("=" * 70)
    print("  임베딩 시스템 기본 기능 단위 테스트")
    print("=" * 70)
    print(Colors.RESET)

    # Test 1: 초기화
    manager = test_initialization()
    if manager is None:
        print(f"\n{Colors.RED}{Colors.BOLD}초기화 실패로 테스트 중단{Colors.RESET}")
        return

    # Test 2: 벡터 검증
    test_embedding_vectors(manager)

    # Test 3: 유사도 계산
    test_cosine_similarity(manager)

    # Test 4: 쿼리 임베딩
    test_query_embedding(manager)

    print(f"\n{Colors.BOLD}{Colors.BLUE}")
    print("=" * 70)
    print("  테스트 완료")
    print("=" * 70)
    print(Colors.RESET)

if __name__ == "__main__":
    main()