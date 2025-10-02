"""
TF-IDF 성능 비교 테스트
Word 기반 vs Character 기반 n-gram 비교

비교 항목:
1. 응답 시간 (인덱스 구축 시간, 검색 시간)
2. 검색 결과 관련성 (상위 10개 결과 분석)
"""

import json
import time
from pathlib import Path
from utils.tfidf_search import WordTfidfSearchEngine, CharTfidfSearchEngine


def load_hs_data(max_docs=None):
    """
    HS 분류 사례 데이터 로드

    Args:
        max_docs: 최대 로드할 문서 수 (None이면 전체)

    Returns:
        documents: 문서 텍스트 리스트
        metadata: 문서 메타데이터 (제목, HS코드 등)
    """
    documents = []
    metadata = []

    knowledge_dir = Path("knowledge")

    # HS분류사례_part1~10.json 로드
    for i in range(1, 11):
        filepath = knowledge_dir / f"HS분류사례_part{i}.json"
        if not filepath.exists():
            print(f"경고: {filepath} 파일을 찾을 수 없습니다.")
            continue

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

            for item in data:
                # 문서 텍스트 생성 (여러 필드 결합)
                text_parts = []

                if 'product_name' in item and item['product_name']:
                    text_parts.append(item['product_name'])
                if 'description' in item and item['description']:
                    text_parts.append(item['description'])
                if 'decision_reason' in item and item['decision_reason']:
                    text_parts.append(item['decision_reason'])

                text = " ".join(text_parts)

                if text.strip():
                    documents.append(text)
                    metadata.append({
                        '품명': item.get('product_name', ''),
                        'HS코드': item.get('hs_code', ''),
                        '분류이유': item.get('decision_reason', '')[:100] + '...' if item.get('decision_reason', '') else ''
                    })

                if max_docs and len(documents) >= max_docs:
                    return documents, metadata

    return documents, metadata


def run_comparison_test(query, word_engine, char_engine, metadata, top_k=10):
    """
    단일 쿼리에 대한 비교 테스트 실행

    Args:
        query: 검색 쿼리
        word_engine: Word 기반 엔진
        char_engine: Char 기반 엔진
        metadata: 문서 메타데이터
        top_k: 상위 몇 개 결과 비교할지
    """
    print(f"\n{'='*80}")
    print(f"테스트 쿼리: {query}")
    print(f"{'='*80}\n")

    # Word 기반 검색
    print("[Word 기반 TF-IDF]")
    start = time.time()
    word_results = word_engine.search(query, top_k)
    word_time = time.time() - start
    print(f"검색 시간: {word_time:.4f}초")
    print(f"\n상위 {top_k}개 결과:")
    for rank, (doc_id, score) in enumerate(word_results, 1):
        meta = metadata[doc_id]
        print(f"{rank}. [유사도: {score:.3f}] {meta['품명']}")
        if meta['HS코드']:
            print(f"   HS코드: {meta['HS코드']}")
        if meta['분류이유']:
            print(f"   {meta['분류이유']}")

    # Character 기반 검색
    print(f"\n{'-'*80}\n")
    print("[Character 기반 TF-IDF]")
    start = time.time()
    char_results = char_engine.search(query, top_k)
    char_time = time.time() - start
    print(f"검색 시간: {char_time:.4f}초")
    print(f"\n상위 {top_k}개 결과:")
    for rank, (doc_id, score) in enumerate(char_results, 1):
        meta = metadata[doc_id]
        print(f"{rank}. [유사도: {score:.3f}] {meta['품명']}")
        if meta['HS코드']:
            print(f"   HS코드: {meta['HS코드']}")
        if meta['분류이유']:
            print(f"   {meta['분류이유']}")

    # 비교 분석
    print(f"\n{'-'*80}")
    print("[비교 분석]")
    print(f"속도: ", end="")
    if word_time < char_time:
        print(f"Word 방식이 {char_time - word_time:.4f}초 빠름")
    else:
        print(f"Char 방식이 {word_time - char_time:.4f}초 빠름")

    print(f"평균 유사도: Word={sum(s for _, s in word_results)/len(word_results):.3f}, "
          f"Char={sum(s for _, s in char_results)/len(char_results):.3f}")

    # 상위 3개 결과 비교
    print("\n상위 3개 결과 품명 비교:")
    print("Word:", [metadata[doc_id]['품명'] for doc_id, _ in word_results[:3]])
    print("Char:", [metadata[doc_id]['품명'] for doc_id, _ in char_results[:3]])


def main():
    """메인 테스트 실행"""

    print("="*80)
    print("TF-IDF 검색 엔진 성능 비교 테스트")
    print("="*80)

    # 데이터 로드
    print("\n[1단계] 데이터 로드 중...")
    documents, metadata = load_hs_data()
    print(f"총 {len(documents)}개 문서 로드 완료")

    # Word 기반 인덱스 구축
    print("\n[2단계] Word 기반 TF-IDF 인덱스 구축 중...")
    start = time.time()
    word_engine = WordTfidfSearchEngine()
    word_engine.fit(documents)
    word_build_time = time.time() - start
    print(f"구축 완료: {word_build_time:.2f}초")

    # Character 기반 인덱스 구축
    print("\n[3단계] Character 기반 TF-IDF 인덱스 구축 중...")
    start = time.time()
    char_engine = CharTfidfSearchEngine()
    char_engine.fit(documents)
    char_build_time = time.time() - start
    print(f"구축 완료: {char_build_time:.2f}초")

    print(f"\n인덱스 구축 시간 비교:")
    print(f"  Word: {word_build_time:.2f}초")
    print(f"  Char: {char_build_time:.2f}초")
    if word_build_time < char_build_time:
        print(f"  -> Word 방식이 {char_build_time - word_build_time:.2f}초 빠름")
    else:
        print(f"  -> Char 방식이 {word_build_time - char_build_time:.2f}초 빠름")

    # 테스트 쿼리 목록
    test_queries = [
        "리튬이온배터리 분류 사례",
        "전기자동차 부품 HS 코드",
        "폴리우레탄폼 관세"
    ]

    print("\n[4단계] 검색 성능 테스트")

    # 각 쿼리별 비교 테스트
    for query in test_queries:
        run_comparison_test(query, word_engine, char_engine, metadata, top_k=10)

    # 최종 요약
    print(f"\n{'='*80}")
    print("테스트 완료")
    print(f"{'='*80}")
    print("\n[결론]")
    print("위 결과를 바탕으로 다음을 평가하세요:")
    print("1. 응답 시간: 어느 방식이 더 빠른가?")
    print("2. 검색 결과 관련성: 어느 방식의 상위 결과가 쿼리와 더 관련 있는가?")
    print("3. 유사도 점수: 어느 방식이 더 높은 신뢰도를 보이는가?")
    print("\n더 우수한 방식을 선택하여 프로젝트에 반영하세요.")


if __name__ == "__main__":
    main()
