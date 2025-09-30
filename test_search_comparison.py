"""
검색 품질 비교 테스트
- 키워드 검색
- 임베딩 검색
- 하이브리드 검색
3가지 방식의 결과를 비교 분석
"""

import time
import json
from typing import Dict, List
from utils.data_loader import HSDataManager

# 컬러 출력 지원
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'

def print_header(title):
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*80}")
    print(f"{title:^80}")
    print(f"{'='*80}{Colors.RESET}\n")

def print_subheader(title):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'-'*80}")
    print(f"{title}")
    print(f"{'-'*80}{Colors.RESET}")

def print_result_item(rank, item_data, show_full=False):
    """검색 결과 항목을 보기 좋게 출력"""
    source = item_data.get('source', 'Unknown')
    score = item_data.get('score', 0)
    item = item_data.get('item', {})
    item_id = item_data.get('item_id', 'N/A')

    # 품목명 추출
    product_name = item.get('품목명', item.get('product_name', 'N/A'))
    hs_code = item.get('HS코드', item.get('hs_code', 'N/A'))

    print(f"{Colors.BOLD}{Colors.YELLOW}[{rank}] {Colors.RESET}", end='')
    print(f"{Colors.GREEN}점수: {score:.4f}{Colors.RESET} | ", end='')
    print(f"{Colors.CYAN}출처: {source}{Colors.RESET}")

    if len(product_name) > 80 and not show_full:
        product_name = product_name[:77] + "..."

    print(f"    품목명: {product_name}")
    print(f"    HS코드: {hs_code}")

    if show_full:
        print(f"    {Colors.DIM}Item ID: {item_id}{Colors.RESET}")

def compare_results(results_dict: Dict[str, List], query: str):
    """여러 검색 방식의 결과를 비교 분석"""
    print_subheader(f"검색 결과 비교 분석: '{query}'")

    # 각 방식의 상위 항목 ID 추출
    top_items = {}
    for method, results in results_dict.items():
        top_items[method] = set([r.get('item_id', '') for r in results[:5]])

    # 교집합 분석
    methods = list(results_dict.keys())
    if len(methods) == 3:
        all_common = top_items[methods[0]] & top_items[methods[1]] & top_items[methods[2]]
        keyword_embedding_common = top_items[methods[0]] & top_items[methods[1]]
        keyword_hybrid_common = top_items[methods[0]] & top_items[methods[2]]
        embedding_hybrid_common = top_items[methods[1]] & top_items[methods[2]]

        print(f"\n{Colors.BOLD}중복도 분석:{Colors.RESET}")
        print(f"  3개 방식 모두 공통: {Colors.GREEN}{len(all_common)}개{Colors.RESET}")
        print(f"  키워드 ∩ 임베딩: {len(keyword_embedding_common)}개")
        print(f"  키워드 ∩ 하이브리드: {len(keyword_hybrid_common)}개")
        print(f"  임베딩 ∩ 하이브리드: {len(embedding_hybrid_common)}개")

        if all_common:
            print(f"\n  {Colors.BOLD}공통 항목:{Colors.RESET}")
            for item_id in list(all_common)[:3]:
                # 첫 번째 방식의 결과에서 항목 찾기
                for result in results_dict[methods[0]]:
                    if result.get('item_id') == item_id:
                        product_name = result.get('item', {}).get('품목명', 'N/A')
                        if len(product_name) > 60:
                            product_name = product_name[:57] + "..."
                        print(f"    - {product_name}")
                        break

    # 고유 항목 분석
    print(f"\n{Colors.BOLD}방식별 고유 항목 (다른 방식에는 없는 상위 5개):{Colors.RESET}")
    for method, items in top_items.items():
        other_methods = [m for m in methods if m != method]
        other_items = set()
        for m in other_methods:
            other_items.update(top_items[m])

        unique = items - other_items
        print(f"  {method}: {Colors.YELLOW}{len(unique)}개{Colors.RESET}")

def run_single_query_test(manager, query, max_results=5):
    """단일 쿼리에 대해 3가지 검색 방식 테스트"""
    print_header(f"쿼리: '{query}'")

    results_dict = {}
    times = {}

    # 1. 키워드 검색
    print_subheader("1️⃣  키워드 검색 (Keyword Search)")
    try:
        start_time = time.time()
        keyword_results = manager._search_keyword(query, max_results=max_results)
        elapsed = time.time() - start_time
        times['키워드 검색'] = elapsed

        if keyword_results:
            results_dict['키워드 검색'] = keyword_results
            print(f"{Colors.GREEN}✓ 검색 완료{Colors.RESET} ({elapsed:.4f}초, {len(keyword_results)}개 결과)\n")
            for i, result in enumerate(keyword_results, 1):
                print_result_item(i, result)
        else:
            print(f"{Colors.RED}✗ 결과 없음{Colors.RESET}")
    except Exception as e:
        print(f"{Colors.RED}✗ 검색 실패: {e}{Colors.RESET}")

    # 2. 임베딩 검색
    print_subheader("2️⃣  임베딩 검색 (Embedding Search)")
    if len(manager.embeddings_cache) == 0:
        print(f"{Colors.YELLOW}⚠ 임베딩 캐시가 비어있어 건너뜁니다{Colors.RESET}")
    else:
        try:
            start_time = time.time()
            embedding_results = manager.search_by_embedding(query, max_results=max_results)
            elapsed = time.time() - start_time
            times['임베딩 검색'] = elapsed

            if embedding_results:
                results_dict['임베딩 검색'] = embedding_results
                print(f"{Colors.GREEN}✓ 검색 완료{Colors.RESET} ({elapsed:.4f}초, {len(embedding_results)}개 결과)\n")
                for i, result in enumerate(embedding_results, 1):
                    print_result_item(i, result)
            else:
                print(f"{Colors.RED}✗ 결과 없음{Colors.RESET}")
        except Exception as e:
            print(f"{Colors.RED}✗ 검색 실패: {e}{Colors.RESET}")

    # 3. 하이브리드 검색
    print_subheader("3️⃣  하이브리드 검색 (Hybrid Search)")
    try:
        start_time = time.time()
        hybrid_results = manager.search_hybrid(
            query,
            max_results=max_results,
            keyword_weight=0.3,
            embedding_weight=0.7
        )
        elapsed = time.time() - start_time
        times['하이브리드 검색'] = elapsed

        if hybrid_results:
            results_dict['하이브리드 검색'] = hybrid_results
            print(f"{Colors.GREEN}✓ 검색 완료{Colors.RESET} ({elapsed:.4f}초, {len(hybrid_results)}개 결과)")
            print(f"  {Colors.DIM}가중치: 키워드 30% + 임베딩 70%{Colors.RESET}\n")
            for i, result in enumerate(hybrid_results, 1):
                print_result_item(i, result)
        else:
            print(f"{Colors.RED}✗ 결과 없음{Colors.RESET}")
    except Exception as e:
        print(f"{Colors.RED}✗ 검색 실패: {e}{Colors.RESET}")

    # 성능 비교
    if times:
        print_subheader("⏱️  성능 비교")
        for method, elapsed in sorted(times.items(), key=lambda x: x[1]):
            print(f"  {method}: {Colors.CYAN}{elapsed:.4f}초{Colors.RESET}")

    # 결과 비교 분석
    if len(results_dict) >= 2:
        compare_results(results_dict, query)

def main():
    print(f"\n{Colors.BOLD}{Colors.MAGENTA}")
    print("=" * 80)
    print("  검색 품질 비교 테스트")
    print("  Keyword vs Embedding vs Hybrid Search")
    print("=" * 80)
    print(Colors.RESET)

    # HSDataManager 초기화
    print(f"{Colors.BLUE}HSDataManager 초기화 중...{Colors.RESET}")
    try:
        manager = HSDataManager()
        print(f"{Colors.GREEN}✓ 초기화 완료{Colors.RESET}")
        print(f"  - 데이터 소스: {len(manager.data)}개")
        print(f"  - 총 아이템: {len(manager.items_by_id)}개")
        print(f"  - 임베딩 캐시: {len(manager.embeddings_cache)}개")

        if len(manager.embeddings_cache) == 0:
            print(f"\n{Colors.YELLOW}⚠ 경고: 임베딩 캐시가 비어있습니다{Colors.RESET}")
            print(f"  Streamlit 앱에서 '임베딩 생성' 버튼을 먼저 클릭하세요")
            print(f"  임베딩 검색과 하이브리드 검색이 제한됩니다\n")

    except Exception as e:
        print(f"{Colors.RED}✗ 초기화 실패: {e}{Colors.RESET}")
        return

    # 테스트 쿼리 정의
    test_queries = [
        "플라스틱으로 만든 병",
        "전자제품용 리튬 배터리",
        "면으로 만든 여성용 반팔 티셔츠",
        "자동차 부품",
    ]

    print(f"\n{Colors.BOLD}테스트 쿼리 목록:{Colors.RESET}")
    for i, q in enumerate(test_queries, 1):
        print(f"  {i}. {q}")

    # 각 쿼리에 대해 테스트 실행
    for query in test_queries:
        run_single_query_test(manager, query, max_results=5)

    # 최종 요약
    print_header("테스트 완료")
    print(f"{Colors.GREEN}모든 쿼리 테스트가 완료되었습니다{Colors.RESET}\n")

    # 추가 분석 정보
    if len(manager.embeddings_cache) > 0:
        print(f"{Colors.BOLD}임베딩 시스템 상태:{Colors.RESET}")
        print(f"  - 캐시된 항목: {len(manager.embeddings_cache)}개")
        print(f"  - 임베딩 모델: {manager.embedding_model}")
        print(f"  - 벡터 차원: 768")

        # 소스별 통계
        from collections import defaultdict
        source_counts = defaultdict(int)
        for item_id in manager.embeddings_cache.keys():
            source = item_id.rsplit('_', 1)[0]
            source_counts[source] += 1

        print(f"\n{Colors.BOLD}소스별 임베딩 분포:{Colors.RESET}")
        for source, count in sorted(source_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  - {source}: {count}개")

    print()

if __name__ == "__main__":
    main()