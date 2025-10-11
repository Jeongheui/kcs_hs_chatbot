import json
import re
import os
import pickle
import gzip
from typing import Dict, List, Any
from collections import defaultdict
from dotenv import load_dotenv
from .tfidf_search import TfidfSearchEngine

# 환경 변수 로드
load_dotenv()

class HSDataManager:
    """
    HS 코드 관련 데이터를 관리하는 클래스
    - HS 분류 사례, 위원회 결정, 협의회 결정 등의 데이터를 로드하고 관리
    - TF-IDF 기반 검색 기능 제공 (Character n-gram)
    - 관련 컨텍스트 생성 기능 제공
    """

    def __init__(self):
        """HSDataManager 초기화"""
        self.data = {}  # 모든 HS 관련 데이터를 저장하는 딕셔너리
        self.search_index = defaultdict(list)  # 키워드 기반 검색을 위한 인덱스 (하위 호환)

        # TF-IDF 검색 엔진
        self.domestic_tfidf = None  # 국내 사례용
        self.overseas_tfidf = None  # 해외 사례용
        self.manual_tfidf = None    # HS 매뉴얼용
        self.domestic_items = []    # 국내 사례 아이템
        self.overseas_items = []    # 해외 사례 아이템

        self.load_all_data()  # 모든 데이터 파일 로드
        self.build_search_index()  # 키워드 검색 인덱스 구축 (하위 호환)

        # TF-IDF 인덱스: pickle 파일이 있으면 로드, 없으면 구축
        if os.path.exists('tfidf_indexes.pkl.gz') or os.path.exists('tfidf_indexes.pkl'):
            self._load_tfidf_indexes()
        else:
            self.build_tfidf_indexes()

    def load_all_data(self):
        """
        모든 HS 데이터 파일을 로드하는 메서드
        - HS분류사례_part1~10.json 파일 로드
        - HS위원회.json, HS협의회.json 파일 로드
        - hs_classification_data_us.json 파일 로드 (미국 관세청 품목분류 사례)
        - hs_classification_data_eu.json 파일 로드 (EU 관세청 품목분류 사례)
        """
        # HS분류사례 파트 로드 (1~10)
        for i in range(1, 11):
            try:
                with open(f'knowledge/HS분류사례_part{i}.json', 'r', encoding='utf-8') as f:
                    self.data[f'HS분류사례_part{i}'] = json.load(f)
            except FileNotFoundError:
                print(f'Warning: HS분류사례_part{i}.json not found')

        # 기타 JSON 파일 로드 (위원회, 협의회 결정)
        other_files = ['knowledge/HS위원회.json', 'knowledge/HS협의회.json']
        for file in other_files:
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    self.data[file.replace('.json', '')] = json.load(f)
            except FileNotFoundError:
                print(f'Warning: {file} not found')

        # 미국 관세청 품목분류 사례 로드
        try:
            with open('knowledge/hs_classification_data_us.json', 'r', encoding='utf-8') as f:
                self.data['hs_classification_data_us'] = json.load(f)
        except FileNotFoundError:
            print('Warning: hs_classification_data_us.json not found')

        # EU 관세청 품목분류 사례 로드
        try:
            with open('knowledge/hs_classification_data_eu.json', 'r', encoding='utf-8') as f:
                self.data['hs_classification_data_eu'] = json.load(f)
        except FileNotFoundError:
            print('Warning: hs_classification_data_eu.json not found')

    def build_search_index(self):
        """
        검색 인덱스 구축 메서드
        - 각 데이터 항목에서 키워드를 추출
        - 추출된 키워드를 인덱스에 저장하여 빠른 검색 가능
        """
        for source, items in self.data.items():
            for item in items:
                # 품목명에서 키워드 추출
                keywords = self._extract_keywords(str(item))
                # 각 키워드에 대해 해당 아이템 참조 저장
                for keyword in keywords:
                    self.search_index[keyword].append((source, item))

    def _extract_keywords(self, text: str) -> List[str]:
        """
        텍스트에서 의미있는 키워드를 추출하는 내부 메서드
        Args:
            text: 키워드를 추출할 텍스트
        Returns:
            추출된 키워드 리스트
        """
        # 특수문자 제거 및 공백 기준 분리
        words = re.sub(r'[^\w\s]', ' ', text).split()
        # 중복 제거 및 길이 2 이상인 단어만 선택
        return list(set(word for word in words if len(word) >= 2))

    def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        쿼리와 관련된 가장 연관성 높은 항목들을 검색하는 메서드
        Args:
            query: 검색할 쿼리 문자열
            max_results: 반환할 최대 결과 수 (기본값: 5)
        Returns:
            검색 결과 리스트 (출처와 항목 정보 포함)
        """
        query_keywords = self._extract_keywords(query)
        results = defaultdict(int)

        # 각 키워드에 대해 매칭되는 항목 찾기
        for keyword in query_keywords:
            for source, item in self.search_index.get(keyword, []):
                # 가중치 계산 (키워드 매칭 횟수 기반)
                results[(source, str(item))] += 1

        # 가중치 기준 정렬
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

        # 상위 결과만 반환
        return [
            {'source': source, 'item': eval(item_str)}
            for (source, item_str), _ in sorted_results[:max_results]
        ]

    def search_domestic_group(self, query: str, group_idx: int, max_results: int = 3) -> List[Dict[str, Any]]:
        """국내 HS 분류 데이터 그룹별 검색 메서드"""
        query_keywords = self._extract_keywords(query)
        results = defaultdict(int)

        # 그룹별 데이터 소스 정의 (5개 그룹)
        group_sources = [
            ['HS분류사례_part1', 'HS분류사례_part2'],  # 그룹1
            ['HS분류사례_part3', 'HS분류사례_part4'],  # 그룹2
            ['HS분류사례_part5', 'HS분류사례_part6'],  # 그룹3
            ['HS분류사례_part7', 'HS분류사례_part8'],  # 그룹4
            ['HS분류사례_part9', 'HS분류사례_part10', 'knowledge/HS위원회', 'knowledge/HS협의회']  # 그룹5
        ]
        sources = group_sources[group_idx]

        for keyword in query_keywords:
            for source, item in self.search_index.get(keyword, []):
                if source in sources:
                    results[(source, str(item))] += 1

        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
        return [
            {'source': source, 'item': eval(item_str)}
            for (source, item_str), _ in sorted_results[:max_results]
        ]

    def get_domestic_context_group(self, query: str, group_idx: int) -> str:
        """국내 HS 분류 관련 컨텍스트(그룹별)를 생성하는 메서드"""
        results = self.search_domestic_group(query, group_idx)
        context = []
        for result in results:
            context.append(f"출처: {result['source']} (국내 관세청)\n항목: {json.dumps(result['item'], ensure_ascii=False)}")
        return "\n\n".join(context)

    def search_overseas_group(self, query: str, group_idx: int, max_results: int = 3) -> List[Dict[str, Any]]:
        """해외 HS 분류 데이터 그룹별 검색 메서드"""
        query_keywords = self._extract_keywords(query)
        results = defaultdict(int)

        # 해외 데이터를 그룹별로 분할 처리
        if group_idx < 3:  # 그룹 0,1,2는 미국 데이터
            target_source = 'hs_classification_data_us'
            # 미국 데이터를 3등분
            us_data = self.data.get(target_source, [])
            chunk_size = len(us_data) // 3
            start_idx = group_idx * chunk_size
            end_idx = start_idx + chunk_size if group_idx < 2 else len(us_data)
            target_items = us_data[start_idx:end_idx]
        else:  # 그룹 3,4는 EU 데이터
            target_source = 'hs_classification_data_eu'
            # EU 데이터를 2등분
            eu_data = self.data.get(target_source, [])
            chunk_size = len(eu_data) // 2
            eu_group_idx = group_idx - 3  # 0 or 1
            start_idx = eu_group_idx * chunk_size
            end_idx = start_idx + chunk_size if eu_group_idx < 1 else len(eu_data)
            target_items = eu_data[start_idx:end_idx]

        # 해당 그룹 데이터에서만 검색
        for keyword in query_keywords:
            for source, item in self.search_index.get(keyword, []):
                if source == target_source and item in target_items:
                    results[(source, str(item))] += 1

        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
        return [
            {'source': source, 'item': eval(item_str)}
            for (source, item_str), _ in sorted_results[:max_results]
        ]

    def get_overseas_context_group(self, query: str, group_idx: int) -> str:
        """해외 HS 분류 관련 컨텍스트(그룹별)를 생성하는 메서드"""
        results = self.search_overseas_group(query, group_idx)
        context = []

        for result in results:
            # 출처에 따라 국가 구분
            if result['source'] == 'hs_classification_data_us':
                country = "미국 관세청"
            elif result['source'] == 'hs_classification_data_eu':
                country = "EU 관세청"
            else:
                country = "해외 관세청"

            context.append(f"출처: {result['source']} ({country})\n항목: {json.dumps(result['item'], ensure_ascii=False)}")

        return "\n\n".join(context)

    def search_domestic(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """국내 HS 분류 데이터에서만 검색하는 메서드"""
        query_keywords = self._extract_keywords(query)
        results = defaultdict(int)

        # 국내 데이터 소스만 필터링
        domestic_sources = [
            'HS분류사례_part1', 'HS분류사례_part2', 'HS분류사례_part3', 'HS분류사례_part4', 'HS분류사례_part5',
            'HS분류사례_part6', 'HS분류사례_part7', 'HS분류사례_part8', 'HS분류사례_part9', 'HS분류사례_part10',
            'knowledge/HS위원회', 'knowledge/HS협의회'
        ]

        for keyword in query_keywords:
            for source, item in self.search_index.get(keyword, []):
                # 국내 데이터 소스만 포함
                if source in domestic_sources:
                    results[(source, str(item))] += 1

        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

        return [
            {'source': source, 'item': eval(item_str)}
            for (source, item_str), _ in sorted_results[:max_results]
        ]

    def get_domestic_context(self, query: str) -> str:
        """국내 HS 분류 관련 컨텍스트를 생성하는 메서드"""
        results = self.search_domestic(query)
        context = []

        for result in results:
            context.append(f"출처: {result['source']} (국내 관세청)\n항목: {json.dumps(result['item'], ensure_ascii=False)}")

        return "\n\n".join(context)


    def get_relevant_context(self, query: str) -> str:
        """
        쿼리에 관련된 컨텍스트를 생성하는 메서드
        Args:
            query: 컨텍스트를 생성할 쿼리 문자열
        Returns:
            관련 컨텍스트 문자열 (출처와 항목 정보 포함)
        """
        results = self.search(query)
        context = []

        for result in results:
            context.append(f"출처: {result['source']}\n항목: {json.dumps(result['item'], ensure_ascii=False)}")

        return "\n\n".join(context)

    def _load_tfidf_indexes(self):
        """
        사전에 구축된 TF-IDF 인덱스를 pickle 파일에서 로드
        """
        try:
            # gzip 압축 파일 우선 시도
            if os.path.exists('tfidf_indexes.pkl.gz'):
                with gzip.open('tfidf_indexes.pkl.gz', 'rb') as f:
                    indexes = pickle.load(f)
            # 기존 비압축 파일 호환성 유지
            elif os.path.exists('tfidf_indexes.pkl'):
                with open('tfidf_indexes.pkl', 'rb') as f:
                    indexes = pickle.load(f)
            else:
                raise FileNotFoundError("No index file found")

            self.domestic_tfidf = indexes['domestic_tfidf']
            self.domestic_items = indexes['domestic_items']
            self.overseas_tfidf = indexes['overseas_tfidf']
            self.overseas_items = indexes['overseas_items']
        except Exception as e:
            print(f"Warning: Failed to load TF-IDF indexes: {e}")
            print("Building indexes from scratch...")
            self.build_tfidf_indexes()

    def build_tfidf_indexes(self):
        """
        TF-IDF 검색 인덱스 구축
        - 국내 사례, 해외 사례, HS 매뉴얼 각각 별도 인덱스 구축
        """
        # 1. 국내 HS 분류 사례 인덱스
        domestic_docs = []
        domestic_items = []

        for i in range(1, 11):
            key = f'HS분류사례_part{i}'
            if key in self.data:
                for item in self.data[key]:
                    # 문서 텍스트 생성
                    text_parts = []
                    if 'product_name' in item and item['product_name']:
                        text_parts.append(item['product_name'])
                    if 'description' in item and item['description']:
                        text_parts.append(item['description'])
                    if 'decision_reason' in item and item['decision_reason']:
                        text_parts.append(item['decision_reason'])

                    text = " ".join(text_parts)
                    if text.strip():
                        domestic_docs.append(text)
                        domestic_items.append(item)

        # 위원회, 협의회 데이터 추가
        for key in ['knowledge/HS위원회', 'knowledge/HS협의회']:
            if key in self.data:
                for item in self.data[key]:
                    text_parts = []
                    if 'product_name' in item and item['product_name']:
                        text_parts.append(item['product_name'])
                    if 'description' in item and item['description']:
                        text_parts.append(item['description'])
                    if 'decision_reason' in item and item['decision_reason']:
                        text_parts.append(item['decision_reason'])

                    text = " ".join(text_parts)
                    if text.strip():
                        domestic_docs.append(text)
                        domestic_items.append(item)

        if domestic_docs:
            self.domestic_tfidf = TfidfSearchEngine()
            self.domestic_tfidf.fit(domestic_docs)
            self.domestic_items = domestic_items

        # 2. 해외 HS 분류 사례 인덱스
        overseas_docs = []
        overseas_items = []

        for key in ['hs_classification_data_us', 'hs_classification_data_eu']:
            if key in self.data:
                for item in self.data[key]:
                    text_parts = []
                    if 'product_name' in item and item['product_name']:
                        text_parts.append(item['product_name'])
                    if 'description' in item and item['description']:
                        text_parts.append(item['description'])
                    if 'decision_reason' in item and item['decision_reason']:
                        text_parts.append(item['decision_reason'])

                    text = " ".join(text_parts)
                    if text.strip():
                        overseas_docs.append(text)
                        overseas_items.append(item)

        if overseas_docs:
            self.overseas_tfidf = TfidfSearchEngine()
            self.overseas_tfidf.fit(overseas_docs)
            self.overseas_items = overseas_items

        # 3. 구축한 인덱스를 gzip 압축하여 pickle 파일로 저장
        try:
            with gzip.open('tfidf_indexes.pkl.gz', 'wb', compresslevel=9) as f:
                pickle.dump({
                    'domestic_tfidf': self.domestic_tfidf,
                    'domestic_items': self.domestic_items,
                    'overseas_tfidf': self.overseas_tfidf,
                    'overseas_items': self.overseas_items
                }, f, protocol=pickle.HIGHEST_PROTOCOL)

            # 파일 크기 정보 출력
            size_mb = os.path.getsize('tfidf_indexes.pkl.gz') / (1024 * 1024)
            print(f"✅ TF-IDF 인덱스가 tfidf_indexes.pkl.gz 파일로 저장되었습니다.")
            print(f"   파일 크기: {size_mb:.2f} MB")
        except Exception as e:
            print(f"⚠️ TF-IDF 인덱스 저장 실패: {e}")

    def search_domestic_tfidf(self, query: str, top_k: int = 100, min_similarity: float = 0.1) -> List[Dict[str, Any]]:
        """
        TF-IDF 기반 국내 HS 분류 사례 검색

        Args:
            query: 검색 쿼리
            top_k: 반환할 상위 결과 개수
            min_similarity: 최소 유사도 임계값 (기본값 0.1)

        Returns:
            검색된 항목 리스트
        """
        if self.domestic_tfidf is None:
            return []

        results = self.domestic_tfidf.search(query, top_k, min_similarity)
        return [self.domestic_items[idx] for idx, score in results]

    def search_overseas_tfidf(self, query: str, top_k: int = 100, min_similarity: float = 0.1) -> List[Dict[str, Any]]:
        """
        TF-IDF 기반 해외 HS 분류 사례 검색

        Args:
            query: 검색 쿼리
            top_k: 반환할 상위 결과 개수
            min_similarity: 최소 유사도 임계값 (기본값 0.1)

        Returns:
            검색된 항목 리스트
        """
        if self.overseas_tfidf is None:
            return []

        results = self.overseas_tfidf.search(query, top_k, min_similarity)
        return [self.overseas_items[idx] for idx, score in results]

    def search_domestic_by_keyword(self, keyword: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        키워드 기반 단순 문자열 포함 검색 (국내 분류사례)

        Args:
            keyword: 검색할 키워드
            top_k: 반환할 최대 결과 개수

        Returns:
            키워드가 포함된 사례 리스트
        """
        domestic_sources = [
            'HS분류사례_part1', 'HS분류사례_part2', 'HS분류사례_part3', 'HS분류사례_part4', 'HS분류사례_part5',
            'HS분류사례_part6', 'HS분류사례_part7', 'HS분류사례_part8', 'HS분류사례_part9', 'HS분류사례_part10',
            'knowledge/HS위원회', 'knowledge/HS협의회'
        ]

        results = []
        keyword_lower = keyword.lower()

        for source in domestic_sources:
            if source in self.data:
                for item in self.data[source]:
                    # 품목명, 설명, 분류근거에서 키워드 검색 (대소문자 구분 없음)
                    searchable_text = ' '.join([
                        str(item.get('product_name', '')),
                        str(item.get('description', '')),
                        str(item.get('decision_reason', ''))
                    ]).lower()

                    if keyword_lower in searchable_text:
                        results.append(item)
                        if len(results) >= top_k:
                            return results

        return results

    def find_domestic_case_by_id(self, ref_id: str) -> Dict[str, Any]:
        """
        참고문서번호로 국내 분류사례 검색

        Args:
            ref_id: 참고문서번호 (예: "품목분류2과-9433")

        Returns:
            해당 사례 딕셔너리 또는 None
        """
        domestic_sources = [
            'HS분류사례_part1', 'HS분류사례_part2', 'HS분류사례_part3', 'HS분류사례_part4', 'HS분류사례_part5',
            'HS분류사례_part6', 'HS분류사례_part7', 'HS분류사례_part8', 'HS분류사례_part9', 'HS분류사례_part10',
            'knowledge/HS위원회', 'knowledge/HS협의회'
        ]

        for source in domestic_sources:
            if source in self.data:
                for item in self.data[source]:
                    if item.get('reference_id') == ref_id:
                        return item
        return None

    def search_overseas_by_keyword(self, keyword: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        키워드 기반 단순 문자열 포함 검색 (해외 분류사례)

        Args:
            keyword: 검색할 키워드
            top_k: 반환할 최대 결과 개수

        Returns:
            키워드가 포함된 사례 리스트 (국가 정보 포함)
        """
        results = []
        keyword_lower = keyword.lower()

        for source in ['hs_classification_data_us', 'hs_classification_data_eu']:
            if source in self.data:
                country = 'US' if 'us' in source else 'EU'
                for item in self.data[source]:
                    # 품목명, 설명, 분류근거에서 키워드 검색 (대소문자 구분 없음)
                    searchable_text = ' '.join([
                        str(item.get('product_name', '')),
                        str(item.get('description', '')),
                        str(item.get('reply', ''))
                    ]).lower()

                    if keyword_lower in searchable_text:
                        results.append(item)
                        if len(results) >= top_k:
                            return results

        return results

    def find_overseas_case_by_id(self, ref_id: str) -> Dict[str, Any]:
        """
        참고문서번호로 해외 분류사례 검색

        Args:
            ref_id: 참고문서번호 (예: "NY N338825")

        Returns:
            {'case': 사례 딕셔너리, 'country': 'US'/'EU'} 또는 None
        """
        for source in ['hs_classification_data_us', 'hs_classification_data_eu']:
            if source in self.data:
                for item in self.data[source]:
                    if item.get('reference_id') == ref_id:
                        country = 'US' if 'us' in source else 'EU'
                        return {'case': item, 'country': country}
        return None

    def search_overseas_by_hs_code(self, hs_code: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        HS 코드로 해외 분류사례 검색

        Args:
            hs_code: HS 코드 (4-10자리, 예: "5515", "5515.12")
            top_k: 반환할 최대 결과 개수

        Returns:
            매칭되는 사례 리스트 (국가 정보 포함)
        """
        results = []

        for source in ['hs_classification_data_us', 'hs_classification_data_eu']:
            if source in self.data:
                country = 'US' if 'us' in source else 'EU'
                for item in self.data[source]:
                    item_hs_code = item.get('hs_code', '')
                    # HS 코드 부분 매칭 (공백, 점, 하이픈 제거 후 비교)
                    if hs_code.replace('.', '').replace(' ', '') in item_hs_code.replace('.', '').replace(' ', '').replace('-', ''):
                        results.append({
                            'case': item,
                            'country': country
                        })
                        if len(results) >= top_k:
                            return results
        return results