import json
import re
from typing import Dict, List, Any
from collections import defaultdict

class HSDataManager:
    """
    HS 코드 관련 데이터를 관리하는 클래스
    - HS 분류 사례, 위원회 결정, 협의회 결정 등의 데이터를 로드하고 관리
    - 키워드 기반 검색 기능 제공 (향후 임베딩 기반으로 교체 예정)
    - 관련 컨텍스트 생성 기능 제공
    """

    def __init__(self):
        """HSDataManager 초기화"""
        self.data = {}  # 모든 HS 관련 데이터를 저장하는 딕셔너리
        self.search_index = defaultdict(list)  # 키워드 기반 검색을 위한 인덱스
        self.load_all_data()  # 모든 데이터 파일 로드
        self.build_search_index()  # 검색 인덱스 구축

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

        NOTE: 향후 임베딩 기반 인덱스로 교체 예정
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

        NOTE: 임베딩 방식으로 교체 후에도 백업용으로 유지
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

        NOTE: 향후 임베딩 기반 semantic search로 교체 예정
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
        """
        국내 HS 분류 데이터 그룹별 검색 메서드

        NOTE: 이 메서드는 임베딩 기반 semantic search로 교체 예정
        """
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
        """
        해외 HS 분류 데이터 그룹별 검색 메서드

        NOTE: 이 메서드는 임베딩 기반 semantic search로 교체 예정
        """
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
        """
        국내 HS 분류 데이터에서만 검색하는 메서드

        NOTE: 향후 임베딩 기반 semantic search로 교체 예정
        """
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