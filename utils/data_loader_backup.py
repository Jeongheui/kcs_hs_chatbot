import json
import re
import os
import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
from dotenv import load_dotenv
from google import genai

# 환경 변수 로드
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

class HSDataManager:
    """
    HS 코드 관련 데이터를 관리하는 클래스
    - HS 분류 사례, 위원회 결정, 협의회 결정 등의 데이터를 로드하고 관리
    - 키워드 기반 검색 및 임베딩 기반 semantic search 지원
    - 관련 컨텍스트 생성 기능 제공
    """

    def __init__(self):
        """HSDataManager 초기화"""
        self.data = {}  # 모든 HS 관련 데이터를 저장하는 딕셔너리
        self.items_by_id = {}  # ID 기반 데이터 참조 (eval() 제거)
        self.search_index = defaultdict(list)  # 키워드 기반 검색을 위한 인덱스

        # 임베딩 관련
        self.client = genai.Client(api_key=GOOGLE_API_KEY) if GOOGLE_API_KEY else None
        self.embeddings_cache = {}  # {item_id: embedding_vector}
        self.embedding_model = 'text-embedding-004'
        self.embedding_cache_dir = 'knowledge/embeddings'

        # 데이터 로드
        self.load_all_data()
        self.build_search_index()

        # 임베딩 캐시 로드 시도
        self.load_embedding_cache()

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
                    self.data[file.replace('knowledge/', '').replace('.json', '')] = json.load(f)
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
        검색 인덱스 구축 메서드 (ID 기반)
        - 각 데이터 항목에서 키워드를 추출
        - 추출된 키워드를 인덱스에 저장하여 빠른 검색 가능
        - eval() 제거: ID 기반 참조 시스템 사용
        """
        item_counter = 0
        for source, items in self.data.items():
            for idx, item in enumerate(items):
                # 고유 ID 생성
                item_id = f"{source}_{idx}"

                # ID로 아이템 저장 (eval() 제거)
                self.items_by_id[item_id] = {
                    'source': source,
                    'item': item
                }

                # 품목명에서 키워드 추출
                keywords = self._extract_keywords(str(item))

                # 각 키워드에 대해 해당 아이템 ID 저장
                for keyword in keywords:
                    self.search_index[keyword].append(item_id)

                item_counter += 1

        print(f"Search index built: {item_counter} items indexed")

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

    # ========== 임베딩 시스템 ==========

    def load_embedding_cache(self):
        """저장된 임베딩 캐시 로드"""
        if not os.path.exists(self.embedding_cache_dir):
            print(f"Embedding cache directory not found: {self.embedding_cache_dir}")
            return

        cache_files = [
            'domestic_embeddings.json',
            'overseas_embeddings.json'
        ]

        for cache_file in cache_files:
            cache_path = os.path.join(self.embedding_cache_dir, cache_file)
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, 'r', encoding='utf-8') as f:
                        cached_data = json.load(f)
                        # numpy 배열로 변환
                        for item_id, embedding_list in cached_data.items():
                            self.embeddings_cache[item_id] = np.array(embedding_list)
                    print(f"Loaded {len(cached_data)} embeddings from {cache_file}")
                except Exception as e:
                    print(f"Error loading embedding cache from {cache_file}: {e}")

    def save_embedding_cache(self):
        """임베딩 캐시를 파일로 저장"""
        if not os.path.exists(self.embedding_cache_dir):
            os.makedirs(self.embedding_cache_dir)

        # 국내/해외 데이터 분리
        domestic_sources = [
            'HS분류사례_part1', 'HS분류사례_part2', 'HS분류사례_part3', 'HS분류사례_part4', 'HS분류사례_part5',
            'HS분류사례_part6', 'HS분류사례_part7', 'HS분류사례_part8', 'HS분류사례_part9', 'HS분류사례_part10',
            'HS위원회', 'HS협의회'
        ]

        domestic_embeddings = {}
        overseas_embeddings = {}

        for item_id, embedding in self.embeddings_cache.items():
            source = item_id.split('_')[0]
            embedding_list = embedding.tolist()

            if source in domestic_sources:
                domestic_embeddings[item_id] = embedding_list
            else:
                overseas_embeddings[item_id] = embedding_list

        # 저장
        if domestic_embeddings:
            cache_path = os.path.join(self.embedding_cache_dir, 'domestic_embeddings.json')
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(domestic_embeddings, f, ensure_ascii=False, indent=2)
            print(f"Saved {len(domestic_embeddings)} domestic embeddings")

        if overseas_embeddings:
            cache_path = os.path.join(self.embedding_cache_dir, 'overseas_embeddings.json')
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(overseas_embeddings, f, ensure_ascii=False, indent=2)
            print(f"Saved {len(overseas_embeddings)} overseas embeddings")

    def generate_embeddings_batch(self, texts: List[str], batch_size: int = 250) -> List[np.ndarray]:
        """
        배치로 임베딩 생성 (Rate limit 준수)
        Args:
            texts: 임베딩을 생성할 텍스트 리스트
            batch_size: 배치 크기 (최대 250)
        Returns:
            임베딩 벡터 리스트
        """
        if not self.client:
            raise ValueError("Gemini client not initialized. Check GOOGLE_API_KEY.")

        embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_num = i // batch_size + 1

            try:
                print(f"Generating embeddings batch {batch_num}/{total_batches} ({len(batch)} items)...")

                response = self.client.models.embed_content(
                    model=self.embedding_model,
                    contents=batch
                )

                # 응답에서 임베딩 추출
                for embedding_data in response.embeddings:
                    embeddings.append(np.array(embedding_data.values))

                # Rate limit 준수: RPM 100 제한
                if batch_num < total_batches:
                    time.sleep(0.7)  # 약 0.7초 대기 (분당 85회 정도)

            except Exception as e:
                print(f"Error generating embeddings for batch {batch_num}: {e}")
                # 실패 시 빈 벡터 추가
                for _ in batch:
                    embeddings.append(np.zeros(768))  # text-embedding-004는 768차원

        return embeddings

    def initialize_all_embeddings(self, force_regenerate: bool = False):
        """
        모든 데이터에 대한 임베딩 생성 및 캐시
        Args:
            force_regenerate: True면 기존 캐시 무시하고 재생성
        """
        if not force_regenerate and len(self.embeddings_cache) > 0:
            print(f"Embeddings already cached: {len(self.embeddings_cache)} items")
            return

        print("Starting embedding initialization...")
        print(f"Total items to embed: {len(self.items_by_id)}")

        # 임베딩 생성할 텍스트 수집
        item_ids = []
        texts = []

        for item_id, data in self.items_by_id.items():
            # 캐시에 이미 있으면 스킵 (force_regenerate=False인 경우)
            if not force_regenerate and item_id in self.embeddings_cache:
                continue

            item_ids.append(item_id)
            # 아이템 전체를 JSON 문자열로 변환하여 임베딩 생성
            texts.append(json.dumps(data['item'], ensure_ascii=False))

        if not texts:
            print("No new items to embed.")
            return

        print(f"Generating embeddings for {len(texts)} new items...")

        # 배치 임베딩 생성
        embeddings = self.generate_embeddings_batch(texts)

        # 캐시에 저장
        for item_id, embedding in zip(item_ids, embeddings):
            self.embeddings_cache[item_id] = embedding

        # 파일로 저장
        self.save_embedding_cache()

        print(f"Embedding initialization complete: {len(self.embeddings_cache)} total embeddings cached")

    def get_query_embedding(self, query: str) -> np.ndarray:
        """쿼리에 대한 임베딩 생성"""
        if not self.client:
            raise ValueError("Gemini client not initialized. Check GOOGLE_API_KEY.")

        try:
            response = self.client.models.embed_content(
                model=self.embedding_model,
                contents=query
            )
            return np.array(response.embeddings[0].values)
        except Exception as e:
            print(f"Error generating query embedding: {e}")
            return np.zeros(768)

    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """코사인 유사도 계산"""
        if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
            return 0.0
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def search_by_embedding(self, query: str, max_results: int = 5,
                           source_filter: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        임베딩 기반 semantic search
        Args:
            query: 검색 쿼리
            max_results: 반환할 최대 결과 수
            source_filter: 검색할 소스 필터링 (예: ['HS분류사례_part1', 'HS분류사례_part2'])
        Returns:
            검색 결과 리스트 (유사도 순)
        """
        if len(self.embeddings_cache) == 0:
            print("Warning: No embeddings cached. Run initialize_all_embeddings() first.")
            return []

        # 쿼리 임베딩 생성
        query_embedding = self.get_query_embedding(query)

        # 모든 캐시된 임베딩과 유사도 계산
        similarities = []

        for item_id, item_embedding in self.embeddings_cache.items():
            # 소스 필터링
            if source_filter:
                source = item_id.split('_')[0]
                if source not in source_filter:
                    continue

            similarity = self.cosine_similarity(query_embedding, item_embedding)
            similarities.append((item_id, similarity))

        # 유사도 순으로 정렬
        similarities.sort(key=lambda x: x[1], reverse=True)

        # 상위 결과 반환
        results = []
        for item_id, similarity in similarities[:max_results]:
            item_data = self.items_by_id.get(item_id)
            if item_data:
                results.append({
                    'source': item_data['source'],
                    'item': item_data['item'],
                    'score': float(similarity),
                    'item_id': item_id
                })

        return results

    def search_hybrid(self, query: str, max_results: int = 5,
                     keyword_weight: float = 0.3, embedding_weight: float = 0.7,
                     source_filter: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        하이브리드 검색 (키워드 + 임베딩)
        Args:
            query: 검색 쿼리
            max_results: 반환할 최대 결과 수
            keyword_weight: 키워드 검색 가중치 (0-1)
            embedding_weight: 임베딩 검색 가중치 (0-1)
            source_filter: 검색할 소스 필터링
        Returns:
            검색 결과 리스트 (통합 점수 순)
        """
        # 키워드 검색 수행
        keyword_results = self._search_keyword(query, max_results=max_results*2, source_filter=source_filter)

        # 임베딩 검색 수행 (캐시가 있는 경우)
        if len(self.embeddings_cache) > 0:
            embedding_results = self.search_by_embedding(query, max_results=max_results*2, source_filter=source_filter)
        else:
            embedding_results = []

        # 점수 통합
        combined_scores = defaultdict(float)
        all_items = {}

        # 키워드 검색 점수 (정규화)
        max_keyword_score = max([r['score'] for r in keyword_results]) if keyword_results else 1.0
        for result in keyword_results:
            item_id = result.get('item_id', f"{result['source']}_unknown")
            normalized_score = result['score'] / max_keyword_score if max_keyword_score > 0 else 0
            combined_scores[item_id] += normalized_score * keyword_weight
            all_items[item_id] = result

        # 임베딩 검색 점수 (이미 0-1 범위)
        for result in embedding_results:
            item_id = result['item_id']
            combined_scores[item_id] += result['score'] * embedding_weight
            all_items[item_id] = result

        # 통합 점수로 정렬
        sorted_items = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

        # 상위 결과 반환
        results = []
        for item_id, combined_score in sorted_items[:max_results]:
            item_data = all_items[item_id]
            item_data['score'] = float(combined_score)
            results.append(item_data)

        return results

    # ========== 기존 키워드 검색 메서드 (ID 기반으로 수정) ==========

    def _search_keyword(self, query: str, max_results: int = 5,
                       source_filter: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        키워드 기반 검색 (내부 메서드)
        Args:
            query: 검색할 쿼리 문자열
            max_results: 반환할 최대 결과 수
            source_filter: 검색할 소스 필터링
        Returns:
            검색 결과 리스트 (출처와 항목 정보 포함)
        """
        query_keywords = self._extract_keywords(query)
        results = defaultdict(int)

        # 각 키워드에 대해 매칭되는 항목 찾기
        for keyword in query_keywords:
            for item_id in self.search_index.get(keyword, []):
                # 소스 필터링
                if source_filter:
                    source = item_id.split('_')[0]
                    if source not in source_filter:
                        continue

                # 가중치 계산 (키워드 매칭 횟수 기반)
                results[item_id] += 1

        # 가중치 기준 정렬
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

        # 상위 결과만 반환 (ID 기반)
        return [
            {
                'source': self.items_by_id[item_id]['source'],
                'item': self.items_by_id[item_id]['item'],
                'score': score,
                'item_id': item_id
            }
            for item_id, score in sorted_results[:max_results]
        ]

    def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        쿼리와 관련된 가장 연관성 높은 항목들을 검색하는 메서드
        - 임베딩이 캐시되어 있으면 하이브리드 검색
        - 없으면 키워드 검색
        Args:
            query: 검색할 쿼리 문자열
            max_results: 반환할 최대 결과 수 (기본값: 5)
        Returns:
            검색 결과 리스트 (출처와 항목 정보 포함)
        """
        if len(self.embeddings_cache) > 0:
            return self.search_hybrid(query, max_results=max_results)
        else:
            return self._search_keyword(query, max_results=max_results)

    def search_domestic_group(self, query: str, group_idx: int, max_results: int = 3) -> List[Dict[str, Any]]:
        """
        국내 HS 분류 데이터 그룹별 검색 메서드
        - 임베딩이 있으면 하이브리드 검색, 없으면 키워드 검색
        """
        # 그룹별 데이터 소스 정의 (5개 그룹)
        group_sources = [
            ['HS분류사례_part1', 'HS분류사례_part2'],  # 그룹1
            ['HS분류사례_part3', 'HS분류사례_part4'],  # 그룹2
            ['HS분류사례_part5', 'HS분류사례_part6'],  # 그룹3
            ['HS분류사례_part7', 'HS분류사례_part8'],  # 그룹4
            ['HS분류사례_part9', 'HS분류사례_part10', 'HS위원회', 'HS협의회']  # 그룹5
        ]
        sources = group_sources[group_idx]

        if len(self.embeddings_cache) > 0:
            return self.search_hybrid(query, max_results=max_results, source_filter=sources)
        else:
            return self._search_keyword(query, max_results=max_results, source_filter=sources)

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
        - 임베딩이 있으면 하이브리드 검색, 없으면 키워드 검색
        """
        # 해외 데이터를 그룹별로 분할 처리
        if group_idx < 3:  # 그룹 0,1,2는 미국 데이터
            target_source = 'hs_classification_data_us'
        else:  # 그룹 3,4는 EU 데이터
            target_source = 'hs_classification_data_eu'

        # 전체 검색 후 그룹별 필터링 (간단한 방식)
        if len(self.embeddings_cache) > 0:
            all_results = self.search_hybrid(query, max_results=max_results*3, source_filter=[target_source])
        else:
            all_results = self._search_keyword(query, max_results=max_results*3, source_filter=[target_source])

        # 그룹별 청크 인덱스 계산
        source_data = self.data.get(target_source, [])
        if not source_data:
            return []

        if group_idx < 3:  # 미국 데이터 3등분
            chunk_size = len(source_data) // 3
            local_group_idx = group_idx
        else:  # EU 데이터 2등분
            chunk_size = len(source_data) // 2
            local_group_idx = group_idx - 3

        start_idx = local_group_idx * chunk_size
        end_idx = start_idx + chunk_size if local_group_idx < (2 if group_idx < 3 else 1) else len(source_data)

        # 해당 그룹 범위의 결과만 필터링
        filtered_results = []
        for result in all_results:
            # item_id에서 인덱스 추출
            item_id = result.get('item_id', '')
            if '_' in item_id:
                try:
                    idx = int(item_id.split('_')[-1])
                    if start_idx <= idx < end_idx:
                        filtered_results.append(result)
                except ValueError:
                    continue

            if len(filtered_results) >= max_results:
                break

        return filtered_results[:max_results]

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
        - 임베딩이 있으면 하이브리드 검색, 없으면 키워드 검색
        """
        # 국내 데이터 소스만 필터링
        domestic_sources = [
            'HS분류사례_part1', 'HS분류사례_part2', 'HS분류사례_part3', 'HS분류사례_part4', 'HS분류사례_part5',
            'HS분류사례_part6', 'HS분류사례_part7', 'HS분류사례_part8', 'HS분류사례_part9', 'HS분류사례_part10',
            'HS위원회', 'HS협의회'
        ]

        if len(self.embeddings_cache) > 0:
            return self.search_hybrid(query, max_results=max_results, source_filter=domestic_sources)
        else:
            return self._search_keyword(query, max_results=max_results, source_filter=domestic_sources)

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