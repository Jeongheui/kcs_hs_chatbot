"""
utils package - HS 품목분류 챗봇 유틸리티 모듈

이 패키지는 기존 utils.py를 기능별로 분할한 모듈들로 구성되어 있습니다.
- text_utils: 텍스트 처리 유틸리티
- question_classifier: LLM 기반 질문 분류
- hs_manual_utils: HS 해설서 관련 함수들
- search_engines: 품목분류표 및 해설서 검색 엔진 (키워드 기반)
- data_loader: HSDataManager 클래스 (임베딩 적용 예정)
- handlers: 질문 유형별 처리 함수들

backward compatibility를 위해 모든 함수와 클래스를 이 __init__.py에서 export합니다.
"""

# Data loader (임베딩 적용 대상)
from .data_loader import HSDataManager

# Search engines (키워드 기반 유지)
from .search_engines import TariffTableSearcher, ParallelHSSearcher

# Text utilities
from .text_utils import clean_text, extract_hs_codes

# HS manual utilities
from .hs_manual_utils import (
    lookup_hscode,
    get_hs_explanations,
    get_tariff_info_for_codes,
    get_manual_info_for_codes,
    prepare_general_rules,
    analyze_user_provided_codes
)

# Question classifier
from .question_classifier import classify_question

# Handlers
from .handlers import (
    handle_web_search,
    handle_hs_classification_cases,
    handle_overseas_hs,
    handle_hs_manual_with_user_codes,
    handle_hs_manual_with_parallel_search
)

__all__ = [
    # Classes
    'HSDataManager',
    'TariffTableSearcher',
    'ParallelHSSearcher',

    # Text utils
    'clean_text',
    'extract_hs_codes',

    # HS manual utils
    'lookup_hscode',
    'get_hs_explanations',
    'get_tariff_info_for_codes',
    'get_manual_info_for_codes',
    'prepare_general_rules',
    'analyze_user_provided_codes',

    # Classification
    'classify_question',

    # Handlers
    'handle_web_search',
    'handle_hs_classification_cases',
    'handle_overseas_hs',
    'handle_hs_manual_with_user_codes',
    'handle_hs_manual_with_parallel_search'
]

__version__ = '2.0.0'
__author__ = 'HS Chatbot Team'