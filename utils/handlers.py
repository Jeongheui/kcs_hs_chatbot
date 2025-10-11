import streamlit as st
import time
import os
import json
import re
from datetime import datetime
from google import genai
from google.genai import types
from google.genai.errors import APIError
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

from .text_utils import clean_text, extract_hs_codes
from .hs_manual_utils import (
    get_tariff_info_for_codes,
    get_manual_info_for_codes,
    prepare_general_rules,
    analyze_user_provided_codes
)
from .search_engines import ParallelHSSearcher

# 환경 변수 로드
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
client = genai.Client(api_key=GOOGLE_API_KEY)

# prompts.py에서 프롬프트 import
from prompts import DOMESTIC_CONTEXT, OVERSEAS_CONTEXT


# ==================== 유틸리티 함수 ====================

def highlight_keywords(text, keywords):
    """텍스트에서 키워드를 형광색으로 하이라이트 (토큰 기반)"""
    if not text or not keywords:
        return text

    # 키워드가 문자열이면 공백으로 분리하여 토큰화
    if isinstance(keywords, str):
        # 특수문자 제거 및 공백 기준 분리
        keywords = re.sub(r'[^\w\s]', ' ', keywords).split()
        # 길이 2 이상인 토큰만 사용
        keywords = [kw.strip() for kw in keywords if len(kw.strip()) >= 2]

    if not keywords:
        return text

    result = text
    # 각 토큰을 개별적으로 하이라이트
    for keyword in keywords:
        if not keyword or len(keyword.strip()) < 2:  # 너무 짧은 키워드는 스킵
            continue
        # 대소문자 구분 없이 치환 (re.IGNORECASE)
        pattern = re.compile(re.escape(keyword), re.IGNORECASE)
        result = pattern.sub(lambda m: f'<mark>{m.group()}</mark>', result)

    return result


# ==================== Multi-Agent 공통 로직 ====================

def _process_single_group(group_id, group_cases, context_prompt, user_input, analysis_type):
    """단일 그룹 처리 함수 (병렬 실행용)"""
    try:
        # 그룹 데이터를 컨텍스트로 변환
        source_label = "국내 관세청" if analysis_type == 'domestic' else "해외 관세청"
        relevant = "\n\n".join([
            f"출처: {source_label}\n항목: {json.dumps(case, ensure_ascii=False)}"
            for case in group_cases
        ])

        prompt = f"{context_prompt}\n\n관련 데이터 ({source_label}, 그룹{group_id+1}):\n{relevant}\n\n사용자: {user_input}\n"

        start_time = datetime.now()
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()

        answer = clean_text(response.text)
        return group_id, answer, start_time, processing_time
    except Exception as e:
        error_msg = f"그룹 {group_id+1} 분석 중 오류 발생: {str(e)}"
        return group_id, error_msg, datetime.now(), 0.0


def _run_group_parallel_analysis(groups, context_prompt, user_input, analysis_type, ui_container=None):
    """5개 그룹을 병렬로 분석하는 공통 함수"""

    # UI 초기화
    if ui_container:
        progress_bar = ui_container.progress(0, text="병렬 AI 분석 시작...")
        responses_container = ui_container.container()

    # 병렬 처리
    results = {}
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(_process_single_group, i, groups[i], context_prompt, user_input, analysis_type)
            for i in range(5)
        ]

        for future in as_completed(futures):
            group_id, answer, start_time, processing_time = future.result()
            results[group_id] = answer

            # session_state에 결과 저장
            if ui_container:
                analysis_result = {
                    'type': analysis_type,
                    'group_id': group_id,
                    'answer': answer,
                    'start_time': start_time.strftime('%H:%M:%S'),
                    'processing_time': processing_time
                }
                st.session_state.ai_analysis_results.append(analysis_result)

                # 실시간 UI 업데이트
                with responses_container:
                    emoji = "🤖" if analysis_type == 'domestic' else "🌐"
                    st.success(f"{emoji} **그룹 {group_id+1} AI 분석 완료** ({processing_time:.1f}초)")
                    with st.container():
                        st.write(f"⏰ {start_time.strftime('%H:%M:%S')}")
                        st.markdown("**분석 결과:**")
                        st.info(answer)
                        st.divider()

                progress_bar.progress(len(results)/5, text=f"완료: {len(results)}/5 그룹")

    # 순서대로 정렬
    group_answers = [results[i] for i in range(5)]
    return group_answers


def _run_head_agent(group_answers, context_prompt, user_input, analysis_type, ui_container=None):
    """Head Agent가 5개 답변을 종합하는 함수"""

    if ui_container:
        ui_container.progress(1.0, text="Head AI 최종 분석 중...")
        ui_container.info("🧠 **Head AI가 모든 분석을 종합하는 중...**")

    try:
        analysis_label = "국내 HS 분류 사례" if analysis_type == 'domestic' else "해외 HS 분류 사례"
        head_prompt = f"{context_prompt}\n\n아래는 {analysis_label} 데이터 5개 그룹별 분석 결과입니다. 각 그룹의 답변을 종합하여 최종 전문가 답변을 작성하세요.\n\n"

        for idx, ans in enumerate(group_answers):
            head_prompt += f"[그룹{idx+1} 답변]\n{ans}\n\n"
        head_prompt += f"\n사용자: {user_input}\n"

        head_response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=head_prompt
        )
        final_answer = clean_text(head_response.text)
    except Exception as e:
        final_answer = f"Head AI 분석 중 오류가 발생했습니다: {str(e)}\n\n그룹별 분석 결과를 참고해주세요."
        if ui_container:
            ui_container.error(f"⚠️ Head AI 오류: {str(e)}")

    if ui_container:
        ui_container.progress(1.0, text="분석 완료!")
        ui_container.success("✅ **모든 AI 분석이 완료되었습니다**")
        ui_container.info("📋 **패널을 접고 아래에서 최종 답변을 확인하세요**")

    return final_answer


# ==================== 통합 Multi-Agent 핸들러 ====================

def handle_multi_agent_analysis(user_input, context, hs_manager, analysis_type, ui_container=None):
    """
    통합 Multi-Agent 분석 핸들러

    Args:
        user_input: 사용자 질문
        context: 대화 컨텍스트 (현재 미사용)
        hs_manager: HS 데이터 매니저
        analysis_type: 'domestic' 또는 'overseas'
        ui_container: Streamlit UI 컨테이너 (optional)

    Returns:
        최종 분석 결과 문자열
    """

    # 분석 타입에 따라 프롬프트 및 검색 엔진 선택
    if analysis_type == 'domestic':
        context_prompt = DOMESTIC_CONTEXT
        search_func = hs_manager.search_domestic_tfidf
        ui_message = "🔍 **국내 HS 분류사례 분석 시작**"
    elif analysis_type == 'overseas':
        context_prompt = OVERSEAS_CONTEXT
        search_func = hs_manager.search_overseas_tfidf
        ui_message = "🌍 **해외 HS 분류사례 분석 시작**"
    else:
        raise ValueError(f"Invalid analysis_type: {analysis_type}. Must be 'domestic' or 'overseas'.")

    # UI 초기화
    if ui_container:
        with ui_container:
            st.info(ui_message)

    # TF-IDF 기반 검색으로 상위 100개 사례 추출
    top_cases = search_func(user_input, top_k=100, min_similarity=0.05)

    # 5개 그룹으로 분할 (각 그룹 20개)
    group_size = len(top_cases) // 5
    groups = [top_cases[i*group_size:(i+1)*group_size if i < 4 else len(top_cases)] for i in range(5)]

    # 5개 그룹 병렬 분석
    group_answers = _run_group_parallel_analysis(groups, context_prompt, user_input, analysis_type, ui_container)

    # Head Agent 최종 종합
    final_answer = _run_head_agent(group_answers, context_prompt, user_input, analysis_type, ui_container)

    return final_answer


# ==================== 기존 인터페이스 유지 (래퍼 함수) ====================

def handle_hs_classification_cases(user_input, context, hs_manager, ui_container=None):
    """국내 HS 분류 사례 처리 (래퍼 함수)"""
    return handle_multi_agent_analysis(user_input, context, hs_manager, 'domestic', ui_container)


def handle_overseas_hs(user_input, context, hs_manager, ui_container=None):
    """해외 HS 분류 사례 처리 (래퍼 함수)"""
    return handle_multi_agent_analysis(user_input, context, hs_manager, 'overseas', ui_container)


# ==================== 기타 핸들러 함수 ====================

def handle_web_search(user_input, context, hs_manager):
    """웹 검색 처리 함수"""
    web_context = """당신은 HS 품목분류 전문가입니다.

사용자의 질문에 대해 최신 웹 정보를 검색하여 물품개요, 용도, 기술개발, 산업동향 등의 정보를 제공해주세요.
"""

    grounding_tool = types.Tool(google_search=types.GoogleSearch())
    config = types.GenerateContentConfig(tools=[grounding_tool])

    prompt = f"{web_context}\n\n사용자: {user_input}\n"

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config=config)

    return clean_text(response.text)


def handle_hs_manual_with_user_codes(user_input, context, hs_manager, logger, extracted_codes, ui_container=None):
    """사용자 제시 HS코드 기반 해설서 분석"""

    # UI 컨테이너가 제공된 경우 분석 과정 표시
    if ui_container:
        with ui_container:
            st.info("🔍 **사용자 제시 HS코드 분석 시작**")
            progress_bar = st.progress(0, text="HS코드 분석 중...")
            analysis_container = st.container()

    logger.log_actual("SUCCESS", f"Found {len(extracted_codes)} HS codes", f"{', '.join(extracted_codes)}")

    if ui_container:
        progress_bar.progress(0.2, text=f"{len(extracted_codes)}개 HS코드 발견...")
        with analysis_container:
            st.success(f"✅ **{len(extracted_codes)}개 HS코드 발견**: {', '.join(extracted_codes)}")

    # 2단계: 각 HS코드별 품목분류표 정보 수집
    logger.log_actual("INFO", "Collecting tariff table information...")
    tariff_info = get_tariff_info_for_codes(extracted_codes)

    if ui_container:
        progress_bar.progress(0.4, text="품목분류표 정보 수집 중...")

    # 3단계: 각 HS코드별 해설서 정보 수집 및 요약
    logger.log_actual("INFO", "Collecting and summarizing manual information...")
    manual_info = get_manual_info_for_codes(extracted_codes, logger)

    if ui_container:
        progress_bar.progress(0.6, text="해설서 정보 수집 및 요약 중...")

        # 수집된 정보 표시
        with analysis_container:
            st.markdown("### 📊 **HS코드별 상세 정보**")

            for code in extracted_codes:
                st.markdown(f"#### 🔢 **HS코드: {code}**")

                col1, col2 = st.columns([1, 1])
                with col1:
                    if code in tariff_info:
                        st.write(f"**📋 국문품명**: {tariff_info[code].get('korean_name', 'N/A')}")
                        st.write(f"**📋 영문품명**: {tariff_info[code].get('english_name', 'N/A')}")

                with col2:
                    if code in manual_info:
                        st.write(f"**📚 해설서**: 수집 완료")
                        if manual_info[code].get('summary_used'):
                            st.write(f"**🤖 요약**: 적용됨")

                st.divider()

    # 4단계: 통칙 준비
    logger.log_actual("INFO", "Preparing general rules...")
    general_rules = prepare_general_rules()

    if ui_container:
        progress_bar.progress(0.8, text="최종 AI 분석 준비 중...")

    # 5단계: 최종 AI 분석
    logger.log_actual("AI", "Starting final AI analysis...")
    final_answer = analyze_user_provided_codes(user_input, extracted_codes, tariff_info, manual_info, general_rules, context)

    if ui_container:
        progress_bar.progress(1.0, text="분석 완료!")
        st.success("🧠 **AI 전문가 분석이 완료되었습니다**")
        st.info("📋 **아래에서 최종 답변을 확인하세요**")

    logger.log_actual("SUCCESS", "User-provided codes analysis completed", f"{len(final_answer)} chars")
    return final_answer


def handle_domestic_case_lookup(user_input, hs_manager):
    """국내 분류사례 원문 검색 처리 함수"""

    # 1. 참고문서번호 직접 검색
    ref_pattern = r'품목분류\d+과-\d+'
    match = re.search(ref_pattern, user_input)

    if match:
        ref_id = match.group()
        case = hs_manager.find_domestic_case_by_id(ref_id)
        if case:
            # 참고문서번호 유효성 검증 (데이터 오류 필터링)
            if case.get('reference_id') and case['reference_id'] != '-1':
                return format_domestic_case_detail(case, query=ref_id)
            else:
                return f"⚠️ 참고문서번호 '{ref_id}'의 데이터에 문제가 있습니다.\n\n키워드 검색을 시도해주세요."
        else:
            return f"⚠️ 참고문서번호 '{ref_id}'에 해당하는 사례를 찾을 수 없습니다.\n\n다른 문서번호나 키워드로 다시 검색해주세요."

    # 2. 키워드 기반 단순 문자열 검색
    results = hs_manager.search_domestic_by_keyword(user_input, top_k=10)

    if not results:
        return f"""⚠️ **"{user_input}"에 대한 검색 결과가 없습니다**

**가능한 원인:**
- 해당 키워드가 포함된 분류사례가 데이터에 없습니다
- 검색어가 원문에 정확히 일치하지 않습니다

**검색 팁:**
- 품목명의 핵심 키워드 사용 (예: '섬유유연제', '폴리아미드')
- 영문 품목명 시도 (예: 'softening', 'polyamide')
- 더 짧고 일반적인 단어 사용 (예: '머그컵' → '컵', 'mug')
- 띄어쓰기 변경 시도 (예: '폴리아미드호스' → '폴리아미드 호스')

**다른 검색 방법:**
- **국내 분류사례 기반 HS 추천**: AI가 유사 사례를 분석하여 HS코드 추천 (TF-IDF 사용)
- **웹 검색**: 최신 정보 및 일반 품목 정보 검색"""

    return format_domestic_case_list(results, query=user_input)


def format_domestic_case_detail(case, query=None):
    """국내 사례 상세 포맷"""
    # 키워드 하이라이트 적용
    product_name = highlight_keywords(case.get('product_name', 'N/A'), query) if query else case.get('product_name', 'N/A')
    description = highlight_keywords(case.get('description', 'N/A'), query) if query else case.get('description', 'N/A')
    decision_reason = highlight_keywords(case.get('decision_reason', 'N/A'), query) if query else case.get('decision_reason', 'N/A')

    return f"""---
<div class="case-detail">

## 📋 국내 분류사례 상세 정보

<div class="info-table">

| 항목 | 내용 |
|------|------|
| 📄 **참고문서번호** | {case.get('reference_id', 'N/A')} |
| 📅 **결정일자** | {case.get('decision_date', 'N/A')} |
| 🏛️ **결정기관** | {case.get('organization', 'N/A')} |
| 🔢 **HS 코드** | {case.get('hs_code', 'N/A')} |

</div>

---

### 📦 품목명
{product_name}

---

### 📝 품목 설명
{description}

---

### ⚖️ 분류 근거
{decision_reason}

</div>
"""


def format_domestic_case_list(results, query):
    """국내 사례 목록 포맷 (Expander 방식)"""
    output = f"## 🔍 \"{query}\" 검색 결과 ({len(results)}건)\n\n"

    for idx, case in enumerate(results, 1):
        product_name = case.get('product_name', 'N/A')
        ref_id = case.get('reference_id', 'N/A')
        hs_code = case.get('hs_code', 'N/A')
        decision_date = case.get('decision_date', 'N/A')

        # 품목명이 너무 길면 자르기 (Expander 제목용)
        product_name_display = product_name[:60] + "..." if len(product_name) > 60 else product_name
        # 제목에도 하이라이트 적용
        product_name_display = highlight_keywords(product_name_display, query)

        # 카드형 Expander
        output += f"""<div class="case-card domestic">
<details>
<summary class="case-summary">
<span class="arrow">▶</span>
<span class="rank">{idx}위</span>
<span class="ref-id">{ref_id}</span>
<span class="hs-code">HS {hs_code}</span>
<span class="product-name">{product_name_display}</span>
<span class="date">{decision_date}</span>
</summary>

<div class="case-content">
"""

        # Expander 내용 (전체 상세 정보, 하이라이트 적용)
        output += format_domestic_case_detail(case, query=query)

        output += """</div>
</details>
</div>

"""

    output += "\n💡 **각 항목을 클릭하면 상세 정보를 확인할 수 있습니다.**"
    return output


def handle_overseas_case_lookup(user_input, hs_manager):
    """해외 분류사례 원문 검색 처리 함수"""

    # 1. 참고문서번호 검색 (미국/EU 패턴)
    us_pattern = r'(NY|HQ|LA|SF|N)\s+[A-Z]?\d+'
    match = re.search(us_pattern, user_input, re.IGNORECASE)

    if match:
        ref_id = match.group()
        result = hs_manager.find_overseas_case_by_id(ref_id)
        if result:
            return format_overseas_case_detail(result['case'], result['country'], query=ref_id)
        else:
            return f"⚠️ 참고문서번호 '{ref_id}'에 해당하는 사례를 찾을 수 없습니다.\n\n다른 문서번호나 키워드로 다시 검색해주세요."

    # 2. HS 코드 검색
    hs_pattern = r'\b\d{4}(\.\d{2}){0,2}\b'
    match = re.search(hs_pattern, user_input)

    if match:
        hs_code = match.group()
        results = hs_manager.search_overseas_by_hs_code(hs_code, top_k=10)
        if results:
            return format_overseas_case_list_by_hs(results, hs_code)

    # 3. 키워드 기반 단순 문자열 검색
    results = hs_manager.search_overseas_by_keyword(user_input, top_k=10)

    if not results:
        return f"""⚠️ **"{user_input}"에 대한 검색 결과가 없습니다**

**가능한 원인:**
- 해당 키워드가 포함된 분류사례가 데이터에 없습니다
- 검색어가 원문에 정확히 일치하지 않습니다

**검색 팁:**
- 영문 품목명 사용 (예: 'fabric', 'textile', 'bag')
- 더 짧고 일반적인 단어 사용 (예: 'ceramic mug' → 'mug', 'ceramic')
- 띄어쓰기 변경 시도
- HS 코드로 검색 (예: '5515.12', '4202.92')
- 참고문서번호로 검색 (예: 'NY N338825')

**다른 검색 방법:**
- **해외 분류사례 기반 HS 추천**: AI가 유사 사례를 분석하여 HS코드 추천 (TF-IDF 사용)
- **웹 검색**: 최신 정보 및 일반 품목 정보 검색"""

    # 결과를 국가별로 분리
    us_results = []
    eu_results = []

    for item in results:
        # 원본 데이터에서 국가 판단
        if 'hs_classification_data_us' in str(item) or item.get('organization', '').startswith('New York'):
            us_results.append(item)
        else:
            eu_results.append(item)

    return format_overseas_case_list(us_results, eu_results, query=user_input)


def format_overseas_case_detail(case, country, query=None):
    """해외 사례 상세 포맷"""
    country_flag = "🇺🇸" if country == "US" else "🇪🇺"
    country_name = "미국 CBP" if country == "US" else "EU 관세청"

    # 키워드 하이라이트 적용
    reply = highlight_keywords(case.get('reply', 'N/A'), query) if query else case.get('reply', 'N/A')
    description = highlight_keywords(case.get('description', 'N/A'), query) if query else case.get('description', 'N/A')

    return f"""---
<div class="case-detail">

## {country_flag} {country_name} 분류사례 상세 정보

<div class="info-table">

| 항목 | 내용 |
|------|------|
| 📄 **참고문서번호** | {case.get('reference_id', 'N/A')} |
| 📅 **결정일자** | {case.get('decision_date', 'N/A')} |
| 🏛️ **결정기관** | {case.get('organization', 'N/A')} |
| 🔢 **HS 코드** | {case.get('hs_code', 'N/A')} |
| 📆 **연도** | {case.get('year', 'N/A')} |

</div>

---

### 📋 요약
{reply}

---

### 📝 상세 내용
{description}

</div>
"""


def format_overseas_case_list_by_hs(results, hs_code):
    """HS 코드 기반 해외 사례 목록 포맷 (Expander 방식)"""
    output = f"## 🔍 HS 코드 \"{hs_code}\" 검색 결과 ({len(results)}건)\n\n"

    us_count = sum(1 for r in results if r['country'] == 'US')
    eu_count = len(results) - us_count

    output += f"- 🇺🇸 미국: {us_count}건\n"
    output += f"- 🇪🇺 EU: {eu_count}건\n\n"

    for idx, item in enumerate(results, 1):
        case = item['case']
        country = item['country']
        flag = "🇺🇸" if country == "US" else "🇪🇺"
        card_class = "us" if country == "US" else "eu"

        reply = case.get('reply', 'N/A')
        reply_short = reply[:80] + "..." if len(reply) > 80 else reply
        # 요약에도 하이라이트 적용
        reply_short = highlight_keywords(reply_short, hs_code)

        ref_id = case.get('reference_id', 'N/A')
        hs_code_display = case.get('hs_code', 'N/A')

        # 카드형 Expander
        output += f"""<div class="case-card {card_class}">
<details>
<summary class="case-summary">
<span class="arrow">▶</span>
<span class="rank">{idx}위 {flag}</span>
<span class="ref-id">{ref_id}</span>
<span class="hs-code">HS {hs_code_display}</span>
<span class="reply-preview">{reply_short}</span>
</summary>

<div class="case-content">
"""

        # Expander 내용 (전체 상세 정보, 하이라이트 적용)
        output += format_overseas_case_detail(case, country, query=hs_code)

        output += """</div>
</details>
</div>

"""

    output += "\n💡 **각 항목을 클릭하면 상세 정보를 확인할 수 있습니다.**"
    return output


def format_overseas_case_list(us_results, eu_results, query):
    """키워드 기반 해외 사례 목록 포맷 (국가별 구분, Expander 방식)"""
    total_count = len(us_results) + len(eu_results)
    output = f"## 🔍 \"{query}\" 검색 결과 ({total_count}건)\n\n"

    if us_results:
        output += f"### 🇺🇸 미국 ({len(us_results)}건)\n\n"
        for idx, case in enumerate(us_results, 1):
            reply = case.get('reply', 'N/A')
            reply_short = reply[:60] + "..." if len(reply) > 60 else reply
            # 요약에도 하이라이트 적용
            reply_short = highlight_keywords(reply_short, query)

            ref_id = case.get('reference_id', 'N/A')
            hs_code = case.get('hs_code', 'N/A')

            # 카드형 Expander
            output += f"""<div class="case-card us">
<details>
<summary class="case-summary">
<span class="arrow">▶</span>
<span class="rank">{idx}위</span>
<span class="ref-id">{ref_id}</span>
<span class="hs-code">HS {hs_code}</span>
<span class="reply-preview">{reply_short}</span>
</summary>

<div class="case-content">
"""

            # Expander 내용 (하이라이트 적용)
            output += format_overseas_case_detail(case, 'US', query=query)

            output += """</div>
</details>
</div>

"""

    if eu_results:
        output += f"\n---\n\n### 🇪🇺 EU ({len(eu_results)}건)\n\n"
        for idx, case in enumerate(eu_results, 1):
            reply = case.get('reply', 'N/A')
            reply_short = reply[:60] + "..." if len(reply) > 60 else reply
            # 요약에도 하이라이트 적용
            reply_short = highlight_keywords(reply_short, query)

            ref_id = case.get('reference_id', 'N/A')
            hs_code = case.get('hs_code', 'N/A')

            # 카드형 Expander
            output += f"""<div class="case-card eu">
<details>
<summary class="case-summary">
<span class="arrow">▶</span>
<span class="rank">{idx}위</span>
<span class="ref-id">{ref_id}</span>
<span class="hs-code">HS {hs_code}</span>
<span class="reply-preview">{reply_short}</span>
</summary>

<div class="case-content">
"""

            # Expander 내용 (하이라이트 적용)
            output += format_overseas_case_detail(case, 'EU', query=query)

            output += """</div>
</details>
</div>

"""

    output += "\n💡 **각 항목을 클릭하면 상세 정보를 확인할 수 있습니다.**"
    return output
