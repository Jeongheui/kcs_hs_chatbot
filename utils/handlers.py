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

# 질문 유형별 처리 함수
def handle_web_search(user_input, context, hs_manager):
    """웹 검색 처리 함수"""
    # 웹검색 전용 컨텍스트
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

def handle_hs_classification_cases(user_input, context, hs_manager, ui_container=None):
    """국내 HS 분류 사례 처리 (그룹별 Gemini + Head Agent)"""

    # 국내 HS 분류사례 전용 컨텍스트
    domestic_context = """당신은 국내 관세청의 HS 품목분류 전문가입니다.

역할과 목표:
- 관세청 HS 분류사례, 위원회 결정, 협의회 결정을 바탕으로 정확한 HS코드 분류 제시
- 국내 관세법과 HS 통칙에 근거한 전문적이고 체계적인 분석 수행
- 기존 분류 사례와의 일관성 유지 및 빈도수 기반 신뢰도 평가

분석 프로세스:
1. **유사 사례 수집 및 그룹화**
   - 사용자가 설명한 품목과 동일하거나 유사한 모든 분류 사례를 찾으세요
   - 찾은 사례들을 HS코드별로 그룹화하세요
   - 각 HS코드 그룹의 사례 개수(빈도수)를 집계하세요

2. **후보군 선정**
   - 빈도수가 가장 높은 최대 3개의 HS코드를 후보군으로 선정하세요
   - 각 후보의 빈도수와 대표 사례를 명시하세요

3. **최적 HS코드 선정**
   - 후보군 중에서 다음 기준으로 가장 적합한 HS코드를 최종 선정하세요:
     * 빈도수 (사례 개수)
     * 품목 설명의 유사도 (재질, 용도, 형상, 기능 등)

주의사항:
- 답변 시 반드시 출처를 명시하세요 (예: "품목분류2과-9433에 따르면...")
- 사용자가 자료에 없는 내용을 물어볼 경우, 반드시 "해당 정보는 자료에 없습니다" 또는 "확인된 사례가 없습니다"라고 답변하세요

답변 구성 (반드시 아래 형식을 따르세요):

## 1. 최종 선정 HS코드
**HS코드: [선정된 코드]**

**선정 사유:**
- 빈도수: [해당 코드의 사례 개수]건
- 유사도 분석: [사용자 품목과의 구체적 유사점]
- 대표 사례: [가장 유사한 1-2개 사례 간략 설명]
- 선정 근거: [해당 코드의 사례에서 사용된 주요 품목분류 근거]

## 2. 기타 후보 HS코드
### 후보 1: HS코드 [두 번째 후보]
- 빈도수: [사례 개수]건
- 미선정 사유: [최종 코드 대비 부족한 점]

### 후보 2: HS코드 [세 번째 후보] (있는 경우)
- 빈도수: [사례 개수]건
- 미선정 사유: [최종 코드 대비 부족한 점]

## 3. 분류 시 주의사항
- [실제 품목분류 신청 시 고려해야 할 요소]
- [추가로 확인이 필요한 품목 특성]

국내 관세청의 일관된 분류 기준을 우선시하고, 빈도수와 유사도를 객관적으로 평가하여 신뢰도 높은 답변을 제공하세요."""

    # UI 컨테이너가 제공된 경우 실시간 표시
    if ui_container:
        with ui_container:
            st.info("🔍 **국내 HS 분류사례 분석 시작**")
            progress_bar = st.progress(0, text="AI 그룹별 분석 진행 중...")
            responses_container = st.container()

    # TF-IDF 기반 검색으로 상위 100개 사례 추출 (임계값 0.05로 낮춤 - AI가 더 많은 데이터 분석)
    top_cases = hs_manager.search_domestic_tfidf(user_input, top_k=100, min_similarity=0.05)

    # 5개 그룹으로 분할 (각 그룹 20개)
    group_size = len(top_cases) // 5
    groups = [top_cases[i*group_size:(i+1)*group_size if i < 4 else len(top_cases)] for i in range(5)]

    # 병렬 처리용 함수
    def process_single_group(i):
        try:
            # 그룹 데이터를 컨텍스트로 변환
            group_cases = groups[i]
            relevant = "\n\n".join([
                f"출처: 국내 관세청\n항목: {json.dumps(case, ensure_ascii=False)}"
                for case in group_cases
            ])

            prompt = f"{domestic_context}\n\n관련 데이터 (국내 관세청, 그룹{i+1}):\n{relevant}\n\n사용자: {user_input}\n"

            start_time = datetime.now()
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()

            answer = clean_text(response.text)
            return i, answer, start_time, processing_time
        except Exception as e:
            error_msg = f"그룹 {i+1} 분석 중 오류 발생: {str(e)}"
            return i, error_msg, datetime.now(), 0.0

    # 5개 그룹 병렬 처리 (max_workers=3)
    if ui_container:
        progress_bar.progress(0, text="병렬 AI 분석 시작...")

    results = {}
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(process_single_group, i) for i in range(5)]

        for future in as_completed(futures):
            group_id, answer, start_time, processing_time = future.result()
            results[group_id] = answer

            # session_state에 결과 저장
            if ui_container:
                analysis_result = {
                    'type': 'domestic',
                    'group_id': group_id,
                    'answer': answer,
                    'start_time': start_time.strftime('%H:%M:%S'),
                    'processing_time': processing_time
                }
                st.session_state.ai_analysis_results.append(analysis_result)

                # 실시간 UI 업데이트 (완료된 순서대로)
                with responses_container:
                    st.success(f"🤖 **그룹 {group_id+1} AI 분석 완료** ({processing_time:.1f}초)")
                    with st.container():
                        st.write(f"⏰ {start_time.strftime('%H:%M:%S')}")
                        st.markdown(f"**분석 결과:**")
                        st.info(answer)
                        st.divider()

                progress_bar.progress(len(results)/5, text=f"완료: {len(results)}/5 그룹")

    # 순서대로 정렬
    group_answers = [results[i] for i in range(5)]

    if ui_container:
        progress_bar.progress(1.0, text="Head AI 최종 분석 중...")
        st.info("🧠 **Head AI가 모든 분석을 종합하는 중...**")

    # Head Agent가 5개 부분 답변을 취합하여 최종 답변 생성
    try:
        head_prompt = f"{domestic_context}\n\n아래는 국내 HS 분류 사례 데이터 5개 그룹별 분석 결과입니다. 각 그룹의 답변을 종합하여 최종 전문가 답변을 작성하세요.\n\n"
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
            st.error(f"⚠️ Head AI 오류: {str(e)}")

    if ui_container:
        progress_bar.progress(1.0, text="분석 완료!")
        st.success("✅ **모든 AI 분석이 완료되었습니다**")
        st.info("📋 **패널을 접고 아래에서 최종 답변을 확인하세요**")

    return final_answer


def handle_overseas_hs(user_input, context, hs_manager, ui_container=None):
    """해외 HS 분류 사례 처리 (그룹별 Gemini + Head Agent)"""

    # 해외 HS 분류사례 전용 컨텍스트
    overseas_context = """당신은 국제 HS 품목분류 전문가입니다.

역할과 목표:
- 미국 관세청(CBP)과 EU 관세청의 HS 분류 사례 분석
- 빈도수 기반 신뢰도 평가를 통한 체계적 분석

분석 프로세스 (미국/EU 각각 적용):
1. **유사 사례 수집 및 그룹화**
   - 사용자가 설명한 품목과 동일하거나 유사한 모든 분류 사례를 찾으세요
   - 찾은 사례들을 HS코드별로 그룹화하세요
   - 각 HS코드 그룹의 사례 개수(빈도수)를 집계하세요

2. **후보군 선정**
   - 빈도수가 가장 높은 최대 3개의 HS코드를 후보군으로 선정하세요
   - 각 후보의 빈도수와 대표 사례를 명시하세요

3. **최적 HS코드 선정**
   - 후보군 중에서 다음 기준으로 가장 적합한 HS코드를 최종 선정하세요:
     * 빈도수 (사례 개수)
     * 품목 설명의 유사도 (재질, 용도, 형상, 기능 등)

주의사항:
- 답변 시 반드시 출처를 명시하세요 (예: "미국 NY N123456에 따르면...", "아일랜드 IEBTIIENEN004-2025-BTI119에 따르면...")
- 사용자가 자료에 없는 내용을 물어볼 경우, 반드시 "해당 정보는 자료에 없습니다" 또는 "확인된 사례가 없습니다"라고 답변하세요

답변 구성 (미국/EU 각각 적용, 반드시 아래 형식을 따르세요):

## 1. 최종 선정 HS코드
**HS코드: [선정된 코드]**

**선정 사유:**
- 빈도수: [해당 코드의 사례 개수]건
- 유사도 분석: [사용자 품목과의 구체적 유사점]
- 대표 사례: [가장 유사한 1-2개 사례 간략 설명]
- 선정 근거: [해당 코드의 사례에서 사용된 주요 품목분류 근거]

## 2. 기타 후보 HS코드
### 후보 1: HS코드 [두 번째 후보]
- 빈도수: [사례 개수]건
- 미선정 사유: [최종 코드 대비 부족한 점]

### 후보 2: HS코드 [세 번째 후보] (있는 경우)
- 빈도수: [사례 개수]건
- 미선정 사유: [최종 코드 대비 부족한 점]

---

# 종합 분석 (미국과 EU 데이터가 둘다 있는 경우에만 작성)

## 미국/EU 분류 비교 
- [두 지역 분류의 공통점과 차이점]

글로벌 무역 관점에서 포괄적으로 분석하고, 빈도수와 유사도를 객관적으로 평가하여 신뢰도 높은 답변을 제공하세요."""

    # UI 컨테이너가 제공된 경우 실시간 표시
    if ui_container:
        with ui_container:
            st.info("🌍 **해외 HS 분류사례 분석 시작**")
            progress_bar = st.progress(0, text="AI 그룹별 분석 진행 중...")
            responses_container = st.container()

    # TF-IDF 기반 검색으로 상위 100개 사례 추출 (임계값 0.05로 낮춤 - AI가 더 많은 데이터 분석)
    top_cases = hs_manager.search_overseas_tfidf(user_input, top_k=100, min_similarity=0.05)

    # 5개 그룹으로 분할 (각 그룹 20개)
    group_size = len(top_cases) // 5
    groups = [top_cases[i*group_size:(i+1)*group_size if i < 4 else len(top_cases)] for i in range(5)]

    # 병렬 처리용 함수
    def process_single_group(i):
        try:
            # 그룹 데이터를 컨텍스트로 변환
            group_cases = groups[i]
            relevant = "\n\n".join([
                f"출처: 해외 관세청\n항목: {json.dumps(case, ensure_ascii=False)}"
                for case in group_cases
            ])

            prompt = f"{overseas_context}\n\n관련 데이터 (해외 관세청, 그룹{i+1}):\n{relevant}\n\n사용자: {user_input}\n"

            start_time = datetime.now()
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()

            answer = clean_text(response.text)
            return i, answer, start_time, processing_time
        except Exception as e:
            error_msg = f"그룹 {i+1} 분석 중 오류 발생: {str(e)}"
            return i, error_msg, datetime.now(), 0.0

    # 5개 그룹 병렬 처리 (max_workers=3)
    if ui_container:
        progress_bar.progress(0, text="병렬 AI 분석 시작...")

    results = {}
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(process_single_group, i) for i in range(5)]

        for future in as_completed(futures):
            group_id, answer, start_time, processing_time = future.result()
            results[group_id] = answer

            # session_state에 결과 저장
            if ui_container:
                analysis_result = {
                    'type': 'overseas',
                    'group_id': group_id,
                    'answer': answer,
                    'start_time': start_time.strftime('%H:%M:%S'),
                    'processing_time': processing_time
                }
                st.session_state.ai_analysis_results.append(analysis_result)

                # 실시간 UI 업데이트 (완료된 순서대로)
                with responses_container:
                    st.success(f"🌐 **그룹 {group_id+1} AI 분석 완료** ({processing_time:.1f}초)")
                    with st.container():
                        st.write(f"⏰ {start_time.strftime('%H:%M:%S')}")
                        st.markdown(f"**분석 결과:**")
                        st.info(answer)
                        st.divider()

                progress_bar.progress(len(results)/5, text=f"완료: {len(results)}/5 그룹")

    # 순서대로 정렬
    group_answers = [results[i] for i in range(5)]

    if ui_container:
        progress_bar.progress(1.0, text="Head AI 최종 분석 중...")
        st.info("🧠 **Head AI가 모든 분석을 종합하는 중...**")

    # Head Agent가 5개 부분 답변을 취합하여 최종 답변 생성
    try:
        head_prompt = f"{overseas_context}\n\n아래는 해외 HS 분류 사례 데이터 5개 그룹별 분석 결과입니다. 각 그룹의 답변을 종합하여 최종 전문가 답변을 작성하세요.\n\n"
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
            st.error(f"⚠️ Head AI 오류: {str(e)}")

    if ui_container:
        progress_bar.progress(1.0, text="분석 완료!")
        st.success("✅ **모든 AI 분석이 완료되었습니다**")
        st.info("📋 **패널을 접고 아래에서 최종 답변을 확인하세요**")

    return final_answer

def handle_hs_manual_with_user_codes(user_input, context, hs_manager, logger, extracted_codes, ui_container=None):
    """사용자 제시 HS코드 기반 해설서 분석

    Args:
        user_input: 사용자 질문
        context: 대화 컨텍스트
        hs_manager: HS 데이터 매니저
        logger: 로거
        extracted_codes: 이미 추출된 HS코드 리스트 (main.py에서 전달)
        ui_container: UI 컨테이너 (optional)
    """

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
                return format_domestic_case_detail(case)
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


def format_domestic_case_detail(case):
    """국내 사례 상세 포맷"""
    return f"""---
## 📋 국내 분류사례 상세 정보

### 기본 정보
- **참고문서번호**: {case.get('reference_id', 'N/A')}
- **결정일자**: {case.get('decision_date', 'N/A')}
- **결정기관**: {case.get('organization', 'N/A')}
- **HS 코드**: {case.get('hs_code', 'N/A')}

---

### 품목명
{case.get('product_name', 'N/A')}

---

### 품목 설명
{case.get('description', 'N/A')}

---

### 분류 근거
{case.get('decision_reason', 'N/A')}
"""


def format_domestic_case_list(results, query):
    """국내 사례 목록 포맷 (Expander 방식)"""
    output = f"## 🔍 \"{query}\" 검색 결과 ({len(results)}건)\n\n"

    for idx, case in enumerate(results, 1):
        product_name = case.get('product_name', 'N/A')
        description = case.get('description', 'N/A')
        ref_id = case.get('reference_id', 'N/A')
        hs_code = case.get('hs_code', 'N/A')
        decision_date = case.get('decision_date', 'N/A')

        # 품목명이 너무 길면 자르기 (Expander 제목용)
        product_name_display = product_name[:60] + "..." if len(product_name) > 60 else product_name

        # Expander 제목
        output += f"<details><summary><b>{idx}위. {ref_id}</b> | HS {hs_code} | {product_name_display}</summary>\n\n"

        # Expander 내용 (전체 상세 정보)
        output += format_domestic_case_detail(case)

        output += "\n</details>\n\n"

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
            return format_overseas_case_detail(result['case'], result['country'])
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


def format_overseas_case_detail(case, country):
    """해외 사례 상세 포맷"""
    country_flag = "🇺🇸" if country == "US" else "🇪🇺"
    country_name = "미국 CBP" if country == "US" else "EU 관세청"

    return f"""---
## {country_flag} {country_name} 분류사례 상세 정보

### 기본 정보
- **참고문서번호**: {case.get('reference_id', 'N/A')}
- **결정일자**: {case.get('decision_date', 'N/A')}
- **결정기관**: {case.get('organization', 'N/A')}
- **HS 코드**: {case.get('hs_code', 'N/A')}
- **연도**: {case.get('year', 'N/A')}

---

### 요약
{case.get('reply', 'N/A')}

---

### 상세 내용
{case.get('description', 'N/A')}
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

        reply = case.get('reply', 'N/A')
        reply_short = reply[:80] + "..." if len(reply) > 80 else reply
        ref_id = case.get('reference_id', 'N/A')
        hs_code_display = case.get('hs_code', 'N/A')

        # Expander 제목
        output += f"<details><summary><b>{idx}위 {flag}. {ref_id}</b> | HS {hs_code_display} | {reply_short}</summary>\n\n"

        # Expander 내용 (전체 상세 정보)
        output += format_overseas_case_detail(case, country)

        output += "\n</details>\n\n"

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
            ref_id = case.get('reference_id', 'N/A')
            hs_code = case.get('hs_code', 'N/A')

            # Expander 제목
            output += f"<details><summary><b>{idx}위. {ref_id}</b> | HS {hs_code} | {reply_short}</summary>\n\n"

            # Expander 내용
            output += format_overseas_case_detail(case, 'US')

            output += "\n</details>\n\n"

    if eu_results:
        output += f"\n---\n\n### 🇪🇺 EU ({len(eu_results)}건)\n\n"
        for idx, case in enumerate(eu_results, 1):
            reply = case.get('reply', 'N/A')
            reply_short = reply[:60] + "..." if len(reply) > 60 else reply
            ref_id = case.get('reference_id', 'N/A')
            hs_code = case.get('hs_code', 'N/A')

            # Expander 제목
            output += f"<details><summary><b>{idx}위. {ref_id}</b> | HS {hs_code} | {reply_short}</summary>\n\n"

            # Expander 내용
            output += format_overseas_case_detail(case, 'EU')

            output += "\n</details>\n\n"

    output += "\n💡 **각 항목을 클릭하면 상세 정보를 확인할 수 있습니다.**"
    return output