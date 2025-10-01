import streamlit as st
import time
import os
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

    # 병렬 처리용 함수
    def process_single_group(i):
        try:
            relevant = hs_manager.get_domestic_context_group(user_input, i)
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

    # 병렬 처리용 함수
    def process_single_group(i):
        try:
            relevant = hs_manager.get_overseas_context_group(user_input, i)
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

def handle_hs_manual_with_parallel_search(user_input, context, hs_manager, logger, ui_container=None):
    """병렬 검색을 활용한 HS 해설서 분석"""

    # UI 컨테이너가 제공된 경우 분석 과정 표시
    if ui_container:
        with ui_container:
            st.info("🔍 **HS 해설서 병렬 분석 시작**")
            progress_bar = st.progress(0, text="병렬 검색 진행 중...")
            analysis_container = st.container()

    # 병렬 검색 수행
    parallel_searcher = ParallelHSSearcher(hs_manager)
    search_results = parallel_searcher.parallel_search(user_input, logger, ui_container)

    # UI 업데이트 - 병렬 검색 결과 먼저 표시
    if ui_container:
        progress_bar.progress(0.6, text="병렬 검색 결과 분석 중...")

        # 1단계: 후보 코드 선정 과정 표시
        with analysis_container:
            st.success("✅ **병렬 검색 완료**")
            st.markdown("### 🎯 **상위 HS코드 후보 선정**")

            for i, result in enumerate(search_results, 1):
                confidence_color = "🟢" if result['confidence'] == 'HIGH' else "🟡"
                st.markdown(f"{confidence_color} **후보 {i}: HS코드 {result['hs_code']}** (신뢰도: {result['confidence']})")

                col1, col2 = st.columns([1, 2])
                with col1:
                    st.write(f"**최종점수**: {result['final_score']:.3f}")
                    st.write(f"**검색경로**: {', '.join(result['sources'])}")
                with col2:
                    if result['tariff_name']:
                        st.write(f"**관세율표 품목명**: {result['tariff_name']}")
                    if result['manual_content']:
                        st.write(f"**📖 해설서 원문**: 발견됨 (요약 예정)")

                st.divider()

    # 결과를 컨텍스트로 변환
    enhanced_context = parallel_searcher.create_enhanced_context(search_results)

    # 각 후보의 해설서 내용 요약 (5회 API 호출)
    if ui_container:
        progress_bar.progress(0.7, text="해설서 내용 요약 중...")

    logger.log_actual("AI", "Starting manual content summarization...")
    summary_start = time.time()

    for i, result in enumerate(search_results):
        if result['manual_content']:
            summary_prompt = f"""다음 HS 해설서 내용을 700자 이내로 핵심 내용만 요약해주세요:

HS코드: {result['hs_code']}
해설서 원문:
{result['manual_content']}

요약 시 포함할 내용:
- 주요 품목 범위
- 포함/제외 품목
- 분류 기준
- 핵심 특징

간결하고 정확하게 요약해주세요."""

            try:
                summary_response = client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=summary_prompt
                )
                result['manual_summary'] = clean_text(summary_response.text)
                logger.log_actual("SUCCESS", f"HS코드 {result['hs_code']} 해설서 요약 완료", f"{len(result['manual_summary'])} chars")
            except APIError as e:
                error_msg = f"Gemini API 오류 (코드: {e.code}): {e.message}"
                logger.log_actual("ERROR", f"HS코드 {result['hs_code']} 요약 실패", error_msg)
                result['manual_summary'] = result['manual_content'][:700] + "..." if len(result['manual_content']) > 700 else result['manual_content']
            except Exception as e:
                logger.log_actual("ERROR", f"HS코드 {result['hs_code']} 요약 실패: {str(e)}")
                result['manual_summary'] = result['manual_content'][:700] + "..." if len(result['manual_content']) > 700 else result['manual_content']
        else:
            result['manual_summary'] = ""

    summary_time = time.time() - summary_start
    logger.log_actual("SUCCESS", f"Manual content summarization completed", f"{summary_time:.2f}s")

    # 2단계: 해설서 요약 완료 후 업데이트된 정보 표시
    if ui_container:
        with analysis_container:
            st.success("✅ **해설서 내용 요약 완료**")
            st.markdown("### 📚 **해설서 요약 결과**")

            for i, result in enumerate(search_results, 1):
                confidence_color = "🟢" if result['confidence'] == 'HIGH' else "🟡"
                st.markdown(f"{confidence_color} **후보 {i}: HS코드 {result['hs_code']}** (신뢰도: {result['confidence']})")

                col1, col2 = st.columns([1, 2])
                with col1:
                    st.write(f"**최종점수**: {result['final_score']:.3f}")
                    st.write(f"**검색경로**: {', '.join(result['sources'])}")
                with col2:
                    if result['tariff_name']:
                        st.write(f"**관세율표 품목명**: {result['tariff_name']}")
                    if result.get('manual_summary'):
                        st.write(f"**📖 해설서 요약**:")
                        st.text(result['manual_summary'][:300] + "...")
                    elif result['manual_content']:
                        st.write(f"**📖 해설서**: 요약 실패 (원문 사용)")

                st.divider()

        progress_bar.progress(0.9, text="AI 전문가 분석 준비 중...")

    # HS 해설서 분석 결과를 세션 상태에 저장 (채팅 기록에서 보기 위해)
    if ui_container:
        if 'hs_manual_analysis_results' not in st.session_state:
            st.session_state.hs_manual_analysis_results = []

        # 현재 분석 결과 저장
        current_analysis = {
            'timestamp': time.time(),
            'search_results': search_results,
            'query': user_input
        }
        st.session_state.hs_manual_analysis_results.append(current_analysis)

        # 최대 5개까지만 보관 (메모리 절약)
        if len(st.session_state.hs_manual_analysis_results) > 5:
            st.session_state.hs_manual_analysis_results.pop(0)

    logger.log_actual("INFO", f"Enhanced context prepared", f"{len(enhanced_context)} chars")

    # HS 해설서 분석 전용 컨텍스트
    manual_context = """당신은 HS 해설서 및 관세율표 전문 분석가입니다.

당신이 받는 데이터:
- 병렬 검색 시스템이 이미 완료한 상위 3개 HS코드 후보
- 각 후보는 관세율표 검색(40% 가중치) + 해설서 검색(60% 가중치) 결과를 통합한 것입니다
- confidence: HIGH(양쪽 검색에서 발견) / MEDIUM(한쪽만 발견)
- sources: 'tariff_to_manual'(관세율표 경로), 'direct_manual'(해설서 직접 경로)
- manual_summary: 해설서 요약본 (우선 참고)

분석 프로세스:
1. **후보 평가**: 3개 후보를 신뢰도(HIGH 우선) + 해설서 적합성으로 평가
2. **최적 선정**: 최고 신뢰도 후보 중 사용자 품목과 가장 일치하는 코드 선정
3. **근거 작성**: 관세율표 품목명 + 해설서 내용을 통합하여 설명

주의사항:
- 답변 시 반드시 출처를 명시하세요 (예: "제39류 HS 해설서에 따르면...", "호 3923 해설서에 따르면...")
- 자료에 없는 내용은 "해당 정보는 HS 해설서에 없습니다"라고 답변하세요

답변 형식 (반드시 아래 구조를 따르세요):

## 1. 최종 추천 HS코드
**HS코드: [코드]**

**선정 사유:**
- 신뢰도: [HIGH/MEDIUM] (검색 경로: [sources 내용])
- 관세율표 품목명: [tariff_name]
- 해설서 근거: [manual_summary 핵심 내용]
- 적합성 분석: [사용자 품목과의 일치점]

## 2. 기타 후보 HS코드
### 후보 2: HS코드 [두 번째]
- 신뢰도: [HIGH/MEDIUM]
- 미선정 사유: [최종 코드 대비 부족한 점]

### 후보 3: HS코드 [세 번째]
- 신뢰도: [HIGH/MEDIUM]
- 미선정 사유: [최종 코드 대비 부족한 점]

## 3. 분류 시 주의사항
- [실무 적용 시 확인 필요 사항]
- [유사 품목과의 구분 기준]"""

    # Gemini에 전달할 프롬프트 구성
    prompt = f"""{manual_context}

[병렬 검색 결과]
{enhanced_context}

사용자 질문: {user_input}
"""

    # Gemini 처리
    logger.log_actual("AI", "Processing with enhanced parallel search context...")
    ai_processing_start = time.time()

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        ai_processing_time = time.time() - ai_processing_start
        final_answer = clean_text(response.text)

        logger.log_actual("SUCCESS", "Gemini processing completed",
                         f"{ai_processing_time:.2f}s, input: {len(prompt)} chars, output: {len(final_answer)} chars")

    except APIError as e:
        ai_processing_time = time.time() - ai_processing_start
        error_msg = f"Gemini API 오류가 발생했습니다.\n\n**오류 코드**: {e.code}\n**오류 메시지**: {e.message}\n\n"

        if e.code == 503:
            error_msg += "**해결 방법**: API 서버가 일시적으로 과부하 상태입니다. 잠시 후 다시 시도해주세요."
        elif e.code == 429:
            error_msg += "**해결 방법**: API 사용량 한도를 초과했습니다. 잠시 후 다시 시도해주세요."
        elif e.code == 404:
            error_msg += "**해결 방법**: 요청한 모델을 찾을 수 없습니다. 모델명을 확인해주세요."
        elif e.code == 400:
            error_msg += "**해결 방법**: 잘못된 요청입니다. 입력 내용을 확인해주세요."
        else:
            error_msg += "**해결 방법**: 문제가 지속되면 관리자에게 문의해주세요."

        logger.log_actual("ERROR", f"Gemini API error (code: {e.code})", e.message)
        final_answer = error_msg

    except Exception as e:
        ai_processing_time = time.time() - ai_processing_start
        error_msg = f"처리 중 예상치 못한 오류가 발생했습니다.\n\n**오류 내용**: {str(e)}\n\n관리자에게 문의해주세요."
        logger.log_actual("ERROR", f"Unexpected error during AI processing: {str(e)}")
        final_answer = error_msg

    # UI 최종 완료 표시
    if ui_container:
        progress_bar.progress(1.0, text="분석 완료!")
        st.success("🧠 **AI 전문가 분석이 완료되었습니다**")
        st.info("📋 **패널을 접고 아래에서 최종 답변을 확인하세요**")

    return final_answer