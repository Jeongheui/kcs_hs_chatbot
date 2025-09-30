import streamlit as st
import time
import os
from datetime import datetime
from google import genai
from google.genai import types
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

사용자의 질문에 대해 최신 웹 정보를 검색하여 물품개요, 용도, 기술개발, 무역동향, 산업동향 등의 정보를 제공해주세요.
국내 HS 분류 사례가 아닌 일반적인 시장 정보와 동향을 중심으로 답변해주세요.

주의사항:
- 답변 시 반드시 출처를 명시하세요 (예: "○○ 사이트에 따르면...", "△△ 보고서에서...")
- 검색된 자료에 없는 내용을 질문할 경우, 반드시 "해당 정보는 검색 결과에 없습니다" 또는 "확인된 자료가 없습니다"라고 답변하세요"""

    grounding_tool = types.Tool(google_search=types.GoogleSearch())
    config = types.GenerateContentConfig(tools=[grounding_tool])

    prompt = f"{web_context}\n\n사용자: {user_input}\n"

    response = client.models.generate_content(
        model="gemini-2.5-flash",
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

def handle_hs_manual_with_user_codes(user_input, context, hs_manager, logger, ui_container=None):
    """사용자 제시 HS코드 기반 해설서 분석"""

    # UI 컨테이너가 제공된 경우 분석 과정 표시
    if ui_container:
        with ui_container:
            st.info("🔍 **사용자 제시 HS코드 분석 시작**")
            progress_bar = st.progress(0, text="HS코드 추출 중...")
            analysis_container = st.container()

    # 1단계: 사용자 제시 HS코드 추출
    logger.log_actual("INFO", "Extracting user-provided HS codes...")
    extracted_codes = extract_hs_codes(user_input)

    if not extracted_codes:
        logger.log_actual("ERROR", "No HS codes found in user input")
        if ui_container:
            progress_bar.progress(1.0, text="분석 완료!")
            st.error("❌ **HS코드를 찾을 수 없습니다**")
            st.info("💡 **사용법**: '3923, 3924, 3926 중에서 플라스틱 용기를 분류해주세요' 형태로 질문하세요")
        return "HS코드를 찾을 수 없습니다. 분석할 HS코드를 포함하여 질문해주세요."

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
            summary_prompt = f"""다음 HS 해설서 내용을 1000자 이내로 핵심 내용만 요약해주세요:

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
            except Exception as e:
                logger.log_actual("ERROR", f"HS코드 {result['hs_code']} 요약 실패: {str(e)}")
                result['manual_summary'] = result['manual_content'][:1000] + "..." if len(result['manual_content']) > 1000 else result['manual_content']
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

역할과 목표:
- 관세율표(품목번호, 한글품명, 영문품명)와 HS 해설서의 병렬 분석
- 관세율표 기반 유사도 검색과 해설서 텍스트 분석의 통합
- 부(部)-류(類)-호(號) 체계와 관세율표 품목명의 정합성 판단

병렬 검색 시스템:
- **관세율표 검색**: hstable.json의 품목명 유사도 매칭 (40% 가중치)
- **해설서 검색**: HS 해설서 본문 텍스트 분석 (60% 가중치)
- **통합 분석**: 두 검색 결과의 교차 검증과 신뢰도 평가

주의사항:
- 답변 시 반드시 출처를 명시하세요 (예: "제39류 HS 해설서에 따르면...", "호 3923 해설서에 따르면...")
- 사용자가 자료에 없는 내용을 물어볼 경우, 반드시 "해당 정보는 HS 해설서에 없습니다" 또는 "확인된 자료가 없습니다"라고 답변하세요

답변 구성요소:
1. **최적 HS코드 추천**: 병렬 검색 결과 기반 최고 신뢰도 코드
2. **관세율표 매칭**: 유사 품목명과 매칭도 분석
3. **해설서 근거**: 해당 부-류-호 해설서의 정확한 적용
4. **신뢰도 평가**: HIGH(양쪽 검색 일치) vs MEDIUM(한쪽만 매칭)
5. **종합 판단**: 관세율표와 해설서 분석의 일치성 검토

관세율표 품목명과 HS 해설서를 모두 활용하여 정확한 분류를 제시해주세요."""

    # Gemini에 전달할 프롬프트 구성
    prompt = f"""{manual_context}

[병렬 검색 결과]
{enhanced_context}

사용자 질문: {user_input}

위의 병렬 검색 결과를 바탕으로 다음을 포함하여 답변해주세요:

1. **가장 적합한 HS 코드 추천**
   - 최고 신뢰도의 HS코드와 그 근거
   - 관세율표 품목명과 해설서 설명 종합

2. **분류 근거 및 분석**
   - 관세율표 기반 검색 결과
   - 해설서 기반 검색 결과
   - 두 검색 경로의 일치성 분석

3. **신뢰도 평가**
   - HIGH: 두 검색 경로 모두에서 발견
   - MEDIUM: 한 검색 경로에서만 발견
   - 각 후보의 신뢰도와 점수

4. **추가 고려사항**
   - 유사 품목과의 구분 기준
   - 분류 시 주의점
   - 필요 시 추가 정보 요청 사항

답변은 전문적이면서도 이해하기 쉽게 작성해주세요.
"""

    # Gemini 처리
    logger.log_actual("AI", "Processing with enhanced parallel search context...")
    ai_processing_start = time.time()

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    ai_processing_time = time.time() - ai_processing_start
    final_answer = clean_text(response.text)

    logger.log_actual("SUCCESS", "Gemini processing completed",
                     f"{ai_processing_time:.2f}s, input: {len(prompt)} chars, output: {len(final_answer)} chars")

    # UI 최종 완료 표시
    if ui_container:
        progress_bar.progress(1.0, text="분석 완료!")
        st.success("🧠 **AI 전문가 분석이 완료되었습니다**")
        st.info("📋 **패널을 접고 아래에서 최종 답변을 확인하세요**")

    return final_answer