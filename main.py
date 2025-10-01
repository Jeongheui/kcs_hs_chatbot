import streamlit as st
from google import genai
from google.genai.errors import APIError
import time
from datetime import datetime

import os
from dotenv import load_dotenv
from utils import HSDataManager, extract_hs_codes, clean_text, classify_question
from utils import handle_web_search, handle_hs_classification_cases, handle_overseas_hs, get_hs_explanations, handle_hs_manual_with_user_codes

# 환경 변수 로드 (.env 파일에서 API 키 등 설정값 로드)
load_dotenv()

# Gemini API 설정
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
client = genai.Client(api_key=GOOGLE_API_KEY)

# Streamlit 페이지 설정
st.set_page_config(
    page_title="HS 품목분류 챗봇",  # 브라우저 탭 제목
    page_icon="📊",  # 브라우저 탭 아이콘
    layout="wide"  # 페이지 레이아웃을 넓게 설정
)

# 사용자 정의 CSS 스타일 추가
st.markdown("""
<style>
.main > div {
    display: flex;
    flex-direction: column;
    height: 85vh;  # 메인 컨테이너 높이 설정
}
.main > div > div:last-child {
    margin-top: auto;  # 마지막 요소를 하단에 고정
}
.stTextInput input {
    border-radius: 10px;  # 입력창 모서리 둥글게
    padding: 8px 12px;
    font-size: 16px;
}
/* 라디오 버튼 크기 및 글자 크기 증가 */
.stRadio > label {
    font-size: 16px !important;
    font-weight: 500 !important;
}
.stRadio > div {
    gap: 4px !important;
}
.stRadio > div > label {
    font-size: 17px !important;
    padding: 4px 0px !important;
}
.stRadio > div > label > div:first-child {
    margin-right: 8px !important;
}
</style>
""", unsafe_allow_html=True)

# HS 데이터 매니저 초기화 (캐싱을 통해 성능 최적화)
@st.cache_resource
def get_hs_manager():
    return HSDataManager()

# 세션 상태 초기화
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []  # 채팅 기록 저장

if 'selected_category' not in st.session_state:
    st.session_state.selected_category = "AI 자동분류"  # 기본값

if 'context' not in st.session_state:
    # 초기 컨텍스트 설정
    st.session_state.context = """당신은 HS 품목분류 전문가로서 관세청에서 오랜 경력을 가진 전문가입니다. 사용자가 물어보는 품목에 대해 아래 네 가지 유형 중 하나로 질문을 분류하여 답변해주세요.

질문 유형:
1. 웹 검색(Web Search): 물품개요, 용도, 기술개발, 무역동향 등 일반 정보 탐색이 필요한 경우.
2. HS 분류 검색(HS Classification Search): HS 코드, 품목분류, 관세, 세율 등 HS 코드 관련 정보가 필요한 경우.
3. HS 해설서 분석(HS Manual Analysis): HS 해설서 본문 심층 분석이 필요한 경우.
4. 해외 HS 분류(Overseas HS Classification): 해외(미국/EU) HS 분류 사례가 필요한 경우.

중요 지침:
1. 사용자가 질문하는 물품에 대해 관련어, 유사품목, 대체품목도 함께 고려하여 가장 적합한 HS 코드를 찾아주세요.
2. 품목의 성분, 용도, 가공상태 등을 고려하여 상세히 설명해주세요.
3. 사용자가 특정 HS code를 언급하며 질문하는 경우, 답변에 해당 HS code 해설서 분석 내용을 포함하여 답변해주세요.
4. 관련 규정이나 판례가 있다면 함께 제시해주세요.
5. 답변은 간결하면서도 전문적으로 제공해주세요.

지금까지의 대화:
"""

if 'ai_analysis_results' not in st.session_state:
    st.session_state.ai_analysis_results = []

class RealTimeProcessLogger:
    def __init__(self, container):
        self.container = container
        self.log_placeholder = container.empty()
        self.logs = []
        self.start_time = time.time()
    
    def log_actual(self, level, message, data=None):
        """실제 진행 상황만 기록"""
        elapsed = time.time() - self.start_time
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        
        log_entry = {
            "time": timestamp,
            "elapsed": f"{elapsed:.2f}s",
            "level": level,
            "message": message,
            "data": data
        }
        self.logs.append(log_entry)
        self.update_display()
    
    def update_display(self):
        log_text = ""
        icons = {"INFO": "ℹ️", "SUCCESS": "✅", "ERROR": "❌", "DATA": "📊", "AI": "🤖", "SEARCH": "🔍"}
        
        for log in self.logs[-8:]:
            icon = icons.get(log['level'], "📝")
            data_str = f" | {log['data']}" if log['data'] else ""
            log_text += f"`{log['time']}` `+{log['elapsed']}` {icon} {log['message']}{data_str}\n\n"
        
        self.log_placeholder.markdown(log_text)
    
    def clear(self):
        self.logs = []
        self.log_placeholder.empty()


def process_query_with_real_logging(user_input):
    """실제 진행사항을 기록하면서 쿼리 처리"""
    
    log_container = st.container()
    logger = RealTimeProcessLogger(log_container)
    
    try:
        logger.log_actual("INFO", "Query processing started", f"Input length: {len(user_input)}")
        
        start_time = time.time()
        hs_manager = get_hs_manager()
        load_time = time.time() - start_time
        logger.log_actual("SUCCESS", "HSDataManager loaded", f"{load_time:.2f}s")
        
        category = st.session_state.selected_category
        logger.log_actual("INFO", "Category selected", category)
        
        if category == "AI 자동분류":
            logger.log_actual("AI", "Starting LLM question classification...")
            start_classify = time.time()
            q_type = classify_question(user_input)
            classify_time = time.time() - start_classify
            logger.log_actual("SUCCESS", "LLM classification completed", f"{q_type} in {classify_time:.2f}s")
        else:
            category_mapping = {
                "웹 검색": "web_search",
                "국내 HS분류사례 검색": "hs_classification",
                "해외 HS분류사례 검색": "overseas_hs",
                "HS해설서 분석(품명 + 후보 HS코드)": "hs_manual",
                "HS해설서 원문 검색(HS코드만 입력)": "hs_manual_raw",
                "AI 자동분류": "auto"  # AI 자동분류는 자동 판별로 처리
            }
            q_type = category_mapping.get(category, "hs_classification")

            # AI 자동분류가 수동 선택된 경우 자동 판별 실행
            if q_type == "auto":
                logger.log_actual("AI", "Starting LLM question classification (manual selection)...")
                start_classify = time.time()
                q_type = classify_question(user_input)
                classify_time = time.time() - start_classify
                logger.log_actual("SUCCESS", "LLM classification completed", f"{q_type} in {classify_time:.2f}s")
            else:
                logger.log_actual("INFO", "Question type mapped", q_type)

        answer_start = time.time()
        
        if q_type == "web_search":
            logger.log_actual("SEARCH", "Initiating Google Search API call...")
            ai_start = time.time()
            answer = "\n\n +++ 웹검색 실시 +++\n\n" + handle_web_search(user_input, st.session_state.context, hs_manager)
            ai_time = time.time() - ai_start
            logger.log_actual("SUCCESS", "Web search completed", f"{ai_time:.2f}s, {len(answer)} chars")
            
        elif q_type == "hs_classification":
            # Multi-Agent 분석 실행 (UI 컨테이너 없이)
            final_answer = handle_hs_classification_cases(user_input, st.session_state.context, hs_manager, None)
            answer = "\n\n +++ HS 분류사례 검색 실시 +++\n\n" + final_answer
            
        elif q_type == "overseas_hs":
            # Multi-Agent 분석 실행 (UI 컨테이너 없이)
            final_answer = handle_overseas_hs(user_input, st.session_state.context, hs_manager, None)
            answer = "\n\n +++ 해외 HS 분류 검색 실시 +++\n\n" + final_answer
            
        elif q_type == "hs_manual":
            # 1단계: HS코드 추출 시도
            logger.log_actual("INFO", "Checking for user-provided HS codes...")
            extracted_codes = extract_hs_codes(user_input)
            logger.log_actual("INFO", f"Extracted codes: {extracted_codes if extracted_codes else 'None'}")

            if extracted_codes:
                # 사용자가 HS코드를 제시한 경우 → 코드 비교 분석
                logger.log_actual("SUCCESS", f"Found {len(extracted_codes)} user-provided HS codes", ", ".join(extracted_codes))
                logger.log_actual("AI", "Starting user-provided codes comparison analysis...")
                ai_start = time.time()
                answer = "\n\n +++ HS 해설서 분석 실시 (사용자 제시 코드 비교) +++ \n\n" + handle_hs_manual_with_user_codes(user_input, st.session_state.context, hs_manager, logger, extracted_codes)
                ai_time = time.time() - ai_start
                logger.log_actual("SUCCESS", "User-provided codes analysis completed", f"{ai_time:.2f}s, {len(answer)} chars")
            else:
                # HS코드 없는 경우 → 에러 메시지
                logger.log_actual("ERROR", "No HS codes found in user input")
                answer = "해설서 분석 모드에서는 반드시 HS 코드를 제시해야 합니다.\n\n예시: '3923.30과 3926.90 중 어느 것이 맞나요?'"
            
        elif q_type == "hs_manual_raw":
            logger.log_actual("SEARCH", "Extracting HS codes...")
            hs_codes = extract_hs_codes(user_input)
            if hs_codes:
                logger.log_actual("SUCCESS", f"Found {len(hs_codes)} HS codes", ", ".join(hs_codes))
                logger.log_actual("DATA", "Retrieving raw HS explanations...")
                raw_start = time.time()
                raw_answer = clean_text(get_hs_explanations(hs_codes))
                raw_time = time.time() - raw_start
                answer = "\n\n +++ HS 해설서 원문 검색 실시 +++ \n\n" + raw_answer
                logger.log_actual("SUCCESS", "Raw HS manual retrieved", f"{raw_time:.2f}s, {len(raw_answer)} chars")
            else:
                logger.log_actual("ERROR", "No valid HS codes found in input")
                answer = "HS 코드를 찾을 수 없습니다. 4자리 HS 코드를 입력해주세요."

        answer_time = time.time() - answer_start
        logger.log_actual("SUCCESS", "Answer generation completed", f"{answer_time:.2f}s, {len(answer)} chars")
        
        total_time = time.time() - logger.start_time
        logger.log_actual("INFO", "Process completed successfully", f"Total time: {total_time:.2f}s")
        
        # Return the answer for external processing
        return answer
        
    except Exception as e:
        logger.log_actual("ERROR", f"Exception occurred: {str(e)}")
        logger.log_actual("ERROR", f"Error type: {type(e).__name__}")
        raise e


# 사이드바 설정 - 챗봇 특성 소개
with st.sidebar:
    st.title("📊 HS 품목분류 전문 AI")

    st.markdown("""
    ### 🎯 챗봇 소개

    한국 관세청 및 글로벌 HS 분류 데이터를 기반으로
    **AI 기술**과 **대규모 데이터베이스**를 결합한
    차세대 품목분류 전문 상담 시스템입니다.

    ---

    ### 🚀 핵심 기술

    **Multi-Agent 시스템**
    - 5개 그룹 병렬 분석 (최대 3개 동시 실행)
    - Head Agent 최종 종합
    
    **Gemini AI 2.5 Flash**
    - Google 최신 LLM 모델
    
    ---

    ### 📚 보유 데이터

    **국내 분류사례**
    - 관세청 분류사례: 899건 (10개 파일로 분할)
    - HS 위원회 결정: 76건
    - HS 협의회 결정: 12건
    - **총 987건**

    **해외 분류사례**
    - 미국 CBP 분류사례: 900건
    - EU 관세청 분류사례: 1,000건
    - **총 1,900건**

    **공식 해설서**
    - HS 품목분류표: 17,966개 코드
    - HS 해설서: 1,448개 항목
    - HS 통칙: 9개 조항

    **실시간 웹 데이터**
    - Google Search API 연동

    ---

    ### ⚡ 성능 특징

    - 캐싱으로 데이터 로딩 최적화
    - 실시간 처리 과정 투명 공개
    - AI 분석 결과 세션 저장
    - 대화 컨텍스트 누적 관리
    """)

    st.divider()
    
    # 새로운 채팅 시작 버튼
    if st.button("새로운 채팅 시작하기", type="primary"):
        st.session_state.chat_history = []  # 채팅 기록 초기화
        # Multi-Agent 및 HS 해설서 분석 결과도 초기화
        if 'ai_analysis_results' in st.session_state:
            st.session_state.ai_analysis_results = []
        if 'hs_manual_analysis_results' in st.session_state:
            st.session_state.hs_manual_analysis_results = []
        # 컨텍스트 초기화 (기본 컨텍스트 재사용)
        st.session_state.context = """당신은 HS 품목분류 전문가로서 관세청에서 오랜 경력을 가진 전문가입니다. 사용자가 물어보는 품목에 대해 아래 네 가지 유형 중 하나로 질문을 분류하여 답변해주세요.

질문 유형:
1. 웹 검색(Web Search): 물품개요, 용도, 기술개발, 무역동향 등 일반 정보 탐색이 필요한 경우.
2. HS 분류 검색(HS Classification Search): HS 코드, 품목분류, 관세, 세율 등 HS 코드 관련 정보가 필요한 경우.
3. HS 해설서 분석(HS Manual Analysis): HS 해설서 본문 심층 분석이 필요한 경우.
4. 해외 HS 분류(Overseas HS Classification): 해외(미국/EU) HS 분류 사례가 필요한 경우.

중요 지침:
1. 사용자가 질문하는 물품에 대해 관련어, 유사품목, 대체품목도 함께 고려하여 가장 적합한 HS 코드를 찾아주세요.
2. 품목의 성분, 용도, 가공상태 등을 고려하여 상세히 설명해주세요.
3. 사용자가 특정 HS code를 언급하며 질문하는 경우, 답변에 해당 HS code 해설서 분석 내용을 포함하여 답변해주세요.
4. 관련 규정이나 판례가 있다면 함께 제시해주세요.
5. 답변은 간결하면서도 전문적으로 제공해주세요.

지금까지의 대화:
"""
        st.success("✅ 새로운 채팅이 시작되었습니다!")

# 메인 페이지 설정
st.title("HS 품목분류 챗봇")
st.markdown("""
<div style='background: #F0F9FF; border-radius: 12px; border-left: 6px solid #3B82F6; padding: 24px 28px 20px 28px; margin-bottom: 18px;'>
  <h4 style='color:#1E40AF; margin-top:0;'>💡 <b>슬기로운 품목분류 생활 (활용 시나리오) </b></h4>
  <ol style='padding-left: 18px;'>
    <li style='margin-bottom: 10px;'>
      <b>[웹 검색] </span> "스마트워치 실리콘 밴드의 정확한 재질 성분과 제조 공정, 주요 용도는 무엇인가?"</b><br>
      <span style='color:#059669;'>✓ 답변: 합성고무(실리콘), 스마트워치 전용 부속품 확인</span>
    </li>
    <li style='margin-bottom: 10px;'>
      <b>[국내 사례] </span> "실리콘 재질로 만든 스마트워치용 교체 밴드는 어떤 HS코드로 분류되나요?"</b><br>
      <span style='color:#059669;'>✓ 답변: 9113.90 (시계 부속품) 12건 / 3926.90 (플라스틱) 5건</span>
    </li>
    <li style='margin-bottom: 10px;'>
      <b>[해외 사례] </span> "미국과 EU에서 스마트워치 실리콘 스트랩(watch strap)의 분류 사례와 관세율을 알려줘"</b><br>
      <span style='color:#059669;'>✓ 답변: 전 세계 동일: 9113.90 분류</span>
    </li>
    <li>
      <b>[해설서 분석] </span> "스마트워치 전용 실리콘 밴드가 9113.90 시계 부속품과 3926.90 기타 플라스틱 제품 중 어디에 분류되는지 해설서와 통칙을 근거로 비교 분석해줘"</b><br>
      <span style='color:#059669;'>✓ 결론: 9113.90-0000 (통칙 1, 구체성 원칙)</span>
    </li>
  </ol>
</div>
""", unsafe_allow_html=True)

# 질문 유형 선택 라디오 버튼 + 설명 카드 (가로 배치)
st.subheader("질문 유형을 선택하세요")

# 2개 컬럼으로 나누기: 왼쪽 질문 유형 선택, 오른쪽 설명 카드
col_left, col_right = st.columns([1, 2])

with col_left:
    selected_category = st.radio(
        "분석 방식을 선택하세요:",
        [
            "웹 검색",
            "국내 HS분류사례 검색",
            "해외 HS분류사례 검색",
            "HS해설서 분석(품명 + 후보 HS코드)",
            "HS해설서 원문 검색(HS코드만 입력)",
            "AI 자동분류"
        ],
        index=0,
        horizontal=False,
        key="category_radio",
        label_visibility="collapsed"
    )

# 카테고리명은 그대로 사용
st.session_state.selected_category = selected_category

# 선택된 유형에 따른 상세 설명 및 예시 표시
category_info = {
    "웹 검색": {
        "icon": "🌐",
        "description": "**Google Search API**를 활용하여 품목의 일반 정보, 시장 동향, 최신 기술 개발, 산업 현황 등을 실시간으로 검색합니다.",
        "data_source": "Google Search API (실시간)",
        "examples": "'반도체 시장 동향', '전기차 배터리 최신 기술', 'AI 칩셋 산업 현황'"
    },
    "국내 HS분류사례 검색": {
        "icon": "🇰🇷",
        "description": "**관세청 분류사례 987건 데이터베이스**를 Multi-Agent 시스템(5그룹 병렬 분석)으로 검색하여 가장 적합한 HS코드를 추천합니다.",
        "data_source": "관세청 분류사례, HS위원회/협의회 결정",
        "examples": "'플라스틱 용기 HS코드', '자동차 엔진 부품의 HS코드', '화장품 용기 분류'"
    },
    "해외 HS분류사례 검색": {
        "icon": "🌍",
        "description": "**미국(CBP) 및 EU 관세청 분류사례 1,900건**을 Multi-Agent 시스템으로 분석하여 글로벌 분류 기준을 비교 제공합니다.",
        "data_source": "미국/EU 관세청 공식 분류사례",
        "examples": "'미국 전자제품 분류 기준', 'EU 화학제품 분류사례', '해외 의료기기 분류 동향'"
    },
    "HS해설서 분석(품명 + 후보 HS코드)": {
        "icon": "📚",
        "description": "**사용자가 제시한 여러 HS코드**를 품목분류표 + HS 해설서 + 통칙 기반으로 심층 비교 분석하여 최적 코드를 추천합니다.",
        "data_source": "HS 품목분류표, HS 해설서 전문, 통칙",
        "examples": "'3923.30과 3926.90 중 플라스틱 용기는?', '8471.30과 8471.50 중 노트북은?'",
        "note": "**주의**: 반드시 비교할 HS코드를 질문에 포함해야 합니다."
    },
    "HS해설서 원문 검색(HS코드만 입력)": {
        "icon": "📖",
        "description": "특정 **HS코드의 해설서 원문**을 통칙/부/류/호 체계로 정리하여 제공합니다.",
        "data_source": "HS 해설서 전문",
        "examples": "'3911', '391190', '8471' (HS코드만 입력)"
    },
    "AI 자동분류": {
        "icon": "🤖",
        "description": "**LLM이 질문 내용을 분석**하여 위 5가지 유형 중 가장 적합한 방식을 자동으로 선택해 답변합니다.",
        "data_source": "상황에 따라 자동 선택",
        "examples": "'플라스틱 용기 분류', '반도체 동향', '미국 자동차 부품 사례' 등 자유롭게 질문"
    }
}

info = category_info[selected_category]

# 오른쪽 컬럼에 선택된 유형 정보를 카드 형식으로 표시
with col_right:
    st.markdown(f"""
    <div style='background-color: #F0F9FF; padding: 20px; border-radius: 10px; border-left: 5px solid #3B82F6; height: 100%;'>
        <h3 style='margin-top: 0; color: #1E40AF;'>{info['icon']} {selected_category}</h3>
        <p style='margin-bottom: 10px;'>{info['description']}</p>
        <p style='margin-bottom: 10px;'><strong>📊 데이터 출처:</strong> {info['data_source']}</p>
        <p style='margin-bottom: 5px;'><strong>💡 예시 질문:</strong></p>
        <p style='margin-bottom: 0; font-style: italic; color: #4B5563;'>{info['examples']}</p>
    </div>
    """, unsafe_allow_html=True)

    # 특수 케이스 주의사항 표시
    if selected_category == "HS해설서 분석(품명 + 후보 HS코드)":
        st.warning(info.get('note', ''))

st.divider()  # 구분선 추가

# 채팅 기록 표시
for message in st.session_state.chat_history:
    if message["role"] == "user":
        st.markdown(f"""<div style='background-color: #e6f7ff; padding: 10px; border-radius: 10px; margin-bottom: 10px;'>
                   <strong>사용자:</strong> {message['content']}
                   </div>""", unsafe_allow_html=True)
    else:
        # 분석 과정이 있는 경우 expander 표시
        if any(keyword in message['content'] for keyword in ["+++ HS 분류사례 검색 실시 +++", "+++ 해외 HS 분류 검색 실시 +++", "+++ HS 해설서 분석 실시 (사용자 제시 코드) +++"]):
            # AI 분석 과정 expander 표시 (채팅 기록에서도 항상 표시)
            with st.expander("🔍 **AI 분석 과정 보기**", expanded=False):
                if "+++ HS 해설서 분석 실시 (사용자 제시 코드) +++" in message['content']:
                    # 새로운 사용자 제시 코드 분석의 경우
                    st.info("🔍 **사용자 제시 HS코드 기반으로 분석되었습니다**")
                    st.markdown("""
                    **분석 과정:**
                    1. 📝 사용자 질문에서 HS코드 추출
                    2. 📊 각 HS코드별 품목분류표 정보 수집
                    3. 📚 각 HS코드별 해설서 정보 수집 및 요약
                    4. 📋 HS 분류 통칙 준비
                    5. 🧠 최종 AI 비교 분석 (Gemini 2.5)
                    """)
                elif st.session_state.ai_analysis_results:
                    # Multi-Agent 분석의 경우 - 저장된 결과 표시
                    for result in st.session_state.ai_analysis_results:
                        emoji = "🤖" if result['type'] == 'domestic' else "🌐"
                        st.success(f"{emoji} **그룹 {result['group_id']+1} AI 분석 완료** ({result['processing_time']:.1f}초)")
                        with st.container():
                            st.write(f"⏰ {result['start_time']}")
                            st.markdown("**분석 결과:**")
                            st.info(result['answer'])
                            st.divider()
                else:
                    st.info("분석 과정 정보가 저장되지 않았습니다.")
            
            # 최종 답변 표시 (마크다운으로 렌더링)
            st.markdown("**품목분류 전문가:**")
            st.markdown(message['content'])
        
        # HS 해설서 원문인지 확인
        elif "+++ HS 해설서 원문 검색 실시 +++" in message['content']:
            # 마크다운으로 렌더링하여 구조화된 형태로 표시
            st.markdown("**품목분류 전문가:**")
            st.markdown(message['content'])
        else:
            st.markdown(f"""<div style='background-color: #f0f2f6; padding: 10px; border-radius: 10px; margin-bottom: 10px;'>
                    <strong>품목분류 전문가:</strong> {message['content']}
                    </div>""", unsafe_allow_html=True)


# 하단 입력 영역 (Form 기반 입력)
input_container = st.container()
st.markdown("<div style='flex: 1;'></div>", unsafe_allow_html=True)

with input_container:
    # Form을 사용하여 안정적인 입력 처리
    with st.form("query_form", clear_on_submit=True):
        # 선택된 유형에 따른 placeholder 메시지
        placeholders = {
            "웹 검색": "예: '반도체 시장 동향', '전기차 산업 현황'",
            "국내 HS분류사례 검색": "예: '플라스틱 용기 HS코드', '자동차 부품 분류'",
            "해외 HS분류사례 검색": "예: '미국 전자제품 분류', 'EU 화학제품 사례'",
            "HS해설서 분석(품명 + 후보 HS코드)": "예: '3923.30과 3926.90 중 플라스틱 용기 분류는?'",
            "HS해설서 원문 검색(HS코드만 입력)": "예: '3911' 또는 '391190' (HS코드만 입력)",
            "AI 자동분류": "예: '플라스틱 용기 분류', '반도체 동향' 등 자유롭게 질문하세요"
        }
        
        user_input = st.text_input(
            "품목에 대해 질문하세요:", 
            placeholder=placeholders.get(st.session_state.selected_category, "여기에 입력 후 Enter 또는 전송 버튼 클릭")
        )
        
        # 두 개의 컬럼으로 나누어 버튼을 오른쪽에 배치
        col1, col2 = st.columns([4, 1])
        with col2:
            submit_button = st.form_submit_button("전송", use_container_width=True)
        
        # 폼이 제출되고 입력값이 있을 때 처리
        if submit_button and user_input and user_input.strip():
            selected_category = st.session_state.selected_category
            
            # HS Manager 인스턴스 가져오기
            hs_manager = get_hs_manager()
            
            # 분석 과정 표시가 필요한 유형들
            if selected_category in ["국내 HS분류사례 검색", "해외 HS분류사례 검색", "HS해설서 분석(품명 + 후보 HS코드)"]:
                if selected_category in ["국내 HS분류사례 검색", "해외 HS분류사례 검색"]:
                    st.session_state.ai_analysis_results = []  # Multi-Agent용 결과 초기화
                analysis_expander = st.expander("🔍 **AI 분석 과정 보기**", expanded=True)

            try:
                # 분석 과정 표시 방식 분기
                if selected_category == "HS해설서 분석(품명 + 후보 HS코드)":
                    # HS 해설서 분석은 HS코드 유무에 따라 분기
                    class DummyLogger:
                        def log_actual(self, level, message, data=None):
                            pass  # UI 표시용이므로 로깅은 생략

                    dummy_logger = DummyLogger()
                    extracted_codes = extract_hs_codes(user_input)

                    if extracted_codes:
                        # HS코드가 있으면 사용자 제시 코드 비교 분석
                        final_answer = handle_hs_manual_with_user_codes(user_input, st.session_state.context, hs_manager, dummy_logger, extracted_codes, analysis_expander)
                        answer = "\n\n +++ HS 해설서 분석 실시 (사용자 제시 코드 비교) +++ \n\n" + final_answer
                    else:
                        # HS코드가 없으면 에러 메시지
                        answer = "해설서 분석 모드에서는 반드시 HS 코드를 제시해야 합니다.\n\n예시: '3923.30과 3926.90 중 어느 것이 맞나요?'"
                elif selected_category not in ["국내 HS분류사례 검색", "해외 HS분류사례 검색"]:
                    # 기타 유형은 로그 패널 표시
                    with st.expander("실시간 처리 과정 로그 보기", expanded=True):
                        answer = process_query_with_real_logging(user_input)
                else:
                    # Multi-Agent 분석용 특별 처리
                    if selected_category == "국내 HS분류사례 검색":
                        # utils 함수를 직접 호출하되 expander 컨테이너 전달
                        final_answer = handle_hs_classification_cases(user_input, st.session_state.context, hs_manager, analysis_expander)
                        answer = "\n\n +++ HS 분류사례 검색 실시 +++\n\n" + final_answer
                    elif selected_category == "해외 HS분류사례 검색":
                        final_answer = handle_overseas_hs(user_input, st.session_state.context, hs_manager, analysis_expander)
                        answer = "\n\n +++ 해외 HS 분류 검색 실시 +++\n\n" + final_answer

                # Update chat history after successful processing
                st.session_state.chat_history.append({"role": "user", "content": user_input})
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                st.session_state.context += f"\n사용자: {user_input}\n품목분류 전문가: {answer}\n"

                # 분석 과정이 표시된 유형들의 최종 답변 표시 (마크다운으로 렌더링)
                if selected_category in ["국내 HS분류사례 검색", "해외 HS분류사례 검색", "HS해설서 분석(품명 + 후보 HS코드)"]:
                    st.markdown("**품목분류 전문가:**")
                    st.markdown(answer)
                
                # Force rerun to display the new chat messages
                st.rerun()
                
            except APIError as e:
                st.error("### Gemini API 오류 발생")
                st.error(f"**오류 코드**: {e.code}")
                st.error(f"**오류 메시지**: {e.message}")

                if e.code == 503:
                    st.warning("**해결 방법**: API 서버가 일시적으로 과부하 상태입니다. 잠시 후 다시 시도해주세요.")
                elif e.code == 429:
                    st.warning("**해결 방법**: API 사용량 한도를 초과했습니다. 잠시 후 다시 시도해주세요.")
                elif e.code == 404:
                    st.warning("**해결 방법**: 요청한 모델을 찾을 수 없습니다. 모델명을 확인해주세요.")
                elif e.code == 400:
                    st.warning("**해결 방법**: 잘못된 요청입니다. 입력 내용을 확인해주세요.")
                else:
                    st.warning("**해결 방법**: 문제가 지속되면 관리자에게 문의해주세요.")

            except Exception as e:
                st.error(f"처리 중 예상치 못한 오류가 발생했습니다: {str(e)}")