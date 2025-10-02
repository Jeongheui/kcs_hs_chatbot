# HS 품목분류 챗봇 (슬기로운 품목분류 생활)

관세청 품목분류 사례, 해외 HS 분류 사례, HS 해설서 등 다양한 데이터를 Multi-Agent 시스템으로 분석하여 전문적인 HS 코드 분류 답변을 제공하는 AI 챗봇입니다.

---

## 💬 채팅 시나리오: LED 무드등 수입 업무

당신은 캐릭터 굿즈를 수입하는 온라인 쇼핑몰 MD입니다.
LED 조명이 내장된 피규어 하우스 무드등을 수입하려는데, 정확한 HS 코드를 찾아야 합니다.

---

### 🔍 1단계: 웹 검색

**질문:** "LED 무드등의 기술 사양과 시장 동향, 주요 용도는?"

**답변:** LED 조명 기술 발전 현황, 인테리어 소품 시장 규모, 수면등/분위기 조명 용도 확인

- LED 광원 충전식 무드등
- 본체, 피규어, USB 케이블 포함
- 크기: 114×59×149mm
- 배터리: 900mAh 내장
- 작동 방식: 마그네틱으로 피규어를 넣으면 LED 점등

---

### 🇰🇷 2단계: 국내 HS 분류사례 검색

**질문:** "LED 충전식 무드등은 어떤 HS코드로 분류되나요?"

**답변:** 9405.21-0000 (전기식 탁상용/침대용 램프) **14건 확인**

- 관세청 분류사례를 5개 그룹으로 분할하여 Multi-Agent 병렬 분석
- Head Agent가 모든 그룹 결과를 종합하여 최종 답변 생성

---

### 🌍 3단계: 해외 HS 분류사례 검색

**질문:** "미국과 EU에서 LED lamp, mood light 분류 사례는?"

**답변:**
- 미국 CBP: **27건** - 주로 9405 류 분류
- EU BTI: **292건** - 9405 류 또는 9503(장난감) 분류 사례 분석
- 국제적 분류 동향 파악

---

### 📚 4단계: HS 해설서 분석

**질문:** "캐릭터 피규어가 포함된 LED 무드등이 9405.21(전기식 램프)과 9503(장난감) 중 어디에 분류되는지 해설서와 통칙을 근거로 비교 분석해줘"

**답변:**
- 9405.21-0000 **최종 선정**
- 근거: 통칙 3(본질적 특성은 조명 기능)
- HS 해설서 제94류 및 제95류 비교 분석
- 관세율표 정보 및 통칙 종합 검토

---

### 📖 5단계: HS 해설서 원문 검색

**질문:** "9405.21

**답변:**
- 특정 HS 코드의 해설서 원문을 구조화된 형태로 제공
- 통칙, 부/류/호 해설을 체계적으로 정리하여 표시

---

## ✨ 이 챗봇의 장점

### ⚡ 가볍고 빠릅니다
- **일반 노트북 또는 Streamlit Cloud 무료 계정**에서도 즉시 실행
- 별도 서버 불필요, 클라우드 비용 제로
- 초기 로딩 후 **실시간 응답 (5-10초 내)**

### 💰 비용이 들지 않습니다
- **Google Gemini 2.5 Flash 무료 API** 사용
- 추가 인프라 비용 없음

### 🔍 TF-IDF로 정확하게 검색합니다
- **복합어 정확 검색**: "폴리우레탄폼" 띄어쓰기 없이 인식
- **중요도 기반 순위화**: 문서별 단어 빈도(TF)와 희소성(IDF) 계산
- 단순한 키워드 검색 대비 월등한 정확도

### 🤖 Multi-Agent로 정확하고 빠릅니다
- 전체 데이터(~900건)에서 **TF-IDF로 상위 100개 사례 추출**
- **5그룹(각 20개 사례)으로 분할 후 병렬 분석**
- **Head Agent**가 5개 분석 결과 종합

### 📊 신뢰할 수 있는 공식 데이터 소스

**국내 분류사례 (총 987건)**
- 관세청 분류사례: 899건
- HS위원회 결정: 76건
- HS협의회 결정: 12건

**해외 분류사례 (총 1,900건)**
- 미국 CBP 분류사례: 900건
- EU 관세청 BTI: 1,000건

**공식 해설서**
- HS 품목분류표: 17,966개 코드
- HS 해설서: 1,448개 항목
- HS 통칙: 9개 조항

**실시간 웹 데이터**
- Google Search API 연동

---

## 🚀 빠른 시작

### 📦 1. 저장소 클론 및 패키지 설치

```bash
git clone https://github.com/YSCHOI-github/kcs_hs_chatbot
cd kcs_hs_chatbot
pip install -r requirements.txt
```

**필수 패키지:**
- `google-genai` - Gemini API 클라이언트
- `streamlit` - 웹 UI 프레임워크
- `python-dotenv` - 환경 변수 관리
- `scikit-learn` - TF-IDF 검색 엔진
- `numpy`, `pandas`, `requests` - 데이터 처리

---

### 🔑 2. API 키 설정

```bash
# .env 파일 생성
echo "GOOGLE_API_KEY=your_api_key_here" > .env
```

Google API 키는 [Google AI Studio](https://aistudio.google.com/app/apikey)에서 **무료로 발급**받을 수 있습니다.

---

### 📁 3. 데이터 준비

`knowledge/` 폴더에 다음 JSON 파일들을 배치하세요:

**국내 분류사례 (10개 파일)**
- `HS분류사례_part1.json` ~ `HS분류사례_part10.json`

**위원회 결정 (2개 파일)**
- `HS위원회.json`
- `HS협의회.json`

**해외 분류사례 (2개 파일)**
- `hs_classification_data_us.json` (미국 CBP)
- `hs_classification_data_eu.json` (EU BTI)

**공식 자료 (3개 파일)**
- `hstable.json` (관세율표)
- `grouped_11_end.json` (HS 해설서)
- `통칙_grouped.json` (HS 통칙)

---

### ▶️ 4. 실행

```bash
streamlit run main.py
```

브라우저에서 자동으로 `http://localhost:8501` 열림

---

## 🎯 핵심 기능

### 🔍 1. 웹 검색
- 물품개요, 용도, 뉴스, 무역동향, 산업동향 등 최신 정보 제공
- Google Search API를 통한 실시간 정보 검색

### 🇰🇷 2. 국내 HS 분류사례 검색
- 관세청 품목분류 사례 987건 데이터베이스 분석
- **TF-IDF Character n-gram 검색**: 복합어 정확 매칭 ("폴리우레탄폼", "리튬이온배터리")
- Multi-Agent 시스템: TF-IDF 상위 100개 사례를 5개 그룹으로 분할 병렬 분석
- Head Agent가 최종 취합하여 전문적인 HS 코드 분류 답변 제공

### 🌍 3. 해외 HS 분류사례 검색
- 미국 CBP 및 EU BTI 품목분류 사례 1,900건 분석
- **TF-IDF Character n-gram 검색**: 해외 사례 정밀 검색
- Multi-Agent 시스템: TF-IDF 상위 100개를 5개 그룹으로 분할 분석
- 국제적인 HS 분류 동향 및 비교 분석 제공

### 📚 4. HS 해설서 분석 (사용자 제시 코드)
- 사용자가 직접 제시한 HS 코드들을 전문적으로 비교 분석
- 5단계 체계적 분석: 코드 추출 → 관세율표 정보 → 해설서 수집 → 통칙 준비 → AI 비교 분석
- 투명한 분석 과정: 각 단계별 진행 상황을 실시간으로 공개
- 여러 HS 코드의 장단점, 적용 가능성, 리스크 등을 종합 평가

### 📖 5. HS 해설서 원문 검색
- 특정 HS 코드의 해설서 원문을 구조화된 형태로 제공
- 통칙, 부/류/호 해설을 체계적으로 정리하여 표시

### 🤖 6. AI 자동분류
- 사용자 질문을 LLM이 자동으로 분석하여 가장 적합한 방식으로 답변
- 5가지 질문 유형 중 최적의 방법을 자동 선택

---

## 🔬 핵심 기술 상세

### TF-IDF Character n-gram 검색

**도입 배경:**
- 기존 키워드 검색의 한계: "폴리우레탄폼" 검색 시 정확히 일치하는 문서만 검색
- 한국어 형태소 분석기 사용 시 속도 저하 (수십 초 소요)

**Character n-gram 방식:**
- 2~4 글자 단위로 텍스트 분해 (예: "리튬이온배터리" → "리튬", "튬이", "이온", "온배", "배터", ...)
- 복합어 부분 매칭 지원

**적용 범위:**
- ✅ 국내 HS 분류사례 검색
- ✅ 해외 HS 분류사례 검색
- ⚪ HS 해설서는 키워드 검색 유지 (원문 조회 목적)

### Multi-Agent 시스템

**병렬 처리 구조:**
1. 전체 데이터(~900건)에서 TF-IDF로 상위 100개 사례 추출 (0.02초)
2. 100개 사례를 5개 그룹으로 분할 (각 그룹 20개 사례)
3. 5개 Group Agent가 병렬 분석 (최대 3개 동시, ThreadPoolExecutor)
4. Head Agent가 5개 분석 결과 종합 → 최종 답변

**사용 모델:**
- Group Agent: Gemini 2.5 Flash (빠른 분석)
- Head Agent: Gemini 2.5 Flash (종합 판단)

---

## 📂 프로젝트 구조

```
kcs_hs_chatbot/
├── main.py                 # Streamlit 메인 애플리케이션
├── utils/                  # 유틸리티 모듈 패키지
│   ├── __init__.py         # 모듈 통합 및 export
│   ├── data_loader.py      # HSDataManager 클래스 (TF-IDF 인덱스 구축)
│   ├── tfidf_search.py     # Character n-gram TF-IDF 검색 엔진
│   ├── handlers.py         # 질문 유형별 처리 함수 (Multi-Agent)
│   ├── question_classifier.py  # LLM 기반 질문 자동 분류
│   ├── hs_manual_utils.py  # HS 해설서 관련 함수들
│   ├── search_engines.py   # 병렬 검색 엔진 (관세율표 + 해설서)
│   └── text_utils.py       # 텍스트 처리 유틸리티
├── hs_search.py            # HS 코드 검색 유틸리티
├── CLAUDE.md              # Claude Code 개발 가이드
├── .env                    # 환경 변수 (API 키)
├── requirements.txt        # 패키지 의존성
├── README.md              # 프로젝트 문서
└── knowledge/             # 핵심 데이터 파일
    ├── HS분류사례_part1.json ~ part10.json  # 국내 분류사례
    ├── HS위원회.json, HS협의회.json          # 위원회 결정사항
    ├── hs_classification_data_us.json       # 미국 관세청 데이터
    ├── hs_classification_data_eu.json       # EU 관세청 데이터
    ├── hstable.json                         # 관세율표
    ├── 통칙_grouped.json                     # HS 통칙
    └── grouped_11_end.json                  # HS 해설서
```

---

## 📄 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다.
