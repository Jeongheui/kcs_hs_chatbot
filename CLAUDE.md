# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an HS (Harmonized System) classification chatbot built with Streamlit and Google Gemini AI. It helps users classify products according to HS codes using multiple data sources including Korean customs cases, overseas classification examples, and official HS manuals.

## Running the Application

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Streamlit application
streamlit run main.py
```

## Environment Setup

- Create a `.env` file with `GOOGLE_API_KEY=your_api_key`
- Required data files in `knowledge/` directory:
  - `HS분류사례_part1.json` through `HS분류사례_part10.json` (domestic cases)
  - `HS위원회.json`, `HS협의회.json` (committee decisions)
  - `hs_classification_data_us.json`, `hs_classification_data_eu.json` (overseas cases)
  - `hstable.json` (tariff table)
  - `grouped_11_end.json` (HS manual)
  - `통칙_grouped.json` (general rules)

## Core Architecture

### Multi-Agent System
The application uses a Multi-Agent architecture for parallel processing:
- **5 Group Agents**: Domestic and overseas HS data is split into 5 groups, each analyzed by a separate Gemini instance running in parallel
- **Head Agent**: Consolidates results from all 5 group agents into a final comprehensive answer
- **ThreadPoolExecutor**: Uses `max_workers=3` for parallel execution in handlers.py

### Question Classification System
User queries are automatically classified into 5 types using LLM (question_classifier.py):
1. `web_search`: General product information, market trends
2. `hs_classification`: Domestic HS classification cases
3. `hs_manual`: HS manual analysis (parallel search or user-provided codes)
4. `overseas_hs`: US/EU classification cases
5. `hs_manual_raw`: Raw HS manual lookup

### Dual-Path Parallel Search
For HS manual analysis (search_engines.py), the system uses two parallel search paths:
- **Path 1 (Tariff → Manual)**: Search tariff table by similarity (40% weight), then lookup manuals for candidate codes
- **Path 2 (Direct Manual)**: Direct keyword search in HS manual texts (60% weight)
- Results are consolidated with confidence scores (HIGH/MEDIUM)

### Handler Functions
All question type handlers are in `utils/handlers.py`:
- `handle_web_search()`: Uses Google Search API via Gemini
- `handle_hs_classification_cases()`: Multi-agent domestic search
- `handle_overseas_hs()`: Multi-agent overseas search
- `handle_hs_manual_with_user_codes()`: User-provided HS codes comparison
- `handle_hs_manual_with_parallel_search()`: Dual-path search system

### Data Management
`utils/data_loader.py` contains `HSDataManager`:
- Loads all JSON data files at initialization
- Provides keyword-based search (marked for future embedding upgrade)
- Group-based search methods for parallel processing
- Methods ending with `_group` split data into 5 groups for Multi-Agent system

## Key Features

### Real-Time Logging
The `RealTimeProcessLogger` class in main.py provides transparent process visualization:
- Tracks timing for each processing step
- Shows only recent logs (last 8 entries)
- Log levels: INFO, SUCCESS, ERROR, DATA, AI, SEARCH

### Session State Management
Streamlit session state tracks:
- `chat_history`: Conversation history
- `context`: Cumulative conversation context for AI
- `ai_analysis_results`: Multi-agent analysis results
- `hs_manual_analysis_results`: HS manual search results
- `selected_category`: Current question type

### UI Components
- Main chat interface with message history
- Expandable analysis panels showing AI reasoning
- Progress bars for multi-step operations
- Category selection radio buttons with examples

## Code Style Notes

- **No Emojis in Python**: Per CLAUDE.md instructions, avoid emojis except in Streamlit UI code
- **Korean Comments**: Many comments and data are in Korean (관세청 = customs office)
- **JSON Data**: All knowledge data is in JSON format with Korean text
- **Present Plan First**: When modifying Python code, always present your plan before implementation

## Important Patterns

### Multi-Agent Execution Pattern
```python
def process_single_group(i):
    # Get group-specific data
    relevant = hs_manager.get_domestic_context_group(user_input, i)
    # Run Gemini on group data
    response = client.models.generate_content(...)
    return i, answer, start_time, processing_time

# Parallel execution
with ThreadPoolExecutor(max_workers=3) as executor:
    futures = [executor.submit(process_single_group, i) for i in range(5)]
    for future in as_completed(futures):
        # Handle results as they complete
```

### UI Container Pattern
Handler functions accept optional `ui_container` parameter:
- If provided: Show real-time progress in Streamlit UI
- If None: Silent processing for non-UI contexts

### Logger Pattern
Handlers accept `logger` parameter (RealTimeProcessLogger):
- `logger.log_actual(level, message, data)` for tracking
- Use in conjunction with UI container for transparency

## Gemini Model Usage

- `gemini-2.0-flash`: Quick tasks (summarization, question classification)
- `gemini-2.5-flash`: Complex analysis (final answers, head agent consolidation)
- All models accessed via `google.genai.Client`

## Testing Notes

- Test with actual Korean HS classification queries
- Ensure `knowledge/` data files are present
- Monitor API usage as multi-agent system makes multiple Gemini calls
- Check parallel execution with different worker counts

## Future Improvements

The codebase includes comments about planned upgrades:
- Replace keyword-based search with embedding-based semantic search
- Upgrade search index to vector database
- Methods marked with "NOTE: 향후 임베딩 기반 semantic search로 교체 예정"