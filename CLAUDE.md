# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an HS (Harmonized System) classification chatbot that uses AI to help classify products according to Korean Customs Service HS codes. The application is built with Streamlit and uses Google's Gemini AI for natural language processing and multi-agent systems for analyzing classification cases.

## Common Commands

### Development and Execution
```bash
# Install dependencies
pip install -r requirements.txt

# Run the Streamlit application
streamlit run main.py

# Activate virtual environment (if needed)
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/macOS
```

### Testing and Quality
The project doesn't have automated tests or linting configured. When making changes:
- Test manually by running the Streamlit app
- Verify all AI agents and search functions work correctly
- Check data loading from knowledge/ directory

## Architecture Overview

### Core Components

1. **main.py** (487 lines) - Streamlit web application
   - UI components and user interaction handling
   - Chat interface with real-time logging
   - Session state management
   - Integration with all AI processing functions

2. **utils.py** (1,435 lines) - Core business logic
   - `HSDataManager` class: Manages all HS classification data
   - `TariffTableSearcher` class: Searches tariff table data
   - `ParallelHSSearcher` class: Multi-agent parallel search system
   - Handler functions for different question types:
     - `handle_web_search()` - Web searches using Google Search API
     - `handle_hs_classification_cases()` - Korean customs classification cases
     - `handle_overseas_hs()` - US/EU customs classification cases
     - `handle_hs_manual_with_user_codes()` - HS code comparison analysis
     - `handle_hs_manual_with_parallel_search()` - HS manual search

3. **hs_search.py** (41 lines) - HS code lookup utilities
   - `lookup_hscode()` function for retrieving HS manual sections

### Multi-Agent System Architecture

The application uses a sophisticated multi-agent approach:
- **Data Partitioning**: Large datasets split into 5 groups for parallel processing
- **Agent Coordination**: Each agent processes its assigned data partition
- **Head Agent**: Consolidates results from all agents for final analysis
- **Real-time Logging**: All agent activities are logged transparently in the UI

### Data Structure

#### Knowledge Base (`knowledge/` directory)
- `HS분류사례_part1.json` to `HS분류사례_part10.json` - Korean customs classification cases (~1,000 cases)
- `HS위원회.json`, `HS협의회.json` - Committee and council decisions
- `hs_classification_data_us.json` - US customs classification cases
- `hs_classification_data_eu.json` - EU customs classification cases
- `hstable.json` - Tariff table for parallel search
- `통칙_grouped.json` - HS classification principles
- `grouped_11_end.json` - HS explanatory notes

#### Supporting Directories
- `hs해설서/` - HS manual processing scripts and PDFs
- `품목분류표_제작/` - Tariff table data processing scripts

### AI Question Classification

The system automatically classifies user questions into 6 types:
1. **AI Auto-classification** - Automatic routing to appropriate handler
2. **Web Search** - Real-time information about products/trends
3. **Korean HS Cases** - Search domestic classification cases
4. **Overseas HS Cases** - Search US/EU classification cases
5. **HS Manual Analysis (User Codes)** - Compare specific HS codes provided by user
6. **HS Manual Search** - Search HS explanatory notes

## Environment Configuration

### Required Environment Variables
Create `.env` file with:
```env
GOOGLE_API_KEY=your_google_gemini_api_key
```

### API Dependencies
- **Google Gemini AI**: Used for natural language processing and multi-agent coordination
- **Google Search API**: Used for real-time web searches (when available)

## Development Guidelines

### Code Style
- No specific linting configuration in place
- Follow existing patterns in the codebase
- Use Korean comments and documentation where appropriate
- Maintain the multi-agent logging transparency

### Data Handling
- All JSON data files use UTF-8 encoding
- Data loading is cached using `@st.cache_resource` for performance
- Handle missing data files gracefully with try/catch blocks

### AI Integration
- All AI calls use Google Gemini client
- Implement proper error handling for API failures
- Maintain context windows for multi-turn conversations
- Use streaming responses where possible for better UX

### Performance Considerations
- HSDataManager is cached to avoid reloading large datasets
- Multi-agent processing reduces latency for large searches
- Parallel search systems combine multiple data sources efficiently

## Key Features to Understand

### Real-time Logging System
The application provides transparent logging of all AI processing steps, making it easy to debug and understand the decision-making process.

### Multi-Agent Coordination
When processing large datasets, the system automatically partitions data and assigns agents to process different sections, then consolidates results.

### Parallel Search Engine
For HS manual searches, the system simultaneously searches both tariff table (`hstable.json`) and explanatory notes (`grouped_11_end.json`) to improve accuracy.

### Question Type Auto-detection
The system uses AI to automatically determine the most appropriate processing method for user questions, eliminating the need for users to select specific modes.