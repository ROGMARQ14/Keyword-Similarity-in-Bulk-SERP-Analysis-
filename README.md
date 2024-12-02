# SERP Similarity Analysis App

This Streamlit application analyzes the similarity between keyword SERPs (Search Engine Results Pages) using the DataforSEO API. It helps you identify keyword cannibalization and understand the relationship between different search terms based on their search results.

## Features

- Upload CSV files containing keywords
- Analyze SERP similarities using DataforSEO API
- Generate similarity matrices and heatmaps
- Download results as CSV
- Interactive visualization of results
- Support for different location codes

## Prerequisites

- Python 3.7+
- DataforSEO API credentials (login and password)

## Installation

1. Clone this repository or download the files
2. Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. Enter your DataforSEO API credentials in the sidebar
3. Upload a CSV file containing your keywords (column should be labeled "Keyword", "Keywords", "keyword", or "keywords")
4. Click "Start Analysis" to begin processing
5. View the results in the similarity matrix and heatmap
6. Download the results as CSV if needed

## CSV File Format

Your CSV file should contain a column with one of these headers:
- "Keyword"
- "Keywords"
- "keyword"
- "keywords"

Example:
```csv
Keyword
seo tools
keyword research
content optimization
```

## Location Codes

The default location code is 2840 (United States). You can change this in the sidebar settings. Common location codes:
- 2840: United States
- 2826: United Kingdom
- 2036: Canada
- 2036: Australia

## Output

The app generates:
1. A similarity matrix showing the percentage similarity between each keyword pair
2. A heatmap visualization of the similarity matrix
3. Downloadable CSV file with the results

## Error Handling

The app includes comprehensive error handling for:
- Invalid API credentials
- Incorrect CSV format
- API rate limiting
- Network issues

## Support

For issues or questions, please open an issue in the repository.
