# SERP Similarity Analysis App

This Streamlit application analyzes the similarity between keyword SERPs (Search Engine Results Pages) using the DataforSEO API. It helps you identify keyword cannibalization and understand the relationship between different search terms based on their search results.

## Features

- Upload CSV files containing keywords
- Analyze SERP similarities using DataforSEO API
- Generate similarity matrices and heatmaps
- Download results as CSV
- Interactive visualization of results
- Support for different location codes
- Parallel processing for faster analysis
- Configurable batch processing
- Caching for improved performance

## Performance Optimizations

- **Parallel Processing**: Processes multiple keywords simultaneously
- **Batch Processing**: Configurable batch size for optimal performance
- **Caching**: Results are cached for 1 hour to avoid redundant API calls
- **Memory Optimization**: Efficient data structures and vectorized operations
- **Duplicate Handling**: Automatic removal of duplicate keywords
- **Smart Rate Limiting**: Configurable batch size to balance speed and API limits

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
3. Configure batch size (1-10 keywords per batch)
   - Higher values = faster processing but may hit API limits
   - Lower values = more reliable but slower processing
4. Upload a CSV file containing your keywords
5. Click "Start Analysis" to begin processing
6. View the results in the similarity matrix and heatmap
7. Download the results as CSV if needed

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

## Performance Tips

1. **Batch Size Selection**:
   - Start with a batch size of 5
   - Increase if processing is too slow and you're not hitting API limits
   - Decrease if you're getting API rate limit errors

2. **Optimal Dataset Size**:
   - The app performs best with up to 100 keywords
   - Larger datasets may require longer processing times
   - Consider splitting very large datasets into smaller chunks

3. **Cache Usage**:
   - Results are cached for 1 hour
   - Rerunning the same analysis within this period will be much faster
   - Clear your browser cache if you need fresh results

## Error Handling

The app includes comprehensive error handling for:
- Invalid API credentials
- Incorrect CSV format
- API rate limiting
- Network issues
- Memory constraints

## Troubleshooting

1. **API Rate Limits**:
   - Reduce batch size
   - Add delays between batches
   - Contact DataforSEO for rate limit increases

2. **Performance Issues**:
   - Check your internet connection
   - Reduce the number of keywords
   - Clear browser cache
   - Restart the application

## Support

For issues or questions, please open an issue in the repository.
