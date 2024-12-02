import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tldextract
import requests
import json
import itertools
import difflib
import base64
from datetime import datetime
import time
import concurrent.futures
from functools import partial
import numpy as np

st.set_page_config(page_title="SERP Similarity Analysis", layout="wide")

@st.cache_data(ttl=3600)
def make_api_request(api_login, api_password, keyword, location_code=2840):
    post_data = {
        "language_code": "en",
        "location_code": location_code,
        "keyword": keyword,
        "calculate_rectangles": True,
        "device": "mobile",
        "depth": 10
    }
    
    try:
        response = requests.post(
            "https://api.dataforseo.com/v3/serp/google/organic/live/regular",
            auth=(api_login, api_password),
            json=[post_data]
        )
        response_json = response.json()
        
        # Debug information
        if response.status_code != 200:
            st.error(f"API Error: Status code {response.status_code}")
            st.error(f"Response: {response_json}")
            return None
            
        if "status_code" in response_json and response_json["status_code"] != 20000:
            st.error(f"API Error: {response_json.get('status_message', 'Unknown error')}")
            st.error(f"Full response: {response_json}")
            return None
            
        return response_json
    except Exception as e:
        st.error(f"Error making API request: {str(e)}")
        return None

@st.cache_data(ttl=3600)
def get_serp_domains(results):
    domains = []
    try:
        if not results:
            st.warning("No results received from API")
            return domains
            
        if not isinstance(results, dict):
            st.warning(f"Unexpected results format: {type(results)}")
            return domains
            
        tasks = results.get('tasks', [])
        if not tasks:
            st.warning("No tasks found in API response")
            st.write("API Response:", results)
            return domains
            
        if len(tasks) == 0:
            st.warning("Empty tasks list in API response")
            return domains
            
        result = tasks[0].get('result', [])
        if not result:
            st.warning("No result found in task")
            st.write("Task data:", tasks[0])
            return domains
            
        if len(result) == 0:
            st.warning("Empty result list in task")
            return domains
            
        items = result[0].get('items', [])
        if not items:
            st.warning("No items found in result")
            st.write("Result data:", result[0])
            return domains
            
        for item in items:
            if 'url' in item:
                ext = tldextract.extract(item['url'])
                domain = f"{ext.domain}.{ext.suffix}"
                domains.append(domain)
            else:
                st.warning(f"No URL found in item: {item}")
                
        if not domains:
            st.warning("No domains were extracted from the results")
            
    except Exception as e:
        st.error(f"Error processing SERP results: {str(e)}")
        st.write("Results that caused error:", results)
    return domains

@st.cache_data(ttl=3600)
def calculate_similarity(serp_comp):
    keyword_diffs = {}
    
    # Process keyword pairs
    for (kw1, domains1), (kw2, domains2) in itertools.combinations(serp_comp.items(), 2):
        # Convert domains to sets for comparison
        set1 = set(domains1)
        set2 = set(domains2)
        
        if not set1 or not set2:  # Skip if either domain list is empty
            continue
            
        # Calculate Jaccard similarity
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        if union > 0:
            similarity = (intersection / union) * 100
        else:
            similarity = 0
            
        keyword_diffs[(kw1, kw2)] = round(similarity, 2)
    
    return keyword_diffs

def create_similarity_matrix(similarity_dict, keywords):
    # Initialize DataFrame with zeros
    similarity_df = pd.DataFrame(0.0, index=keywords, columns=keywords, dtype=float)
    
    # Fill the matrix with similarity scores
    for (kw1, kw2), score in similarity_dict.items():
        similarity_df.at[kw1, kw2] = score
        similarity_df.at[kw2, kw1] = score  # Mirror the score
    
    # Set diagonal to 100%
    np.fill_diagonal(similarity_df.values, 100.0)
    
    return similarity_df

def plot_heatmap(similarity_df):
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(similarity_df.astype(float), 
                annot=True, 
                fmt=".2f", 
                cmap="YlGnBu",
                ax=ax)
    plt.title("Keyword SERP Similarity Matrix")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    return fig

def get_similar_keyword_pairs(similarity_df, threshold):
    """Extract keyword pairs that have similarity above the threshold."""
    similar_pairs = []
    # Get upper triangle of matrix to avoid duplicates
    for i in range(len(similarity_df.index)):
        for j in range(i + 1, len(similarity_df.columns)):
            similarity = similarity_df.iloc[i, j]
            if similarity >= threshold:
                keyword1 = similarity_df.index[i]
                keyword2 = similarity_df.columns[j]
                similar_pairs.append({
                    'Keyword 1': keyword1,
                    'Keyword 2': keyword2,
                    'Similarity Score': similarity
                })
    return pd.DataFrame(similar_pairs)

def main():
    st.title("SERP Similarity Analysis")
    
    st.write("""
    This app analyzes the similarity between keyword SERPs using DataforSEO API.
    Upload a CSV file with your keywords and enter your API credentials to begin.
    """)
    
    with st.sidebar:
        st.header("API Credentials")
        api_login = st.text_input("DataforSEO API Login", type="password")
        api_password = st.text_input("DataforSEO API Password", type="password")
        
        st.header("Settings")
        location_code = st.number_input("Location Code (default: 2840 for US)", value=2840)
        batch_size = st.slider(
            "Parallel Processing Batch Size", 
            min_value=1, 
            max_value=10, 
            value=5,
            help="Number of keywords to process simultaneously. Higher values = faster processing but may hit API rate limits. This does NOT affect the number of SERP results per keyword."
        )
        
        st.markdown("""
        ℹ️ **About Batch Size:**
        - This controls how many keywords are processed in parallel
        - Example: With batch size 5 and 100 keywords, the app will process 5 keywords simultaneously
        - Higher values = faster processing but more API load
        - Lower values = slower but more reliable
        - Does NOT affect the number of SERP results per keyword
        """)
        
        st.header("Similarity Analysis")
        similarity_threshold = st.slider(
            "Similarity Threshold (%)", 
            min_value=0, 
            max_value=100, 
            value=65,
            help="Export keyword pairs with similarity score above this threshold"
        )
        
        st.markdown("""
        ℹ️ **About Similarity Threshold:**
        - Higher threshold = stricter similarity requirement
        - Lower threshold = more keyword pairs included
        - 65% is a good starting point for finding related keywords
        - Adjust based on your specific needs
        """)
    
    st.subheader("Upload Keywords")
    uploaded_file = st.file_uploader(
        "Upload your keywords CSV file", 
        type=['csv'],
        help="Make sure your CSV file has a column named 'Keyword', 'Keywords', 'keyword', or 'keywords'"
    )
    
    if uploaded_file is not None:
        try:
            file_details = {
                "Filename": uploaded_file.name,
                "File size": f"{uploaded_file.size} bytes"
            }
            st.write("File uploaded successfully:")
            st.json(file_details)
            
            try:
                df = pd.read_csv(uploaded_file)
            except UnicodeDecodeError:
                df = pd.read_csv(uploaded_file, encoding='latin1')
            
            # Find keyword column
            keyword_columns = [col for col in df.columns if col.lower().strip() in ['keyword', 'keywords']]
            
            if not keyword_columns:
                st.error("No 'keyword' or 'keywords' column found in the CSV file.")
                st.write("Available columns:", df.columns.tolist())
                return
            
            keyword_column = keyword_columns[0]
            keywords = df[keyword_column].dropna().unique().tolist()
            
            st.subheader("Preview of Keywords")
            st.write(f"Found {len(keywords)} unique keywords in column '{keyword_column}'")
            st.write("First 5 keywords:", keywords[:5])
            
            if st.button("Start Analysis"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Add debug section
                debug_expander = st.expander("Debug Information")
                
                serp_comp = {}
                batches = [keywords[i:i + batch_size] for i in range(0, len(keywords), batch_size)]
                
                for batch_idx, batch in enumerate(batches):
                    status_text.text(f"Processing batch {batch_idx + 1}/{len(batches)}")
                    
                    with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
                        futures = [
                            executor.submit(make_api_request, api_login, api_password, keyword, location_code)
                            for keyword in batch
                        ]
                        
                        for keyword, future in zip(batch, futures):
                            with debug_expander:
                                st.write(f"Processing keyword: {keyword}")
                                
                            results = future.result()
                            if results:
                                domains = get_serp_domains(results)
                                if domains:
                                    with debug_expander:
                                        st.write(f"Found domains for {keyword}:", domains)
                                    serp_comp[keyword] = domains
                                else:
                                    with debug_expander:
                                        st.write(f"No domains found for keyword: {keyword}")
                            else:
                                with debug_expander:
                                    st.write(f"No results returned for keyword: {keyword}")
                    
                    progress = (batch_idx + 1) / len(batches)
                    progress_bar.progress(progress)
                    time.sleep(0.2)
                
                with debug_expander:
                    st.write("Final serp_comp dictionary:", serp_comp)
                
                if len(serp_comp) < 2:
                    st.error("""
                    Not enough valid results to calculate similarities. Please check:
                    1. Your API credentials are correct
                    2. You have sufficient API credits
                    3. The API is responding correctly (check Debug Information below)
                    """)
                    return
                
                status_text.text("Calculating similarity matrix...")
                similarity_dict = calculate_similarity(serp_comp)
                similarity_df = create_similarity_matrix(similarity_dict, keywords)
                
                st.subheader("Similarity Matrix")
                st.dataframe(similarity_df)
                
                st.subheader("Similarity Heatmap")
                fig = plot_heatmap(similarity_df)
                st.pyplot(fig)
                
                # Get similar keyword pairs based on threshold
                similar_pairs_df = get_similar_keyword_pairs(similarity_df, similarity_threshold)
                
                st.subheader(f"Similar Keyword Pairs (Similarity ≥ {similarity_threshold}%)")
                if not similar_pairs_df.empty:
                    st.dataframe(similar_pairs_df)
                    
                    # Download buttons
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Full similarity matrix download
                        csv_full = similarity_df.to_csv()
                        st.download_button(
                            label="Download Full Similarity Matrix (CSV)",
                            data=csv_full,
                            file_name=f"similarity_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    
                    with col2:
                        # Similar pairs download
                        csv_pairs = similar_pairs_df.to_csv(index=False)
                        st.download_button(
                            label=f"Download Similar Pairs ≥{similarity_threshold}% (CSV)",
                            data=csv_pairs,
                            file_name=f"similar_pairs_{similarity_threshold}pct_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                else:
                    st.info(f"No keyword pairs found with similarity ≥ {similarity_threshold}%")
                
                status_text.text("Analysis complete!")
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
