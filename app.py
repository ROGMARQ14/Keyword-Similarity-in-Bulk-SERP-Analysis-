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
import asyncio
import aiohttp
from functools import partial
import numpy as np

st.set_page_config(page_title="SERP Similarity Analysis", layout="wide")

# Cache the API response to avoid redundant calls
@st.cache_data(ttl=3600)  # Cache for 1 hour
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
        return response.json()
    except Exception as e:
        st.error(f"Error making API request: {str(e)}")
        return None

# Optimize domain extraction with caching
@st.cache_data(ttl=3600)
def get_serp_domains(results):
    domains = []
    try:
        if results and isinstance(results, dict):
            tasks = results.get('tasks', [])
            if tasks and len(tasks) > 0:
                result = tasks[0].get('result', [])
                if result and len(result) > 0:
                    items = result[0].get('items', [])
                    # Use set for faster lookups
                    seen_domains = set()
                    for item in items:
                        if 'url' in item:
                            ext = tldextract.extract(item['url'])
                            domain = f"{ext.domain}.{ext.suffix}"
                            if domain not in seen_domains:
                                domains.append(domain)
                                seen_domains.add(domain)
    except Exception as e:
        st.error(f"Error processing SERP results: {str(e)}")
    return domains

# Optimize similarity calculation
@st.cache_data(ttl=3600)
def calculate_similarity(serp_comp):
    keyword_diffs = {}
    # Pre-compute strings for comparison
    domain_strings = {
        kw: ' '.join(sorted(set(domains))) # Remove duplicates and sort for consistency
        for kw, domains in serp_comp.items()
    }
    
    # Use list comprehension for faster processing
    combinations = list(itertools.combinations(serp_comp.items(), 2))
    
    # Process in chunks for better performance
    def process_chunk(chunk):
        chunk_results = {}
        for (kw1, _), (kw2, _) in chunk:
            sm = difflib.SequenceMatcher(None, domain_strings[kw1], domain_strings[kw2])
            chunk_results[(kw1, kw2)] = round(sm.ratio() * 100, 2)
        return chunk_results
    
    # Split into chunks and process
    chunk_size = 100
    chunks = [combinations[i:i + chunk_size] for i in range(0, len(combinations), chunk_size)]
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        chunk_results = list(executor.map(process_chunk, chunks))
    
    # Combine results
    for result in chunk_results:
        keyword_diffs.update(result)
    
    return keyword_diffs

@st.cache_data(ttl=3600)
def create_similarity_matrix(similarity_dict, keywords):
    similarity_df = pd.DataFrame(index=keywords, columns=keywords)
    
    # Vectorized operations for better performance
    for (kw1, kw2), ratio in similarity_dict.items():
        similarity_df.at[kw1, kw2] = ratio
        similarity_df.at[kw2, kw1] = ratio
    
    # Set diagonal to 100% efficiently
    np.fill_diagonal(similarity_df.values, 100.0)
    
    return similarity_df.fillna(0)

@st.cache_data(ttl=3600)
def plot_heatmap(similarity_df):
    plt.figure(figsize=(12, 10))
    sns.heatmap(similarity_df.astype(float), annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title("Keyword SERP Similarity Matrix")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    return plt

def main():
    st.title("SERP Similarity Analysis")
    
    st.write("""
    This app analyzes the similarity between keyword SERPs using DataforSEO API.
    Upload a CSV file with your keywords and enter your API credentials to begin.
    """)
    
    # API Credentials input
    with st.sidebar:
        st.header("API Credentials")
        api_login = st.text_input("DataforSEO API Login", type="password")
        api_password = st.text_input("DataforSEO API Password", type="password")
        
        st.header("Settings")
        location_code = st.number_input("Location Code (default: 2840 for US)", value=2840)
        
        # Add batch size control
        batch_size = st.slider("Batch Size (keywords per request)", 
                             min_value=1, 
                             max_value=10, 
                             value=5,
                             help="Higher values are faster but may hit API limits")
    
    # File upload with better error handling and feedback
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
            
            # Read CSV efficiently
            try:
                df = pd.read_csv(uploaded_file, encoding='utf-8', usecols=lambda x: x.lower().strip() in ['keyword', 'keywords'])
            except UnicodeDecodeError:
                df = pd.read_csv(uploaded_file, encoding='latin1', usecols=lambda x: x.lower().strip() in ['keyword', 'keywords'])
            
            keyword_columns = df.columns.tolist()
            
            if not keyword_columns:
                st.error("""
                No 'keyword' column found in the CSV file.
                Please make sure your CSV has one of these column names: 'Keyword', 'Keywords', 'keyword', or 'keywords'
                """)
                return
            
            keyword_column = keyword_columns[0]
            keywords = df[keyword_column].dropna().unique().tolist()  # Use unique to remove duplicates
            
            st.subheader("Preview of Keywords")
            st.write(f"Found {len(keywords)} unique keywords in column '{keyword_column}'")
            st.write("First 5 keywords:", keywords[:5])
            
            if st.button("Start Analysis"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Process keywords in batches
                serp_comp = {}
                batches = [keywords[i:i + batch_size] for i in range(0, len(keywords), batch_size)]
                
                for batch_idx, batch in enumerate(batches):
                    status_text.text(f"Processing batch {batch_idx + 1}/{len(batches)}")
                    
                    # Process batch in parallel
                    with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
                        futures = [
                            executor.submit(make_api_request, api_login, api_password, keyword, location_code)
                            for keyword in batch
                        ]
                        
                        for keyword, future in zip(batch, futures):
                            results = future.result()
                            if results:
                                domains = get_serp_domains(results)
                                serp_comp[keyword] = domains
                    
                    progress = (batch_idx + 1) / len(batches)
                    progress_bar.progress(progress)
                    time.sleep(0.2)  # Reduced delay between batches
                
                status_text.text("Calculating similarity matrix...")
                similarity_dict = calculate_similarity(serp_comp)
                similarity_df = create_similarity_matrix(similarity_dict, keywords)
                
                st.subheader("Similarity Matrix")
                st.dataframe(similarity_df)
                
                st.subheader("Similarity Heatmap")
                fig = plot_heatmap(similarity_df)
                st.pyplot(fig)
                
                csv = similarity_df.to_csv()
                st.download_button(
                    label="Download Similarity Matrix (CSV)",
                    data=csv,
                    file_name=f"similarity_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
                status_text.text("Analysis complete!")
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
