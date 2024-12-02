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

# Configure Streamlit page
st.set_page_config(
    page_title="SERP Similarity Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'similarity_df' not in st.session_state:
    st.session_state.similarity_df = None
if 'similar_pairs_df' not in st.session_state:
    st.session_state.similar_pairs_df = None

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
            return domains
            
        if len(tasks) == 0:
            st.warning("Empty tasks list in API response")
            return domains
            
        result = tasks[0].get('result', [])
        if not result:
            st.warning("No result found in task")
            return domains
            
        if len(result) == 0:
            st.warning("Empty result list in task")
            return domains
            
        items = result[0].get('items', [])
        if not items:
            st.warning("No items found in result")
            return domains
            
        for item in items:
            if 'url' in item:
                ext = tldextract.extract(item['url'])
                domain = f"{ext.domain}.{ext.suffix}"
                domains.append(domain)
                
    except Exception as e:
        st.error(f"Error processing SERP results: {str(e)}")
    return list(set(domains))  # Remove duplicates

@st.cache_data(ttl=3600)
def calculate_similarity(serp_comp):
    keyword_diffs = {}
    
    for (kw1, domains1), (kw2, domains2) in itertools.combinations(serp_comp.items(), 2):
        set1 = set(domains1)
        set2 = set(domains2)
        
        if not set1 or not set2:
            continue
            
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        if union > 0:
            similarity = (intersection / union) * 100
        else:
            similarity = 0
            
        keyword_diffs[(kw1, kw2)] = round(similarity, 2)
    
    return keyword_diffs

def create_similarity_matrix(similarity_dict, keywords):
    similarity_df = pd.DataFrame(0.0, index=keywords, columns=keywords, dtype=float)
    
    for (kw1, kw2), score in similarity_dict.items():
        similarity_df.at[kw1, kw2] = score
        similarity_df.at[kw2, kw1] = score
    
    np.fill_diagonal(similarity_df.values, 100.0)
    return similarity_df

def plot_heatmap(similarity_df):
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        similarity_df.astype(float),
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        ax=ax
    )
    plt.title("Keyword SERP Similarity Matrix")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    return fig

def get_similar_keyword_pairs(similarity_df, threshold):
    similar_pairs = []
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

def sidebar_content():
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
            help="Number of keywords to process simultaneously"
        )
        
        st.header("Similarity Analysis")
        similarity_threshold = st.slider(
            "Similarity Threshold (%)",
            min_value=0,
            max_value=100,
            value=65,
            help="Export keyword pairs with similarity score above this threshold"
        )
        
        return api_login, api_password, location_code, batch_size, similarity_threshold

def main():
    st.title("SERP Similarity Analysis")
    
    # Get sidebar inputs
    api_login, api_password, location_code, batch_size, similarity_threshold = sidebar_content()
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload your keywords CSV file",
        type=['csv'],
        help="CSV file with a column named 'Keyword' or 'Keywords'"
    )
    
    if uploaded_file is not None:
        try:
            # Read and process CSV
            try:
                df = pd.read_csv(uploaded_file)
            except UnicodeDecodeError:
                df = pd.read_csv(uploaded_file, encoding='latin1')
            
            keyword_columns = [col for col in df.columns if col.lower().strip() in ['keyword', 'keywords']]
            
            if not keyword_columns:
                st.error("No 'keyword' or 'keywords' column found in the CSV file.")
                st.write("Available columns:", df.columns.tolist())
                return
            
            keyword_column = keyword_columns[0]
            keywords = df[keyword_column].dropna().unique().tolist()
            
            st.write(f"Found {len(keywords)} unique keywords")
            st.write("First 5 keywords:", keywords[:5])
            
            if st.button("Start Analysis"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Process keywords
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
                            results = future.result()
                            if results:
                                domains = get_serp_domains(results)
                                if domains:
                                    serp_comp[keyword] = domains
                    
                    progress = (batch_idx + 1) / len(batches)
                    progress_bar.progress(progress)
                    time.sleep(0.2)
                
                if len(serp_comp) < 2:
                    st.error("Not enough valid results. Please check your API credentials.")
                    return
                
                # Calculate similarities
                status_text.text("Calculating similarities...")
                similarity_dict = calculate_similarity(serp_comp)
                st.session_state.similarity_df = create_similarity_matrix(similarity_dict, keywords)
                
                # Generate similar pairs
                st.session_state.similar_pairs_df = get_similar_keyword_pairs(
                    st.session_state.similarity_df,
                    similarity_threshold
                )
                
                # Display results
                st.subheader("Similarity Matrix")
                st.dataframe(st.session_state.similarity_df)
                
                st.subheader("Similarity Heatmap")
                fig = plot_heatmap(st.session_state.similarity_df)
                st.pyplot(fig)
                
                # Display similar pairs
                st.subheader(f"Similar Keyword Pairs (Similarity ≥ {similarity_threshold}%)")
                if not st.session_state.similar_pairs_df.empty:
                    st.dataframe(st.session_state.similar_pairs_df)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            "Download Full Matrix (CSV)",
                            data=st.session_state.similarity_df.to_csv(),
                            file_name=f"similarity_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    
                    with col2:
                        st.download_button(
                            f"Download Similar Pairs ≥{similarity_threshold}% (CSV)",
                            data=st.session_state.similar_pairs_df.to_csv(index=False),
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
