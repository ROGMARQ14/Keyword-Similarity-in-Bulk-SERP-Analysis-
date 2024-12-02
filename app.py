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

st.set_page_config(page_title="SERP Similarity Analysis", layout="wide")

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

def get_serp_domains(results):
    domains = []
    try:
        if results and isinstance(results, dict):
            tasks = results.get('tasks', [])
            if tasks and len(tasks) > 0:
                result = tasks[0].get('result', [])
                if result and len(result) > 0:
                    items = result[0].get('items', [])
                    for item in items:
                        if 'url' in item:
                            ext = tldextract.extract(item['url'])
                            domain = f"{ext.domain}.{ext.suffix}"
                            domains.append(domain)
    except Exception as e:
        st.error(f"Error processing SERP results: {str(e)}")
    return domains

def calculate_similarity(serp_comp):
    keyword_diffs = {}
    for (kw1, comp1), (kw2, comp2) in itertools.combinations(serp_comp.items(), 2):
        comp1_str = ' '.join(comp1)
        comp2_str = ' '.join(comp2)
        sm = difflib.SequenceMatcher(None, comp1_str, comp2_str)
        ratio = round(sm.ratio() * 100, 2)
        keyword_diffs[(kw1, kw2)] = ratio
    return keyword_diffs

def create_similarity_matrix(similarity_dict, keywords):
    similarity_df = pd.DataFrame(index=keywords, columns=keywords)
    
    for (kw1, kw2), ratio in similarity_dict.items():
        similarity_df.at[kw1, kw2] = ratio
        similarity_df.at[kw2, kw1] = ratio
    
    for kw in keywords:
        similarity_df.at[kw, kw] = 100.0
    
    return similarity_df.fillna(0)

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
    
    # File upload
    uploaded_file = st.file_uploader("Upload your keywords CSV file", type=['csv'])
    
    if uploaded_file and api_login and api_password:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Find the keyword column
            keyword_columns = [col for col in df.columns if col.lower().strip() in ['keyword', 'keywords']]
            
            if not keyword_columns:
                st.error("No 'keyword' or 'keywords' column found in the CSV file.")
                return
                
            keyword_column = keyword_columns[0]
            keywords = df[keyword_column].dropna().tolist()
            
            if st.button("Start Analysis"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                serp_comp = {}
                for idx, keyword in enumerate(keywords):
                    status_text.text(f"Processing keyword: {keyword}")
                    results = make_api_request(api_login, api_password, keyword, location_code)
                    
                    if results:
                        domains = get_serp_domains(results)
                        serp_comp[keyword] = domains
                        
                    progress = (idx + 1) / len(keywords)
                    progress_bar.progress(progress)
                    time.sleep(0.5)  # Avoid rate limiting
                
                status_text.text("Calculating similarity matrix...")
                similarity_dict = calculate_similarity(serp_comp)
                similarity_df = create_similarity_matrix(similarity_dict, keywords)
                
                # Display results
                st.subheader("Similarity Matrix")
                st.dataframe(similarity_df)
                
                # Plot heatmap
                st.subheader("Similarity Heatmap")
                fig = plot_heatmap(similarity_df)
                st.pyplot(fig)
                
                # Download buttons
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
