import streamlit as st
import torch
from transformers import pipeline
from datasets import load_dataset
import pandas as pd
import time

MODELS = {
    "BART-CNN (Production)": {
        "name": "facebook/bart-large-cnn",
        "max_len": 142,
        "min_len": 56,
        "info": "Industry standard - 3-4 sentence summaries"
    },
    "PEGASUS-XSum": {
        "name": "google/pegasus-xsum",
        "max_len": 64,
        "min_len": 10,
        "info": "Single sentence summaries"
    },
    "LED-16K": {
        "name": "allenai/led-base-16384",
        "max_len": 142,
        "min_len": 56,
        "info": "For very long documents"
    }
}

@st.cache_resource
def load_model(model_name):
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("summarization", model=model_name, device=device)

@st.cache_data
def load_examples():
    dataset = load_dataset("xsum", split="test")
    return dataset.select(range(20))

def main():
    st.set_page_config(page_title="Text Summarizer", page_icon="📝", layout="wide")
    
    st.title("📝 Text Summarization (Production-Ready)")
    st.markdown("**Local execution** | BART-CNN (industry standard) as default")
    
    st.sidebar.header("⚙️ Settings")
    
    selected_model = st.sidebar.selectbox(
        "Model",
        list(MODELS.keys()),
        help="BART-CNN recommended for production"
    )
    
    config = MODELS[selected_model]
    st.sidebar.info(f"**{config['info']}**\nModel: `{config['name']}`")
    
    max_length = st.sidebar.slider("Max Length", 30, 300, config['max_len'])
    min_length = st.sidebar.slider("Min Length", 10, 100, config['min_len'])
    
    device = "GPU" if torch.cuda.is_available() else "CPU"
    st.sidebar.success(f"🖥️ Running on: {device}")
    
    with st.spinner(f"Loading {selected_model}..."):
        summarizer = load_model(config['name'])
    
    tab1, tab2, tab3 = st.tabs(["📄 Summarize", "🎯 Examples", "⚖️ Compare"])
    
    with tab1:
        st.header("Summarize Your Text")
        
        text = st.text_area(
            "Enter text:",
            height=200,
            placeholder="Paste your article here..."
        )
        
        if st.button("🚀 Summarize", type="primary"):
            if text:
                with st.spinner("Generating..."):
                    start = time.time()
                    result = summarizer(
                        text,
                        max_length=max_length,
                        min_length=min_length,
                        do_sample=False
                    )
                    elapsed = time.time() - start
                
                summary = result[0]['summary_text']
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Input", f"{len(text.split())} words")
                col2.metric("Summary", f"{len(summary.split())} words")
                col3.metric("Time", f"{elapsed:.2f}s")
                
                st.success("**Summary:**")
                st.write(summary)
            else:
                st.warning("Please enter text to summarize")
    
    with tab2:
        st.header("Try Examples from XSum Dataset")
        
        examples = load_examples()
        
        idx = st.selectbox(
            "Select example:",
            range(len(examples)),
            format_func=lambda x: f"Example {x+1}"
        )
        
        example = examples[idx]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📰 Article")
            st.write(example['document'])
        
        with col2:
            st.subheader("🎯 XSum Reference")
            st.info(example['summary'])
        
        if st.button("Generate Summary"):
            with st.spinner("Generating..."):
                start = time.time()
                result = summarizer(
                    example['document'],
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=False
                )
                elapsed = time.time() - start
            
            st.success(f"**{selected_model} Summary:**")
            st.write(result[0]['summary_text'])
            st.caption(f"Generated in {elapsed:.2f}s")
    
    with tab3:
        st.header("⚖️ Compare Models")
        
        compare_text = st.text_area(
            "Text to compare:",
            height=150,
            placeholder="Enter text to compare across models..."
        )
        
        models_compare = st.multiselect(
            "Select models:",
            list(MODELS.keys()),
            default=["BART-CNN (Production)", "PEGASUS-XSum"]
        )
        
        if st.button("🔍 Compare") and compare_text and models_compare:
            results = []
            
            for model_name in models_compare:
                cfg = MODELS[model_name]
                temp_sum = load_model(cfg['name'])
                
                start = time.time()
                res = temp_sum(
                    compare_text,
                    max_length=cfg['max_len'],
                    min_length=cfg['min_len'],
                    do_sample=False
                )
                elapsed = time.time() - start
                
                summary = res[0]['summary_text']
                
                results.append({
                    'Model': model_name,
                    'Summary': summary,
                    'Words': len(summary.split()),
                    'Time': f"{elapsed:.2f}s"
                })
            
            st.subheader("Results")
            for r in results:
                with st.expander(f"{r['Model']} - {r['Words']} words"):
                    st.write(f"**Summary:** {r['Summary']}")
                    st.caption(f"Time: {r['Time']}")
            
            df = pd.DataFrame(results)
            st.dataframe(df, use_container_width=True)
            
            st.info("""
            **💡 Recommendation:**
            - **BART-CNN**: Production use (informative, 3-4 sentences)
            - **PEGASUS-XSum**: Headlines only (1 sentence)
            - **LED-16K**: Very long documents
            """)

if __name__ == "__main__":
    main()
