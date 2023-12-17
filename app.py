from flashrank.Ranker import Ranker, RerankRequest
import streamlit as st
import json

print("Import successful!")

def get_result(query, passages, choice):
    if choice == "Nano":
        ranker = Ranker()
    elif choice == "Small":
        ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="/opt")
    elif choice == "Medium":
        ranker = Ranker(model_name="rank-T5-flan", cache_dir="/opt")
    elif choice == "Large":
        ranker = Ranker(model_name="ms-marco-MultiBERT-L-12", cache_dir="/opt")

    rerankrequest = RerankRequest(query=query, passages=passages)
    results = ranker.rerank(rerankrequest)
    print(results)

    return results
 
st.set_page_config(
    layout="wide",
    page_title="ReRanking App"
)

def main():
    st.title("ReRanking using Flash Rank")
    st.sidebar.write("According to the Model Size 👇")
    menu = ["Nano", "Small", "Medium", "Large"]
    choice = st.sidebar.selectbox("Choose", menu)

    st.sidebar.info("""
**Model Options:**
- **Nano**: ~4MB, blazing fast model with competitive performance (ranking precision).
- **Small**: ~34MB, slightly slower with the best performance (ranking precision).
- **Medium**: ~110MB, slower model with the best zero-shot performance (ranking precision).
- **Large**: ~150MB, slower model with competitive performance (ranking precision) for 100+ languages.
""")

    with st.expander("About Flash Rank"):
        st.markdown("""
        **Flash Rank**: Ultra-lite & Super-fast Python library for search & retrieval re-ranking.

        - **Ultra-lite**: No heavy dependencies. Runs on CPU with a tiny ~4MB reranking model.
        - **Super-fast**: Speed depends on the number of tokens in passages and query, plus model depth.
        - **Cost-efficient**: Ideal for serverless deployments with low memory and time requirements.
        - **Based on State-of-the-Art Cross-encoders**: Includes models like ms-marco-TinyBERT-L-2-v2 (default), ms-marco-MiniLM-L-12-v2, rank-T5-flan, and ms-marco-MultiBERT-L-12.
        - **Sleek Models for Efficiency**: Designed for minimal overhead in user-facing scenarios.

        _Flash Rank is tailored for scenarios requiring efficient and effective reranking, balancing performance with resource usage._
        """)

    with st.form(key='input_form'):
        query_input = st.text_area("Query Input")
        context_input = st.text_area("Context Input")
        submit_button = st.form_submit_button(label='ReRank')

    if submit_button:
        with st.spinner('Processing...'):
            passages = json.loads(context_input)
            print("passages: ", passages, type(passages))
            result = get_result(query_input, passages, choice)
            st.subheader("Please find the ReRanked results below 👇")
            st.json(result)



if __name__ == "__main__":
    main()