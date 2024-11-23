from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from streamlit_chat import message

import langid
from utils.model import run_llm

import pprint


st.title("TACO | Two Armies Chat Once")
st.text("A chatbot to help KATUSA soldiers navigate U.S. and ROK Army regulations in both English and Korean language.")
st.text("카투사 병사들이 한미 육군 규정을 영어와 한국어로 탐색할 수 있도록 돕는 챗봇입니다.")
prompt = st.text_area("Prompt", placeholder="Enter your prompt")


# Sidebar user information
with st.sidebar:
    st.title("User Profile")

    # You can replace these with actual user data
    user_name = "JIUK KIM"
    user_email = "alex.jiuk.kim@gmail.com"

    profile_pic = get_profile_picture(user_email)
    st.image(profile_pic, width=150)
    st.write(f"**Name:** {user_name}")
    st.write(f"**Email:** {user_email}")


if prompt:
    with st.spinner("Generating reponse..."):
        detected_language = langid.classify(prompt) 
        
        generated_response = run_llm(query=prompt)
        documents_metadata_source = list([document.metadata["source"].replace("rag/src/", "") for document in generated_response["context"]])
        documents_metadata_page = list([document.metadata["page"] for document in generated_response["context"]])
        documents_page_contents = list([document.page_content for document in generated_response["context"]])
        sources_string = "\n\n".join([f"{documents_metadata_source} Page{documents_metadata_page}\n{documents_page_contents}" for documents_metadata_source, documents_metadata_page, documents_page_contents in zip(documents_metadata_source, documents_metadata_page, documents_page_contents)])
        
        st.subheader("Question")
        st.text(body=generated_response['input'])
        st.subheader("Answer")
        st.text(body=generated_response['answer'])
        st.subheader("Sources")
        st.text(body=sources_string)

# Add a footer
st.markdown("---")
st.markdown("Powered by LangChain and Streamlit")


if __name__ == "__main__":
    print("Two Armies Chat Once is running...")