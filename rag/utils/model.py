from dotenv import load_dotenv
load_dotenv()

from langchain_ollama import ChatOllama, OllamaLLM

from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate


from langchain import hub

from langchain_community.vectorstores import FAISS
from .embed import get_embeddings

import pprint

def run_llm(query: str):
    llm = OllamaLLM(
        model="llama3.2:1b",
        )
    
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=retrieval_qa_chat_prompt
    )
    
    vector_store = FAISS.load_local(
        folder_path="faiss_index_taco",
        embeddings=get_embeddings(),
        allow_dangerous_deserialization=True
    )
    retrieval_chain = create_retrieval_chain(
        retriever=vector_store.as_retriever(),
        combine_docs_chain=combine_docs_chain,
    )
    
    result = retrieval_chain.invoke(
        input={
            "input": query
            }
        )
    result = StrOutputParser().parse(text=result)
    return result

def run_llm_multilangual(query: str, query_lang: str):
    
    if query_lang != "en":
        llm_translate = OllamaLLM(
            model="mistral:latest",
            temperature=0
        )
        template_translation_to_en = f"""
        Translate the following korean text to English language. Answer just the translated text.

        <text>

        {query}

        </text>
        """
        input_translation = llm_translate.invoke(template_translation_to_en)
        # print(f"Translated query in English... \n {input_translation}")
        target_query = input_translation
    else:
        target_query = query
        
    llm = OllamaLLM(
        model="mistral:latest",
        )
    
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=retrieval_qa_chat_prompt
    )
    
    vector_store = FAISS.load_local(
        folder_path="faiss_index_taco",
        embeddings=get_embeddings(),
        allow_dangerous_deserialization=True
    )
    retrieval_chain = create_retrieval_chain(
        retriever=vector_store.as_retriever(),
        combine_docs_chain=combine_docs_chain,
    )
    
    result = retrieval_chain.invoke(
        input={
            "input": target_query
            }
        )
    result = StrOutputParser().parse(text=result)
    print("result generated")
    
    if query_lang != "en":
        llm_translate_result = OllamaLLM(
            model="ollama-ko-0502:latest",
        )
        template_translation_to_ko = f"""
        Translate the following english text to korean language. Answer just the translated text.

        <text>

        {result['answer']}

        </text>
        """
        result_translation = llm_translate_result.invoke(template_translation_to_ko)
        # print(f"Translated result in Korean... \n {result_translation}")
        result['input'] = query
        result['answer'] = result_translation
    
    return result


if __name__ == "__main__":
    print("utils/model.py is running...")
    context_result = run_llm("tell me about uniform standard")
    pprint.pprint(context_result)
