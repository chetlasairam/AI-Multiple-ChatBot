from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os
import pickle
import sqlite3
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyPDFLoader
import streamlit as st_app
from langchain.callbacks import get_openai_callback
import tempfile

def establish_db_connection():
    db_conn = sqlite3.connect('assistant.db')
    db_cursor = db_conn.cursor()

    db_cursor.execute("PRAGMA table_info(conversation_log)")
    column_data = [column[1] for column in db_cursor.fetchall()]
    if "user_query" in column_data and "assistant_id" not in column_data:
        db_cursor.execute('''
            CREATE TABLE IF NOT EXISTS new_conversation_log (
                entry_id INTEGER PRIMARY KEY AUTOINCREMENT,
                assistant_id TEXT,
                user_query TEXT,
                assistant_reply TEXT
            )
        ''')
        db_conn.commit()

        db_cursor.execute('''
            INSERT INTO new_conversation_log (user_query, assistant_reply)
            SELECT user_query, assistant_reply FROM conversation_log
        ''')
        db_conn.commit()

        db_cursor.execute("DROP TABLE conversation_log")
        db_cursor.execute("ALTER TABLE new_conversation_log RENAME TO conversation_log")
        db_conn.commit()

    else:
        db_cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversation_log (
                entry_id INTEGER PRIMARY KEY AUTOINCREMENT,
                assistant_id TEXT,
                user_query TEXT,
                assistant_reply TEXT
            )
        ''')
        db_conn.commit()

    return db_conn, db_cursor

def history_clear(db_conn, db_cursor, bot_id):
    db_cursor.execute('DELETE FROM conversation_log WHERE assistant_id = ?', (bot_id,))
    db_conn.commit()

def process_pdf_for_vector_storage(pdf_path):
    document_loader = PyPDFLoader(file_path=pdf_path)
    document_data = document_loader.load()
    doc_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
    split_docs = doc_splitter.split_documents(documents=document_data)
    doc_embeddings = OpenAIEmbeddings()
    vector_storage = FAISS.from_documents(split_docs, doc_embeddings)
    return vector_storage

def store_vector_data(vector_store, store_file="vector_data.pkl"):
    with open(store_file, "wb") as file:
        pickle.dump(vector_store, file)

def retrieve_vector_data(store_file="vector_data.pkl"):
    with open(store_file, "rb") as file:
        vector_store = pickle.load(file)
    return vector_store

answer_prompt_template = """Based on the context in the attached PDF,
                            provide an answer using only this context. Include text from the "answer" 
                            section in the document where possible without errors.
                            If the answer isn't in the context, say "I don't know." Avoid inventing an answer.

                            CONTEXT: {context}
                            QUERY: {question}
                         """

ANSWER_PROMPT = PromptTemplate(
    template=answer_prompt_template, 
    input_variables=["context", "question"]
)
def initiate_assistant_bot():
    st_app.set_page_config(page_title="Assistant Bot", layout="wide")

    st_app.markdown("""
        <style>
        .userbox {
            color: #4f8bf9;
            background-color: #eef2f7;
            padding: 10px;
            border-radius: 10px;
            margin: 10px 0;
        }
        .responsebox {
            color: #f94f8b;
            background-color: #f7eef2;
            padding: 10px;
            border-radius: 10px;
            margin: 10px 0;
        }
        .streamlit-sidebar {
            background-color: #f4f4f8;
        }
        .stButton>button {
            width: 100%;
            border-radius: 10px;
            padding: 10px;
        }
        </style>
    """, unsafe_allow_html=True)

    st_app.title("BOT")
    st_app.sidebar.header("Assistant Bot Settings")

    db_conn, db_cursor = establish_db_connection()

    if 'assistants' not in st_app.session_state:
        st_app.session_state.assistants = {}

    new_bot_id = st_app.sidebar.text_input("Enter new bot ID:")
    if st_app.sidebar.button("Create Bot"):
        if new_bot_id and new_bot_id not in st_app.session_state.assistants:
            st_app.session_state.assistants[new_bot_id] = {"ai_model": "gpt-3.5-turbo-0613", "reply_temperature": 0.7, "token_limit": 500}

    chosen_bot_id = st_app.sidebar.selectbox("Choose a bot:", list(st_app.session_state.assistants.keys()))

    if chosen_bot_id:
        configure_bot(chosen_bot_id)

    if st_app.sidebar.button("Remove Bot"):
        if chosen_bot_id and chosen_bot_id in st_app.session_state.assistants:
            db_cursor.execute("DELETE FROM conversation_log WHERE assistant_id = ?", (chosen_bot_id,))
            db_conn.commit()
            del st_app.session_state.assistants[chosen_bot_id]
            st_app.experimental_rerun()

    if chosen_bot_id:
        manage_bot_conversation(chosen_bot_id, db_conn, db_cursor)

    db_conn.close()

def configure_bot(bot_id):
    bot_settings = st_app.session_state.assistants[bot_id]
    bot_settings['ai_model'] = st_app.sidebar.selectbox(f"Choose AI model for {bot_id}", ["gpt-3.5-turbo-0613", "gpt-3.5-turbo-0301", "gpt-4-0314"], key=f"ai_model_{bot_id}")
    bot_settings['reply_temperature'] = st_app.sidebar.slider(f"Adjust reply temperature for {bot_id}", min_value=0.20, max_value=1.00, value=bot_settings['reply_temperature'], step=0.20, key=f"temp_{bot_id}")
    bot_settings['token_limit'] = st_app.sidebar.slider(f"Set token limit for {bot_id}", min_value=200, max_value=1200, value=bot_settings['token_limit'], step=200, key=f"tokens_{bot_id}")

def manage_bot_conversation(bot_id, db_conn, db_cursor):
    st_app.sidebar.markdown(f"## Assistant Bot: {bot_id}")

    load_dotenv('key.env')

    if bot_id not in st_app.session_state.assistants:
        st_app.session_state.assistants[bot_id] = {"ai_model": None, "reply_temperature": 0.7, "token_limit": 500, "vector_store": None}

    assistant_bot = st_app.session_state.assistants[bot_id]

    if 'pdf_doc' not in assistant_bot:
        assistant_bot['pdf_doc'] = None

    pdf_upload = st_app.sidebar.file_uploader(f"Upload PDF for {bot_id}", type="pdf", key=f"pdf_uploader_{bot_id}")
    if pdf_upload is not None:
        temp_pdf_name = pdf_upload.name
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(pdf_upload.getvalue())
            temp_pdf_path = temp_pdf.name

        assistant_bot['vector_store'] = process_pdf_for_vector_storage(temp_pdf_path)
        store_vector_data(assistant_bot['vector_store'])
        assistant_bot['pdf_doc'] = temp_pdf_name

    if assistant_bot['pdf_doc']:
        st_app.sidebar.markdown(f"<div style='margin: 10px 0; padding: 10px; background-color: #f4f4f8; border-radius: 10px;'>Uploaded PDF: <b>{assistant_bot['pdf_doc']}</b></div>", unsafe_allow_html=True)

    chat_bot = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), model=assistant_bot['ai_model'], temperature=assistant_bot['reply_temperature'], max_tokens=assistant_bot['token_limit'])

    user_query = st_app.text_input("Your Question:", key=f"query_{bot_id}")
    if user_query.strip():  # Checks if the query is not just empty or whitespace
        query_prompt = f"{user_query}\n"

        if os.path.exists("vector_data.pkl"):
            vector_store = retrieve_vector_data()

            answer_chain = RetrievalQA.from_chain_type(llm=chat_bot,
                                    chain_type="stuff",
                                    retriever=vector_store.as_retriever(),
                                    input_key="query",
                                    return_source_documents=True,
                                    chain_type_kwargs={"prompt":ANSWER_PROMPT})

            st_app.sidebar.subheader("Response Tokens:")
            with get_openai_callback() as callback:
                bot_response = answer_chain(query_prompt)
            st_app.sidebar.write(callback)
            bot_response = bot_response['result']

            db_cursor.execute("INSERT INTO conversation_log (assistant_id, user_query, assistant_reply) VALUES (?, ?, ?)", (bot_id, query_prompt, bot_response))
            db_conn.commit()
            response_placeholder = st_app.empty()
            response_placeholder.markdown(f"<div class='userbox'>Assistant Bot {bot_id}: {bot_response}</div>", unsafe_allow_html=True)

    st_app.header("Conversation History for " + bot_id)

    if st_app.button("Clear History for " + bot_id):
        history_clear(db_conn, db_cursor, bot_id)
        
    conversation_history = db_cursor.execute("SELECT user_query, assistant_reply FROM conversation_log WHERE assistant_id=?", (bot_id,)).fetchall()
    for query, response in conversation_history[::-1]:
        st_app.markdown(f"<div class='userbox'>User: {query}</div>", unsafe_allow_html=True)
        st_app.markdown(f"<div class='responsebox'>Bot {bot_id}: {response}</div>", unsafe_allow_html=True)

    st_app.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

if __name__ == "__main__":
    initiate_assistant_bot()
