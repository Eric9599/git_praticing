import os
import requests
import pandas as pd
import numpy as np
from uuid import uuid4
from dotenv import load_dotenv

# LangChain 相關
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import CharacterTextSplitter, TokenTextSplitter
from semantic_text_splitter import TextSplitter

# --- 1. 載入設定 ---
load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
SUBMIT_URL = os.getenv("HOMEWORK_SUBMIT_URL")
DATA_PATH = os.getenv("DATA_SOURCE_FOLDER_PATH", "./HW")
OUTPUT_CSV = os.getenv("OUTPUT_FILE_PATH", "2411232025_RAG_HW_01.csv")

if not API_KEY:
    print("* 錯誤：找不到 OPENAI_API_KEY，請檢查 .env 檔案")
    exit()

# --- 2. 初始化模型 ---
print(">>> 初始化 OpenAI 模型...")
try:
    llm = ChatOpenAI(
        api_key=API_KEY,
        model="gpt-4o",
        temperature=0
    )

    embedding_model = OpenAIEmbeddings(
        api_key=API_KEY,
        model="text-embedding-3-small"
    )
except Exception as e:
    print(f"* 模型初始化失敗: {e}")
    exit()

# --- 3. 資料讀取 (修改邏輯：保存檔名與內容的對應) ---
print(f">>> 正在從 {DATA_PATH} 讀取資料...")

# 我們不再合併成 all_text，而是存成一個列表，每項包含檔名與內容
documents_data = []
text_list = ["data_01.txt", "data_02.txt", "data_03.txt", "data_04.txt", "data_05.txt"]

for file_name in text_list:
    combined_path = os.path.join(DATA_PATH, file_name)
    try:
        with open(combined_path, "r", encoding="utf-8-sig") as f:
            content = f.read()
            if content:
                documents_data.append({
                    "file_name": file_name,  # 記住這個檔名！
                    "text": content
                })
    except FileNotFoundError:
        print(f"* 找不到檔案: {combined_path}")

if not documents_data:
    print("* 錯誤：沒有讀取到任何文本")
    exit()

# 讀取問題 CSV
csv_path = os.path.join(DATA_PATH, "questions.csv")
try:
    df_questions = pd.read_csv(csv_path, encoding="utf-8-sig")
    print(f" * 已讀取 {len(df_questions)} 個問題。")
except Exception as e:
    print(f" * 讀取 CSV 失敗: {csv_path} - {e}")
    exit()


# --- 4. 定義切塊函式 (保持不變) ---

def fixed_size_splitter(text):
    text_splitter = CharacterTextSplitter(chunk_size=400, chunk_overlap=50, separator="\n", length_function=len)
    return text_splitter.split_text(text)


def sliding_window_splitter(text):
    text_splitter = TokenTextSplitter(chunk_size=300, chunk_overlap=30, model_name="gpt-4")
    return text_splitter.split_text(text)


def semantic_splitter(text):
    splitter = TextSplitter((200, 1000))
    return splitter.chunks(text)


# --- 5. 處理切塊與來源標記 (新增函式) ---

def process_chunks_with_source(docs_data, splitter_func):
    """
    輸入: 文件列表 (含檔名)
    輸出: (所有 chunks 列表, 對應的 sources 列表)
    """
    all_chunks = []
    all_sources = []

    for doc in docs_data:
        file_name = doc["file_name"]
        text = doc["text"]

        # 對單一檔案進行切塊
        chunks = splitter_func(text)

        # 將切出來的每一塊，都標記上這個檔名
        for chunk in chunks:
            all_chunks.append(chunk)
            all_sources.append(file_name)

    return all_chunks, all_sources


# --- 6. 定義評分函式 ---

def get_score_from_api(q_id, answer):
    if not SUBMIT_URL: return 0
    payload = {"q_id": q_id, "student_answer": answer}
    try:
        response = requests.post(SUBMIT_URL, json=payload)
        response.raise_for_status()
        return response.json().get("score", 0)
    except Exception as e:
        print(f"  * API 評分連線失敗 (QID: {q_id}): {e}")
        return 0


# --- 7. 核心：向量計算邏輯 ---

def calculate_similarity(query_vector, chunk_vectors):
    q_vec = np.array(query_vector)
    c_vecs = np.array(chunk_vectors)
    scores = np.dot(c_vecs, q_vec)
    best_idx = np.argmax(scores)
    return best_idx, scores[best_idx]


def run_rag_pipeline(chunks, sources, method_name, questions_df):
    """
    現在多接收一個參數: sources (來源列表)
    """
    print(f"\n* 執行方法: {method_name} (Chunk數: {len(chunks)})")

    # 1. 批次向量化
    print("  正在將文字轉換為向量...")
    try:
        chunk_vectors = embedding_model.embed_documents(chunks)
    except Exception as e:
        print(f"* 向量化失敗: {e}")
        return []

    results = []

    # 自動偵測問題欄位
    possible_names = ['question', 'questions', 'Question', 'Questions', 'content']
    q_col = next((col for col in possible_names if col in questions_df.columns), None)

    if not q_col:
        print(f"* 找不到問題欄位! 現有: {questions_df.columns.tolist()}")
        return []

    # 2. 逐題檢索與回答
    for index, row in questions_df.iterrows():
        q_id = row['q_id']
        question = row[q_col]

        # A. 檢索
        try:
            query_vector = embedding_model.embed_query(question)
            best_idx, best_score = calculate_similarity(query_vector, chunk_vectors)

            #  修改點：根據 index 找回對應的 chunk 和 source
            best_chunk = chunks[best_idx]
            best_source = sources[best_idx]  # 這裡就能拿到正確的檔名 (如 data_02.txt)

        except Exception as e:
            print(f"  * 檢索計算失敗: {e}")
            best_chunk = "無資料"
            best_source = "Unknown"

        # B. 生成
        prompt = f"參考文件：{best_chunk}\n問題：{question}\n請回答問題，若無法回答請說不知道。"
        try:
            student_answer = llm.invoke(prompt).content
        except Exception as e:
            print(f"  * 生成失敗: {e}")
            student_answer = "生成錯誤"

        # C. 評分
        score = get_score_from_api(q_id, student_answer)
        print(f"  [Q{q_id}] 分數: {score}, 來源: {best_source}")

        results.append({
            "id": str(uuid4()),  # UUID
            "q_id": q_id,
            "method": method_name,
            "retrieve_text": best_chunk,
            "score": score,
            "source": best_source  # 這裡存入正確的檔名
        })

    return results


# --- 8. 執行與存檔 ---

print("\n>>> 開始切塊與處理來源...")

# 這裡改用新的處理函式，會同時回傳 chunks 和 sources
chunks_fixed, sources_fixed = process_chunks_with_source(documents_data, fixed_size_splitter)
chunks_sliding, sources_sliding = process_chunks_with_source(documents_data, sliding_window_splitter)
chunks_semantic, sources_semantic = process_chunks_with_source(documents_data, semantic_splitter)

final_data = []

# 呼叫 pipeline 時，把 sources 也傳進去
if chunks_fixed:
    final_data.extend(run_rag_pipeline(chunks_fixed, sources_fixed, "fixed_size", df_questions))

if chunks_sliding:
    final_data.extend(run_rag_pipeline(chunks_sliding, sources_sliding, "sliding_window", df_questions))

if chunks_semantic:
    final_data.extend(run_rag_pipeline(chunks_semantic, sources_semantic, "semantic", df_questions))

if final_data:
    df = pd.DataFrame(final_data)
    cols = ['id', 'q_id', 'method', 'retrieve_text', 'score', 'source']
    for c in cols:
        if c not in df.columns: df[c] = ""
    df[cols].to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n作業流程全部執行完畢！檔案已儲存為: {OUTPUT_CSV}")
else:
    print("\n️ 無數據產生。")