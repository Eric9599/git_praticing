import os
import requests
import pandas as pd
import torch
from uuid import uuid4
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import VectorParams, Distance, PointStruct
from semantic_text_splitter import TextSplitter
from langchain_openai import ChatOpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- 1. 設定區 ---
EMBEDDING_API_URL = "http://ws-04.wade0426.me/embed"
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "data_collection_final_v2"  # 改名以確保乾淨
MODEL_PATH = os.path.expanduser("./Qwen3-Reranker-0.6B")
TEXT_PATH = "./CW"
TEXT_LIST = ["data_01.txt", "data_02.txt", "data_03.txt", "data_04.txt", "data_05.txt"]
CSV_INPUT_PATH = "./CW/Re_Write_questions.csv"
CSV_OUTPUT_PATH = "Completed_Questions_with_Sources_Final.csv"

# Reranker 設定
MAX_RERANKER_LENGTH = 8192
DEVICE = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"🖥️ 運行裝置: {DEVICE}")


# --- 2. 自動偵測與 Embedding 函數 (最關鍵的修正) ---

def get_embedding(texts):
    """
    修正後的 Embedding 函數，針對 ws-04 API 格式
    Request: {"texts": [...]}
    Response: {"embeddings": [[...]]}
    """
    if isinstance(texts, str):
        texts = [texts]

    # 修正 1: Key 必須是 'texts'
    payload = {"texts": texts}

    try:
        response = requests.post(EMBEDDING_API_URL, json=payload, timeout=30)

        # 如果不是 200，印出詳細錯誤
        if response.status_code != 200:
            print(f"❌ API 報錯 ({response.status_code}): {response.text}")
            return []

        result = response.json()

        # 修正 2: 優先抓取 'embeddings'
        if "embeddings" in result:
            return result["embeddings"]
        elif "data" in result:  # 相容性保留
            return [item["embedding"] for item in result["data"]]
        else:
            print(f"❌ API 回傳格式不符: {result.keys()}")
            return []
    except Exception as e:
        print(f"❌ API 連線失敗: {e}")
        return []


def check_and_set_dimension():
    """自動偵測 API 的向量維度，避免設定錯誤"""
    print("🕵️‍♂️ 正在偵測 Embedding API 的向量維度...")
    vecs = get_embedding(["test"])
    if vecs and len(vecs) > 0:
        dim = len(vecs[0])
        print(f"✅ 偵測成功！向量維度為: {dim}")
        return dim
    else:
        print("❌ 無法偵測維度，將使用預設值 1536 (可能會失敗)")
        return 1536


# 自動設定正確的維度
VECTOR_SIZE = check_and_set_dimension()

# --- 3. 初始化 Qdrant 與 LLM ---
client = QdrantClient(url=QDRANT_URL)
llm = ChatOpenAI(
    base_url="https://ws-02.wade0426.me/v1",
    api_key="day6hw",
    model="gpt-4o",
    temperature=0
)

# --- 4. 初始化 Reranker 模型 ---
reranker_model = None
reranker_tokenizer = None

if os.path.exists(MODEL_PATH):
    try:
        print(f"⏳ 正在載入 Reranker 模型 ({MODEL_PATH})...")
        reranker_tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH, local_files_only=True, trust_remote_code=True
        )
        reranker_model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, local_files_only=True, trust_remote_code=True
        ).to(DEVICE).eval()

        token_false_id = reranker_tokenizer.convert_tokens_to_ids("no")
        token_true_id = reranker_tokenizer.convert_tokens_to_ids("yes")

        # Prompt 模板
        PREFIX = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"
        PREFIX_TOKENS = reranker_tokenizer.encode(PREFIX, add_special_tokens=False)
        SUFFIX_TOKENS = reranker_tokenizer.encode(SUFFIX, add_special_tokens=False)
        print("✅ Reranker 模型載入完成。")
    except Exception as e:
        print(f"⚠️ Reranker 模型載入失敗: {e}")
else:
    print(f"⚠️ 找不到 Reranker 路徑，將跳過。")


# --- 5. Reranker 邏輯 ---

def format_instruction(instruction, query, doc):
    if instruction is None: instruction = '根據查詢檢索相關文件'
    return "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(
        instruction=instruction, query=query, doc=doc
    )


def calculate_rerank_scores(pairs):
    if not reranker_model or not reranker_tokenizer: return [0.0] * len(pairs)

    processed_pairs = []
    for pair in pairs:
        pair_ids = reranker_tokenizer.encode(
            pair, add_special_tokens=False, truncation=True,
            max_length=MAX_RERANKER_LENGTH - len(PREFIX_TOKENS) - len(SUFFIX_TOKENS)
        )
        full_ids = PREFIX_TOKENS + pair_ids + SUFFIX_TOKENS
        processed_pairs.append(reranker_tokenizer.decode(full_ids))

    inputs = reranker_tokenizer(
        processed_pairs, padding=True, truncation=True, return_tensors="pt", max_length=MAX_RERANKER_LENGTH
    )
    for key in inputs: inputs[key] = inputs[key].to(DEVICE)

    with torch.no_grad():
        batch_scores = reranker_model(**inputs).logits[:, -1, :]
        true_vector = batch_scores[:, token_true_id]
        false_vector = batch_scores[:, token_false_id]
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        return batch_scores[:, 1].exp().tolist()


def rerank_documents(query, documents, task_instruction=None):
    if not reranker_model: return [(doc, 1.0) for doc in documents]  # Fallback

    pairs = [format_instruction(task_instruction, query, doc) for doc in documents]
    scores = calculate_rerank_scores(pairs)
    doc_scores = list(zip(documents, scores))
    doc_scores.sort(key=lambda x: x[1], reverse=True)
    return doc_scores


# --- 6. 檢索 Pipeline ---

def rewrite_query(original_question):
    try:
        sys_prompt = "你是一個搜尋專家。請將使用者的問題「重寫」為精確的搜尋查詢，去除口語詞，補充關鍵字，直接輸出結果。"
        response = llm.invoke([("system", sys_prompt), ("user", original_question)])
        return response.content.strip()
    except:
        return original_question


def retrieve_pipeline(query, top_k=3, initial_k=20):
    # 1. Embedding
    query_vecs = get_embedding([query])
    if not query_vecs: return "", ""
    query_vector = query_vecs[0]

    # 2. Qdrant Search
    try:
        results = client.query_points(
            collection_name=COLLECTION_NAME, query=query_vector, limit=initial_k
        ).points
    except:
        results = client.search(
            collection_name=COLLECTION_NAME, query_vector=query_vector, limit=initial_k
        )

    if not results: return "", ""

    # 3. Data Prep for Reranker
    docs_map = {}
    docs_text = []
    for hit in results:
        payload = hit.payload
        text = payload.get("text", "")
        file_name = payload.get("file_name", "Unknown")
        if text not in docs_map:
            docs_text.append(text)
            docs_map[text] = file_name

    # 4. Rerank
    ranked_results = rerank_documents(query, docs_text)
    final_docs = ranked_results[:top_k]

    # 5. Format
    context_segments = []
    source_set = set()
    for text, score in final_docs:
        fname = docs_map.get(text, "Unknown")
        source_set.add(fname)
        context_segments.append(f"【來源：{fname} | 分數：{score:.4f}】\n{text}")

    return "\n\n".join(context_segments), ", ".join(list(source_set))


def answer_question_pipeline(original_question):
    rewritten = rewrite_query(original_question)
    print(f"🔍 Rewrite: {rewritten}")
    context, sources = retrieve_pipeline(rewritten)

    if not context: return "查無相關資料。", ""

    prompt = f"""
    請根據【背景資料】回答問題。必須引用來源。若資料不足請回答不知道。

    【背景資料】：
    {context}

    【問題】：
    {original_question}
    """
    try:
        ans = llm.invoke(prompt).content
        return ans, sources
    except Exception as e:
        return f"Error: {e}", ""


# --- 7. 主程式執行 (強制重置資料庫) ---

print(f"💥 正在重建 Collection: {COLLECTION_NAME} (維度: {VECTOR_SIZE})...")
try:
    client.delete_collection(COLLECTION_NAME)
    print("🗑️ 舊資料已刪除。")
except:
    pass

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
)

print("📂 開始上傳資料...")
splitter = TextSplitter((200, 1000))
total_uploaded = 0

for file_name in TEXT_LIST:
    path = os.path.join(TEXT_PATH, file_name)
    if not os.path.exists(path):
        print(f"⚠️ 找不到檔案: {path}")
        continue

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    if not content: continue

    chunks = splitter.chunks(content)
    # 這裡會呼叫修正後的 get_embedding
    vectors = get_embedding(chunks)

    if vectors and len(vectors) == len(chunks):
        points = [
            PointStruct(
                id=str(uuid4()),
                vector=v,
                payload={"file_name": file_name, "text": c}
            ) for c, v in zip(chunks, vectors)
        ]
        client.upsert(collection_name=COLLECTION_NAME, points=points)
        total_uploaded += len(points)
        print(f"   ✅ {file_name}: 成功上傳 {len(points)} 筆。")
    else:
        print(f"   ❌ {file_name}: 向量生成失敗 (維度可能不符或API錯誤)")

print(f"🏁 資料庫準備完成，共 {total_uploaded} 筆資料。")

# --- 8. 執行 CSV ---
if os.path.exists(CSV_INPUT_PATH) and total_uploaded > 0:
    print("\n🚀 開始回答 CSV 問題...")
    df = pd.read_csv(CSV_INPUT_PATH)
    ans_list, src_list = [], []

    for idx, row in df.iterrows():
        q = row.get('questions', row.get('question'))
        print(f"📝 ({idx + 1}) {q}")
        a, s = answer_question_pipeline(q)
        ans_list.append(a)
        src_list.append(s)

    df['answer'] = ans_list
    df['source'] = src_list
    df.to_csv(CSV_OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"✨ 完成！結果已存至 {CSV_OUTPUT_PATH}")
else:
    print(f"⚠️ 無法執行問答：找不到 CSV 或資料庫為空。")