import os
import uuid
import torch
import requests
import pandas as pd
import gc
from typing import List
from tqdm import tqdm
from openai import OpenAI
from qdrant_client import QdrantClient, models
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_core.embeddings import Embeddings

# --- 0. 設定環境變數 (解決 MPS 記憶體限制) ---
# 這行必須在 import torch 之前或者是程式最開頭設定
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# --- 1. 設定與常數 ---
EMBEDDING_API_URL = "http://ws-04.wade0426.me/embed"

TEXT_PATH = "HW/qa_data.txt"
PREDICT_INPUT = "HW/day6_HW_questions.csv.xlsx"
PREDICT_OUTPUT = "HW/day6_HW_questions_result.xlsx"

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "rag_homework_day6_api"

LLM_BASE_URL = "https://ws-03.wade0426.me/v1/chat/completions"
LLM_MODEL_NAME = "/models/Llama-3_3-Nemotron-Super-49B-v1_5-NVFP4"
LLM_API_KEY = "day6hw"

# Reranker 設定
RERANKER_MODEL_PATH = os.path.expanduser("Qwen3-Reranker-0.6B")

# 設定運算裝置
if torch.cuda.is_available():
    device_obj = torch.device("cuda")
elif torch.backends.mps.is_available():
    device_obj = torch.device("mps")
else:
    device_obj = torch.device("cpu")

print(f"⏳ 使用裝置: {device_obj}")


# --- 2. Embedding 類別 ---
class CustomAPIEmbeddings(Embeddings):
    def __init__(self, api_url):
        self.api_url = api_url

    def _call_api(self, texts: List[str]) -> List[List[float]]:
        data = {
            "texts": texts,
            "normalize": True,
            "batch_size": 32
        }

        try:
            response = requests.post(self.api_url, json=data, timeout=60)
            if response.status_code == 200:
                result = response.json()
                return result.get('embeddings', [])
            else:
                print(f"❌ API Error Code: {response.status_code}")
                return []
        except Exception as e:
            print(f"❌ API Exception: {e}")
            return []

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._call_api(texts)

    def embed_query(self, text: str) -> List[float]:
        results = self._call_api([text])
        if results and len(results) > 0:
            return results[0]
        return []


print(f"⏳ 初始化 Embedding API ({EMBEDDING_API_URL})...")
embedding_model = CustomAPIEmbeddings(EMBEDDING_API_URL)

# 測試連線
try:
    print("* 測試 Embedding API 連線...")
    test_vec = embedding_model.embed_query("測試")
    if test_vec:
        print(f"✅ API 連線成功！向量維度: {len(test_vec)}")
    else:
        print("❌ API 連線失敗，回傳為空")
        exit()
except Exception as e:
    print(f"❌ API 測試發生例外錯誤: {e}")
    exit()


# --- 3. LLM Client ---
class SimpleLLMClient:
    def __init__(self, base_url, model_name, api_key):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model_name = model_name

    def generate(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"LLM Error: {e}")
            return "Error generating response."


print("⏳ 初始化 LLM Client...")
llm_client = SimpleLLMClient(LLM_BASE_URL, LLM_MODEL_NAME, LLM_API_KEY)

# --- 4. Qdrant 初始化 ---
client = QdrantClient(url=QDRANT_URL)


def simple_text_splitter(text, chunk_size=500, chunk_overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += (chunk_size - chunk_overlap)
    return chunks


def init_qdrant_collection():
    """檢查並建立 Qdrant Collection"""

    print(f"* 正在重置集合 {COLLECTION_NAME}...")
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"🗑️ 已刪除舊集合 {COLLECTION_NAME}")
    except Exception:
        pass

    print(f"* 建立新 Collection: {COLLECTION_NAME}...")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "dense": models.VectorParams(
                distance=models.Distance.COSINE,
                size=4096,
            ),
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)
        },
    )

    if os.path.exists(TEXT_PATH):
        print(f"* 讀取 {TEXT_PATH} 並執行 Chunking...")

        with open(TEXT_PATH, "r", encoding="utf-8") as f:
            full_text = f.read()

        documents = simple_text_splitter(full_text, chunk_size=500, chunk_overlap=50)
        print(f"* 原始文本已切分為 {len(documents)} 個 Chunks")

        print("* 計算 Embeddings...")
        doc_embeddings = embedding_model.embed_documents(documents)

        if not doc_embeddings:
            print("❌ Embedding 計算失敗")
            return

        points = [
            models.PointStruct(
                id=uuid.uuid4().hex,
                vector={
                    "dense": embedding,
                    "sparse": models.Document(text=doc, model="Qdrant/bm25"),
                },
                payload={"text": doc},
            )
            for doc, embedding in zip(documents, doc_embeddings)
        ]

        batch_size = 50
        print(f"⏳ 開始寫入 {len(points)} 筆資料至 Qdrant...")
        for i in tqdm(range(0, len(points), batch_size), desc="Upserting"):
            client.upsert(
                collection_name=COLLECTION_NAME,
                points=points[i:i + batch_size]
            )
        print("✅ 資料寫入完成")
    else:
        print(f"⚠️ 找不到 {TEXT_PATH}，跳過資料寫入。")


init_qdrant_collection()

# --- 5. Reranker 模型載入 ---
print("⏳ 載入 Reranker 模型...")
reranker_tokenizer = AutoTokenizer.from_pretrained(
    RERANKER_MODEL_PATH, local_files_only=True, trust_remote_code=True
)
reranker_model = AutoModelForCausalLM.from_pretrained(
    RERANKER_MODEL_PATH, local_files_only=True, trust_remote_code=True
).to(device_obj).eval()

token_false_id = reranker_tokenizer.convert_tokens_to_ids("no")
token_true_id = reranker_tokenizer.convert_tokens_to_ids("yes")
max_reranker_length = 8192

prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
suffix = "<|im_end|>\n<|im_start|>assistant\n"
prefix_tokens = reranker_tokenizer.encode(prefix, add_special_tokens=False)
suffix_tokens = reranker_tokenizer.encode(suffix, add_special_tokens=False)


# 【修正重點】 改為分批處理 (Batch Processing)
def compute_rerank_scores(pairs, batch_size=4):
    """
    分批計算 Reranker 分數，避免 MPS Out Of Memory
    batch_size: 建議設小一點 (例如 2 或 4)
    """
    all_scores = []

    # 使用 tqdm 顯示 Rerank 進度 (可選)
    # for i in range(0, len(pairs), batch_size):

    for i in range(0, len(pairs), batch_size):
        batch_pairs = pairs[i: i + batch_size]

        processed_inputs = []
        for pair in batch_pairs:
            pair_ids = reranker_tokenizer.encode(
                pair, add_special_tokens=False, truncation=True,
                max_length=max_reranker_length - len(prefix_tokens) - len(suffix_tokens)
            )
            full_ids = prefix_tokens + pair_ids + suffix_tokens
            processed_inputs.append(reranker_tokenizer.decode(full_ids))

        inputs = reranker_tokenizer(
            processed_inputs, padding=True, truncation=True, return_tensors="pt", max_length=max_reranker_length
        )

        # 移動到 GPU
        for key in inputs:
            inputs[key] = inputs[key].to(device_obj)

        with torch.no_grad():
            logits = reranker_model(**inputs).logits[:, -1, :]
            scores = logits[:, token_true_id].exp().tolist()
            all_scores.extend(scores)

        # 清理 GPU 記憶體
        del inputs, logits, scores
        if device_obj.type == "mps":
            torch.mps.empty_cache()
        elif device_obj.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    return all_scores


def rerank_documents(query, documents):
    if not documents: return []

    formatted_pairs = [
        f"<Instruct>: 根據查詢檢索相關文件\n<Query>: {query}\n<Document>: {doc}"
        for doc in documents
    ]

    scores = compute_rerank_scores(formatted_pairs, batch_size=4)

    doc_scores = list(zip(documents, scores))
    doc_scores.sort(key=lambda x: x[1], reverse=True)
    return doc_scores


# --- 6. 核心流程函數 ---

def query_rewrite(query: str) -> str:
    prompt = f"""
    你是一個搜尋引擎優化專家。請將以下使用者的問題改寫為更精確、適合做語義檢索的關鍵字查詢。
    保留核心意圖，去除贅詞，並針對自來水公司相關業務進行優化。
    只輸出改寫後的句子，不要有任何解釋。

    使用者問題: {query}
    改寫後查詢:
    """
    rewritten = llm_client.generate(prompt).strip()
    return rewritten


def hybrid_search_with_rerank(query: str, initial_limit=20, final_limit=3):
    query_vec = embedding_model.embed_query(query)

    try:
        response = client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                models.Prefetch(
                    query=models.Document(text=query, model="Qdrant/bm25"),
                    using="sparse",
                    limit=initial_limit,
                ),
                models.Prefetch(
                    query=query_vec,
                    using="dense",
                    limit=initial_limit,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=initial_limit,
        )
        candidate_docs = [point.payload["text"] for point in response.points]
    except Exception as e:
        print(f"Search Error: {e}")
        return []

    if not candidate_docs:
        return []

    # 這裡進行 Rerank
    top_results = rerank_documents(query, candidate_docs)[:final_limit]
    return top_results


def main():
    print(f"📂 讀取 Excel: {PREDICT_INPUT}")
    if not os.path.exists(PREDICT_INPUT):
        print("❌ 檔案不存在")
        return

    df = pd.read_excel(PREDICT_INPUT)

    # 初始化 DataFrame 的欄位
    if 'q_id' not in df.columns: df['q_id'] = None
    if 'answer' not in df.columns: df['answer'] = None

    df['q_id'] = df['q_id'].astype('object')
    df['answer'] = df['answer'].astype('object')

    # 【新增】用來暫存 Ground Truth (Context) 的列表
    ground_truth_list = []

    print("* 開始處理問題...")

    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        original_question = str(row['questions'])

        # 產生 ID
        current_uuid = str(uuid.uuid4())

        # 1. RAG 檢索
        refined_query = query_rewrite(original_question)
        search_results = hybrid_search_with_rerank(refined_query)
        retrieval_context = [doc for doc, score in search_results]

        # 轉成字串供 Prompt 使用
        context_str = "\n".join(retrieval_context)

        ground_truth_list.append({
            "q_id": current_uuid,
            "questions": original_question,
            "contexts": retrieval_context,  # DeepEval 需要 list
            "ground_truth": ""  # 預留欄位，DeepEval 的 Recall 需要這個 (標準答案)
        })

        # 3. 生成 Answer
        qa_prompt = f"""
        你是一個專業的自來水公司客服助手。請根據【參考資料】回答使用者的【問題】。

        規範：
        1. 答案必須基於參考資料，不要編造。
        2. 如果參考資料不足以回答，請回答「目前資訊不足，建議聯繫客服」。
        3. 語氣親切、專業。

        【參考資料】：
        {context_str}

        【問題】：{original_question}

        【回答】：
        """
        answer = llm_client.generate(qa_prompt)

        # 4. 將 Answer 寫回原本的 DataFrame
        df.at[index, 'q_id'] = current_uuid
        df.at[index, 'answer'] = answer

    # --- 迴圈結束，開始存檔 ---

    # 檔案 1：存 Answer 的 Excel
    df.to_excel(PREDICT_OUTPUT, index=False)
    print(f"✅ Answer 處理完成！結果已儲存至: {PREDICT_OUTPUT}")

    # 檔案 2：存 Ground Truth (Context) 的 CSV
    gt_df = pd.DataFrame(ground_truth_list)

    # 存成 CSV (建議用 utf-8-sig 以免中文亂碼)
    gt_df.to_csv("ground_truth.csv", index=False, encoding='utf-8-sig')
    print(f"✅ Ground Truth (Context) 已儲存至: ground_truth.csv")


if __name__ == "__main__":
    main()
