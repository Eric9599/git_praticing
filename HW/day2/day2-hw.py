import time
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

llm = ChatOpenAI(
    base_url="https://ws-02.wade0426.me/v1",
    api_key="day2hw",
    model="Qwen/Qwen3-VL-8B-Instruct",
    temperature=0,
    max_tokens=50 # 不知道，為何無效，Gemini 說可能是 API server 的最低限制token，遠高於 我的 50 token，  所以我改用 prompt 方式限制。
)
user_input = input("請輸入topic: ")

prompt1 = ChatPromptTemplate.from_messages([("human", "請用【幽默風】針對主題 {topic} 寫一篇貼文, 內容嚴格限制在 50 個字以內")])
prompt2 = ChatPromptTemplate.from_messages([("human", "請用【專業風】針對主題 {topic} 寫一篇貼文, 內容嚴格限制在 50 個字以內")])

chain1 = prompt1 | llm | StrOutputParser()
chain2 = prompt2 | llm | StrOutputParser()

parallel_agent = RunnableParallel(style_humor=chain1, style_professional=chain2)

target_topic = {"topic": user_input}

print("="*20, "1. Streaming 模式開始", "="*20)
for chunk in parallel_agent.stream(target_topic):
    print(chunk, end="", flush=True)
print("\n")

print("-" * 50)
print("="*20, "2. Batch 模式開始", "="*20)

start_time = time.perf_counter()
batch_result = parallel_agent.invoke(target_topic)
end_time = time.perf_counter()

print("批次處理結果：", batch_result)