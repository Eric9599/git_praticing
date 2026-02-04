import re
import os
from typing import TypedDict, List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from langchain_core.tools import tool

# 1. 初始化模型
model = ChatOpenAI(
    base_url="https://ws-02.wade0426.me/v1",
    api_key="day3hw",
    model="Qwen/Qwen3-VL-8B-Instruct",
    temperature=0.1
)


# 2. 定義資料狀態 (State)
class GraphState(TypedDict):
    raw_segments: List[dict]
    full_text: str
    transcript_result: str
    summary_result: str


# 3. 定義工具 (@tool)
@tool
def summarize_tool(content: str) -> str:
    """
    會議摘要專用工具。
    用途：輸入長篇逐字稿內容，輸出結構化的重點摘要。
    """
    print("[Tool] 摘要工具被呼叫，正在進行 AI 推論...")

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一個專業的會議記錄員，擅長歸納重點。"),
        ("human", "請將以下內容整理成 3-5 點關鍵摘要（請務必使用繁體中文回答）：\n\n{content}")
    ])

    chain = prompt | model | StrOutputParser()
    return chain.invoke({"content": content})


# 4. SRT 解析工具
def parse_srt_file(file_path):
    segments = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        pattern = re.compile(r'(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\s+(.*?)(?=\n\n|\Z)', re.DOTALL)
        matches = pattern.findall(content)

        for start, end, text in matches:
            segments.append({
                "start": start,
                "end": end,
                "text": text.replace('\n', ' ').strip()
            })
    except FileNotFoundError:
        print(f"錯誤：找不到檔案 {file_path}")
    return segments


# 5. 定義節點 (Nodes)
def asr_node(state: GraphState):
    """節點 A：讀取檔案"""
    print("[ASR Node] 讀取檔案中...")

    srt_path = "out/133.srt"
    txt_path = "out/133.txt"

    segments = parse_srt_file(srt_path)

    if os.path.exists(txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            full_text = f.read()
    else:
        full_text = " ".join([seg['text'] for seg in segments])

    return {"raw_segments": segments, "full_text": full_text}


def transcript_node(state: GraphState):
    """節點 B"""
    print("[Transcript Node] 格式化逐字稿...")

    segments = state.get("raw_segments", [])
    output_str = ""
    for seg in segments:
        output_str += f"[{seg['start']} -> {seg['end']}] {seg['text']}\n"

    return {"transcript_result": output_str}


def summary_node(state: GraphState):
    """節點 C"""
    print("[Summary Node] 呼叫摘要工具...")

    text = state.get("full_text", "")

    if not text:
        tool_output = "無內容可摘要"
    else:
        tool_output = summarize_tool.invoke(text)

    return {"summary_result": tool_output}


def writer_node(state: GraphState):
    """節點 D"""
    print("[Writer Node] 輸出結果如下：")
    print("\n" + "=" * 50)

    # 取出資料
    transcript = state.get("transcript_result", "")
    summary = state.get("summary_result", "")

    print("【重點摘要】")
    print(summary)
    print("\n" + "-" * 50 + "\n")
    print("【逐字稿時間軸】")
    print(transcript)

    print("=" * 50 + "\n")

    return {}


# 6. 建構 LangGraph 圖形
workflow = StateGraph(GraphState)

# 6.1 加入節點
workflow.add_node("asr_agent", asr_node)
workflow.add_node("transcript_writer", transcript_node)
workflow.add_node("summarizer", summary_node)
workflow.add_node("writer", writer_node)

# 6.2 設定邊
workflow.set_entry_point("asr_agent")

workflow.add_edge("asr_agent", "transcript_writer")
workflow.add_edge("asr_agent", "summarizer")
workflow.add_edge("transcript_writer", "writer")
workflow.add_edge("summarizer", "writer")
workflow.add_edge("writer", END)

# 6.3 編譯
app = workflow.compile()
print(app.get_graph().draw_ascii())

# 7. 執行
print("執行 LangGraph...\n")
app.invoke({})
print("執行結束")