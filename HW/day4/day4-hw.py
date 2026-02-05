import time
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from search_searxng import search_searxng
from vlm_read_website import vlm_read_website

llm = ChatOpenAI(
    base_url="https://ws-02.wade0426.me/v1",
    api_key="day4hw",
    model="google/gemma-2-27b-it",
    temperature=0
)


class AgentState(TypedDict):
    input: str  # 使用者問題
    knowledge_base: str  # 已知資訊
    messages: List[str]  # 過程紀錄
    search_results: List[dict]  # 搜尋結果暫存
    current_result_index: int  # 目前 VLM 讀到第幾篇
    vlm_temp_content: str  # VLM 剛讀完的內容
    final_answer: str  # 最終產出
    is_sufficient: bool  # 資訊是否足夠
    search_exhausted: bool  # 搜尋結果是否看完
    valuable_found: bool  # VLM 是否發現有價值資訊



def node_check_cache(state: AgentState):
    print("\n🔹 [Node] 檢查快取")
    # (此處可實作 Redis/VectorDB)
    hit = False
    if hit:
        return {"final_answer": "Cached Answer"}
    return {}


def node_decision(state: AgentState):
    print("\n* [Node] 決策層評估")
    kb = state.get("knowledge_base", "無")

    system_prompt = f"""
    使用者問題: {state['input']}
    目前收集到的資訊: {kb}

    請評估：目前的資訊是否足以「完整」回答問題？
    - 若足夠，請根據資訊生成回答。
    - 若不足，請回覆 'SEARCH'。
    """
    response = llm.invoke(system_prompt).content

    if "SEARCH" in response:
        print("   => 決定：需要更多資訊 (SEARCH)")
        return {"is_sufficient": False}
    else:
        print("   => 決定：資訊足夠，生成答案")
        return {"is_sufficient": True, "final_answer": response}


def node_gen_keywords(state: AgentState):
    print("\n* [Node] 生成關鍵字")

    system_prompt = f"""
    指令：你是一個搜尋引擎關鍵字產生器。
    任務：將使用者的問題轉換為一個簡單的 Google 搜尋關鍵字。

    使用者問題：{state['input']}

    嚴格限制：
    1. 只准輸出關鍵字 (例如: "Apple stock news")
    2. 禁止輸出任何標點符號、解釋、前言或後語。
    3. 禁止輸出 "Here is the keyword" 之類的廢話。
    4. 直接給答案，不要囉嗦。
    """

    print("   (等待 LLM 生成中...)")
    try:
        response = llm.invoke(system_prompt, config={"timeout": 10})
        raw_content = response.content
    except Exception as e:
        print(f"* LLM 生成超時或錯誤，使用預設關鍵字。錯誤: {e}")
        raw_content = state['input']

    print(f"* [Debug] LLM 原始回應: {raw_content[:100]}...")

    keyword = raw_content.replace('"', '').replace("'", "").replace("\n", " ").strip()

    remove_words = ["search query:", "keyword:", "google search:", "here is"]
    for word in remove_words:
        if keyword.lower().startswith(word):
            keyword = keyword[len(word):].strip()

    if len(keyword) > 30:
        print(f"   [警告] 關鍵字太長，強制截斷...")
        keyword = keyword[:30]

    print(f"   => 最終使用關鍵字: {keyword}")
    return {"messages": [f"Query: {keyword}"]}


def node_search_tool(state: AgentState):
    # 呼叫 search_searxng.py
    last_msg = state['messages'][-1]
    query = last_msg.replace("Query: ", "")

    results = search_searxng(query, limit=3)

    print(f"   => 取得 {len(results)} 筆結果")
    return {"search_results": results, "current_result_index": 0}


def node_vlm_process(state: AgentState):
    print("\n[Node] VLM 視覺閱讀")
    idx = state.get("current_result_index", 0)
    results = state.get("search_results", [])

    if not results or idx >= len(results):
        print("   => 無更多搜尋結果")
        return {"search_exhausted": True}

    target = results[idx]

    # 呼叫 vlm_read_website.py
    # 傳入 URL 和 Title 幫助 LLM 理解
    content = vlm_read_website(target['url'], target.get('title', ''))

    return {"vlm_temp_content": content, "search_exhausted": False}


def node_value_check(state: AgentState):
    print("\n[Node] 價值評估")
    content = state.get("vlm_temp_content", "")[:2000]

    system_prompt = f"""
    使用者問題: {state['input']}
    剛讀取的網頁內容摘要: {content}

    請問這段內容對回答使用者問題「有幫助/有價值」嗎？
    請回覆 YES 或 NO。
    """
    res = llm.invoke(system_prompt).content.upper()
    print(f"   => 評估結果: {res}")

    if "YES" in res:
        # 將新資訊加入知識庫
        old_kb = state.get("knowledge_base", "")
        new_kb = f"{old_kb}\n\n[新資訊]: {content}"
        return {"knowledge_base": new_kb, "valuable_found": True}
    else:
        return {"valuable_found": False}


def node_update_index(state: AgentState):
    return {"current_result_index": state["current_result_index"] + 1}


# draw graph
workflow = StateGraph(AgentState)
start_time = time.time()
workflow.add_node("check_cache", node_check_cache)
workflow.add_node("decision", node_decision)
workflow.add_node("gen_keywords", node_gen_keywords)
workflow.add_node("search_tool", node_search_tool)
workflow.add_node("vlm_process", node_vlm_process)
workflow.add_node("value_check", node_value_check)
workflow.add_node("update_index", node_update_index)
end_time = time.time()
workflow.set_entry_point("check_cache")

# 邏輯連線
workflow.add_conditional_edges(
    "check_cache",
    lambda x: "end" if x.get("final_answer") else "decision",
    {"end": END, "decision": "decision"}
)

workflow.add_conditional_edges(
    "decision",
    lambda x: "end" if x.get("is_sufficient") else "gen_keywords",
    {"end": END, "gen_keywords": "gen_keywords"}
)

workflow.add_edge("gen_keywords", "search_tool")
workflow.add_edge("search_tool", "vlm_process")

# VLM 讀完後的路由：沒資料了 -> 回決策層；還有資料 -> 檢查價值
workflow.add_conditional_edges(
    "vlm_process",
    lambda x: "decision" if x.get("search_exhausted") else "value_check",
    {"decision": "decision", "value_check": "value_check"}
)

# 價值檢查後的路由：有價值 -> 回決策層重判；沒價值 -> 讀下一篇
workflow.add_conditional_edges(
    "value_check",
    lambda x: "decision" if x.get("valuable_found") else "update_index",
    {"decision": "decision", "update_index": "update_index"}
)

workflow.add_edge("update_index", "vlm_process")

app = workflow.compile()
print(app.get_graph().draw_ascii())


# Executing
if __name__ == "__main__":
    print("* AI Agent 啟動中...")
    q = input("請輸入您的問題: ")

    initial = {
        "input": q,
        "knowledge_base": "",
        "messages": [],
        "search_results": [],
        "current_result_index": 0
    }

    final_output = None

    print("\n--- 開始處理 ---")

    for event in app.stream(initial):
        for value in event.values():
            # 如果節點沒有回傳任何東西 (None)，就跳過。
            if value and "final_answer" in value:
                final_output = value["final_answer"]

    print("\n" + "=" * 30)
    print("* 最終答案：")
    print("=" * 30)

    if final_output:
        print(final_output)
    else:
        print("* 任務結束，但沒有生成答案。")

    print("\n" + "=" * 30)