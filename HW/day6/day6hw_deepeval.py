import pandas as pd
import os
import ast
import json
import re
import time
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from deepeval.test_case import LLMTestCase
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRecallMetric,
    ContextualPrecisionMetric,
    ContextualRelevancyMetric
)


class Gemma3Wrapper(DeepEvalBaseLLM):
    def __init__(self, model_name="gemma-3-27b-it"):
        self.model_name = model_name
        self.model = ChatOpenAI(
            base_url="https://ws-02.wade0426.me/v1",
            api_key="day6hwdeepeval",
            model=model_name,
            temperature=0.1,
            max_retries=3,
            request_timeout=120
        )

    def load_model(self):
        return self.model

    def _clean_output(self, text: str) -> str:
        # 1. 移除 Markdown
        text = text.strip()
        text = re.sub(r"```[a-zA-Z]*", "", text)
        text = text.replace("```", "")

        # 2. 抓取最外層的括號
        first_brace = text.find("{")
        first_bracket = text.find("[")
        start_idx = -1
        char_pair = ""

        # 判斷是 Dict 還是 List
        if first_brace != -1 and (first_bracket == -1 or first_brace < first_bracket):
            start_idx = first_brace
            char_pair = "}"
        elif first_bracket != -1:
            start_idx = first_bracket
            char_pair = "]"

        if start_idx != -1:
            end_idx = text.rfind(char_pair)
            if end_idx != -1 and end_idx > start_idx:
                candidate = text[start_idx: end_idx + 1]
                return candidate
        return text

    def generate(self, prompt: str, schema=None) -> str:
        # 【新增】重試迴圈：如果 JSON 解析失敗，最多重試 3 次
        max_retries = 3
        for attempt in range(max_retries):
            try:
                messages = [
                    SystemMessage(
                        content="You are a strict JSON generator. Output ONLY valid JSON. Use double quotes for strings. Do not use trailing commas."),
                    HumanMessage(content=prompt)
                ]

                response = self.model.invoke(messages)
                content = self._clean_output(response.content)

                # 【關鍵】嘗試解析 JSON
                # 如果這裡是壞的 JSON，會直接跳到 except 並重試
                try:
                    json.loads(content)
                    return content  # 成功解析，回傳結果
                except json.JSONDecodeError:
                    try:
                        fixed_content = content.replace("'", '"').replace("True", "true").replace("False", "false")
                        json.loads(fixed_content)
                        return fixed_content
                    except:
                        raise ValueError("Invalid JSON")

            except Exception as e:
                print(f"   [JSON Error] 第 {attempt + 1} 次生成失敗，正在重試... ({e})")
                time.sleep(1)  # 休息一下再試

        # 如果 3 次都失敗，回傳空 JSON 避免程式崩潰
        print("   [Failure] 重試 3 次仍無法產生有效 JSON，放棄此題。")
        return "{}"

    async def a_generate(self, prompt: str, schema=None) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return self.model_name


# ================= 2. 配置區域 =================
INPUT_FILE_EXCEL = "HW/day6_HW_questions_result.xlsx"
INPUT_FILE_CSV = "ground_truth.csv"
OUTPUT_FILE = "deep_eval_results_5metrics.xlsx"

COL_QUESTION = "questions"
COL_ANSWER = "answer"
COL_CONTEXTS = "contexts"
COL_GROUND_TRUTH = "ground_truth"


def main():
    if not os.path.exists(INPUT_FILE_EXCEL) or not os.path.exists(INPUT_FILE_CSV):
        print("❌ 錯誤: 找不到輸入檔案")
        return

    print(f"正在讀取檔案...")
    df_excel = pd.read_excel(INPUT_FILE_EXCEL)
    df_csv = pd.read_csv(INPUT_FILE_CSV)

    # 合併資料
    if COL_GROUND_TRUTH in df_csv.columns:
        df_excel[COL_GROUND_TRUTH] = df_csv[COL_GROUND_TRUTH]
    if COL_CONTEXTS in df_csv.columns:
        df_excel[COL_CONTEXTS] = df_csv[COL_CONTEXTS]

    # 取前 5 筆
    df = df_excel.head(3)

    try:
        custom_llm = Gemma3Wrapper()
    except Exception as e:
        print(f"模型初始化失敗: {e}")
        return

    metrics_to_run = [
        FaithfulnessMetric(threshold=0.7, model=custom_llm, include_reason=True, strict_mode=False),
        AnswerRelevancyMetric(threshold=0.7, model=custom_llm, include_reason=True, strict_mode=False),
        ContextualRecallMetric(threshold=0.7, model=custom_llm, include_reason=True, strict_mode=False),
        ContextualPrecisionMetric(threshold=0.7, model=custom_llm, include_reason=True, strict_mode=False),
        ContextualRelevancyMetric(threshold=0.7, model=custom_llm, include_reason=True, strict_mode=False)
    ]

    print(f"開始評估 {len(df)} 筆資料...")
    results_data = []

    for index, row in df.iterrows():
        print(f"\n--- 正在處理第 {index + 1} / {len(df)} 題 ---")
        row_data = row.to_dict()

        # Context 處理
        raw_context = row.get(COL_CONTEXTS, [])
        retrieval_context = []
        if isinstance(raw_context, str):
            raw_context = raw_context.strip()
            if (raw_context.startswith("[") and raw_context.endswith("]")):
                try:
                    retrieval_context = ast.literal_eval(raw_context)
                except:
                    retrieval_context = [raw_context]
            else:
                retrieval_context = [raw_context]
        elif isinstance(raw_context, list):
            retrieval_context = raw_context
        retrieval_context = [str(c) for c in retrieval_context if str(c).strip()]

        # Ground Truth 處理
        ground_truth = row.get(COL_GROUND_TRUTH, None)
        expected_output = str(ground_truth) if pd.notna(ground_truth) else ""

        # 自動填補 Ground Truth
        if not expected_output.strip() or expected_output.lower() == "nan":
            print("   ⚠️ Ground Truth 為空，使用 Contexts 替代...")
            expected_output = "\n".join(retrieval_context)
            row_data[COL_GROUND_TRUTH] = "[Auto-Filled] " + expected_output[:30] + "..."

        if not expected_output.strip():
            expected_output = None

        test_case = LLMTestCase(
            input=str(row[COL_QUESTION]),
            actual_output=str(row[COL_ANSWER]),
            retrieval_context=retrieval_context,
            expected_output=expected_output
        )

        for metric in metrics_to_run:
            metric_name = metric.__class__.__name__

            if ("Recall" in metric_name or "Precision" in metric_name) and not expected_output:
                print(f"   ! 跳過 {metric_name}: 資料全空")
                row_data[f"{metric_name}_Score"] = -1
                continue

            try:
                metric.measure(test_case)
                row_data[f"{metric_name}_Score"] = metric.score
                row_data[f"{metric_name}_Reason"] = metric.reason
                print(f"   > {metric_name}: {metric.score}")

                # 【關鍵修改】每跑完一個指標，休息 2 秒
                time.sleep(2)

            except Exception as e:
                # 簡化錯誤訊息
                err_msg = str(e)
                if "<html" in err_msg or "524" in err_msg:
                    print(f"   ! {metric_name} Error: 伺服器超時 (524)")
                else:
                    print(f"   ! {metric_name} Error: {err_msg[:50]}...")

                row_data[f"{metric_name}_Score"] = -1
                row_data[f"{metric_name}_Reason"] = "API Timeout/Error"

        results_data.append(row_data)

        # 【關鍵修改】每跑完一整題，大休息 5 秒
        print("   (等待 5 秒讓伺服器冷卻...)")
        time.sleep(5)

    output_df = pd.DataFrame(results_data)
    output_df.to_excel(OUTPUT_FILE, index=False)
    print(f"\n✅ 評估完成！結果已儲存至: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()