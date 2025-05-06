import json
import os
from PyPDF2 import PdfReader
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda, RunnableMap
from dotenv import load_dotenv

load_dotenv()

# === Setup LLM ===
llm = ChatOpenAI(model="o1-mini", api_key=os.getenv("OPENAI_API_KEY"))

# === Prompt Template ===
CAB_PROMPT = PromptTemplate.from_template("""
You are a document analysis assistant.

Given the following Central Analysis Bureau (CAB) document content, analyze it based on the user's needs. We specifically want to identify and extract the Department of Transportation (DOT) score, the financial score, and basic scores.

Document Content:
{document_text}

Return a Python dictionary as a string containing your analysis. Example format:
{{
    "dot_score": "..."
    "financial_score": "...",
    "basic_scores": ["...","..."],
}}

Do NOT add any additional tags, quotation marks, or other symbols before or after the dictionary.
Perform the analysis now.
""")

JUDICIAL_HELLHOLE_PROMPT = PromptTemplate.from_template("""
You are a document analysis assistant that specializes in extracting information on Judicial Hellholes.

Given the following document text reporting the year's worth of reporting the locations and details of current judicial hellholes, analyze it based on the user's needs.

Document Content:
{document_text}

Return a Python list of dictionaries as a string containing your analysis. Example format:
[
    {{
        "city": "...",
        "district": "...",
        "risk-level": "..."
    }},
    ...
]

Do NOT add any additional tags, quotation marks, or other symbols before or after the list. Each dictionary item in the list should have the exact same format.
Perform the analysis now.
""")

analysis_prompt = JUDICIAL_HELLHOLE_PROMPT

# === Step 1: Extract text from PDF ===
def extract_text_step(inputs):
    pdf_path = inputs["pdf_path"]
    reader = PdfReader(pdf_path)
    text = "".join(page.extract_text() or "" for page in reader.pages)
    return {"document_text": text, "output_path": inputs.get("output_path", "analysis_output.json")}

extract_text_chain = RunnableLambda(extract_text_step)

# === Step 2: Run LLM ===
analysis_chain = analysis_prompt | llm

# === Step 3: Parse LLM output and save to JSON ===
def parse_and_save_step(inputs):
    content = inputs["llm_response"].content if hasattr(inputs["llm_response"], "content") else inputs["llm_response"]
    output_path = inputs["output_path"]
    try:
        result_dict = json.loads(content)
    except json.JSONDecodeError:
        print("⚠️ Failed to parse the response into JSON. Here's the raw output:")
        print(content)
        return None

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=2)

    print(f"✅ Analysis saved to {output_path}")
    return result_dict

parse_and_save_chain = RunnableLambda(parse_and_save_step)

# === Full Chain: PDF → Text → LLM → Parse + Save ===
full_analysis_chain = (
        extract_text_chain
        | RunnableLambda(lambda x: {
    "document_text": x["document_text"],
    "output_path": x["output_path"]
})
        | RunnableMap({
    "llm_response": analysis_chain,
    "output_path": lambda x: x["output_path"]
})
        | parse_and_save_chain
)

# === Run Full Pipeline ===
if __name__ == "__main__":
    result = full_analysis_chain.invoke({
        "pdf_path": "example.pdf",
        "output_path": "analysis_output.json"
    })