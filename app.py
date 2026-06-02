"""
Phase 6 - Streamlit Web 界面
把前面所有模块串成一个能点的应用：
  ① 粘贴 JD + 批量上传简历 PDF → ② 排序表 → ③ 点开候选人看详情（匹配分析 + 面试题）

运行：
    pip install streamlit
    streamlit run app.py
需已设置 DEEPSEEK_API_KEY（或在侧边栏直接填）。

要点：
  - 上传的 PDF 是内存文件对象，直接喂给 pdfplumber，无需落盘。
  - 筛选结果存进 st.session_state，点详情 / 生成面试题不会重跑整批、不重复烧 API。
  - 面试题在点开详情后按需生成并按候选人缓存。
"""
import os
import json

import streamlit as st
import pdfplumber
from openai import OpenAI
from dotenv import load_dotenv

from resume_extract import extract_profile
from jd_parser import parse_jd
from matcher import match
from interview_gen import generate_questions
from export_excel import report_to_xlsx_bytes

load_dotenv()
st.set_page_config(page_title="AI 简历筛选助手", layout="wide")


@st.cache_resource
def get_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


def pdf_to_text(uploaded) -> str:
    """从上传的 PDF 文件对象提取文本。"""
    parts = []
    with pdfplumber.open(uploaded) as pdf:
        for page in pdf.pages:
            parts.append(page.extract_text() or "")
    text = "\n".join(parts).strip()
    if not text:
        raise ValueError("没提取到文本（可能是扫描件 / 图片型 PDF，需 OCR）")
    return text


def screen(client, jd_text, files, use_embedding):
    jd = parse_jd(client, jd_text)
    results, failed = [], []
    prog = st.progress(0.0, text="开始……")
    for i, f in enumerate(files, 1):
        prog.progress(i / len(files), text=f"处理 {f.name}（{i}/{len(files)}）")
        try:
            profile = extract_profile(client, pdf_to_text(f))
            r = match(profile, jd, use_embedding=use_embedding)
            results.append({"file": f.name, "profile": profile, "result": r})
        except Exception as e:
            failed.append({"file": f.name, "error": str(e)})
    prog.empty()
    results.sort(key=lambda x: x["result"].score, reverse=True)
    return jd, results, failed


# ---------------- 侧边栏：配置 ----------------
st.sidebar.header("配置")
api_key = st.sidebar.text_input(
    "DeepSeek API Key", value=os.environ.get("DEEPSEEK_API_KEY", ""), type="password")
use_embedding = st.sidebar.toggle("启用 Embedding 语义匹配", value=True)

# ---------------- 主区：输入 ----------------
st.title("AI 简历筛选助手")
jd_text = st.text_area("① 粘贴岗位 JD", height=180, placeholder="把招聘 JD 贴在这里……")
files = st.file_uploader("② 上传简历 PDF（可多选）", type=["pdf"], accept_multiple_files=True)
run = st.button("③ 开始筛选", type="primary",
                disabled=not (jd_text and files and api_key))

if run:
    client = get_client(api_key)
    with st.spinner("解析 JD、提取并匹配中……（首次启用 embedding 会下载模型）"):
        jd, results, failed = screen(client, jd_text, files, use_embedding)
    st.session_state["jd"] = jd
    st.session_state["results"] = results
    st.session_state["failed"] = failed
    st.session_state.pop("questions", None)   # 清掉上一轮缓存的面试题

# ---------------- 结果展示 ----------------
results = st.session_state.get("results")
if not results:
    st.info("填入 JD、上传简历 PDF，然后点「开始筛选」。")
    st.stop()

jd = st.session_state["jd"]
failed = st.session_state.get("failed", [])

st.subheader(f"筛选结果（{len(results)} 份成功，{len(failed)} 份失败）")
st.dataframe(
    [{"排名": i, "分数": x["result"].score, "姓名": x["profile"].name or "(未知)",
      "文件": x["file"], "缺口": "、".join(x["result"].gaps) or "无"}
     for i, x in enumerate(results, 1)],
    width='stretch', hide_index=True,
)

if failed:
    with st.expander(f"未能处理的 {len(failed)} 份"):
        for fl in failed:
            st.write(f"- {fl['file']}：{fl['error']}")

# 导出（Phase 7）
report = {"jd": jd.model_dump(),
          "ranking": [{"rank": i, "file": x["file"], "name": x["profile"].name,
                       "score": x["result"].score, "highlights": x["result"].highlights,
                       "gaps": x["result"].gaps,
                       "dimensions": [d.model_dump() for d in x["result"].dimensions]}
                      for i, x in enumerate(results, 1)]}
e1, e2, e3 = st.columns([1, 1, 2])
with e1:
    top_n = st.number_input("导出 Top N", 1, len(results), min(10, len(results)))
with e2:
    st.download_button(
        "导出 Excel", report_to_xlsx_bytes(report, int(top_n)),
        file_name="screening_result.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
with e3:
    st.download_button("下载完整 JSON", json.dumps(report, ensure_ascii=False, indent=2),
                       file_name="batch_results.json", mime="application/json")

st.divider()

# ---------------- 候选人详情 ----------------
st.subheader("候选人详情")
names = [f"#{i}　{x['profile'].name or '(未知)'}　· {x['result'].score}分　· {x['file']}"
         for i, x in enumerate(results, 1)]
idx = st.selectbox("选择候选人", range(len(results)), format_func=lambda i: names[i])
item = results[idx]
profile, r = item["profile"], item["result"]

c1, c2 = st.columns(2)
with c1:
    st.markdown("**提取到的简历信息**")
    st.json(profile.model_dump(), expanded=False)
with c2:
    st.markdown(f"**匹配分析　总分 {r.score} / 100**")
    for d in r.dimensions:
        st.markdown(f"**[{d.name}]**　得分 {d.score} × 权重 {d.weight}")
        if isinstance(d.detail, list):
            for h in d.detail:
                mark = "✓" if h["matched"] else "✗"
                ev = f"　← {h['evidence']}" if h.get("evidence") else ""
                st.write(f"{mark} {h['required']}（{h['how']}）{ev}")
        elif d.detail:
            st.caption(str(d.detail))

st.markdown("**面试问题**")
if st.button("生成面试问题", key=f"genq_{idx}"):
    client = get_client(api_key)
    with st.spinner("生成中……"):
        kit = generate_questions(client, profile, r, jd, n=5)
    st.session_state.setdefault("questions", {})[idx] = kit.model_dump()

qkit = st.session_state.get("questions", {}).get(idx)
if qkit:
    for i, q in enumerate(qkit["questions"], 1):
        st.markdown(f"{i}. **[{q['category']}]** {q['question']}")
        st.caption(f"理由：{q['rationale']}")
else:
    st.caption("点上面的按钮，按该候选人的亮点与缺口生成针对性问题。")
