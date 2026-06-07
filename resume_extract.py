import json
import os
import re
import sys
from typing import Optional, List

import pdfplumber
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError

load_dotenv()


class _Model(BaseModel):
    model_config = ConfigDict(coerce_numbers_to_str=True)


class Education(_Model):
    degree: Optional[str] = Field(None, description="学历，如 本科/硕士/博士")
    major: Optional[str] = None
    school: Optional[str] = None
    year: Optional[str] = None


class WorkExperience(_Model):
    company: Optional[str] = None
    title: Optional[str] = None
    period: Optional[str] = Field(None, description="时间段，如 2021-至今")
    highlights: List[str] = Field(default_factory=list, description="该段经历的关键点")


class ResumeProfile(_Model):
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    years_of_experience: Optional[float] = Field(None, description="总工作年限(年)")
    education: List[Education] = Field(default_factory=list)
    skills: List[str] = Field(default_factory=list)
    work_experience: List[WorkExperience] = Field(default_factory=list)


def extract_text_from_pdf(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到文件: {path}")

    parts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            parts.append(page.extract_text() or "")
    text = "\n".join(parts).strip()

    if not text:
        raise ValueError(
            "没提取到文本，可能是扫描件/图片型 PDF，需要 OCR（本 MVP 暂不支持）。"
        )
    return text


SYSTEM_PROMPT = (
    "你是专业的简历信息提取助手。只输出一个 JSON 对象，"
    "不要任何解释文字，不要 markdown 代码块。"
)


def build_user_prompt(resume_text: str) -> str:
    return f"""从下面的简历文本中提取信息，严格按这个 JSON 结构输出：

{{
  "name": "姓名，没有则 null",
  "email": "邮箱或 null",
  "phone": "电话或 null",
  "years_of_experience": "总工作年限(数字，可估算)，没有则 null",
  "education": [{{"degree": "本科/硕士/博士", "major": "专业", "school": "学校", "year": "毕业年份"}}],
  "skills": ["技能1", "技能2"],
  "work_experience": [{{"company": "公司", "title": "职位", "period": "时间段", "highlights": ["关键点1"]}}]
}}

规则：
- 只提取简历里确实出现的信息，不要编造；找不到的字段用 null 或空数组 []。
- skills 拆成单个技能词（如 Python、Django、Redis），不要写整句。
- highlights 提炼每段经历最关键的 1~3 条。

简历文本：
\"\"\"
{resume_text}
\"\"\"
"""


def _strip_json(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^```(?:json)?|```$", "", s, flags=re.MULTILINE).strip()
    if not s.startswith("{"):
        m = re.search(r"\{.*\}", s, re.DOTALL)
        if m:
            s = m.group(0)
    return s


def extract_profile(
        client: OpenAI,
        resume_text: str,
        model: str = "deepseek-v4-pro",
        max_retries: int = 2,
) -> ResumeProfile:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(resume_text)},
    ]
    last_err = None
    for attempt in range(1, max_retries + 1):
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0,
        )
        raw = resp.choices[0].message.content or ""
        try:
            data = json.loads(_strip_json(raw))
            return ResumeProfile.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as e:
            last_err = e
            print(f"[第 {attempt} 次解析失败] {e}", file=sys.stderr)
            messages.append({"role": "assistant", "content": raw})
            messages.append({
                "role": "user",
                "content": f"上面的输出无法解析为目标 JSON（错误：{e}）。请只重新输出修正后的合法 JSON。",
            })
    raise RuntimeError(f"重试 {max_retries} 次仍失败：{last_err}")


def main():
    if len(sys.argv) < 2:
        print("用法: python resume_extract.py <简历PDF路径>", file=sys.stderr)
        sys.exit(1)
    pdf_path = sys.argv[1]

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("请先设置环境变量 DEEPSEEK_API_KEY", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    print(f"[1/3] 解析 PDF: {pdf_path}", file=sys.stderr)
    text = extract_text_from_pdf(pdf_path)
    print(f"      提取到约 {len(text)} 字", file=sys.stderr)

    print("[2/3] 调用 DeepSeek V4-Pro 提取信息 ...", file=sys.stderr)
    profile = extract_profile(client, text)

    print("[3/3] 完成\n", file=sys.stderr)
    result = json.dumps(profile.model_dump(), ensure_ascii=False, indent=2)
    print(result)

    out_path = os.path.splitext(pdf_path)[0] + ".json"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(result)
    print(f"\n已保存到 {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
