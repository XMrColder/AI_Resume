import json
import os
import sys
from typing import Optional, List

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

from resume_extract import _strip_json  # 复用 JSON 容错解析

load_dotenv()


class JDRequirement(BaseModel):
    title: Optional[str] = None
    required_skills: List[str] = Field(default_factory=list, description="必备技能")
    preferred_skills: List[str] = Field(default_factory=list, description="加分项技能")
    min_years: float = Field(0, description="最低经验年限要求(年)")
    responsibilities: List[str] = Field(default_factory=list, description="岗位职责")


SYSTEM_PROMPT = "你是招聘 JD 解析助手。只输出一个 JSON 对象，不要解释文字，不要 markdown。"


def build_prompt(jd_text: str) -> str:
    return f"""从下面的招聘 JD 中提取要求，严格按这个 JSON 结构输出：

{{
  "title": "职位名称或 null",
  "required_skills": ["必须具备的硬技能"],
  "preferred_skills": ["加分/优先项技能"],
  "min_years": "最低经验年限(数字)，没写则 0",
  "responsibilities": ["核心岗位职责"]
}}

规则：
- required 是"必须/要求"的，preferred 是"优先/加分/熟悉者佳"的，注意区分。
- 技能拆成单词（Python、Kubernetes），不要写整句。
- 只提取 JD 里写了的，不要脑补。

JD 文本：
\"\"\"
{jd_text}
\"\"\"
"""


def parse_jd(client: OpenAI, jd_text: str,
             model: str = "deepseek-v4-pro", max_retries: int = 2) -> JDRequirement:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_prompt(jd_text)},
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
            return JDRequirement.model_validate(json.loads(_strip_json(raw)))
        except (json.JSONDecodeError, ValidationError) as e:
            last_err = e
            print(f"[JD 第 {attempt} 次解析失败] {e}", file=sys.stderr)
            messages.append({"role": "assistant", "content": raw})
            messages.append({
                "role": "user",
                "content": f"上面的输出无法解析为目标 JSON（{e}）。请只重新输出修正后的合法 JSON。",
            })
    raise RuntimeError(f"JD 解析重试 {max_retries} 次仍失败：{last_err}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python jd_parser.py <jd.txt>", file=sys.stderr)
        sys.exit(1)
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        print("请先设置 DEEPSEEK_API_KEY", file=sys.stderr)
        sys.exit(1)
    client = OpenAI(api_key=key, base_url="https://api.deepseek.com")
    with open(sys.argv[1], encoding="utf-8") as f:
        jd = parse_jd(client, f.read())
    print(json.dumps(jd.model_dump(), ensure_ascii=False, indent=2))
