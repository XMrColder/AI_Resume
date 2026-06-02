import argparse
import json
import os
import sys
from typing import Optional, List

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ValidationError

from jd_parser import JDRequirement, parse_jd
from matcher import MatchResult, match
from resume_extract import ResumeProfile, _strip_json

load_dotenv()


class InterviewQuestion(BaseModel):
    category: str
    question: str
    rationale: str


class InterviewKit(BaseModel):
    candidate: Optional[str] = None
    questions: List[InterviewQuestion]


SYSTEM_PROMPT = "你是资深技术面试官。只输出一个 JSON 对象，不要解释文字，不要 markdown。"


def build_prompt(profile: ResumeProfile, result: MatchResult,
                 jd: JDRequirement, n: int) -> str:
    exp = "；".join(h for w in profile.work_experience for h in w.highlights) or "（简历未提供经历要点）"
    matched = "、".join(result.highlights) or "（无明显匹配技能）"
    gaps = "、".join(result.gaps) or "（无明显缺口）"
    return f"""基于候选人与岗位的匹配情况，设计 {n} 个有针对性的面试问题。

岗位：{jd.title or '（未指定）'}
岗位职责：{'；'.join(jd.responsibilities) or '（未提供）'}

候选人姓名：{profile.name or '（未知）'}
匹配到的技能：{matched}
经历要点：{exp}
岗位要求但简历未体现的缺口：{gaps}

要求：
- 覆盖两类——「深挖亮点」：针对其匹配技能/经历，提开放问题验证真实深度，而非让其简单复述；
  「探查缺口」：针对缺口技能，评估其是否有相邻经验或快速补足的能力。
- 问题要具体、可考察，避免空泛（不要问"你了解 X 吗"，而要能暴露真实水平）。
- 每题给出 category（深挖亮点 / 探查缺口 / 综合）和 rationale（为什么问、关联哪个亮点或缺口）。

只输出以下结构的 JSON：
{{
  "candidate": "姓名或 null",
  "questions": [{{"category": "...", "question": "...", "rationale": "..."}}]
}}
"""


def generate_questions(client: OpenAI, profile: ResumeProfile, result: MatchResult,
                       jd: JDRequirement, n: int = 5,
                       model: str = "deepseek-v4-pro", max_retries: int = 2) -> InterviewKit:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_prompt(profile, result, jd, n)},
    ]
    last_err = None
    for attempt in range(1, max_retries + 1):
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.4,
        )
        raw = resp.choices[0].message.content or ""
        try:
            return InterviewKit.model_validate(json.loads(_strip_json(raw)))
        except (json.JSONDecodeError, ValidationError) as e:
            last_err = e
            print(f"[面试题第 {attempt} 次解析失败] {e}", file=sys.stderr)
            messages.append({"role": "assistant", "content": raw})
            messages.append({
                "role": "user",
                "content": f"上面的输出无法解析为目标 JSON（{e}）。请只重新输出修正后的合法 JSON。",
            })
    raise RuntimeError(f"面试题生成重试 {max_retries} 次仍失败：{last_err}")


def render(kit: InterviewKit, result: MatchResult) -> str:
    lines = [
        f"候选人：{kit.candidate or '(未知)'}    匹配分：{result.score}/100",
        f"亮点：{'、'.join(result.highlights) or '无'}    缺口：{'、'.join(result.gaps) or '无'}",
        "=" * 46,
    ]
    for i, q in enumerate(kit.questions, 1):
        lines.append(f"{i}. [{q.category}] {q.question}")
        lines.append(f"   理由：{q.rationale}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Phase 5 面试问题生成")
    ap.add_argument("resume_json", help="Phase 1 产出的简历 JSON")
    ap.add_argument("jd_txt", help="岗位描述纯文本")
    ap.add_argument("-n", type=int, default=5, help="生成问题数量（默认 5）")
    ap.add_argument("--no-embedding", action="store_true", help="匹配时退回纯规则")
    args = ap.parse_args()

    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        print("请先设置 DEEPSEEK_API_KEY", file=sys.stderr)
        sys.exit(1)
    client = OpenAI(api_key=key, base_url="https://api.deepseek.com")

    with open(args.resume_json, encoding="utf-8") as f:
        profile = ResumeProfile.model_validate(json.load(f))
    with open(args.jd_txt, encoding="utf-8") as f:
        jd = parse_jd(client, f.read())

    result = match(profile, jd, use_embedding=not args.no_embedding)
    kit = generate_questions(client, profile, result, jd, n=args.n)
    print(render(kit, result))

    out = os.path.splitext(args.resume_json)[0] + ".questions.json"
    with open(out, "w", encoding="utf-8") as f:
        f.write(json.dumps(kit.model_dump(), ensure_ascii=False, indent=2))
    print(f"\n面试题已存到 {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
