import json
import os
import sys
from typing import Optional, List, Any

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

from jd_parser import JDRequirement, parse_jd
from resume_extract import ResumeProfile

load_dotenv()

ALIAS = {
    "k8s": "kubernetes", "js": "javascript", "ts": "typescript",
    "py": "python", "postgres": "postgresql", "es": "elasticsearch",
}


def norm(s: str) -> str:
    s = s.strip().lower()
    return ALIAS.get(s, s)


class SkillHit(BaseModel):
    required: str
    matched: bool
    how: str  # exact / alias / miss
    evidence: Optional[str] = None
    credit: float


class Dimension(BaseModel):
    name: str
    weight: float
    score: float  # 0~1
    detail: Any = None


class MatchResult(BaseModel):
    score: int  # 0~100
    dimensions: List[Dimension]
    highlights: List[str]
    gaps: List[str]


def match_skill(req: str, resume_skills: List[str]) -> SkillHit:
    rn = norm(req)
    for s in resume_skills:
        if norm(s) == rn:
            how = "exact" if s.strip().lower() == req.strip().lower() else "alias"
            return SkillHit(required=req, matched=True, how=how, evidence=s, credit=1.0)
    return SkillHit(required=req, matched=False, how="miss", evidence=None, credit=0.0)


def match(profile: ResumeProfile, jd: JDRequirement) -> MatchResult:
    req_hits = [match_skill(r, profile.skills) for r in jd.required_skills]
    skill_score = (sum(h.credit for h in req_hits) / len(req_hits)) if req_hits else 1.0

    pref_hits = [match_skill(p, profile.skills) for p in jd.preferred_skills]
    pref_score = (sum(h.credit for h in pref_hits) / len(pref_hits)) if pref_hits else 0.0

    yoe = profile.years_of_experience or 0
    years_score = 1.0 if not jd.min_years else min(1.0, yoe / jd.min_years)

    dims = [
        Dimension(name="必备技能", weight=0.60, score=round(skill_score, 2),
                  detail=[h.model_dump() for h in req_hits]),
        Dimension(name="经验年限", weight=0.25, score=round(years_score, 2),
                  detail=f"要求≥{jd.min_years}年，实际 {yoe} 年"),
        Dimension(name="加分项", weight=0.15, score=round(pref_score, 2),
                  detail=[h.model_dump() for h in pref_hits]),
    ]
    total = round(100 * sum(d.weight * d.score for d in dims))

    highlights = [h.evidence for h in req_hits + pref_hits if h.matched]
    gaps = [h.required for h in req_hits if not h.matched]
    return MatchResult(score=total, dimensions=dims, highlights=highlights, gaps=gaps)


def render(profile: ResumeProfile, jd: JDRequirement, r: MatchResult) -> str:
    lines = [
        f"候选人：{profile.name or '(未知)'}    岗位：{jd.title or '(未知)'}",
        f"总分：{r.score} / 100",
        "-" * 42,
    ]
    for d in r.dimensions:
        lines.append(f"[{d.name}]  得分 {d.score}  ×  权重 {d.weight}")
        if isinstance(d.detail, list):
            for h in d.detail:
                mark = "✓" if h["matched"] else "✗"
                ev = f"  ← {h['evidence']}" if h["evidence"] else ""
                lines.append(f"    {mark} {h['required']} ({h['how']}){ev}")
        elif d.detail:
            lines.append(f"    {d.detail}")
    lines += [
        "-" * 42,
        f"亮点：{'、'.join(r.highlights) or '无'}",
        f"缺口：{'、'.join(r.gaps) or '无'}",
    ]
    return "\n".join(lines)


def main():
    if len(sys.argv) < 3:
        print("用法: python matcher.py <简历.json> <jd.txt>", file=sys.stderr)
        sys.exit(1)
    resume_json, jd_path = sys.argv[1], sys.argv[2]

    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        print("请先设置 DEEPSEEK_API_KEY", file=sys.stderr)
        sys.exit(1)
    client = OpenAI(api_key=key, base_url="https://api.deepseek.com")

    with open(resume_json, encoding="utf-8") as f:
        profile = ResumeProfile.model_validate(json.load(f))
    with open(jd_path, encoding="utf-8") as f:
        jd_text = f.read()

    print("解析 JD ...", file=sys.stderr)
    jd = parse_jd(client, jd_text)

    result = match(profile, jd)
    print(render(profile, jd, result))

    out = os.path.splitext(resume_json)[0] + ".match.json"
    with open(out, "w", encoding="utf-8") as f:
        f.write(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))
    print(f"\n匹配结果已存到 {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
