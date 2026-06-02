import os
import sys
import json
import argparse
from typing import Optional, List, Any

from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

from resume_extract import ResumeProfile
from jd_parser import JDRequirement, parse_jd

load_dotenv()

ALIAS = {
    "k8s": "kubernetes", "js": "javascript", "ts": "typescript",
    "py": "python", "postgres": "postgresql", "es": "elasticsearch",
}
SEM_THRESHOLD = 0.65


def norm(s: str) -> str:
    s = s.strip().lower()
    return ALIAS.get(s, s)


class SkillHit(BaseModel):
    required: str
    matched: bool
    how: str
    evidence: Optional[str] = None
    credit: float


class Dimension(BaseModel):
    name: str
    weight: float
    score: float
    detail: Any = None


class MatchResult(BaseModel):
    score: int
    dimensions: List[Dimension]
    highlights: List[str]
    gaps: List[str]


def match_skill(req: str, resume_skills: List[str],
                use_embedding: bool = True, threshold: float = SEM_THRESHOLD) -> SkillHit:
    rn = norm(req)
    for s in resume_skills:
        if norm(s) == rn:
            how = "exact" if s.strip().lower() == req.strip().lower() else "alias"
            return SkillHit(required=req, matched=True, how=how, evidence=s, credit=1.0)

    if use_embedding and resume_skills:
        from embedding import best_match
        i, score = best_match(req, resume_skills)
        if i >= 0 and score >= threshold:
            return SkillHit(required=req, matched=True, how=f"semantic({score:.2f})",
                            evidence=resume_skills[i], credit=1.0)
    return SkillHit(required=req, matched=False, how="miss", evidence=None, credit=0.0)


def experience_relevance(profile: ResumeProfile, jd: JDRequirement):
    from embedding import similarity
    jd_text = "；".join(jd.responsibilities) or (jd.title or "")
    res_text = "；".join(h for w in profile.work_experience for h in w.highlights)
    if not jd_text or not res_text:
        return 0.0, 0.0
    cos = similarity(jd_text, res_text)
    rel = max(0.0, min(1.0, (cos - 0.2) / 0.6))
    return rel, cos


def match(profile: ResumeProfile, jd: JDRequirement, use_embedding: bool = True) -> MatchResult:
    req_hits = [match_skill(r, profile.skills, use_embedding) for r in jd.required_skills]
    skill_score = (sum(h.credit for h in req_hits) / len(req_hits)) if req_hits else 1.0

    pref_hits = [match_skill(p, profile.skills, use_embedding) for p in jd.preferred_skills]
    pref_score = (sum(h.credit for h in pref_hits) / len(pref_hits)) if pref_hits else 0.0

    yoe = profile.years_of_experience or 0
    years_score = 1.0 if not jd.min_years else min(1.0, yoe / jd.min_years)

    if use_embedding:
        w = {"必备技能": 0.50, "经历相关": 0.25, "经验年限": 0.15, "加分项": 0.10}
    else:
        w = {"必备技能": 0.60, "经验年限": 0.25, "加分项": 0.15}

    dims = [
        Dimension(name="必备技能", weight=w["必备技能"], score=round(skill_score, 2),
                  detail=[h.model_dump() for h in req_hits]),
        Dimension(name="经验年限", weight=w["经验年限"], score=round(years_score, 2),
                  detail=f"要求≥{jd.min_years}年，实际 {yoe} 年"),
        Dimension(name="加分项", weight=w["加分项"], score=round(pref_score, 2),
                  detail=[h.model_dump() for h in pref_hits]),
    ]
    if use_embedding:
        rel, cos = experience_relevance(profile, jd)
        dims.append(Dimension(name="经历相关", weight=w["经历相关"], score=round(rel, 2),
                              detail=f"JD职责 vs 简历经历：余弦 {cos:.2f} → 归一 {rel:.2f}"))

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
    ap = argparse.ArgumentParser(description="Phase 3 简历-JD 匹配（规则 + Embedding）")
    ap.add_argument("resume_json", help="Phase 1 产出的简历 JSON")
    ap.add_argument("jd_txt", help="岗位描述纯文本")
    ap.add_argument("--no-embedding", action="store_true",
                    help="退回纯规则模式（不加载 embedding 模型）")
    args = ap.parse_args()
    use_embedding = not args.no_embedding

    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        print("请先设置 DEEPSEEK_API_KEY", file=sys.stderr)
        sys.exit(1)
    client = OpenAI(api_key=key, base_url="https://api.deepseek.com")

    with open(args.resume_json, encoding="utf-8") as f:
        profile = ResumeProfile.model_validate(json.load(f))
    with open(args.jd_txt, encoding="utf-8") as f:
        jd_text = f.read()

    print("解析 JD ……", file=sys.stderr)
    jd = parse_jd(client, jd_text)

    if use_embedding:
        print("加载 embedding 模型（首次会下载，需联网一次）……", file=sys.stderr)
    result = match(profile, jd, use_embedding=use_embedding)
    print(render(profile, jd, result))

    out = os.path.splitext(args.resume_json)[0] + ".match.json"
    with open(out, "w", encoding="utf-8") as f:
        f.write(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))
    print(f"\n匹配结果已存到 {out}", file=sys.stderr)


if __name__ == "__main__":
    main()