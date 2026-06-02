import argparse
import glob
import json
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

from jd_parser import JDRequirement, parse_jd
from matcher import match
from resume_extract import ResumeProfile, extract_text_from_pdf, extract_profile

load_dotenv()


def load_or_extract(client: OpenAI, pdf_path: str, refresh: bool = False) -> ResumeProfile:
    cache = os.path.splitext(pdf_path)[0] + ".json"
    if not refresh and os.path.exists(cache):
        with open(cache, encoding="utf-8") as f:
            return ResumeProfile.model_validate(json.load(f))
    text = extract_text_from_pdf(pdf_path)
    profile = extract_profile(client, text)
    with open(cache, "w", encoding="utf-8") as f:
        f.write(json.dumps(profile.model_dump(), ensure_ascii=False, indent=2))
    return profile


def find_pdfs(folder: str):
    return sorted(p for p in glob.glob(os.path.join(folder, "*"))
                  if p.lower().endswith(".pdf"))


def run_batch(client: OpenAI, folder: str, jd: JDRequirement,
              use_embedding: bool = True, refresh: bool = False):
    pdfs = find_pdfs(folder)
    results, failed = [], []
    for i, pdf in enumerate(pdfs, 1):
        name = os.path.basename(pdf)
        try:
            print(f"[{i}/{len(pdfs)}] {name} ……", file=sys.stderr)
            profile = load_or_extract(client, pdf, refresh)
            r = match(profile, jd, use_embedding=use_embedding)
            results.append({"file": name, "profile": profile, "result": r})
        except Exception as e:
            print(f"    跳过（失败）：{e}", file=sys.stderr)
            failed.append({"file": name, "error": str(e)})
    results.sort(key=lambda x: x["result"].score, reverse=True)
    return results, failed


def render_table(results, failed) -> str:
    lines = ["排名 / 分数 / 候选人 / 文件", "-" * 56]
    for rank, item in enumerate(results, 1):
        p, r = item["profile"], item["result"]
        gaps = "、".join(r.gaps) or "无"
        lines.append(f"{rank:>2}.  [{r.score:>3}]  {p.name or '(未知)'}  ·  {item['file']}")
        lines.append(f"          缺口：{gaps}")
    if failed:
        lines.append("")
        lines.append(f"未能处理（{len(failed)} 份）：")
        for f in failed:
            lines.append(f"  - {f['file']}：{f['error']}")
    return "\n".join(lines)


def build_report(jd: JDRequirement, results, failed) -> dict:
    ranking = []
    for rank, item in enumerate(results, 1):
        p, r = item["profile"], item["result"]
        ranking.append({
            "rank": rank,
            "file": item["file"],
            "name": p.name,
            "score": r.score,
            "highlights": r.highlights,
            "gaps": r.gaps,
            "dimensions": [d.model_dump() for d in r.dimensions],
        })
    return {"jd": jd.model_dump(), "ranking": ranking, "failed": failed}


def main():
    ap = argparse.ArgumentParser(description="Phase 4 批量简历筛选 + 排序")
    ap.add_argument("folder", help="存放简历 PDF 的文件夹")
    ap.add_argument("jd_txt", help="岗位描述纯文本")
    ap.add_argument("--no-embedding", action="store_true", help="退回纯规则模式")
    ap.add_argument("--refresh", action="store_true", help="忽略缓存，强制重新提取")
    args = ap.parse_args()
    use_embedding = not args.no_embedding

    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        print("请先设置 DEEPSEEK_API_KEY", file=sys.stderr)
        sys.exit(1)
    client = OpenAI(api_key=key, base_url="https://api.deepseek.com")

    with open(args.jd_txt, encoding="utf-8") as f:
        jd_text = f.read()
    print("解析 JD ……", file=sys.stderr)
    jd = parse_jd(client, jd_text)

    results, failed = run_batch(client, args.folder, jd, use_embedding, args.refresh)
    if not results and not failed:
        print(f"文件夹里没找到 PDF：{args.folder}", file=sys.stderr)
        sys.exit(1)

    print(render_table(results, failed))

    report = build_report(jd, results, failed)
    out_path = os.path.join(args.folder, "batch_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n共 {len(results)} 份成功、{len(failed)} 份失败，详细结果已存到 {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
