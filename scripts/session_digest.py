#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
把 DSH 会话日志（~/.dsh/sessions/<工作区>/<会话>/session.v4.jsonl.zstd）
导出成「两级、按需加载」的纯文本，供另一个会话低成本查阅。

设计目标：
  1. 别的会话**能**知道这里发生过什么  -> 内容落盘成普通 md
  2. 但**不是每次**都进上下文           -> 不进 AGENTS.md，只有主动 read 才加载
  3. 也**不用**扫全部上下文             -> 先看 <id>-index.md（每轮一行，约 1.5k tokens），
                                          再 grep <id>-full.md 定位行号，最后 read 指定行段

用法：
  python scripts/session_digest.py list [--cwd .]
  python scripts/session_digest.py build <会话ID前缀> [--out notes/sessions]
                                          [--max-result 400] [--max-text 12000]

输出（默认 notes/sessions/）：
  INDEX.md            所有已导出会话的一行总表
  <id8>-index.md      小：每轮一行 + 工具统计 + 涉及文件
  <id8>-full.md       全：逐轮完整转录（工具结果按 --max-result 截断），供 grep
"""

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time

DSH_HOME = os.environ.get("DSH_HOME") or os.path.join(os.path.expanduser("~"), ".dsh")
LOG_NAME = "session.v4.jsonl.zstd"

# 这些是系统注入到 user 位置的内容，不算"用户说的话"
INJECT_PATTERNS = (
    "Current runtime context",
    "<system-reminder>",
    "Time sampled while preparing",
    "The approval policy changed",
    "This snapshot supersedes earlier runtime-context",
    "background job ",
    "Background subagent ",
    "[model changed:",
)


def is_injected(text):
    t = (text or "").lstrip()
    if any(t.startswith(p) for p in INJECT_PATTERNS):
        return True
    return " sent a message:" in t[:120] and t.startswith("Agent ")


# ---------------------------------------------------------------- 基础工具


def zstd_bin(explicit=None):
    if explicit:
        return explicit
    found = shutil.which("zstd")
    if found:
        return found
    for cand in (r"E:\anaconda3\Library\bin\zstd.exe",
                 r"C:\Program Files\zstd\zstd.exe",
                 "/usr/bin/zstd"):
        if os.path.exists(cand):
            return cand
    sys.exit("找不到 zstd 可执行文件，请用 --zstd 指定")


def iter_sessions():
    root = os.path.join(DSH_HOME, "sessions")
    if not os.path.isdir(root):
        sys.exit("没有找到会话目录: %s" % root)
    out = []
    for ws in os.listdir(root):
        wsdir = os.path.join(root, ws)
        if not os.path.isdir(wsdir):
            continue
        for name in os.listdir(wsdir):
            sdir = os.path.join(wsdir, name)
            log = os.path.join(sdir, LOG_NAME)
            if os.path.isfile(log):
                out.append({"name": name, "dir": sdir, "log": log,
                            "size": os.path.getsize(log)})
    return out


def decompress(log, zstd):
    p = subprocess.run([zstd, "-d", "-c", log],
                       stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    if p.returncode != 0:
        sys.exit("解压失败: %s" % log)
    return p.stdout.decode("utf-8", "replace")


def load_events(text):
    """按 \\n 切分——不能用 str.splitlines()，它会在 U+2028 等字符上误切。"""
    out = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def short_id(sid):
    s = sid or ""
    if s.startswith("session-"):
        s = s[len("session-"):]
    return s[:8]


def ts(ms):
    try:
        return time.strftime("%Y-%m-%d %H:%M", time.localtime(ms / 1000.0))
    except Exception:
        return "?"


def hhmm(ms):
    try:
        return time.strftime("%H:%M", time.localtime(ms / 1000.0))
    except Exception:
        return "?"


def parts_text(content):
    out = []
    for p in content or []:
        if isinstance(p, str):
            out.append(p)
        elif isinstance(p, dict) and p.get("type") == "text":
            out.append(p.get("text") or "")
    return "\n".join(x for x in out if x.strip())


def clip(s, n):
    s = re.sub(r"[ \t]+\n", "\n", (s or "").strip())
    s = re.sub(r"\n{3,}", "\n\n", s)
    if len(s) <= n:
        return s
    return s[:n].rstrip() + "\n…(截断，原文 %d 字)" % len(s)


def one_line(s, n):
    s = re.sub(r"\s+", " ", (s or "").strip())
    return s if len(s) <= n else s[:n] + "…"


# ---------------------------------------------------------------- 解析


def annotate(events):
    """给每个事件打上轮次号；返回 (meta, annotated)。"""
    meta = {"id": "", "cwd": "", "created": 0, "title": "", "parent": "", "origin": ""}
    cur = 0
    out = []
    for ev in events:
        et = ev.get("type")
        d = ev.get("data") or {}
        if et == "session":
            meta["id"] = ev.get("id") or d.get("id") or ""
            meta["cwd"] = ev.get("cwd") or meta["cwd"]
            meta["created"] = ev.get("createdAt") or 0
            meta["parent"] = ev.get("parentSession") or ""
            meta["origin"] = ev.get("origin") or ""
            continue
        if et == "session/title":
            meta["title"] = d.get("title") or meta["title"]
            continue
        if et == "turn/start":
            cur = d.get("turn") or cur
            continue
        if et == "turn/end":
            if cur:
                out.append(dict(ev, _turn=cur, _closing=True))
            cur = 0
            continue
        if cur == 0:
            continue
        out.append(dict(ev, _turn=cur))
    return meta, out


def turn_stats(annotated):
    turns = {}
    order = []
    for ev in annotated:
        t = ev["_turn"]
        if t not in turns:
            turns[t] = {"turn": t, "start": ev.get("time") or 0, "end": ev.get("time") or 0,
                        "users": [], "injected": 0, "assistant": [], "tools": {},
                        "steps": set(), "files": set(), "reason": ""}
            order.append(t)
        c = turns[t]
        c["start"] = min(c["start"] or ev.get("time") or 0, ev.get("time") or c["start"])
        c["end"] = ev.get("time") or c["end"]
        et = ev.get("type")
        d = ev.get("data") or {}
        if et == "user/message":
            txt = parts_text(d.get("content"))
            if not txt.strip():
                continue
            if is_injected(txt):
                c["injected"] += 1
            else:
                c["users"].append(txt)
        elif et == "assistant/message":
            c["steps"].add(d.get("step"))
            txt = parts_text((d.get("message") or {}).get("content"))
            if txt.strip():
                c["assistant"].append(txt)
        elif et == "tool/call":
            c["steps"].add(d.get("step"))
            name = d.get("name") or "?"
            c["tools"][name] = c["tools"].get(name, 0) + 1
            try:
                args = json.loads(d.get("arguments") or "{}")
            except ValueError:
                args = {}
            for k in ("file_path", "path"):
                v = args.get(k)
                if isinstance(v, str) and v:
                    c["files"].add(v)
        elif et == "turn/end":
            c["reason"] = ((d.get("reason") or {}).get("kind") or "")
    return [turns[t] for t in order]


# ---------------------------------------------------------------- 渲染


def render_full(meta, annotated, max_result, max_text):
    lines = []
    lines.append("# 会话转录 %s" % short_id(meta["id"]))
    lines.append("")
    lines.append("- 工作目录: `%s`" % meta["cwd"])
    lines.append("- 创建时间: %s" % ts(meta["created"]))
    lines.append("- 标题: %s" % (meta["title"] or "(无)"))
    if meta["parent"]:
        lines.append("- 父会话: `%s`" % meta["parent"])
    lines.append("")
    lines.append("> 按需取证：`grep` 本文件关键词拿行号，再 `read(offset=行号-10, limit=60)`；")
    lines.append("> 不要整篇读。")
    lines.append("")

    cur = None
    for ev in annotated:
        t = ev["_turn"]
        if t != cur:
            cur = t
            lines.append("")
            lines.append("---")
            lines.append("")
            lines.append("## 第 %d 轮 · %s" % (t, hhmm(ev.get("time"))))
            lines.append("")
        et = ev.get("type")
        d = ev.get("data") or {}
        if et == "user/message":
            txt = parts_text(d.get("content"))
            if not txt.strip():
                continue
            if is_injected(txt):
                if txt.startswith("Current runtime context"):
                    continue  # 噪声，直接丢
                lines.append("### ⚙ 系统注入")
                lines.append("")
                lines.append(clip(txt, 200))
                lines.append("")
            else:
                lines.append("### 👤 用户")
                lines.append("")
                lines.append(clip(txt, max_text))
                lines.append("")
        elif et == "assistant/message":
            txt = parts_text((d.get("message") or {}).get("content"))
            if txt.strip():
                lines.append("### 🤖 助手")
                lines.append("")
                lines.append(clip(txt, max_text))
                lines.append("")
        elif et == "tool/call":
            try:
                args = json.loads(d.get("arguments") or "{}")
            except ValueError:
                args = {}
            arg_s = json.dumps(args, ensure_ascii=False)
            lines.append("- **调用** `%s` `%s`" % (d.get("name") or "?", one_line(arg_s, 260)))
        elif et == "tool/result":
            msg = d.get("message") or {}
            txt = parts_text(msg.get("content"))
            if txt.strip():
                lines.append("- 结果: %s" % clip(txt, max_result).replace("\n", "\n  "))
    return "\n".join(lines)


def render_index(meta, turns):
    total_steps = sum(len(t["steps"]) for t in turns)
    total_tools = sum(sum(t["tools"].values()) for t in turns)
    lines = []
    lines.append("# 会话 %s · %s · %s" % (short_id(meta["id"]), meta["cwd"], ts(meta["created"])))
    lines.append("")
    lines.append("**标题**: %s" % (meta["title"] or "(无)"))
    lines.append("**规模**: %d 轮 / %d 步 / %d 次工具调用" % (len(turns), total_steps, total_tools))
    lines.append("")
    lines.append("| 轮 | 时间 | 工具 | 用户说了什么 | 这一轮收尾 |")
    lines.append("|---|---|---|---|---|")
    prev = ""
    for t in turns:
        tools = ", ".join("%s×%d" % (k, v)
                          for k, v in sorted(t["tools"].items(), key=lambda kv: -kv[1])[:5])
        if not t["users"]:
            user = "(仅系统注入)"
        else:
            txt = t["users"][0]
            # 客户端每轮都会重发整段用户消息，只显示"这轮新增的部分"
            if prev and txt.startswith(prev) and len(txt) > len(prev):
                user = "…" + txt[len(prev):]
            elif prev and prev.startswith(txt) and len(txt) <= len(prev):
                user = "(同上)"
            else:
                user = txt
            prev = txt if len(txt) > len(prev) else prev
            user = one_line(user, 70)
        user = user.replace("|", "\\|")
        tail = one_line(t["assistant"][-1], 70).replace("|", "\\|") if t["assistant"] else "-"
        lines.append("| %d | %s-%s | %s | %s | %s |"
                     % (t["turn"], hhmm(t["start"]), hhmm(t["end"]), tools or "-", user, tail or "-"))
    lines.append("")
    files = sorted({f for t in turns for f in t["files"]})
    if files:
        lines.append("**涉及文件**（前 50）:")
        lines.append("")
        for f in files[:50]:
            lines.append("- `%s`" % f)
        if len(files) > 50:
            lines.append("- …共 %d 个" % len(files))
    return "\n".join(lines)


def rebuild_index(outdir):
    rows = []
    for f in sorted(glob.glob(os.path.join(outdir, "*-index.md"))):
        short = os.path.basename(f)[: -len("-index.md")]
        with open(f, "r", encoding="utf-8") as fh:
            head = [next(fh, "") for _ in range(6)]
        title = ""
        for line in head:
            if line.startswith("**标题**:"):
                title = line.split(":", 1)[1].strip()
        parts = head[0].strip().lstrip("#").strip().split(" · ")
        loc = " · ".join(parts[1:]) if len(parts) >= 3 else head[0].strip()
        rows.append((short, loc, title))
    lines = ["# 会话档案索引", "",
             "查法：在本表挑一个会话 → 打开 `<id>-index.md`（每轮一行，0.1k~3k tokens）。",
             "只到这一级即可；不要去找或重建原文转录。",
             "",
             "| 会话 | 位置 · 时间 | 标题 | 摘要 |", "|---|---|---|---|"]
    for short, loc, title in rows:
        lines.append("| `%s` | %s | %s | `%s-index.md` |"
                     % (short, loc, title.replace("|", "\\|"), short))
    with open(os.path.join(outdir, "INDEX.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------- 命令


def cmd_list(args):
    zstd = zstd_bin(args.zstd)
    rows = []
    for s in iter_sessions():
        p = subprocess.run([zstd, "-d", "-c", s["log"]],
                           stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        head = None
        for line in p.stdout.decode("utf-8", "replace").split("\n"):
            if line.strip():
                try:
                    head = json.loads(line)
                except ValueError:
                    continue
                break
        if not head or head.get("type") != "session":
            continue
        rows.append({"id": short_id(head.get("id") or s["name"]),
                     "cwd": head.get("cwd") or "",
                     "created": head.get("createdAt") or 0,
                     "origin": head.get("origin") or "",
                     "size": s["size"]})
    if args.cwd:
        want = os.path.abspath(args.cwd).lower()
        rows = [r for r in rows if os.path.abspath(r["cwd"]).lower() == want]
    rows.sort(key=lambda r: r["created"], reverse=True)
    if not rows:
        print("没有匹配的会话")
        return
    print("%-9s %-17s %-9s %-8s %s" % ("ID", "创建时间", "来源", "大小", "工作目录"))
    for r in rows:
        print("%-9s %-17s %-9s %-8s %s" % (r["id"], ts(r["created"]),
                                           r["origin"] or "-",
                                           "%.0fKB" % (r["size"] / 1024.0), r["cwd"]))


def cmd_build(args):
    zstd = zstd_bin(args.zstd)
    cands = [s for s in iter_sessions() if args.id in s["name"]]
    if not cands:
        sys.exit("找不到会话: %s（先跑 list）" % args.id)
    if len(cands) > 1:
        sys.exit("前缀不唯一，匹配到 %d 个，请给更长的 ID" % len(cands))

    events = load_events(decompress(cands[0]["log"], zstd))
    meta, annotated = annotate(events)
    turns = turn_stats(annotated)
    sid = short_id(meta["id"]) or short_id(cands[0]["name"])

    outdir = os.path.abspath(args.out)
    os.makedirs(outdir, exist_ok=True)
    idx_name = "%s-index.md" % sid
    idx = render_index(meta, turns)
    with open(os.path.join(outdir, idx_name), "w", encoding="utf-8") as fh:
        fh.write(idx + "\n")

    written = [(idx_name, idx)]
    if args.with_full:                      # 默认不产出原文转录，只有显式要求才写
        full_name = "%s-full.md" % sid
        full = render_full(meta, annotated, args.max_result, args.max_text)
        with open(os.path.join(outdir, full_name), "w", encoding="utf-8") as fh:
            fh.write(full + "\n")
        written.append((full_name, full))

    rebuild_index(outdir)

    print("会话 %s  (%s)" % (sid, meta["cwd"]))
    for name, body in written:
        print("  %-26s %7.1f KB  %6d 字  ~%d tokens"
              % (name, len(body.encode("utf-8")) / 1024.0, len(body), len(body) // 3))
    print("  输出目录: %s" % outdir)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="DSH 会话日志 -> 两级可检索文本")
    ap.add_argument("--zstd", help="zstd.exe 路径")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("list", help="列出会话")
    p1.add_argument("--cwd", default=None)
    p1.set_defaults(func=cmd_list)

    p2 = sub.add_parser("build", help="导出某个会话（默认只出摘要索引）")
    p2.add_argument("id", help="会话 ID 前缀")
    p2.add_argument("--out", default="notes/sessions")
    p2.add_argument("--with-full", action="store_true",
                    help="额外产出 <id>-full.md 原文转录（默认不产出）")
    p2.add_argument("--max-result", type=int, default=400)
    p2.add_argument("--max-text", type=int, default=12000)
    p2.set_defaults(func=cmd_build)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
