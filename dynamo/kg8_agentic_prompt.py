#!/usr/bin/env python3
"""Rewrap known_good_8 prompts with a SWE agent instruction template.

Root cause this fixes: uni_agent/tasks/swe_rebench/preprocess.py:52 emits only
[{"role":"user","content":problem_statement}] — no agent behavior instructions.
Qwen3-32B then answers the issue as a chat question (plain text, no tool call)
and the ReAct loop terminates at step 1 (agent.py: plain text => "finished").
Measured: 86/126 sessions ended at step 1; median 3 turns. With this template
episodes use the tool loop and grow to the long contexts the replay needs.
"""
import pandas as pd

SYS = """You are an autonomous software engineer working in a sandboxed repository checkout at /testbed. Your job is to resolve the GitHub issue given by the user by modifying the codebase.

Rules you must follow:
- ALWAYS respond with exactly one tool call. Never reply with plain text: a plain-text reply terminates the session as a failure.
- Explore before editing: locate the relevant files and read the code (shell with grep/find, editor view).
- Reproduce the problem first (a small script or command), then fix the root cause with minimal changes, then re-run to verify the fix.
- Run the relevant tests to check for regressions.
- Do NOT modify tests. Do NOT call submit until you have verified your fix works. Early submission without a verified fix counts as failure. You have 20 steps: use them thoroughly."""

USER_TMPL = """<issue>
{stmt}
</issue>

The repository is checked out at /testbed (git repo, dependencies pre-installed). Resolve the issue described above. Begin by exploring the relevant code."""

SRC = "/workspace/data/uni_agent/swe_rebench_known_good_8.parquet"
DST = "/workspace/data/uni_agent/swe_rebench_known_good_8_agentic.parquet"

df = pd.read_parquet(SRC)
df["prompt"] = df["prompt"].apply(lambda p: [
    {"role": "system", "content": SYS},
    {"role": "user", "content": USER_TMPL.format(stmt=p[0]["content"])},
])
df.to_parquet(DST)
print(f"wrote {DST} rows={len(df)}")
