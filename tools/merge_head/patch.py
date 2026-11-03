#!/usr/bin/env python3
"""Move the public head's final hash-bound source_edits pass of apply_merged_policy to the end of build(), after our
passes, applied by rebind_edits (sites found by their ops and context, not by position).

    patch.py <worktree>      edits src/point_add/mod.rs and adds src/point_add/rebind_edits.rs
"""
import re, shutil, sys
from pathlib import Path

wt = Path(sys.argv[1])
mod = wt / "src/point_add/mod.rs"
s = mod.read_text()
i = s.index("fn apply_merged_policy(")
j = s.index("\n}\n", i)
body = s[i:j]
m = re.search(r"\n(\s*)source_edits::run\(ops,\s*include_str!\(\"([^\"]+)\"\)\)\s*$", body)
if not m:
    sys.exit("patch: apply_merged_policy does not end with a source_edits pass; nothing to move")
table = m.group(2)
head = body[:m.start()]
k = head.rindex("\n")
lastline = head[k + 1:]
lm = re.fullmatch(r"(\s*)let ops = (.+);", lastline)
if not lm:
    sys.exit("patch: the line before the final pass is not 'let ops = ...;'")
s = s[:i] + head[:k + 1] + f"{lm.group(1)}{lm.group(2)}" + s[j:]
b = s.index("pub fn build() -> Vec<Op> {")
e = s.index("\n}\n", b)
build = s[b:e]
call = "apply_merged_policy(divide_prefix::prefix())"
if call not in build or 'let nonce: u64 = required_env("TAIL_NONCE");' not in build:
    sys.exit("patch: build() does not look as expected")
k = build.index("\n", build.index("clear_process_env();"))
build = (build[:k] + f"\n    let policy = {call};\n    let policy_edits = rebind_edits::prepare(&policy, include_str!(\"{table}\"));"
         + build[k:].replace(call, "policy", 1))
build = build.replace('    let nonce: u64 = required_env("TAIL_NONCE");',
                      '    ops = rebind_edits::run(ops, &policy_edits);\n    let nonce: u64 = required_env("TAIL_NONCE");', 1)
s = s[:b] + build + s[e:]
s = s.replace("mod hold_pass;\n", "mod hold_pass;\nmod rebind_edits;\n", 1)
mod.write_text(s)
shutil.copy(Path(__file__).with_name("rebind_edits.rs"), wt / "src/point_add/rebind_edits.rs")
print(f"patch: {table} now applied after our passes by rebind_edits")
