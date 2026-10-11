"""Read-only Git/source inventory. Writes only a requested release-audit report.

Static references are evidence for review, NOT proof that an unreferenced file
is safe to delete. Never reads ignored .env files or prints credential values.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEXT = {".py", ".ps1", ".tsx", ".ts", ".js", ".mjs", ".json", ".md", ".yaml", ".yml", ".toml", ".css"}


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT).decode("utf-8", "replace")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--history", action="store_true", help="also scan Git text history for token/private-key patterns; only counts are recorded")
    args = ap.parse_args()
    tracked = git("ls-files", "-z").strip("\0").split("\0")
    untracked = git("ls-files", "--others", "--exclude-standard", "-z").strip("\0").split("\0")
    ignored = git("ls-files", "--others", "--ignored", "--exclude-standard", "--directory", "-z").strip("\0").split("\0")
    paths = sorted(set(tracked + untracked) - {""})
    texts = {p: (ROOT / p).read_text(encoding="utf-8", errors="replace") for p in paths
             if (ROOT / p).is_file() and (ROOT / p).suffix in TEXT and (ROOT / p).stat().st_size < 2_000_000}
    scripts = []
    syntax_errors = []
    local_path_findings = []
    secret_findings = []
    duplicates = defaultdict(list)
    for name, source in texts.items():
        duplicates[hashlib.sha256(source.encode()).hexdigest()].append(name)
        for number, line in enumerate(source.splitlines(), 1):
            if re.search(r"[A-Za-z]:[\\/]+Users[\\/]", line):
                local_path_findings.append({"path": name, "line": number})
            if re.search(r"gh[pousr]_[A-Za-z0-9]{36,}|sk-[A-Za-z0-9_-]{32,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----", line):
                secret_findings.append({"path": name, "line": number, "kind": "token/private-key pattern; value redacted"})
            uri = re.search(r"mongodb(?:\+srv)?://[^\s:/]+:([^\s@]+)@", line)
            if uri and not any(marker in uri.group(1).lower() for marker in ("change-me", "password", "<", "${", "example")):
                secret_findings.append({"path": name, "line": number, "kind": "credential URI; value redacted"})
        if name.startswith("scripts/") and name.endswith(".py"):
            try:
                tree = ast.parse(source)
            except SyntaxError as error:
                syntax_errors.append({"path": name, "line": error.lineno}); continue
            arguments = []
            imports = []
            calls = Counter()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import): imports.extend(alias.name for alias in node.names)
                if isinstance(node, ast.ImportFrom): imports.append(node.module or "")
                if isinstance(node, ast.Call):
                    method = node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id if isinstance(node.func, ast.Name) else ""
                    calls[method] += 1
                    if method in ("add_argument", "add_parser"):
                        arguments.extend(a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str))
            scripts.append({"path": name, "purpose": (ast.get_docstring(tree) or "No module docstring").split("\n")[0],
                            "arguments": arguments, "imports": sorted(set(imports)),
                            "subprocess_calls": {key: calls[key] for key in ("Popen", "run", "kill", "terminate", "wait", "communicate") if calls[key]},
                            "finally_blocks": sum(isinstance(node, ast.Try) and bool(node.finalbody) for node in ast.walk(tree)),
                            "reference_files": [p for p, content in texts.items() if p != name and (name in content or Path(name).stem in content)][:30]})
        elif name.startswith("scripts/") and name.endswith(".ps1"):
            scripts.append({"path": name, "purpose": "PowerShell operator entry point", "reference_files": [p for p, c in texts.items() if p != name and name in c]})
    component_candidates = []
    for name in texts:
        if name.startswith("frontend/apps/web/components/ui/") and name.endswith(".tsx"):
            specifier = "components/ui/" + Path(name).stem
            refs = [p for p, source in texts.items() if p != name and (specifier in source or f'"./{Path(name).stem}"' in source)]
            if not refs: component_candidates.append(name)
    packages = {}
    for name in paths:
        if name.endswith("package.json") and name in texts:
            package = json.loads(texts[name])
            packages[name] = {k: package.get(k) for k in ("name", "packageManager", "engines", "scripts", "dependencies", "devDependencies")}
    sensitive = [p for p in tracked if Path(p).name == ".env" or p.endswith((".key", ".pem"))]
    generated = [p for p in tracked if any(part in ("node_modules", ".venv", ".next", "__pycache__", "output") for part in Path(p).parts) or p.endswith(".tsbuildinfo")]
    history_counts = None
    if args.history:
        history = git("log", "--all", "-p", "--", ".", ":(exclude)*.pt")
        history_counts = {"commits": int(git("rev-list", "--all", "--count")),
                          "github_tokens": len(re.findall(r"gh[pousr]_[A-Za-z0-9]{36,}", history)),
                          "api_key_tokens": len(re.findall(r"sk-[A-Za-z0-9_-]{32,}", history)),
                          "private_keys": len(re.findall(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----", history))}
    counts = Counter(p.split("/")[0] if "/" in p else "(root)" for p in tracked)
    report = {"base_commit": git("rev-parse", "HEAD").strip(), "tracked_count": len(tracked), "tracked_by_directory": dict(sorted(counts.items())),
              "tracked": tracked, "untracked": untracked, "ignored_directory_inventory": ignored,
              "scripts": scripts, "packages": packages, "python_syntax_errors": syntax_errors,
              "developer_path_mentions": local_path_findings, "secret_pattern_findings_redacted": secret_findings,
              "tracked_sensitive_files": sensitive, "tracked_generated_outputs": generated,
              "duplicate_text_groups": [names for names in duplicates.values() if len(names) > 1],
              "unreferenced_ui_review_only": component_candidates,
              "git_history_pattern_counts_redacted": history_counts,
              "note": "Static analysis only. No deletion. Ignored directories collapsed; local databases/caches/data are not submission sources."}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"tracked": len(tracked), "untracked": len(untracked), "scripts": len(scripts), "secret_findings": len(secret_findings), "syntax_errors": syntax_errors, "generated_tracked": generated}))
    return int(bool(syntax_errors or sensitive or secret_findings))


if __name__ == "__main__":
    raise SystemExit(main())
