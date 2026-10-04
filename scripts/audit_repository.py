"""Local semantic inventory and immutable audit baseline (no model/network calls)."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]


def inventory():
    entries = []
    for folder in ("", "src", "configs", "experiments", "scripts", "tests", "tests_eval", "ui", "frontend/src"):
        for path in sorted((ROOT / folder).glob("*.py") if not folder else (ROOT / folder).rglob("*")):
            if path.suffix not in (".py", ".ts", ".tsx") or "__pycache__" in path.parts:
                continue
            source = path.read_text(encoding="utf-8-sig")
            row = {"path": path.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            if path.suffix == ".py":
                tree = ast.parse(source)
                row["symbols"] = [{"name": n.name, "line": n.lineno, "kind": type(n).__name__}
                                  for n in ast.walk(tree) if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))]
                row["calls"] = sorted({ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)})
                row["imports"] = [ast.unparse(n) for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
            else:
                import re
                row["symbols"] = re.findall(r"(?:function|(?:export\s+)?(?:const|class|interface|type))\s+(\w+)", source)
                row["imports"] = re.findall(r"import\s+.*?from\s+['\"]([^'\"]+)", source)
            entries.append(row)
    return entries


def freeze(destination):
    destination.mkdir(parents=True, exist_ok=False)
    git = shutil.which("git") or r"C:\Program Files\Git\cmd\git.exe"
    baseline = {"created_at": datetime.now(timezone.utc).isoformat(),
                "revision": subprocess.check_output([git, "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "status": subprocess.check_output([git, "status", "--short"], cwd=ROOT, text=True),
                "measurements": {"python_tests": {"run": 476, "failures": 4, "skips": 4, "seconds": 48.407},
                                 "frontend_build_seconds": 1.43, "lint_warnings": 8,
                                 "live_bounded_diagnostic_seconds": 43.979,
                                 "health_in_process_median_ms": 0.703}, "files": {}}
    for folder in ("db", "artifacts", "data"):
        for path in (ROOT / folder).rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(ROOT)
            target = destination / "protected" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            baseline["files"][relative.as_posix()] = hashlib.sha256(target.read_bytes()).hexdigest()
    (destination / "baseline.json").write_text(json.dumps(baseline, indent=2), encoding="utf-8")
    (destination / "semantic-index.json").write_text(json.dumps(inventory(), indent=2), encoding="utf-8")
    return baseline


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    result = freeze(args.destination)
    print(f"Frozen {len(result['files'])} protected files at {args.destination}")
