"""Update the code of this working copy in seconds, without the 350 MB archive and without touching data/.

The repository carries ~375 MB of downloaded data next to ~2 MB of code, so re-downloading the whole ZIP after
every change is mostly waste. This script fetches only the code:

    python scripts/get_code.py --dry-run     # what would change, nothing written
    python scripts/get_code.py               # git backend if git is installed (recommended)
    python scripts/get_code.py --http        # no git: GitHub API, only changed files

git backend: fetches the branch with `--filter=blob:none --depth 1` (the tree only, 0.5 MB) and then checks
out just the files that differ, so git downloads their blobs one by one and nothing else. Works in a folder
unpacked from a ZIP as well: it creates a repository there (`.git`), which later runs reuse (2-5 s).

http backend: one request for the whole file tree, then the files whose git blob hash differs from the local
one; concurrent downloads from raw.githubusercontent.com, with the contents API as a fallback.

Neither backend deletes anything outside the code paths, and `data/raw`, `data/interim`, `results/*.parquet`
and `reports/` are never downloaded: those come from the exchanges, not from git.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DEFAULT_REMOTE = "https://github.com/nikalekcandr/quant_research.git"
DEFAULT_BRANCH = "main"

# what "code" means: small text files that make the pipeline run; everything else is data
CODE_DIRS = ("src", "studies", "tests", "scripts", "docs", "data/lit", "results/studies")
CODE_EXT = {".py", ".md", ".txt", ".toml", ".cfg", ".ini", ".json", ".yml", ".yaml", ".sh", ".bat", ".ps1", ".csv"}
ROOT_FILES = ("README.md", "AGENTS.md", "CLAUDE.md", "PLAN.md", "ROADMAP.md", "STRATEGIES.md", "RUN_ME.md",
              "config.py", "requirements.txt", "requirements-dev.txt", ".gitignore")
MAX_BYTES = 2_000_000          # the 16 MB report CSV and the parquet files stay out


def is_code_path(path: str, size: int | None = None) -> bool:
    """True for the files this script is allowed to download (never data/, never big binaries)."""
    p = path.replace("\\", "/")
    if p.startswith("../") or "/../" in p:
        return False
    if size is not None and size > MAX_BYTES:
        return False
    if "/" not in p:
        return p in ROOT_FILES
    top = p.split("/")[0]
    if not any(p.startswith(d + "/") for d in CODE_DIRS):
        return False
    if top == "results":                       # only the readable reports: tables and parquet are outputs
        return p.startswith("results/studies/") and p.endswith(".md")
    return os.path.splitext(p)[1].lower() in CODE_EXT


def blob_sha(data: bytes) -> str:
    """Git's object id for a blob: sha1("blob <len>\\0" + data). Identical to `git hash-object`."""
    return hashlib.sha1(b"blob %d\x00" % len(data) + data).hexdigest()


def local_sha(path: Path) -> str | None:
    try:
        return blob_sha(path.read_bytes())
    except OSError:
        return None


def changed_paths(root: Path, remote: dict[str, str], only: set[str] | None = None) -> tuple[list[str], list[str]]:
    """(paths to write, paths that exist locally but are gone upstream) — decided by content hashes."""
    write, stale = [], []
    for path, sha in sorted(remote.items()):
        if only and path not in only:
            continue
        if local_sha(root / path) != sha:
            write.append(path)
    return write, stale


def _git(args, cwd, check=True, quiet=True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=cwd, check=check, text=True,
                          stdout=subprocess.PIPE if quiet else None,
                          stderr=subprocess.PIPE if quiet else None)


def git_backend(root: Path, remote: str, branch: str, dry_run: bool, force: bool) -> int:
    if not shutil.which("git"):
        raise RuntimeError("git is not installed")
    if not (root / ".git").exists():
        print(f"  initialising a repository in {root} (one time)")
        _git(["init", "-q"], root)
        _git(["remote", "add", "origin", remote], root)
    remotes = _git(["remote"], root).stdout.split()
    if "origin" not in remotes:
        _git(["remote", "add", "origin", remote], root)

    t0 = time.time()
    print(f"  fetching the tree of {branch} (no file contents yet)")
    try:
        _git(["fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", branch], root)
    except subprocess.CalledProcessError:
        print("  (server does not support partial fetch, falling back to a plain shallow fetch)")
        _git(["fetch", "-q", "--depth", "1", "origin", branch], root)
    head = _git(["rev-parse", "FETCH_HEAD"], root).stdout.strip()

    listing = _git(["ls-tree", "-r", "FETCH_HEAD"], root).stdout.splitlines()
    tree: dict[str, str] = {}
    for line in listing:
        meta, path = line.split("\t", 1)
        mode, kind, sha = meta.split()
        if kind == "blob" and is_code_path(path):
            tree[path] = sha
    write, _ = changed_paths(root, tree)
    print(f"  tree fetched in {time.time() - t0:.1f} s: {len(tree)} code files upstream, {len(write)} differ")
    if not write:
        print("  already up to date")
        return 0
    for p in write[:25]:
        print(f"    {'new ' if not (root / p).exists() else 'upd '} {p}")
    if len(write) > 25:
        print(f"    ... and {len(write) - 25} more")
    if dry_run:
        return 0

    if not force:
        dirty = _git(["status", "--porcelain", "--", *write], root, check=False).stdout.splitlines()
        # only edits in the working tree matter: "M ", "AM" etc. Untracked ("??") means never tracked here,
        # and a staged copy ("A ") is just what the previous sync checked out.
        dirty = [d for d in dirty if len(d) > 1 and d[1] not in (" ", "?")]
        if dirty:
            print("\n  local changes in these files, refusing to overwrite:")
            for d in dirty[:10]:
                print(f"    {d}")
            print("  run with --force to replace them anyway")
            return 2

    t1 = time.time()
    print(f"  checking out {len(write)} file(s): git downloads only their contents")
    _git(["checkout", "FETCH_HEAD", "--", *write], root)
    print(f"  done in {time.time() - t1:.1f} total; commit {head[:8]}")
    return 0


def _parse_remote(remote: str) -> tuple[str, str] | None:
    r = remote.rstrip("/").removesuffix(".git")
    if "github.com" in r:
        parts = r.split("github.com")[-1].strip("/:").split("/")
        if len(parts) >= 2:
            return parts[0], parts[1]
    return None


def _get(url: str, token: str | None = None) -> bytes:
    """One GET; QRR_CA_BUNDLE names a CA file for the corporate proxies that break the default store."""
    import requests
    headers = {"User-Agent": "quant-research-get-code", "Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    r = requests.get(url, headers=headers, timeout=60, verify=os.environ.get("QRR_CA_BUNDLE") or True)
    r.raise_for_status()
    return r.content


def http_backend(root: Path, remote: str, branch: str, dry_run: bool, limit: int | None) -> int:
    owner_repo = _parse_remote(remote)
    if not owner_repo:
        raise RuntimeError(f"cannot parse a GitHub repository out of {remote}")
    owner, repo = owner_repo
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    print(f"  asking GitHub for the file tree of {owner}/{repo}@{branch}")
    tree_json = json.loads(_get(f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1", token))
    entries = [e for e in tree_json.get("tree", []) if e.get("type") == "blob"]
    tree = {e["path"]: e["sha"] for e in entries if is_code_path(e["path"], e.get("size"))}
    if tree_json.get("truncated"):
        print("  (warning: GitHub truncated the tree listing)")
    write, _ = changed_paths(root, tree)
    print(f"  {len(tree)} code files upstream, {len(write)} differ")
    if limit:
        write = write[:limit]
    if not write:
        print("  already up to date")
        return 0
    for p in write[:25]:
        print(f"    {'new ' if not (root / p).exists() else 'upd '} {p}")
    if dry_run:
        return 0

    commit = tree_json.get("sha") or branch

    raw_ok = {"v": None}                         # once the CDN is unreachable, stop paying for its timeout

    def one(path: str) -> tuple[str, bytes | None]:
        want = tree[path]
        data = None
        if raw_ok["v"] is not False:
            try:                                 # a CDN, not the rate-limited REST API
                data = _get(f"https://raw.githubusercontent.com/{owner}/{repo}/{commit}/{path}")
                raw_ok["v"] = True
            except Exception:
                raw_ok["v"] = False
        if data is None:
            blob = json.loads(_get(f"https://api.github.com/repos/{owner}/{repo}/contents/{path}?ref={commit}", token))
            data = base64.b64decode(blob["content"]) if "content" in blob else None
        if data is not None and blob_sha(data) != want:
            raise RuntimeError(f"hash mismatch for {path}: got {blob_sha(data)[:8]}, expected {want[:8]}")
        return path, data

    t0 = time.time()
    written = 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        for path, data in ex.map(one, write):
            if data is None:
                continue
            dest = root / path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            written += 1
    (root / ".code_sync.json").write_text(json.dumps({"commit": commit, "when": time.time()}, indent=2), encoding="utf-8")
    print(f"  wrote {written} file(s) in {time.time() - t0:.1f} s (hashes verified)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="update only the code of this working copy (whole tree in data/ is left alone)")
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[1]), help="working copy to update")
    ap.add_argument("--remote", default=os.environ.get("QRR_REMOTE", DEFAULT_REMOTE))
    ap.add_argument("--branch", default=os.environ.get("QRR_BRANCH", DEFAULT_BRANCH))
    ap.add_argument("--http", action="store_true", help="use the GitHub API instead of git")
    ap.add_argument("--dry-run", action="store_true", help="only show what would change")
    ap.add_argument("--force", action="store_true", help="overwrite locally modified files")
    ap.add_argument("--limit", type=int, default=None, help="http backend: stop after N files (for tests)")
    a = ap.parse_args(argv)
    root = Path(a.root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    print(f"code update: {root} <- {a.remote} [{a.branch}]")
    backend = http_backend if a.http else git_backend
    try:
        if a.http:
            return backend(root, a.remote, a.branch, a.dry_run, a.limit)
        return backend(root, a.remote, a.branch, a.dry_run, a.force)
    except RuntimeError as e:
        print(f"  git backend unavailable: {e}; trying the GitHub API")
        return http_backend(root, a.remote, a.branch, a.dry_run, a.limit)


if __name__ == "__main__":
    sys.exit(main())
