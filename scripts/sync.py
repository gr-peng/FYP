#!/usr/bin/env python3
"""Safely commit, merge and push this project to the FYP GitHub repository.

Authentication is deliberately delegated to Git's credential helper or SSH agent.
This script never accepts a token argument and never places a token in a remote URL.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

# SSH avoids a stale editor credential helper and does not put a PAT in a URL.
DEFAULT_REMOTE = "git@github.com:gr-peng/FYP.git"
DEFAULT_BRANCH = "main"
DEFAULT_REMOTE_NAME = "origin"

SENSITIVE_NAME = re.compile(
    r"(?:^|/)(?:\.env(?:\..*)?|credentials?[^/]*|secrets?[^/]*|"
    r".*\.(?:pem|key|p12|pfx)|id_(?:rsa|dsa|ecdsa|ed25519))(?:/|$)",
    re.IGNORECASE,
)

# Match assignments, not prose such as "API key handling". Placeholder values are
# ignored so that documented examples can remain in source and README files.
SENSITIVE_ASSIGNMENT = re.compile(
    r"(?ix)(?:api[_-]?key|secret|token|password|passwd|private[_-]?key|"
    r"access[_-]?key|authorization|bearer)\s*[:=]\s*"
    r"[\"']?([A-Za-z0-9_./+=:@-]{8,})"
)
KNOWN_TOKEN = re.compile(
    r"(?i)\b(?:sk-[A-Za-z0-9_-]{16,}|sk-proj-[A-Za-z0-9_-]{16,}|"
    r"ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|"
    r"xox[baprs]-[A-Za-z0-9-]{16,}|AKIA[0-9A-Z]{16})\b"
)
PLACEHOLDER = re.compile(
    r"(?i)^(?:your[-_ ]?(?:api[-_ ]?)?key|replace[-_ ]?me|change[-_ ]?me|"
    r"example|dummy|placeholder|changeme|<[^>]+>|\$\{[^}]+\})$"
)
URL_CREDENTIAL = re.compile(r"(?i)(https?://)[^/@\s]+@")
URL_QUERY_CREDENTIAL = re.compile(r"(?i)[?&](?:token|access_token|api_key|apikey|password)=[^&\s]+")


class SyncError(RuntimeError):
    """A recoverable synchronization failure."""


def sanitize_text(value: str) -> str:
    """Remove likely credentials before anything from Git reaches the terminal."""
    value = URL_CREDENTIAL.sub(r"\1***@", value)
    value = KNOWN_TOKEN.sub("***REDACTED***", value)

    def replace_assignment(match: re.Match[str]) -> str:
        return f"{match.group(0)[: match.group(0).find(match.group(1))]}***REDACTED***"

    return SENSITIVE_ASSIGNMENT.sub(replace_assignment, value)


def run_git(
    repo: Path,
    args: Sequence[str],
    *,
    check: bool = True,
    quiet: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run Git without a shell and sanitize captured output."""
    process = subprocess.run(
        ["git", *args],
        cwd=repo,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "1"},
        text=True,
        capture_output=True,
        check=False,
    )
    if check and process.returncode:
        detail = sanitize_text((process.stderr or process.stdout or "").strip())
        raise SyncError(detail or f"git {' '.join(args)} failed")
    if not quiet:
        output = sanitize_text((process.stdout or "") + (process.stderr or ""))
        if output.strip():
            print(output.rstrip())
    return process


def find_repo() -> Path:
    process = run_git(Path.cwd(), ["rev-parse", "--show-toplevel"], quiet=True)
    if process.returncode:
        raise SyncError("当前目录不在 Git 仓库内")
    return Path(process.stdout.strip()).resolve()


def canonical_remote(url: str) -> str:
    value = url.strip().lower().removesuffix(".git")
    if value.startswith("git@github.com:"):
        value = "https://github.com/" + value.removeprefix("git@github.com:")
    elif value.startswith("ssh://git@github.com/"):
        value = "https://github.com/" + value.removeprefix("ssh://git@github.com/")
    return value.rstrip("/")


def validate_branch(repo: Path, branch: str) -> None:
    result = run_git(repo, ["check-ref-format", "--branch", branch], check=False, quiet=True)
    if result.returncode:
        raise SyncError(f"无效的 branch 名称: {branch!r}")


def ensure_remote(repo: Path, name: str, expected_url: str) -> None:
    if URL_CREDENTIAL.search(expected_url) or URL_QUERY_CREDENTIAL.search(expected_url):
        raise SyncError("--remote-url 不能包含 token、password 或其他 URL 凭据")
    existing = run_git(repo, ["remote", "get-url", name], check=False, quiet=True)
    if existing.returncode:
        run_git(repo, ["remote", "add", name, expected_url])
        print(f"已添加远程仓库 {name}（凭据由 Git 管理）")
        return
    actual = existing.stdout.strip()
    if canonical_remote(actual) != canonical_remote(expected_url):
        raise SyncError(
            f"远程 {name!r} 已指向另一个仓库；未自动改写。"
            "如确需更换目标，请先用 git remote set-url，或传入 --remote-url。"
        )


def is_authentication_error(message: str) -> bool:
    lowered = message.lower()
    return any(
        marker in lowered
        for marker in (
            "authentication failed",
            "no anonymous write access",
            "missing or invalid credentials",
            "could not read username",
            "permission denied (publickey)",
            "credential helper",
        )
    )


def authentication_hint(repo: Path, remote: str) -> str:
    """Give a credential-free recovery path for GitHub write authentication."""
    return (
        "GitHub 写权限认证失败。若 HTTPS credential helper 不可用，可执行：\n"
        f"  git -C {repo} remote set-url {remote} git@github.com:gr-peng/FYP.git\n"
        "  ssh -T git@github.com\n"
        "然后重新运行同步脚本。也可以先配置本机 Git credential helper；不要把 PAT/API key"
        " 写进 remote URL。"
    )


def staged_sensitive_paths(repo: Path) -> list[str]:
    result = run_git(
        repo,
        ["diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z"],
        quiet=True,
    )
    paths = [path for path in result.stdout.split("\0") if path]
    return [path for path in paths if SENSITIVE_NAME.search(path) and path != ".env.example"]


def staged_sensitive_lines(repo: Path) -> list[str]:
    result = run_git(
        repo,
        ["diff", "--cached", "--unified=0", "--no-color", "--binary"],
        quiet=True,
    )
    violations: list[str] = []
    for raw_line in result.stdout.splitlines():
        if not raw_line.startswith("+") or raw_line.startswith("+++"):
            continue
        line = raw_line[1:]
        assignment = SENSITIVE_ASSIGNMENT.search(line)
        if assignment:
            value = assignment.group(1).strip("\"'")
            # Avoid flagging scanner declarations such as
            # ``KNOWN_TOKEN = re.compile(...)`` in this script itself.
            is_code_declaration = value.lower().startswith("re.")
            if not is_code_declaration and not PLACEHOLDER.fullmatch(value):
                violations.append("疑似敏感赋值")
                continue
        if KNOWN_TOKEN.search(line):
            violations.append("疑似已知 token 格式")
    return violations


def scan_staged(repo: Path) -> None:
    paths = staged_sensitive_paths(repo)
    lines = staged_sensitive_lines(repo)
    if not paths and not lines:
        return
    print("检测到可能的敏感信息，已停止同步；未执行 commit 或 push。")
    for path in paths:
        print(f"  敏感文件名: {path}")
    for violation in lines:
        print(f"  {violation}（具体值已隐藏）")
    raise SyncError("请移除密钥、token、密码或敏感文件后重试")


def validate_comment(comment: str) -> None:
    if not comment.strip():
        raise SyncError("comment 不能为空")
    if SENSITIVE_ASSIGNMENT.search(comment) or KNOWN_TOKEN.search(comment):
        raise SyncError("comment 看起来包含敏感信息；具体值不会被输出")


@contextmanager
def message_file(comment: str) -> Iterator[Path]:
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", prefix="fyp-sync-", suffix=".msg", delete=False
    )
    try:
        os.chmod(handle.name, 0o600)
        handle.write(comment.rstrip() + "\n")
        handle.close()
        yield Path(handle.name)
    finally:
        try:
            Path(handle.name).unlink()
        except FileNotFoundError:
            pass


def remote_branch_exists(repo: Path, remote: str, branch: str) -> bool:
    result = run_git(repo, ["ls-remote", "--heads", remote, branch], check=False, quiet=True)
    if result.returncode:
        detail = sanitize_text((result.stderr or result.stdout or "").strip())
        raise SyncError(detail or "无法读取远程仓库；请检查网络和 Git 凭据")
    return bool(result.stdout.strip())


def merge_in_progress(repo: Path) -> bool:
    """Return whether Git has an unfinished merge in this worktree."""
    result = run_git(repo, ["rev-parse", "--git-path", "MERGE_HEAD"], quiet=True)
    merge_head = Path(result.stdout.strip())
    if not merge_head.is_absolute():
        merge_head = repo / merge_head
    return merge_head.exists()


def continue_existing_merge(repo: Path, comment: str) -> str:
    """Finish a user-resolved merge while keeping the requested message."""
    unmerged = run_git(
        repo,
        ["diff", "--name-only", "--diff-filter=U"],
        check=False,
        quiet=True,
    )
    if unmerged.returncode:
        raise SyncError("无法检查 merge 冲突状态")
    if unmerged.stdout.strip():
        raise SyncError("仍有未解决的 merge 冲突；请编辑文件并执行 git add 后重试")
    staged = run_git(repo, ["diff", "--cached", "--quiet"], check=False, quiet=True)
    if staged.returncode == 0:
        raise SyncError("merge 冲突已标记解决前，请先执行 git add <已解决文件>")
    if staged.returncode != 1:
        raise SyncError("无法读取 merge 的 staged 改动")
    scan_staged(repo)
    with message_file(comment) as path:
        run_git(repo, ["commit", "--file", str(path)])
    return "continued"


def switch_to_branch(repo: Path, branch: str, remote: str, has_remote: bool) -> None:
    current = run_git(repo, ["branch", "--show-current"], quiet=True).stdout.strip()
    dirty = bool(run_git(repo, ["status", "--porcelain"], quiet=True).stdout.strip())
    local_exists = (
        run_git(
            repo, ["show-ref", "--verify", f"refs/heads/{branch}"], check=False, quiet=True
        ).returncode
        == 0
    )
    if current != branch and dirty:
        raise SyncError(
            f"当前 branch 是 {current or '(detached HEAD)'} 且工作区有改动；"
            f"请先处理改动，再切换到目标 branch {branch!r}。"
        )
    if current == branch:
        return
    if local_exists:
        run_git(repo, ["switch", branch])
    elif has_remote:
        run_git(repo, ["switch", "--track", "-c", branch, f"{remote}/{branch}"])
    else:
        run_git(repo, ["switch", "-c", branch])


def commit_local_changes(repo: Path, comment: str) -> bool:
    run_git(repo, ["add", "--all"])
    status = run_git(repo, ["diff", "--cached", "--quiet"], check=False, quiet=True)
    if status.returncode == 0:
        return False
    if status.returncode != 1:
        raise SyncError("无法读取 staged 改动")
    scan_staged(repo)
    with message_file(comment) as path:
        run_git(repo, ["commit", "--file", str(path)])
    print("已提交本地改动")
    return True


def merge_remote(repo: Path, remote: str, branch: str, comment: str, has_remote: bool) -> str:
    if not has_remote:
        return "remote-empty"
    remote_ref = f"refs/remotes/{remote}/{branch}"
    local_ref = "HEAD"
    remote_included = run_git(
        repo,
        ["merge-base", "--is-ancestor", remote_ref, local_ref],
        check=False,
        quiet=True,
    )
    if remote_included.returncode == 0:
        return "remote-included"
    local_included = run_git(
        repo,
        ["merge-base", "--is-ancestor", local_ref, remote_ref],
        check=False,
        quiet=True,
    )
    if local_included.returncode == 0:
        run_git(repo, ["merge", "--ff-only", remote_ref])
        return "fast-forward"
    merge_args = ["merge", "--no-ff"]
    if run_git(repo, ["merge-base", local_ref, remote_ref], check=False, quiet=True).returncode:
        merge_args.append("--allow-unrelated-histories")
    with message_file(comment) as path:
        run_git(repo, [*merge_args, "--file", str(path), remote_ref])
    return "merge"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-b", "--branch", default=DEFAULT_BRANCH, help="目标 branch（默认 main）")
    parser.add_argument(
        "-m", "--comment", default=None, help="本次同步/merge 的 comment；不传则使用默认说明"
    )
    parser.add_argument("--comment-file", type=Path, help="从文件读取多行 comment")
    parser.add_argument("--remote", default=DEFAULT_REMOTE_NAME, help="远程名称（默认 origin）")
    parser.add_argument("--remote-url", default=DEFAULT_REMOTE, help="目标仓库 URL")
    parser.add_argument("--no-push", action="store_true", help="只提交和合并，不 push，用于预检查")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.comment_file and args.comment is not None:
            raise SyncError("--comment 和 --comment-file 只能选择一个")
        if args.comment_file:
            comment = args.comment_file.read_text(encoding="utf-8")
        else:
            comment = args.comment or f"Sync project to {args.branch}"
        validate_comment(comment)
        repo = find_repo()
        validate_branch(repo, args.branch)
        ensure_remote(repo, args.remote, args.remote_url)

        has_remote = remote_branch_exists(repo, args.remote, args.branch)
        if has_remote:
            run_git(repo, ["fetch", args.remote, args.branch])
        switch_to_branch(repo, args.branch, args.remote, has_remote)
        if merge_in_progress(repo):
            committed = False
            merge_result = continue_existing_merge(repo, comment)
        else:
            committed = commit_local_changes(repo, comment)
            merge_result = merge_remote(repo, args.remote, args.branch, comment, has_remote)
        if args.no_push:
            print(
                f"预检查完成（commit={committed}, merge={merge_result}）；由于 --no-push 未上传。"
            )
            return 0
        run_git(repo, ["push", "--set-upstream", args.remote, args.branch])
        print(f"同步完成：{args.remote}/{args.branch}（commit={committed}, merge={merge_result}）")
        return 0
    except (OSError, SyncError, subprocess.SubprocessError) as error:
        detail = sanitize_text(str(error))
        if is_authentication_error(detail):
            print(
                authentication_hint(repo if "repo" in locals() else Path.cwd(), args.remote),
                file=sys.stderr,
            )
        else:
            print(f"同步失败：{detail}", file=sys.stderr)
        print("未执行强制 push；若发生 merge 冲突，请解决冲突后重新运行。", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
