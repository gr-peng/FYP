import subprocess
from pathlib import Path

import pytest

import scripts.sync_to_github as sync_script
from scripts.sync_to_github import (
    SyncError,
    canonical_remote,
    sanitize_text,
    validate_comment,
)


def test_canonical_remote_accepts_https_and_ssh() -> None:
    assert canonical_remote("https://github.com/gr-peng/FYP.git") == (
        "https://github.com/gr-peng/fyp"
    )
    assert canonical_remote("git@github.com:gr-peng/FYP.git") == ("https://github.com/gr-peng/fyp")


def test_sanitize_text_hides_credentials() -> None:
    fake_token = "sk-" + "1234567890123456"
    fake_key = "top" + "secret123456"
    value = f"https://user:secret@example.test/x API_KEY={fake_key} {fake_token}"
    cleaned = sanitize_text(value)
    assert "secret" not in cleaned
    assert fake_key not in cleaned
    assert fake_token not in cleaned
    assert "REDACTED" in cleaned


def test_validate_comment_rejects_sensitive_assignment() -> None:
    with pytest.raises(SyncError):
        validate_comment("sync API_KEY=" + "topsecret123456")


def test_validate_comment_allows_normal_multiline_message(tmp_path: Path) -> None:
    message = "Implement exchange core\n\nAdd price-time priority tests"
    validate_comment(message)


def test_staged_scanner_blocks_without_printing_secret(monkeypatch, capsys) -> None:
    secret = "top" + "secret123456"

    def fake_run_git(repo, args, **kwargs):
        if args[:4] == ["diff", "--cached", "--name-only", "--diff-filter=ACMR"]:
            return subprocess.CompletedProcess(args, 0, ".env\0", "")
        return subprocess.CompletedProcess(args, 0, f"+OPENAI_API_KEY={secret}\n", "")

    monkeypatch.setattr(sync_script, "run_git", fake_run_git)
    with pytest.raises(SyncError):
        sync_script.scan_staged(Path("/tmp/project"))
    output = capsys.readouterr().out
    assert secret not in output
    assert "敏感文件名: .env" in output
