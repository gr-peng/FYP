# GitHub 同步脚本

`scripts/sync.py` 面向本地项目目录执行以下流程：

1. 校验目标 branch 名称，并确保 `origin`（或 `--remote` 指定的名称）指向目标仓库。
2. 查询并 fetch 目标远程 branch；本地没有该 branch 时创建或跟踪它。
3. 执行 `git add --all`，扫描 staged 文件名和新增内容中的常见敏感信息。
4. 有本地改动时，用 comment 创建提交。
5. 远程有分叉时，用同一 comment 创建真正的 merge commit；可快进或远程已包含时
   不额外创建 merge commit。
6. 默认执行 `git push --set-upstream`；传 `--no-push` 则停在本地预检查。

## 示例

```bash
# 默认同步 main，使用单行说明
python scripts/sync.py -m "Add exchange accounting tests"

# 同步其他 branch，并提供多行 merge message
python scripts/sync.py -b research --comment-file /tmp/research-message.txt

# 只验证本地提交/合并，不上传
python scripts/sync.py --no-push -b main -m "Dry-run sync"
```

如果 `origin` 尚不存在，脚本会添加不含凭据的目标 URL。HTTPS 认证使用 Git 已配置的
credential helper，SSH 认证使用 SSH agent；脚本不读取、保存或接收 API key/PAT，也不
支持把 token 嵌进 `--remote-url`。需要 GitHub 权限时，应先在本机配置 credential helper
或 SSH key。

扫描器会阻止 `.env`、凭据/密钥文件、私钥和新增内容中的常见 key/token/password 赋值，
并且只报告文件名或“疑似敏感信息”，不会打印具体值。扫描命中后改动保持 staged，便于
用户自行移除；脚本不会自动删除文件。merge 冲突会保留在工作区，解决并执行 `git add`
后重新运行即可继续完成 merge。

脚本的 `--comment` 是 Git commit/merge message，不是 GitHub Pull Request 的网页评论。
同步策略不依赖 GitHub REST API，因此不需要额外的 API key。
