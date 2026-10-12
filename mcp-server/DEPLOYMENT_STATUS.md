# AI 角斗场 (Arena) · MCP 服务器 — 四通道部署状态

> 生成时间：2026-10-12（v0.5）
> 项目根目录：`/workspace`（AI-Arena MCP 服务器 v0.5，pytest 45/45 全绿）

## 通道总览

| # | 通道 | 状态 | 访问方式 |
|---|---|---|---|
| ① | 直接 `/workspace` | ✅ 完成 | 沙箱工作目录，52 个文件齐全 |
| ② | 项目资产盘 (tdrive) | ✅ 完成 | `AI-Arena-src.zip` 全量源码包（v0.5，MD5 已校验）|
| ③ | git 本地仓库 | 🟡 本地完成 / 无 remote | v0.5 = commit `be0e62e` |
| ④ | 乐享知识库 (lexiang) | ✅ 完成（v0.1 内容） | 见下方链接 |

---

## ① 直接 `/workspace`（已完成）

52 个文件全部位于 `/workspace`，源码已提交 git。用户可直接在沙箱内查看、运行。

```bash
cd /workspace && python run.py        # 启动 MCP 服务器（stdio）
pip install pytest && pytest          # 45/45 通过
```

v0.5 新增能力（无需 API key）：

```python
from mcp_server import server as srv
# 辩论模式：比较制，谁的论证更有说服力谁赢
mid = srv.start_match("ts-001", ["heuristic-conservative", "heuristic-aggressive"],
                      mode="debate", max_rounds=16)["match_id"]
srv.run_match(mid)
srv.analyze_match(mid)     # 含 debate 辩论指标（说服力/论据充分度）
```

---

## ② 项目资产盘 tdrive（✅ 已完成全量落盘）

资产盘 root `DKobgCurEQbo` → 项目目录 **`AI-Arena` = `DoXEjAujQUMq`**。

### 主交付：全量源码包

| 项 | 值 |
|---|---|
| 文件名 | `AI-Arena-src.zip` |
| file_id | `UjyMmsXWcJOY`（同名覆盖，ID 稳定不变）|
| 大小 | 262950 字节 |
| MD5 | `9107170fe1787085cce7f80a126d676c` |
| 条目 | 52 个文件 + 目录 |

**校验结果（v0.5）**：下载回本地比对 MD5 完全一致，`unzip -t` CRC 无错误，
包内 `MEMORY.md`、`README.md`、`tests/test_debate.py` 为最新版。

> 说明：zip 包内的 `DEPLOYMENT_STATUS.md` 为打包时快照（其 MD5 行写的是前一版
> `9bfa17d7...`），属文档自指的固有差一版；**以本行数值与资产盘实际下载为准**，
> 源码/题库/测试/报告等核心内容与本地逐字节一致。

包含内容（`mcp_server/` 全量源码、`data/puzzles.json` 题库、`tests/`（45 项）、
`reports/`（排行榜图 + 雷达图 + 文字报告）、`data/matches/`（10 个样例对局日志，
历史 261 个测试对局已清理——运行时数据可随时重新生成）、
`pyproject.toml`、`run.py`、`README.md`、`.env.example`、`.gitignore`、
`.workbuddy/memory/MEMORY.md`、`DEPLOYMENT_STATUS.md`）。

### 散落单文件（压缩包之外）

- `README.md` → `UXqQriRHhOhf`
- `mcp_server/engine/__init__.py` → `UrCanHujNRAL`

> ⚠️ 注意：`tests/`、`eval/`、`store/`、`reports/`、`.workbuddy/memory/`、`data/`
> 这些目录在资产盘上是**空壳**（历史遗留，只建了目录没传文件）。
> **完整内容以 `AI-Arena-src.zip` 为准**，解压即可还原全部目录结构。

### 踩坑记录

- 网盘上传是三步：`file_upload` 取预签名 → `curl -T` PUT → `file_upload_complete`。
  **只做第一步不会落库**——这正是此前"9 个文件已获凭证"最终全是空目录的原因。
- **PUT 必须带 `x-cos-security-token` 头**：漏带报 `InvalidAccessKeyId`
  （COS 临时 AK 不配安全令牌就"不存在"）。
- 网盘拒绝点开头文件名（`.gitignore` / `.env.example` 会被判为"文件名为空"），进压缩包可规避。
- **打包 zip 一次上传是海量小文件的最优解**：把 75+ 次顺序调用压缩到 3 次，目录结构完整保留。
- `dir_list` 对刚上传文件可能返回 `size=0`（元数据延迟），必须用 `file_download` + MD5 对比验证。
- `file_upload` 的 `file_size` 要用 `stat -c %s` 的**文件实际字节数**，不是 `unzip -l`
  末尾的"内容总大小"（v0.5 曾误传 393038 vs 实际 260967；PUT 文件流无损，MD5 仍通过）。

---

## ③ git 本地仓库（本地完成）

```bash
git log --oneline
# be0e62e feat: v0.5 辩论模式（比较制+平局加时赛+辩论指标），修复风格名冒号解析
# 007bfe7 feat: AI 角斗场 MCP 服务器 v0.1（海龟汤引擎 + 多模型评测脚手架）
```

- 无 remote 配置 → 暂无法 `git push`。
- 若要推送到远端，请提供仓库 URL（GitHub / 工蜂 / CNB 均可），我来加 remote 并 push。

---

## ④ 乐享知识库 lexiang（v0.1 内容，未随版本更新）

个人知识库：`用户2015的个人知识库`（`space_id=459b0f14579d4ba98001b5aef3a7d595`）

- 📁 文件夹 **AI-Arena** → `96e71c854a2d4f80964e3c179350f94c`
- 📄 知识页 **README · 项目说明** → `79c3b93a4298478f8064cef6132dd169`
  - https://mcp.lexiang-app.com/pages/79c3b93a4298478f8064cef6132dd169
- 📄 知识页 **架构设计 · Architecture** → `7271888faec348efa3e35cf94ae89f68`
  - https://mcp.lexiang-app.com/pages/7271888faec348efa3e35cf94ae89f68

---

## 下一步可选动作

1. ~~**补齐网盘**~~ ✅ 已完成：`AI-Arena-src.zip` 全量落盘，MD5 校验通过。
2. **配置 git remote + push**：提供仓库 URL 即可（当前仅本地 commit，**未推送**）。
3. **v0.6 破解 misleading**：实现"反向利用规律"的求解者策略（识别稳定反向回答并取反），
   验证后更新乐享知识页。
4. **真实多模型对战**：接入 litellm，把 `heuristic-*` 换成真实模型名（需 `.env` key）。
