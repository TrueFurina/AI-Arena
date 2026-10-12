# 5 分钟快速上手（QUICKSTART）

> 从零到出第一份评测报告。无需任何 API key 也能跑通整条链路。

---

## 1. 环境

系统 Python 受 PEP 668 保护，用隔离 venv：
```bash
python3.11 -m venv /root/venv_aiarena
source /root/venv_aiarena/bin/activate
pip install litellm pandas matplotlib python-dotenv pydantic
# 或：pip install -r requirements.txt
```

---

## 2. 零 key 跑通（最重要的一步）

```bash
# 单局（规则化假玩家，无需 key）
python run/single_game.py --mode mock --puzzle turtle_001

# 多模型对战脚手架（mock 基线，零 key）
python run/tournament.py --mock
# 预期：mock-strong 还原度≈1.0，mock-weak≈0.0

# 看评测报告
ls analysis/        # metrics.csv / report.txt / restoration.png
```

---

## 3. 预览真实模型提示词（不发请求）

```bash
python run/single_game.py --mode llm --model deepseek --gm smart --dry-run
```

---

## 4. 接入真实模型（需要 key）

1. `cp .env.example .env`，编辑填入 key（见 `doc/KEYS.md` 对照表）：
   ```
   DEEPSEEK_API_KEY=sk-xxxx
   ```
2. 先验证连通性：
   ```bash
   python run/test_api.py --model deepseek
   ```
3. 跑真实单局：
   ```bash
   python run/single_game.py --mode llm --model deepseek --gm smart --puzzle turtle_001
   ```
4. （可选）批量对战：把 `config.toml` 里对应 `[[models]]` 的 `enabled` 改为 `true`，再跑：
   ```bash
   python run/tournament.py
   ```

---

## 5. 出评测报告

```bash
python run/evaluate.py
# 读 logs/ -> analysis/metrics.csv + report.txt + restoration.png
```

---

## 6. 测试

```bash
python -m pytest tests/ -q     # 零 key 全绿
```

---

## 下一步

- 方法论与扩展：见 `doc/METHODOLOGY.md` / `doc/EXTENDING.md`
- 指标含义：见 `doc/EVALUATION.md`
- 长期路线：见 `doc/ROADMAP.md`
- 常用操作已沉淀为 skills（`arena-run` / `arena-evaluate` / `arena-puzzle` / `arena-metrics` / `arena-tournament`），可用 Skill 工具一键调用。
