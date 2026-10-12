import os
import sys

# 让 tests/ 下的用例能直接 import engine / players（与 run_*.py 脚本一致）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
