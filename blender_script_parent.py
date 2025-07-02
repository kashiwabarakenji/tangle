import bpy
import sys
import os

# --- ここを書き換えてください ---
script_path = "/Users/kashiwa/tangle/blender_script.py"
# 例: script_path = "/Users/ユーザー名/blender_script.py"

# --- sys.pathへ追加（必要なら） ---
script_dir = os.path.dirname(os.path.abspath(script_path))
if script_dir not in sys.path:
    sys.path.append(script_dir)

# --- ファイル存在チェック ---
if not os.path.exists(script_path):
    raise FileNotFoundError(f"指定されたスクリプトが見つかりません: {script_path}")

# --- ファイルをexecで実行 ---
with open(script_path, 'r', encoding='utf-8') as f:
    code = f.read()
    exec(code, globals())

print(f"スクリプト {script_path} を実行しました。")