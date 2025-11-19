#!/usr/bin/env bash
set -euo pipefail

# 以脚本所在目录为基准的相对路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 本脚本位于项目目录下，因此直接使用脚本目录作为项目目录
PROJECT_DIR="${SCRIPT_DIR}"

# 激活 conda 环境 graphpae（尽量兼容多种安装方式）
# 注意：conda 的激活脚本可能引用未设置的变量，在 set -u 下会报错；
# 因此在激活前临时关闭 -u，激活后再恢复。
set +u
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1090
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1090
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
  echo "[ERROR] conda 未找到，请先安装或手动 source conda.sh" >&2
  exit 1
fi
conda activate graphpae
set -u

# 等待指定进程完成
wait_for_process() {
  local pid="$1"
  if [ -z "${pid}" ]; then
    echo "[ERROR] 进程ID不能为空" >&2
    return 1
  fi
  if ! ps -p "${pid}" > /dev/null 2>&1; then
    echo "[INFO] 进程 ${pid} 不存在，可能已经完成"
    return 0
  fi
  echo "[INFO] 等待进程 ${pid} 完成..."
  echo "[INFO] 进程信息: $(ps -p ${pid} -o pid,cmd --no-headers 2>/dev/null || echo '进程不存在')"
  while ps -p "${pid}" > /dev/null 2>&1; do
    sleep 10
    echo "[INFO] $(date '+%F %T') 进程 ${pid} 仍在运行，继续等待..."
  done
  echo "[INFO] 进程 ${pid} 已完成"
  return 0
}

# 等待进程374482完成后再启动
if ps -p 473272 > /dev/null 2>&1; then
  echo "=============================="
  echo "检测到进程 473272 正在运行，等待其完成后开始执行"
  echo "=============================="
  wait_for_process 473272
  echo ""
else
  echo "[INFO] 进程 473272 不存在，直接开始执行"
  echo ""
fi

run_one() {
  local dataset="$1"
  local exit_code=0
  echo "=============================="
  echo "[START] $(date '+%F %T') dataset=${dataset}"
  echo "=============================="
  cd "${PROJECT_DIR}" || {
    echo "[ERROR] 无法切换到项目目录: ${PROJECT_DIR}" >&2
    return 1
  }
  # 使用 set +e 临时禁用错误退出，确保即使 Python 脚本返回非零退出码也继续执行
  set +e
  python train_node.py --dataset "${dataset}"
  exit_code=$?
  set -e
  if [ ${exit_code} -eq 0 ]; then
    echo "[DONE ] $(date '+%F %T') dataset=${dataset}"
  else
    echo "[ERROR] $(date '+%F %T') dataset=${dataset} 执行失败，退出码: ${exit_code}" >&2
    echo "[WARN] 继续执行下一个数据集..." >&2
  fi
  return 0  # 总是返回 0，确保脚本继续执行
}

# 依次运行（Cornell -> BlogCatalog -> Chameleon -> Squirrel -> Cora）
run_one cornell
run_one wisconsin
run_one texas
run_one actor
run_one deezereurope
run_one blog
run_one flickr
run_one chameleon
run_one squirrel

run_one cora
run_one citeseer
run_one pubmed
run_one minesweeper


echo "All datasets finished."