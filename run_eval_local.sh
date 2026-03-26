#!/bin/bash
# SimLingo 本地单机评估脚本（无 SLURM，无 debugpy）
# 用法：bash run_eval_local.sh [路由xml文件]
# 例：  bash run_eval_local.sh bench2drive_00.xml
# 不传参数则默认跑 bench2drive_00.xml

set -e  # 遇到错误立即退出

# ─────────────────────────────────────────────
# 路径配置（根据你的机器已自动填好）
# ─────────────────────────────────────────────
WORK_DIR="/home/robot/Document/python_code/VLA/simlingo"
CARLA_ROOT="/home/robot/software/carla0915"
CHECKPOINT="/home/robot/Document/python_code/VLA/simlingo/checkpoints/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt"
ROUTE_DIR="${WORK_DIR}/leaderboard/data/bench2drive_split"
AGENT="${WORK_DIR}/team_code/agent_simlingo.py"
AGENT_CONFIG="${WORK_DIR}/team_code/config_simlingo.py"
OUT_DIR="${WORK_DIR}/eval_results/local"

# 端口（本地跑单路线，固定端口即可）
PORT=2000
TM_PORT=2500
GPU_RANK=0

# 要跑哪条路由（默认第一条，也可以传参指定）
ROUTE_XML="${1:-bench2drive_00.xml}"
ROUTE="${ROUTE_DIR}/${ROUTE_XML}"
ROUTE_NAME="${ROUTE_XML%.xml}"

# ─────────────────────────────────────────────
# 环境变量
# ─────────────────────────────────────────────
export CARLA_ROOT
export CARLA_SERVER="${CARLA_ROOT}/CarlaUE4.sh"
export PYTHONPATH="${WORK_DIR}/team_code:${WORK_DIR}:${CARLA_ROOT}/PythonAPI/carla:${WORK_DIR}/Bench2Drive/leaderboard:${WORK_DIR}/Bench2Drive/scenario_runner:${PYTHONPATH}"
# export PYTHONPATH="${PYTHONPATH}:${CARLA_ROOT}/PythonAPI/carla"
# export PYTHONPATH="${PYTHONPATH}:${WORK_DIR}/Bench2Drive/leaderboard"
# export PYTHONPATH="${PYTHONPATH}:${WORK_DIR}/Bench2Drive/scenario_runner"
export SCENARIO_RUNNER_ROOT="${WORK_DIR}/Bench2Drive/scenario_runner"
export LEADERBOARD_ROOT="${WORK_DIR}/Bench2Drive/leaderboard"
export CHALLENGE_TRACK_CODENAME=SENSORS
export DEBUG_CHALLENGE=0
export REPETITIONS=1
export RESUME=False
export IS_BENCH2DRIVE=True

# 结果保存路径
mkdir -p "${OUT_DIR}"
CHECKPOINT_ENDPOINT="${OUT_DIR}/${ROUTE_NAME}_result.json"
SAVE_PATH="${OUT_DIR}/${ROUTE_NAME}_frames"

# ─────────────────────────────────────────────
# 第一步：启动 CARLA（后台运行）
# ─────────────────────────────────────────────
echo "========================================"
echo " 启动 CARLA (离屏渲染，端口 ${PORT})"
echo "========================================"

# 杀掉可能残留的 CARLA 进程
pkill -f CarlaUE4 2>/dev/null || true
sleep 2

# # 后台启动 CARLA，-RenderOffScreen 节省显存
# ${CARLA_ROOT}/CarlaUE4.sh \
#     #-RenderOffScreen \
#     -nosound \
#     -carla-port=${PORT} \
#     -carla-streaming-port=0 \
#     &
# CARLA_PID=$!
# echo "CARLA PID: ${CARLA_PID}"

# # 等待 CARLA 初始化（通常需要 15-20 秒）
# echo "等待 CARLA 初始化 (20秒)..."
# sleep 20

# ─────────────────────────────────────────────
# 第二步：运行评估
# ─────────────────────────────────────────────
echo "========================================"
echo " 开始评估路由: ${ROUTE_XML}"
echo " 结果保存到:   ${CHECKPOINT_ENDPOINT}"
echo "========================================"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=${GPU_RANK} python \
    "${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py" \
    --routes="${ROUTE}" \
    --repetitions=${REPETITIONS} \
    --track=${CHALLENGE_TRACK_CODENAME} \
    --checkpoint="${CHECKPOINT_ENDPOINT}" \
    --agent="${AGENT}" \
    --agent-config="${AGENT_CONFIG}" \
    --debug=${DEBUG_CHALLENGE} \
    --resume=${RESUME} \
    --port=${PORT} \
    --traffic-manager-port=${TM_PORT}

# ─────────────────────────────────────────────
# 第三步：关闭 CARLA
# ─────────────────────────────────────────────
echo "评估完成，关闭 CARLA..."
kill ${CARLA_PID} 2>/dev/null || pkill -f CarlaUE4 2>/dev/null || true

# ─────────────────────────────────────────────
# 第四步：显示结果摘要
# ─────────────────────────────────────────────
echo ""
echo "========================================"
echo " 结果摘要"
echo "========================================"
if [ -f "${CHECKPOINT_ENDPOINT}" ]; then
    python3 -c "
import json
with open('${CHECKPOINT_ENDPOINT}') as f:
    data = json.load(f)
records = data.get('_checkpoint', data).get('records', [])
if records:
    scores = [r.get('scores', {}) for r in records]
    ds  = [s.get('score_composed', 0) for s in scores]
    rc  = [s.get('score_route', 0) for s in scores]
    inf = [s.get('score_penalty', 1) for s in scores]
    print(f'  路线数:          {len(records)}')
    print(f'  Driving Score:   {sum(ds)/len(ds):.2f}')
    print(f'  Route Completion:{sum(rc)/len(rc):.2f}')
    print(f'  Infraction Score:{sum(inf)/len(inf):.4f}')
else:
    print('  (结果文件存在但暂无记录，可能评估未完成)')
    print(f'  原始内容: {list(data.keys())}')
"
else
    echo "  结果文件未生成，评估可能中途出错。"
fi

echo ""
echo "完整结果: ${CHECKPOINT_ENDPOINT}"