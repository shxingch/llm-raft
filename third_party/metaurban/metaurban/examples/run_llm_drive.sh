#!/bin/bash

# MetaUrban LLM驾驶控制启动脚本
# 使用方法: ./run_llm_drive.sh [API_KEY]

echo "========================================"
echo "MetaUrban LLM驾驶控制系统启动脚本"
echo "========================================"

# 检查OpenAI API密钥
if [ -z "$OPENAI_API_KEY" ] && [ -z "$1" ]; then
    echo "错误: 未设置OpenAI API密钥"
    echo ""
    echo "请使用以下方式之一设置API密钥:"
    echo "1. 环境变量: export OPENAI_API_KEY='your-key-here'"
    echo "2. 命令行参数: ./run_llm_drive.sh 'your-key-here'"
    echo ""
    exit 1
fi

# 设置API密钥（如果通过参数提供）
if [ ! -z "$1" ]; then
    export OPENAI_API_KEY="$1"
    echo "使用命令行提供的API密钥"
else
    echo "使用环境变量中的API密钥"
fi

# 检查Python依赖
echo "检查Python依赖..."
python3 -c "import openai, numpy, metaurban" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "错误: 缺少必要的Python依赖"
    echo "请运行: pip install openai numpy metaurban"
    exit 1
fi

echo "依赖检查通过!"
echo ""

# 显示启动选项
echo "启动选项:"
echo "1. 基础模式 (GPT-4, 激光雷达观察)"
echo "2. 完整模式 (GPT-4, 多传感器观察)"
echo "3. 快速模式 (GPT-3.5-turbo, 激光雷达观察)"
echo "4. 自定义模式"
echo ""

read -p "请选择模式 (1-4): " choice

case $choice in
    1)
        echo "启动基础模式..."
        python3 drive_in_dynamic_env_llm.py
        ;;
    2)
        echo "启动完整模式..."
        python3 drive_in_dynamic_env_llm.py --observation all
        ;;
    3)
        echo "启动快速模式..."
        python3 drive_in_dynamic_env_llm.py --model gpt-3.5-turbo
        ;;
    4)
        echo "自定义模式 - 请输入参数:"
        read -p "观察模式 (lidar/all) [lidar]: " obs_mode
        read -p "物体密度 (0.0-1.0) [0.3]: " obj_density
        read -p "行人密度倍数 [1.0]: " ped_density
        read -p "LLM模型 [gpt-4]: " model
        read -p "最大步数 [1000]: " max_steps
        
        # 设置默认值
        obs_mode=${obs_mode:-lidar}
        obj_density=${obj_density:-0.3}
        ped_density=${ped_density:-1.0}
        model=${model:-gpt-4}
        max_steps=${max_steps:-1000}
        
        echo "启动自定义模式..."
        python3 drive_in_dynamic_env_llm.py \
            --observation "$obs_mode" \
            --density_obj "$obj_density" \
            --density_ped "$ped_density" \
            --model "$model" \
            --max_steps "$max_steps"
        ;;
    *)
        echo "无效选择，使用默认基础模式..."
        python3 drive_in_dynamic_env_llm.py
        ;;
esac

echo ""
echo "LLM驾驶控制系统已退出" 