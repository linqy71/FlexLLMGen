#!/bin/bash

# 修复 FlexGen 权重路径问题
echo "🔧 修复 FlexGen 权重路径..."

cd ~/opt_weights

echo "📁 当前权重目录："
ls -la

# 检查是否存在正确的目录名
if [ -d "opt-1.3b-np" ] && [ ! -d "facebook-opt-1.3b-np" ]; then
    echo "🔄 重命名权重目录..."
    mv opt-1.3b-np facebook-opt-1.3b-np
    echo "✅ 权重目录已重命名为: facebook-opt-1.3b-np"
elif [ -d "facebook-opt-1.3b-np" ]; then
    echo "✅ 权重目录已经正确: facebook-opt-1.3b-np"
else
    echo "❌ 权重目录不存在，请检查路径"
    exit 1
fi

echo "📁 修复后的权重目录："
ls -la

echo "🎉 权重路径修复完成！"
