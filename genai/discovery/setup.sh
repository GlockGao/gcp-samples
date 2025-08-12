#!/bin/bash

echo "安装 Google Cloud Discovery Engine 依赖..."

# 安装 Google Cloud Discovery Engine 客户端库
pip install -U 'google-cloud-discoveryengine'

# 安装其他可能需要的依赖
pip install -U 'google-cloud-core'

echo "依赖安装完成！"
echo ""
echo "接下来请设置环境变量："
echo "export GOOGLE_CLOUD_PROJECT='project-easongy-poc'"
echo "export GOOGLE_CLOUD_LOCATION='global'"
echo ""
echo "然后运行："
echo "python rerank.py"
