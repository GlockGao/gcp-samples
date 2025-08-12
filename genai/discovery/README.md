# Discovery Engine Rerank 使用指南

这个项目演示了如何使用Google Cloud Discovery Engine进行文档重排序（rerank）。

## 功能特性

- 使用Discovery Engine的语义排序功能对文档进行重排序
- 支持自定义查询和文档集合
- 提供相关性分数和排序位置
- 包含完整的错误处理和日志记录
- 支持灵活的配置选项

## 环境设置

### 1. 安装依赖

```bash
# 运行安装脚本
bash setup.sh

# 或者手动安装
pip install -U 'google-cloud-discoveryengine'
```

### 2. 设置环境变量

```bash
# 设置Google Cloud项目ID
export GOOGLE_CLOUD_PROJECT="your-project-id"

# 设置地理位置（可选，默认为global）
export GOOGLE_CLOUD_LOCATION="global"
```

### 3. 认证设置

确保您已经设置了Google Cloud认证：

```bash
# 使用gcloud认证
gcloud auth application-default login

# 或者设置服务账号密钥
export GOOGLE_APPLICATION_CREDENTIALS="path/to/your/service-account-key.json"
```

## 使用方法

### 基本使用

```python
from rerank import DiscoveryEngineReranker

# 初始化reranker
reranker = DiscoveryEngineReranker()

# 准备文档数据
documents = [
    {
        "id": "doc1",
        "title": "文档标题1",
        "content": "文档内容1..."
    },
    {
        "id": "doc2", 
        "title": "文档标题2",
        "content": "文档内容2..."
    }
]

# 执行重排序
query = "您的查询"
reranked_docs = reranker.rerank_documents(
    query=query,
    documents=documents,
    top_n=5  # 返回前5个最相关的结果
)

# 查看结果
for doc in reranked_docs:
    print(f"标题: {doc['title']}")
    print(f"相关性分数: {doc['rank_score']}")
    print(f"排序位置: {doc['rank_position']}")
```

### 运行示例

```bash
# 直接运行示例代码
python rerank.py
```

## API 参考

### DiscoveryEngineReranker 类

#### 初始化参数

- `project_id` (可选): Google Cloud项目ID
- `location` (可选): 地理位置，默认为'global'

#### rerank_documents 方法

**参数:**
- `query` (str): 查询字符串
- `documents` (List[Dict]): 文档列表
- `top_n` (可选): 返回结果数量
- `model` (str): 排序模型，默认为"semantic-ranker-512@latest"

**返回:**
- 重排序后的文档列表，包含原始字段和新增的排序信息

### 文档格式

每个文档应包含以下字段：

```python
{
    "id": "唯一标识符",
    "title": "文档标题", 
    "content": "文档内容"
}
```

重排序后会添加：

```python
{
    "rank_score": 0.8542,  # 相关性分数
    "rank_position": 1     # 排序位置
}
```

## 高级用法

### 自定义排序模型

```python
# 使用不同的排序模型
reranked_docs = reranker.rerank_documents(
    query=query,
    documents=documents,
    model="semantic-ranker-512@latest"  # 或其他可用模型
)
```

### 批量处理

```python
# 处理大量文档
def batch_rerank(reranker, query, all_documents, batch_size=100):
    results = []
    for i in range(0, len(all_documents), batch_size):
        batch = all_documents[i:i+batch_size]
        batch_results = reranker.rerank_documents(query, batch)
        results.extend(batch_results)
    return results
```

### 错误处理

```python
try:
    reranked_docs = reranker.rerank_documents(query, documents)
except Exception as e:
    print(f"重排序失败: {e}")
    # 处理错误或使用原始顺序
    reranked_docs = documents
```

## 注意事项

1. **配额限制**: Discovery Engine有API调用配额限制，请注意使用频率
2. **文档大小**: 单个文档内容不应过长，建议控制在合理范围内
3. **批次大小**: 一次请求的文档数量有限制，大量文档需要分批处理
4. **地理位置**: 确保选择正确的地理位置以获得最佳性能

## 故障排除

### 常见错误

1. **认证错误**
   ```
   解决方案: 检查GOOGLE_APPLICATION_CREDENTIALS或运行gcloud auth
   ```

2. **项目ID未设置**
   ```
   解决方案: 设置GOOGLE_CLOUD_PROJECT环境变量
   ```

3. **API未启用**
   ```
   解决方案: 在Google Cloud Console中启用Discovery Engine API
   ```

4. **权限不足**
   ```
   解决方案: 确保服务账号有Discovery Engine的使用权限
   ```

## 性能优化

1. **缓存结果**: 对于相同的查询和文档集合，可以缓存结果
2. **并行处理**: 对于独立的查询，可以使用多线程处理
3. **文档预处理**: 提前清理和格式化文档内容

## 扩展功能

这个基础实现可以扩展为：

- 支持多种文档格式（PDF、Word等）
- 集成到搜索系统中
- 添加结果缓存机制
- 实现批量处理优化
- 添加性能监控和日志记录

## 相关资源

- [Google Cloud Discovery Engine 文档](https://cloud.google.com/discovery-engine)
- [Discovery Engine API 参考](https://cloud.google.com/discovery-engine/docs/reference)
- [Python 客户端库文档](https://cloud.google.com/python/docs/reference/discoveryengine/latest)
