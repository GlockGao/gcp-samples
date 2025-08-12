import os
from typing import List, Dict, Any, Optional
from google.cloud import discoveryengine_v1 as discoveryengine
from google.cloud.discoveryengine_v1.types import RankRequest, RankingRecord


class DiscoveryEngineReranker:
    """使用Google Cloud Discovery Engine进行文档重排序的类"""
    
    def __init__(self, project_id: Optional[str] = None, location: Optional[str] = None):
        """
        初始化Discovery Engine Reranker
        
        Args:
            project_id: Google Cloud项目ID，如果不提供则从环境变量获取
            location: 地理位置，如果不提供则从环境变量获取
        """
        self.project_id = project_id or os.getenv('GOOGLE_CLOUD_PROJECT')
        self.location = location or os.getenv('GOOGLE_CLOUD_LOCATION', 'global')
        
        if not self.project_id:
            raise ValueError("项目ID未设置。请设置GOOGLE_CLOUD_PROJECT环境变量或在初始化时提供project_id参数。")
        
        print(f"使用项目ID: {self.project_id}")
        print(f"使用地理位置: {self.location}")
        
        # 初始化Discovery Engine客户端
        self.client = discoveryengine.RankServiceClient()
        
        # 构建ranking config路径
        self.ranking_config = f"projects/{self.project_id}/locations/{self.location}/rankingConfigs/default_ranking_config"
        self.ranking_config = self.client.ranking_config_path(
            project=self.project_id,
            location=self.location,
            ranking_config="default_ranking_config"
        )
    
    def rerank_documents(
        self, 
        query: str, 
        documents: List[Dict[str, Any]], 
        top_n: Optional[int] = None,
        model: str = "semantic-ranker-default-004"
    ) -> List[Dict[str, Any]]:
        """
        使用Discovery Engine对文档进行重排序
        
        Args:
            query: 查询字符串
            documents: 文档列表，每个文档应包含'id', 'title', 'content'等字段
            top_n: 返回前N个结果，如果不指定则返回所有结果
            model: 使用的排序模型
            
        Returns:
            重排序后的文档列表，按相关性从高到低排序
        """
        try:
            # 准备ranking records
            records = []
            for i, doc in enumerate(documents):
                record = RankingRecord(
                    id=doc.get('id', str(i)),
                    title=doc.get('title', ''),
                    content=doc.get('content', '')
                )
                records.append(record)
            
            # 创建rerank请求
            request = RankRequest(
                ranking_config=self.ranking_config,
                model=model,
                query=query,
                records=records,
                top_n=top_n or len(documents)
            )
            
            print(f"正在对 {len(documents)} 个文档进行重排序...")
            print(f"查询: {query}")
            
            # 执行rerank
            response = self.client.rank(request=request)
            
            # 处理响应
            reranked_docs = []
            for record in response.records:
                # 找到原始文档
                original_doc = None
                for doc in documents:
                    if doc.get('id', str(documents.index(doc))) == record.id:
                        original_doc = doc.copy()
                        break
                
                if original_doc:
                    # 添加排序分数
                    original_doc['rank_score'] = record.score
                    original_doc['rank_position'] = len(reranked_docs) + 1
                    reranked_docs.append(original_doc)
            
            print(f"重排序完成，返回 {len(reranked_docs)} 个结果")
            return reranked_docs
            
        except Exception as e:
            print(f"重排序过程中发生错误: {str(e)}")
            raise


def create_sample_documents() -> List[Dict[str, Any]]:
    """创建示例文档用于测试"""
    return [
        {
            "id": "doc1",
            "title": "Python编程基础",
            "content": "Python是一种高级编程语言，具有简洁的语法和强大的功能。它广泛用于Web开发、数据科学、人工智能等领域。"
        },
        {
            "id": "doc2", 
            "title": "机器学习入门",
            "content": "机器学习是人工智能的一个分支，通过算法让计算机从数据中学习模式。常见的机器学习算法包括线性回归、决策树、神经网络等。"
        },
        {
            "id": "doc3",
            "title": "Web开发技术",
            "content": "Web开发涉及前端和后端技术。前端技术包括HTML、CSS、JavaScript，后端技术包括Python、Java、Node.js等。"
        },
        {
            "id": "doc4",
            "title": "数据库设计原理",
            "content": "数据库是存储和管理数据的系统。关系型数据库使用SQL语言，NoSQL数据库适合处理大规模非结构化数据。"
        },
        {
            "id": "doc5",
            "title": "云计算服务",
            "content": "云计算提供按需的计算资源和服务。主要的云服务提供商包括AWS、Google Cloud、Azure等，提供IaaS、PaaS、SaaS等服务模式。"
        }
    ]


def main():
    """主函数，演示如何使用Discovery Engine进行文档重排序"""
    try:
        # 检查环境变量
        project = os.getenv('GOOGLE_CLOUD_PROJECT')
        location = os.getenv('GOOGLE_CLOUD_LOCATION', 'global')
        
        if not project:
            print("错误: 环境变量 'GOOGLE_CLOUD_PROJECT' 未设置。")
            print("请运行: export GOOGLE_CLOUD_PROJECT=your-project-id")
            return
        
        print(f"获取到的 GOOGLE_CLOUD_PROJECT: {project}")
        print(f"获取到的 GOOGLE_CLOUD_LOCATION: {location}")
        
        # 初始化reranker
        reranker = DiscoveryEngineReranker()
        
        # 创建示例文档
        documents = create_sample_documents()
        
        # 定义查询
        query = "Python机器学习"
        
        print(f"\n原始文档顺序:")
        for i, doc in enumerate(documents, 1):
            print(f"{i}. {doc['title']}")
        
        # 执行重排序
        print(f"\n执行重排序...")
        reranked_docs = reranker.rerank_documents(
            query=query,
            documents=documents,
            top_n=3  # 只返回前3个最相关的结果
        )
        
        # 显示结果
        print(f"\n重排序后的结果 (查询: '{query}'):")
        for doc in reranked_docs:
            print(f"{doc['rank_position']}. {doc['title']}")
            print(f"   相关性分数: {doc['rank_score']:.4f}")
            print(f"   内容: {doc['content'][:100]}...")
            print()
        
    except Exception as e:
        print(f"程序执行出错: {str(e)}")


if __name__ == "__main__":
    main()
