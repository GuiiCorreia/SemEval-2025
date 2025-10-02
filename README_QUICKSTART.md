# Guia Rápido - MTRAG Benchmark

## 1. Setup Inicial

```bash
# Criar ambiente
conda create -n mtrag python=3.10
conda activate mtrag

# Instalar dependências de avaliação
pip install -c scripts/evaluation/constraints.txt -r scripts/evaluation/requirements.txt

# Instalar rank_bm25 para o exemplo de retriever
pip install rank-bm25
```

## 2. Teste de Avaliação com Dados de Exemplo

O repositório já vem com resultados de retrieval que podem ser avaliados:

```bash
# Avaliar retrieval no arquivo RAG.jsonl
python scripts/evaluation/run_retrieval_eval.py \
  --input_file human/generation_tasks/RAG.jsonl \
  --output_file test_output.jsonl
```

**Saída esperada:**
- `test_output.jsonl`: Arquivo original enriquecido com scores por query
- `test_output_aggregate.csv`: Métricas agregadas por coleção

**Métricas reportadas:**
- **Recall@1, @3, @5**: Proporção de documentos relevantes recuperados
- **nDCG@1, @3, @5**: Normalized Discounted Cumulative Gain

## 3. Rodando Seu Próprio Retriever

### Passo 1: Entender o formato dos dados

**Corpus** (`corpora/passage_level/*.jsonl`):
```json
{
  "_id": "822086267_6698-7277-0-579",
  "text": "Texto do documento...",
  "title": "Título do documento"
}
```

**Queries** (`human/retrieval_tasks/*/clapnq_rewrite.jsonl`):
```json
{
  "_id": "unique_task_id",
  "text": "Texto da query",
  "metadata": {...}
}
```

### Passo 2: Usar o exemplo fornecido

O arquivo `example_retriever.py` demonstra como:
- Carregar um corpus
- Indexar com BM25
- Recuperar documentos para queries
- Gerar output no formato correto

**Rodar exemplo com ClapNQ:**

```bash
python example_retriever.py \
  --corpus corpora/passage_level/clapnq.jsonl \
  --queries human/retrieval_tasks/clapnq/clapnq_rewrite.jsonl \
  --output my_clapnq_results.jsonl \
  --collection mt-rag-clapnq-elser-512-100-20240503 \
  --top_k 5
```

### Passo 3: Avaliar seus resultados

```bash
python scripts/evaluation/run_retrieval_eval.py \
  --input_file my_clapnq_results.jsonl \
  --output_file my_clapnq_evaluated.jsonl
```

## 4. Rodando em Todas as Coleções

Para participar do benchmark, você precisa rodar em todas as 4 coleções:

### ClapNQ (Wikipedia)
```bash
python example_retriever.py \
  --corpus corpora/passage_level/clapnq.jsonl \
  --queries human/retrieval_tasks/clapnq/clapnq_rewrite.jsonl \
  --output results_clapnq.jsonl \
  --collection mt-rag-clapnq-elser-512-100-20240503
```

### Cloud (Documentação Técnica)
```bash
python example_retriever.py \
  --corpus corpora/passage_level/cloud.jsonl \
  --queries human/retrieval_tasks/cloud/cloud_rewrite.jsonl \
  --output results_cloud.jsonl \
  --collection mt-rag-ibmcloud-elser-512-100-20240502
```

### FiQA (Finanças)
```bash
python example_retriever.py \
  --corpus corpora/passage_level/fiqa.jsonl \
  --queries human/retrieval_tasks/fiqa/fiqa_rewrite.jsonl \
  --output results_fiqa.jsonl \
  --collection mt-rag-fiqa-beir-elser-512-100-20240501
```

### Govt (Governo)
```bash
python example_retriever.py \
  --corpus corpora/passage_level/govt.jsonl \
  --queries human/retrieval_tasks/govt/govt_rewrite.jsonl \
  --output results_govt.jsonl \
  --collection mt-rag-govt-elser-512-100-20240611
```

### Combinar e Avaliar Todos
```bash
# Combinar todos os resultados em um único arquivo
cat results_clapnq.jsonl results_cloud.jsonl results_fiqa.jsonl results_govt.jsonl > all_results.jsonl

# Avaliar
python scripts/evaluation/run_retrieval_eval.py \
  --input_file all_results.jsonl \
  --output_file final_evaluation.jsonl
```

## 5. Implementando Seu Próprio Retriever

Você pode substituir o BM25 no `example_retriever.py` por:

### Opção 1: Modelo de embeddings (ex: sentence-transformers)
```python
from sentence_transformers import SentenceTransformer
import numpy as np

class EmbeddingRetriever:
    def __init__(self, corpus_file, model_name='BAAI/bge-base-en-v1.5'):
        self.model = SentenceTransformer(model_name)
        # Carregar corpus...
        self.embeddings = self.model.encode([doc['text'] for doc in self.corpus])

    def retrieve(self, query, top_k=5):
        query_emb = self.model.encode([query])
        scores = np.dot(self.embeddings, query_emb.T).flatten()
        # Retornar top_k...
```

### Opção 2: API (Elasticsearch, OpenSearch, etc)
```python
class ElasticsearchRetriever:
    def __init__(self, es_client, index_name):
        self.es = es_client
        self.index = index_name

    def retrieve(self, query, top_k=5):
        results = self.es.search(
            index=self.index,
            body={"query": {"match": {"text": query}}},
            size=top_k
        )
        # Formatar resultados...
```

### Opção 3: Modelos mais avançados
- ColBERT
- ELSER (usado no paper)
- Modelos proprietários (OpenAI, Cohere, etc)

## 6. Formato de Saída Esperado

Seu retriever deve gerar um JSONL onde cada linha tem:

```json
{
  "task_id": "unique_task_id",
  "Collection": "mt-rag-clapnq-elser-512-100-20240503",
  "contexts": [
    {
      "document_id": "doc_id_1",
      "text": "Texto do documento",
      "title": "Título (opcional)",
      "score": 0.95
    }
  ],
  "input": [{"speaker": "user", "text": "query text"}],
  "targets": [{"speaker": "agent", "text": ""}]
}
```

## 7. Comparando com Baselines

Do paper, os resultados baseline são:

| Retriever | Setup | R@1 | R@3 | R@5 | nDCG@1 | nDCG@3 | nDCG@5 |
|-----------|-------|-----|-----|-----|--------|--------|--------|
| BM25 | Last Turn | 0.08 | 0.15 | 0.20 | 0.17 | 0.16 | 0.18 |
| BM25 | Query Rewrite | 0.09 | 0.18 | 0.25 | 0.20 | 0.19 | 0.22 |
| BGE-base | Last Turn | 0.13 | 0.24 | 0.30 | 0.26 | 0.25 | 0.27 |
| BGE-base | Query Rewrite | 0.17 | 0.30 | 0.37 | 0.34 | 0.31 | 0.34 |
| **Elser** | **Query Rewrite** | **0.20** | **0.43** | **0.52** | **0.46** | **0.45** | **0.48** |

## 8. Próximos Passos

1. ✅ Rodar avaliação com dados de exemplo
2. ✅ Entender formato dos dados
3. ⬜ Implementar seu retriever
4. ⬜ Rodar em todas as 4 coleções
5. ⬜ Avaliar resultados
6. ⬜ Iterar e melhorar!

## Dúvidas?

- **Paper**: https://arxiv.org/abs/2501.03468
- **Issues**: https://github.com/IBM/mt-rag-benchmark/issues
- **Contato**: sjrosenthal@us.ibm.com