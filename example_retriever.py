"""
Exemplo de como implementar e rodar um retriever para o benchmark MTRAG

Este script demonstra:
1. Como carregar os corpora
2. Como carregar as queries
3. Como implementar um retriever simples (BM25)
4. Como gerar o arquivo de saída no formato esperado para avaliação
"""

import json
from typing import List, Dict
from pathlib import Path
from rank_bm25 import BM25Okapi
import argparse


class SimpleRetriever:
    """Retriever simples usando BM25"""

    def __init__(self, corpus_file: str):
        """
        Carrega o corpus e inicializa o BM25

        Args:
            corpus_file: Caminho para o arquivo .jsonl do corpus
        """
        print(f"Carregando corpus de {corpus_file}...")
        self.corpus = []
        self.corpus_ids = []

        with open(corpus_file, 'r', encoding='utf-8') as f:
            for line in f:
                doc = json.loads(line)
                self.corpus.append(doc)
                self.corpus_ids.append(doc['_id'])

        print(f"Corpus carregado: {len(self.corpus)} documentos")

        # Tokenizar o corpus para BM25
        print("Indexando corpus com BM25...")
        tokenized_corpus = [doc['text'].lower().split() for doc in self.corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print("Indexação completa!")

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Recupera top_k documentos mais relevantes para a query

        Args:
            query: Texto da query
            top_k: Número de documentos a retornar

        Returns:
            Lista de dicionários com document_id, text, title, score
        """
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        # Pegar top_k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for idx in top_indices:
            doc = self.corpus[idx]
            results.append({
                'document_id': doc['_id'],
                'text': doc['text'],
                'title': doc.get('title', ''),
                'score': float(scores[idx])
            })

        return results


def load_queries(queries_file: str) -> List[Dict]:
    """
    Carrega queries do arquivo JSONL

    Args:
        queries_file: Caminho para arquivo de queries (ex: clapnq_rewrite.jsonl)

    Returns:
        Lista de dicionários com task_id, query, collection, etc
    """
    queries = []
    with open(queries_file, 'r', encoding='utf-8') as f:
        for line in f:
            queries.append(json.loads(line))
    return queries


def run_retrieval_experiment(
    corpus_file: str,
    queries_file: str,
    output_file: str,
    collection_name: str,
    top_k: int = 5
):
    """
    Roda experimento completo de retrieval

    Args:
        corpus_file: Caminho para corpus .jsonl
        queries_file: Caminho para queries .jsonl
        output_file: Caminho para salvar resultados
        collection_name: Nome da coleção (ex: mt-rag-clapnq-elser-512-100-20240503)
        top_k: Número de documentos a recuperar
    """
    # 1. Inicializar retriever
    retriever = SimpleRetriever(corpus_file)

    # 2. Carregar queries
    queries = load_queries(queries_file)
    print(f"\nProcessando {len(queries)} queries...")

    # 3. Rodar retrieval para cada query
    results = []
    for i, query_item in enumerate(queries):
        if i % 50 == 0:
            print(f"Processado {i}/{len(queries)} queries...")

        query_text = query_item['text']
        task_id = query_item['_id']

        # Recuperar documentos
        retrieved_docs = retriever.retrieve(query_text, top_k=top_k)

        # Montar resultado no formato esperado
        result = {
            'task_id': task_id,
            'Collection': collection_name,
            'contexts': retrieved_docs,
            'input': [{'speaker': 'user', 'text': query_text}],
            'targets': [{'speaker': 'agent', 'text': ''}]  # Vazio para retrieval
        }
        results.append(result)

    # 4. Salvar resultados
    print(f"\nSalvando resultados em {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')

    print(f"✓ Resultados salvos! Total: {len(results)} queries processadas")


def main():
    parser = argparse.ArgumentParser(description='Rodar retriever no benchmark MTRAG')
    parser.add_argument('--corpus', type=str, required=True, help='Caminho para corpus .jsonl')
    parser.add_argument('--queries', type=str, required=True, help='Caminho para queries .jsonl')
    parser.add_argument('--output', type=str, required=True, help='Caminho para salvar resultados')
    parser.add_argument('--collection', type=str, required=True,
                       help='Nome da coleção (ex: mt-rag-clapnq-elser-512-100-20240503)')
    parser.add_argument('--top_k', type=int, default=5, help='Número de documentos a recuperar')

    args = parser.parse_args()

    run_retrieval_experiment(
        corpus_file=args.corpus,
        queries_file=args.queries,
        output_file=args.output,
        collection_name=args.collection,
        top_k=args.top_k
    )


if __name__ == '__main__':
    # Exemplo de uso direto (sem argumentos)
    # Descomente as linhas abaixo para rodar um exemplo

    # run_retrieval_experiment(
    #     corpus_file='corpora/passage_level/clapnq.jsonl',
    #     queries_file='human/retrieval_tasks/clapnq/clapnq_rewrite.jsonl',
    #     output_file='my_retrieval_results.jsonl',
    #     collection_name='mt-rag-clapnq-elser-512-100-20240503',
    #     top_k=5
    # )

    main()
