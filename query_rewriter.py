"""
Query Rewriter para MTRAG Benchmark

Este script permite criar suas próprias reescritas de queries usando diferentes estratégias:
1. Concatenação simples (baseline)
2. LLM-based rewriting (GPT, Claude, modelos locais)
3. Template-based rewriting
4. Reescrita com histórico seletivo

Uso:
    python query_rewriter.py --input human/conversations/conversations.json \
                             --output my_custom_rewrites/ \
                             --method llm \
                             --model gpt-4o-mini
"""

import json
import argparse
from typing import List, Dict
from pathlib import Path


class QueryRewriter:
    """Base class para diferentes estratégias de reescrita"""

    def rewrite(self, current_question: str, conversation_history: List[Dict]) -> str:
        """
        Reescreve a query atual com base no histórico da conversa

        Args:
            current_question: Pergunta atual do usuário
            conversation_history: Lista de mensagens anteriores [{"speaker": "user/agent", "text": "..."}]

        Returns:
            Query reescrita
        """
        raise NotImplementedError


class SimpleRewriter(QueryRewriter):
    """Estratégia 1: Concatenação simples do último turno"""

    def rewrite(self, current_question: str, conversation_history: List[Dict]) -> str:
        if not conversation_history or len(conversation_history) < 2:
            return f"|user|: {current_question}"

        # Pegar apenas o último Q&A
        prev_question = None
        prev_answer = None

        for i in range(len(conversation_history) - 1, -1, -1):
            msg = conversation_history[i]
            if msg['speaker'] == 'agent' and prev_answer is None:
                prev_answer = msg['text']
            elif msg['speaker'] == 'user' and prev_question is None:
                prev_question = msg['text']

            if prev_question and prev_answer:
                break

        if prev_question:
            return f"|user|: Given the context about '{prev_question}', {current_question}"

        return f"|user|: {current_question}"


class TemplateRewriter(QueryRewriter):
    """Estratégia 2: Template com contexto relevante"""

    def rewrite(self, current_question: str, conversation_history: List[Dict]) -> str:
        if not conversation_history:
            return f"|user|: {current_question}"

        # Identificar entidades mencionadas no histórico
        entities = self._extract_entities(conversation_history)

        if entities:
            context = ", ".join(entities[:3])  # Top 3 entidades
            return f"|user|: Regarding {context}, {current_question}"

        return f"|user|: {current_question}"

    def _extract_entities(self, history: List[Dict]) -> List[str]:
        """Extração simples de entidades (pode ser melhorada com NER)"""
        entities = []
        for msg in history:
            if msg['speaker'] == 'user':
                # Pegar substantivos próprios capitalizados (simplificado)
                words = msg['text'].split()
                entities.extend([w for w in words if w[0].isupper() and len(w) > 2])
        return list(set(entities))


class LLMRewriter(QueryRewriter):
    """Estratégia 3: Reescrita usando LLM (GPT, Claude, etc)"""

    def __init__(self, model: str = "gpt-4o-mini", provider: str = "openai", api_key: str = None):
        self.model = model
        self.provider = provider

        if provider == "openai":
            import openai
            self.client = openai.OpenAI(api_key=api_key)
        elif provider == "anthropic":
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key)
        elif provider == "local":
            # Para modelos locais via transformers
            from transformers import pipeline
            self.client = pipeline("text2text-generation", model=model)
        else:
            raise ValueError(f"Provider {provider} não suportado")

    def rewrite(self, current_question: str, conversation_history: List[Dict]) -> str:
        # Construir contexto da conversa
        context = self._build_context(conversation_history)

        prompt = f"""You are a query rewriter for an information retrieval system.

Conversation history:
{context}

Current question: {current_question}

Rewrite the current question to be self-contained and include all necessary context from the conversation history. The rewritten query should be optimized for semantic search.

Rewritten query:"""

        if self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=150
            )
            rewritten = response.choices[0].message.content.strip()

        elif self.provider == "anthropic":
            response = self.client.messages.create(
                model=self.model,
                max_tokens=150,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )
            rewritten = response.content[0].text.strip()

        elif self.provider == "local":
            response = self.client(prompt, max_length=150, do_sample=False)
            rewritten = response[0]['generated_text'].strip()

        return f"|user|: {rewritten}"

    def _build_context(self, history: List[Dict]) -> str:
        """Constrói string do contexto da conversa"""
        if not history:
            return "No previous context."

        # Pegar últimos 3 turnos (ou menos)
        recent = history[-6:] if len(history) > 6 else history

        lines = []
        for msg in recent:
            speaker = "User" if msg['speaker'] == 'user' else "Assistant"
            lines.append(f"{speaker}: {msg['text']}")

        return "\n".join(lines)


class FullHistoryRewriter(QueryRewriter):
    """Estratégia 4: Incluir todo o histórico relevante"""

    def __init__(self, max_history: int = 3):
        self.max_history = max_history

    def rewrite(self, current_question: str, conversation_history: List[Dict]) -> str:
        if not conversation_history:
            return f"|user|: {current_question}"

        # Pegar últimos N turnos
        recent = conversation_history[-self.max_history*2:] if len(conversation_history) > self.max_history*2 else conversation_history

        # Construir query com histórico
        history_parts = []
        for msg in recent:
            if msg['speaker'] == 'user':
                history_parts.append(msg['text'])

        if history_parts:
            full_context = " | ".join(history_parts)
            return f"|user|: Context: [{full_context}] | Current: {current_question}"

        return f"|user|: {current_question}"


def process_conversations(
    input_file: str,
    output_dir: str,
    rewriter: QueryRewriter,
    domain_filter: str = None
):
    """
    Processa arquivo de conversações e gera queries reescritas

    Args:
        input_file: Caminho para conversations.json
        output_dir: Diretório de saída
        rewriter: Instância de QueryRewriter
        domain_filter: Filtrar por domínio específico (clapnq, cloud, fiqa, govt)
    """
    print(f"Carregando conversações de {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        conversations = json.load(f)

    print(f"Total de conversações: {len(conversations)}")

    # Agrupar por domínio
    by_domain = {}
    for conv in conversations:
        collection = conv['retriever']['collection']['name']

        # Mapear nome da coleção para domínio
        if 'clapnq' in collection:
            domain = 'clapnq'
        elif 'cloud' in collection or 'ibmcloud' in collection:
            domain = 'cloud'
        elif 'fiqa' in collection:
            domain = 'fiqa'
        elif 'govt' in collection:
            domain = 'govt'
        else:
            continue

        if domain_filter and domain != domain_filter:
            continue

        if domain not in by_domain:
            by_domain[domain] = []
        by_domain[domain].append(conv)

    print(f"Domínios encontrados: {list(by_domain.keys())}")

    # Processar cada domínio
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for domain, convs in by_domain.items():
        print(f"\nProcessando {len(convs)} conversações de {domain}...")
        queries = []

        for conv in convs:
            conv_id = conv.get('id', str(hash(str(conv))))
            messages = conv['messages']

            # Processar cada turno
            history = []
            for i, msg in enumerate(messages):
                if msg['speaker'] == 'user':
                    # Criar ID do task
                    task_id = f"{conv_id}<::>{i+1}"

                    # Reescrever query
                    original_question = msg['text']
                    rewritten_question = rewriter.rewrite(original_question, history)

                    queries.append({
                        "_id": task_id,
                        "text": rewritten_question,
                        "metadata": {
                            "original": original_question,
                            "turn": i+1,
                            "conversation_id": conv_id
                        }
                    })

                # Adicionar ao histórico
                history.append({
                    'speaker': msg['speaker'],
                    'text': msg['text']
                })

        # Salvar queries reescritas
        output_file = output_path / f"{domain}_custom_rewrite.jsonl"
        print(f"Salvando {len(queries)} queries em {output_file}...")

        with open(output_file, 'w', encoding='utf-8') as f:
            for query in queries:
                f.write(json.dumps(query) + '\n')

        print(f"✓ {domain}: {len(queries)} queries salvas")


def main():
    parser = argparse.ArgumentParser(description='Reescrever queries para MTRAG')
    parser.add_argument('--input', type=str, required=True,
                       help='Caminho para conversations.json')
    parser.add_argument('--output', type=str, required=True,
                       help='Diretório de saída')
    parser.add_argument('--method', type=str, default='simple',
                       choices=['simple', 'template', 'llm', 'full_history'],
                       help='Método de reescrita')
    parser.add_argument('--model', type=str, default='gpt-4o-mini',
                       help='Modelo LLM (se method=llm)')
    parser.add_argument('--provider', type=str, default='openai',
                       choices=['openai', 'anthropic', 'local'],
                       help='Provider do LLM (se method=llm)')
    parser.add_argument('--api_key', type=str, default=None,
                       help='API key para o provider')
    parser.add_argument('--domain', type=str, default=None,
                       choices=['clapnq', 'cloud', 'fiqa', 'govt'],
                       help='Filtrar por domínio específico')
    parser.add_argument('--max_history', type=int, default=3,
                       help='Número máximo de turnos no histórico (se method=full_history)')

    args = parser.parse_args()

    # Criar rewriter
    if args.method == 'simple':
        rewriter = SimpleRewriter()
    elif args.method == 'template':
        rewriter = TemplateRewriter()
    elif args.method == 'llm':
        if not args.api_key:
            print("⚠️  API key necessária para method=llm. Use --api_key ou variável de ambiente")
            import os
            args.api_key = os.getenv('OPENAI_API_KEY') if args.provider == 'openai' else os.getenv('ANTHROPIC_API_KEY')
        rewriter = LLMRewriter(model=args.model, provider=args.provider, api_key=args.api_key)
    elif args.method == 'full_history':
        rewriter = FullHistoryRewriter(max_history=args.max_history)

    print(f"Usando método: {args.method}")

    # Processar conversações
    process_conversations(
        input_file=args.input,
        output_dir=args.output,
        rewriter=rewriter,
        domain_filter=args.domain
    )

    print("\n✓ Processamento concluído!")
    print(f"\nPróximos passos:")
    print(f"1. Rodar retrieval com suas queries customizadas:")
    print(f"   python example_retriever.py --corpus corpora/passage_level/clapnq.jsonl \\")
    print(f"       --queries {args.output}/clapnq_custom_rewrite.jsonl \\")
    print(f"       --output results_custom.jsonl \\")
    print(f"       --collection mt-rag-clapnq-elser-512-100-20240503")
    print(f"\n2. Avaliar resultados:")
    print(f"   python scripts/evaluation/run_retrieval_eval.py \\")
    print(f"       --input_file results_custom.jsonl \\")
    print(f"       --output_file results_custom_evaluated.jsonl")


if __name__ == '__main__':
    main()