# Query Rewriting para MTRAG

Query rewriting é crucial para retrieval em conversações multi-turn. Este guia mostra como criar suas próprias reescritas.

## Por que Query Rewriting?

Em conversações multi-turn, queries frequentemente dependem de contexto anterior:

**Turno 1:** "Where do the Arizona Cardinals play?"
**Turno 2:** "Do they play outside the US?"  ← Precisa de contexto!

**Query reescrita:** "Do the Arizona Cardinals play outside the US?" ← Auto-contida!

## Estratégias Implementadas

### 1. **Simple Rewriter** (Baseline)
Concatenação simples com contexto do último turno.

```bash
python query_rewriter.py \
  --input human/conversations/conversations.json \
  --output my_rewrites/ \
  --method simple
```

**Exemplo:**
- Original: "Do they play outside the US?"
- Reescrita: "Given the context about 'Where do the Arizona Cardinals play', Do they play outside the US?"

### 2. **Template-Based Rewriter**
Usa templates com extração de entidades.

```bash
python query_rewriter.py \
  --input human/conversations/conversations.json \
  --output my_rewrites/ \
  --method template
```

**Exemplo:**
- Original: "What is their schedule?"
- Reescrita: "Regarding Arizona Cardinals, NFL, 2017, what is their schedule?"

### 3. **LLM-Based Rewriter** (Melhor qualidade)
Usa LLMs (GPT-4, Claude, etc) para reescrita inteligente.

#### Com OpenAI:
```bash
export OPENAI_API_KEY="your-key-here"

python query_rewriter.py \
  --input human/conversations/conversations.json \
  --output my_rewrites/ \
  --method llm \
  --provider openai \
  --model gpt-4o-mini
```

#### Com Anthropic Claude:
```bash
export ANTHROPIC_API_KEY="your-key-here"

python query_rewriter.py \
  --input human/conversations/conversations.json \
  --output my_rewrites/ \
  --method llm \
  --provider anthropic \
  --model claude-3-haiku-20240307
```

#### Com modelo local (HuggingFace):
```bash
python query_rewriter.py \
  --input human/conversations/conversations.json \
  --output my_rewrites/ \
  --method llm \
  --provider local \
  --model google/flan-t5-large
```

**Exemplo:**
- Original: "Do they play outside the US?"
- Contexto: User asked about Arizona Cardinals games, Assistant mentioned they play in London
- Reescrita: "Do the Arizona Cardinals play games outside the United States?"

### 4. **Full History Rewriter**
Inclui histórico completo (últimos N turnos).

```bash
python query_rewriter.py \
  --input human/conversations/conversations.json \
  --output my_rewrites/ \
  --method full_history \
  --max_history 3
```

**Exemplo:**
- Original: "Who won?"
- Reescrita: "Context: [Where do Cardinals play | Do they play outside US | What was the score] | Current: Who won?"

## Workflow Completo

### 1. Gerar suas queries reescritas

```bash
# Para todas as coleções
python query_rewriter.py \
  --input human/conversations/conversations.json \
  --output my_custom_rewrites/ \
  --method llm \
  --model gpt-4o-mini

# Ou para uma coleção específica
python query_rewriter.py \
  --input human/conversations/conversations.json \
  --output my_custom_rewrites/ \
  --method llm \
  --model gpt-4o-mini \
  --domain clapnq
```

**Output:**
- `my_custom_rewrites/clapnq_custom_rewrite.jsonl`
- `my_custom_rewrites/cloud_custom_rewrite.jsonl`
- `my_custom_rewrites/fiqa_custom_rewrite.jsonl`
- `my_custom_rewrites/govt_custom_rewrite.jsonl`

### 2. Rodar retrieval com suas queries customizadas

```bash
# ClapNQ
python example_retriever.py \
  --corpus corpora/passage_level/clapnq.jsonl \
  --queries my_custom_rewrites/clapnq_custom_rewrite.jsonl \
  --output results_clapnq_custom.jsonl \
  --collection mt-rag-clapnq-elser-512-100-20240503

# Repetir para cloud, fiqa, govt...
```

### 3. Combinar e avaliar

```bash
# Combinar resultados
cat results_*_custom.jsonl > all_results_custom.jsonl

# Avaliar
python scripts/evaluation/run_retrieval_eval.py \
  --input_file all_results_custom.jsonl \
  --output_file evaluation_custom.jsonl
```

### 4. Comparar com baseline

```bash
# Suas queries customizadas
cat evaluation_custom_aggregate.csv

# Queries originais (dos autores)
python scripts/evaluation/run_retrieval_eval.py \
  --input_file human/generation_tasks/RAG.jsonl \
  --output_file evaluation_baseline.jsonl

cat evaluation_baseline_aggregate.csv
```

## Estratégias Avançadas

### A. Filtrar contexto relevante

Modificar `LLMRewriter._build_context()` para incluir apenas turnos relevantes:

```python
def _build_context(self, history: List[Dict]) -> str:
    # Filtrar apenas perguntas do usuário
    user_questions = [msg for msg in history if msg['speaker'] == 'user']
    recent_questions = user_questions[-3:]  # Últimas 3 perguntas

    return " | ".join([q['text'] for q in recent_questions])
```

### B. Adicionar informações da resposta anterior

```python
def rewrite(self, current_question: str, conversation_history: List[Dict]) -> str:
    # Pegar última resposta do agente
    last_answer = None
    for msg in reversed(conversation_history):
        if msg['speaker'] == 'agent':
            last_answer = msg['text']
            break

    if last_answer:
        # Usar resposta anterior como contexto
        prompt = f"""Previous answer: {last_answer}

        Current question: {current_question}

        Rewrite the question to be self-contained:"""
```

### C. Query expansion com sinônimos

```python
from nltk.corpus import wordnet

def expand_query(query: str) -> str:
    words = query.split()
    expanded = []

    for word in words:
        synonyms = [syn.name() for syn in wordnet.synsets(word)][:2]
        expanded.extend([word] + synonyms)

    return " ".join(expanded)
```

### D. Usar embeddings para detectar mudança de tópico

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def detect_topic_change(current: str, history: List[str]) -> bool:
    if not history:
        return False

    current_emb = model.encode([current])
    last_emb = model.encode([history[-1]])

    similarity = cosine_similarity(current_emb, last_emb)[0][0]

    # Se similaridade baixa, tópico mudou
    return similarity < 0.5
```

## Métricas Esperadas

Baseado no paper, query rewriting melhora significativamente:

| Método | R@5 | nDCG@5 |
|--------|-----|--------|
| Last Turn (sem rewrite) | 0.49 | 0.45 |
| Query Rewrite (baseline) | 0.52 | 0.48 |
| **Sua reescrita customizada** | **?** | **?** |

**Objetivo:** Superar 0.52 R@5 e 0.48 nDCG@5!

## Tips para Melhorar

1. **Analyze failures:** Veja queries com baixo score e ajuste a estratégia
2. **Combine methods:** Use LLM para casos ambíguos, template para casos simples
3. **Domain-specific:** Crie estratégias diferentes por domínio
4. **Iterate:** Compare resultados e refine o prompt do LLM
5. **Cost vs Quality:** LLM é caro mas melhor; template é rápido mas menos preciso

## Troubleshooting

**Erro: API rate limit**
- Use batch processing
- Adicione delays entre requests
- Use modelo local

**Queries muito longas**
- Limite histórico (max_history)
- Resuma contexto ao invés de concatenar

**Resultados piores que baseline**
- Verifique se está usando o mesmo corpus (passage_level)
- Compare queries reescritas manualmente
- Teste diferentes temperaturas do LLM (0.3-0.7)

## Recursos

- **Paper MTRAG:** https://arxiv.org/abs/2501.03468
- **Query rewriting em IR:** https://arxiv.org/abs/2305.14283
- **Conversational search:** https://arxiv.org/abs/2203.08808