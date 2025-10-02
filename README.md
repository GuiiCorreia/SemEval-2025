# 📋 Plano de Melhorias para MTRAGEval - SemEval

## 🎯 Visão Geral do Contexto

Com base no paper MTRAGEval e nas distribuições observadas:
- **84%** das tarefas são **Answerable**
- **8%** são **Partially Answerable** 
- **7%** são **Unanswerable**
- **85%** são conversações **Multi-turn** (Follow-up)
- **15%** são **Clarification** (Conversacional)

---

## 💡 Estratégias Propostas

### 1. **Otimização de Query Rewriting**

#### Problema Identificado
O paper usa `rewritten_query` com um único prompt simples, mas há espaço para melhorias.

#### Soluções Propostas
- **Usar modelos maiores** para query rewriting (ex: GPT-4, Claude, Gemini 2.5 Flash)
- **Few-shot prompts** com exemplos de boas reformulações
- **In-context learning** para adaptar ao domínio
- **Chain-of-Thought (CoT)** no prompt para raciocínio explícito

#### Implementação Sugerida
```
Ideia: Fine-tuning em cascata
1. Gerar dados sintéticos com Gemini 2.5 Flash a partir dos outputs do Mixtral
2. Usar esses dados para fine-tuning do modelo original
3. Adicionar CoT no prompt para melhor raciocínio contextual
```

---

### 2. **Multi-Query Retrieval Strategy** ⭐

#### Problema Identificado
Pegar todo o contexto anterior e fazer apenas 1 query limita a cobertura da busca.

#### Solução Proposta: Pipeline de 5 Queries
```
┌─────────────────────────────────────────────┐
│  Contexto Conversacional Completo           │
└──────────────────┬──────────────────────────┘
                   │
         ┌─────────▼──────────┐
         │  Gerar 5 Queries   │
         │  Diversificadas    │
         └─────────┬──────────┘
                   │
    ┌──────────────┼──────────────┐
    │              │              │
┌───▼───┐     ┌───▼───┐     ┌───▼───┐  ...
│Query 1│     │Query 2│     │Query 5│
└───┬───┘     └───┬───┘     └───┬───┘
    │             │             │
┌───▼──────┐ ┌───▼──────┐ ┌───▼──────┐
│Top-K docs│ │Top-K docs│ │Top-K docs│
└───┬──────┘ └───┬──────┘ └───┬──────┘
    └─────────────┼─────────────┘
                  │
         ┌────────▼─────────┐
         │   RERANK MODEL   │
         │  (ex: Cohere)    │
         └────────┬─────────┘
                  │
         ┌────────▼─────────┐
         │  Top 5 Finais    │
         │  para Geração    │
         └──────────────────┘
```

**Vantagens:**
- Maior cobertura semântica
- Diferentes perspectivas da mesma necessidade informacional
- Reranking final garante os melhores documentos

---

### 3. **Guardrails e Prompt Engineering**

#### 3.1 Melhorar o Prompt de Geração

Com base na **Imagem 1**, o prompt atual inclui avaliação de:
- **[Faithfulness]**: Fidelidade aos documentos
- **[Appropriateness]**: Relevância à questão
- **[Completeness]**: Completude da resposta

#### Melhorias Propostas:

```markdown
PROMPT APRIMORADO:

Você é um assistente especializado em RAG conversacional. 

[CONTEXTO DA CONVERSA]
{previous_turns}

[DOCUMENTOS RECUPERADOS]
{passages}

[QUESTÃO ATUAL]
{current_question}

[INSTRUÇÕES]
1. Analise se a questão É RESPONDÍVEL com os documentos fornecidos
   - Se NÃO for respondível → Responda: "Não tenho informações suficientes nos documentos para responder."
   - Se PARCIALMENTE respondível → Indique o que pode ser respondido e o que falta

2. [CLARIFICATION CHECK] Se a questão do usuário for ambígua:
   - Solicite esclarecimentos ANTES de responder
   - Exemplo: "Você quer saber sobre X ou Y?"

3. [FOLLOW-UP HANDLING] Se for uma pergunta de follow-up:
   - Conecte com informações anteriores da conversa
   - Mantenha coerência com respostas passadas

4. [FAITHFULNESS] Seja fiel aos documentos:
   - Use APENAS informações presentes nos documentos
   - Cite trechos quando apropriado
   
5. [COMPLETENESS] Responda de forma completa mas concisa

6. [ERROR CORRECTION] Se o usuário cometer um erro factual:
   - Corrija educadamente com base nos documentos
   - NÃO aceite/propague informações incorretas
```

---

### 4. **Estratégia de Answerability Detection** 🛡️

#### Problema Identificado
Os 7% de questões **Unanswerable** e 8% **Partially Answerable** precisam ser detectados para evitar alucinações.

#### Implementação Proposta

```python
# Pipeline de Detecção

1. Filtrar dataset por categoria:
   - answerable_subset (84%)
   - partial_subset (8%)
   - unanswerable_subset (7%)

2. Análise de Padrões:
   - O que leva o modelo a classificar como unanswerable?
   - Quais padrões nas perguntas/documentos?
   
3. Criar Dataset de Teste Específico:
   - Balanceado com os 3 tipos
   - Testar guardrails de answerability

4. Implementar Classificador Pre-Generation:
   ANTES de gerar resposta:
   ┌────────────────────────────┐
   │ Classificador Answerability│
   │ (LLM Judge ou Fine-tuned)  │
   └──────────┬─────────────────┘
              │
     ┌────────┼────────┐
     │        │        │
 Answerable  Partial  Unanswerable
     │        │        │
  Gerar    Gerar +    "Não tenho
  Normal   Avisar     informações"
```

---

### 5. **Tratamento de Question Types (Tabela 9 no paper)**

#### 5.1 Follow-up Questions
- Implementar memória conversacional efetiva
- Resolver correferências (ex: "ele", "isso", "aquilo")
- Contexto deve incluir entidades mencionadas

#### 5.2 Clarification Questions
**CRÍTICO**: O modelo deve saber quando pedir esclarecimentos

```
Cenários para pedir clarificação:
1. Usuário faz pergunta ambígua
2. Múltiplas interpretações possíveis
3. Informação insuficiente na pergunta

Exemplo (da Tabela):
User: "graphql"
Agent: "GraphQL is an open-source data query..."
User [Clarification]: "No, I meant, how do I set it up."
```

**Solução**: Adicionar no prompt:
- Detecção de ambiguidade
- Template para pedir clarificações
- Não assumir intenção do usuário

#### 5.3 Troubleshooting Questions
- Formato: "I have error X... what should I do?"
- Requer busca em documentação técnica
- Resposta step-by-step

---

## 📊 Roadmap de Implementação

### Fase 1: Retrieval (Subtask A)
- [ ] Implementar pipeline de 5 queries diversificadas
- [ ] Integrar modelo de rerank (Cohere/BGE-reranker)
- [ ] Testar com query rewriting melhorado (CoT + Few-shot)

### Fase 2: Answerability & Guardrails
- [ ] Criar dataset filtrado por answerability
- [ ] Implementar classificador pre-generation
- [ ] Desenvolver prompts específicos para cada tipo

### Fase 3: Generation (Subtasks B & C)
- [ ] Aplicar prompt aprimorado com guardrails
- [ ] Implementar detecção de clarification needs
- [ ] Adicionar correção de erros factuais do usuário
- [ ] Testar com few-shot examples por question type

### Fase 4: Fine-tuning (Opcional)
- [ ] Gerar dados sintéticos com Gemini 2.5 Flash
- [ ] Fine-tuning em modelo menor para speed/cost
- [ ] Validar performance vs. baselines

---

## 🎯 Métricas de Sucesso Esperadas

| Componente | Baseline (Paper) | Meta |
|------------|------------------|------|
| Retrieval nDCG@5 | 0.49 (Elser) | **0.60+** |
| Generation RBllm | 0.76 (GPT-4o) | **0.85+** |
| Answerability F1 | - | **0.90+** |

---

## ⚠️ Pontos de Atenção

1. **Evitar over-engineering**: Testar incrementalmente
2. **Balancear custo/performance**: Few-shot pode ser suficiente vs. fine-tuning
3. **Validação humana**: 20 tarefas serão avaliadas por humanos (conforme paper)
4. **Conformidade**: Não usar Mixtral 8x7B (usado na criação do dataset)

---

**Status**: Pronto para implementação por fases
**Próximo passo**: Escolher qual fase começar e definir modelos/ferramentas
