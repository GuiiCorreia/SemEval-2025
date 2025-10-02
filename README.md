# Estratégia Completa de Implementação - MTRAGEval v3
## Síntese: Paper + Análise de Dados + Propostas

---

## 1. Visão Geral do Desafio

### 1.1 Contexto da Competição (do Paper)

**MTRAGEval** é uma shared task focada em avaliar sistemas de RAG conversacional multi-turno. Diferente de competições anteriores (TREC RAG, iKAT):
- Usa **active retrieval** (real-time, não pré-computado)
- Fornece **respostas longas** (não apenas sim/não)
- Inclui **casos unanswerable** (16% do dataset)
- Cobre **múltiplos domínios** (Cloud, Government)
- Avalia **todos os componentes** do pipeline RAG

**Três subtasks:**
- **A) Retrieval**: Recuperar passagens relevantes
- **B) Generation with Reference**: Gerar resposta com passagens gold
- **C) Full RAG**: Pipeline completo end-to-end

### 1.2 Principais Desafios Identificados

**Do Paper:**
1. **Answerability** (16% unanswerable, 8% partially answerable): Saber quando não responder
2. **Later turns** (85% são multi-turn): Turnos posteriores dependem de contexto não-standalone
3. **Faithfulness**: Manter resposta fiel às passagens recuperadas

**Dos Dados (exemplo analisado):**
1. **Domain shift**: Conversa muda de tornados → preparação geral → terremotos
2. **Context mismatch**: Docs recuperados (todos sobre earthquakes) não cobrem contexto completo (safe rooms, tornados)
3. **Score proximity**: Retrieval scores muito próximos (7.74-8.43) tornam reranking essencial
4. **Reference expansion**: Resposta gold menciona conceitos não presentes nos docs (SMC Alert, CERT)

---

## 2. Estratégia Unificada: Retrieval (Subtask A)

### 2.1 Análise do Problema

**Baseline atual (do paper):**
- BM25: nDCG@5 = 0.18
- BGE-base: nDCG@5 = 0.27
- **Elser: nDCG@5 = 0.45** (melhor)

**Problema identificado nos dados:**
- Query literal ("Is it the same for earthquakes?") perde contexto crítico
- Docs recuperados focam apenas no tópico atual, ignoram histórico
- Campo `rewritten_query` mostra que query standalone é essencial

### 2.2 Proposta: Multiple Query Strategy + Hybrid Retrieval

**Arquitetura de 5 queries diversas:**

**Query 1: Standalone Rewrite**
- Usar campo `rewritten_query` como referência
- Objetivo: Tornar pergunta independente do histórico
- Exemplo: "Is it the same?" → "Is the same three-day supply recommendation applicable for earthquakes?"
- Peso: 30% (mais importante)

**Query 2: Entity-Focused**
- Extrair entidades-chave do histórico completo
- Combinar com pergunta atual
- Exemplo: "sheltered rooms, safe rooms, three-day supply, earthquakes"
- Peso: 20%

**Query 3: Contextual Window**
- Últimos 2-3 turnos completos (user + agent)
- Captura contexto recente sem overhead
- Peso: 20%

**Query 4: Comparison/Relationship**
- Ativar quando pergunta tem padrão comparativo ("is it the same", "how about", "what about")
- Buscar explicitamente relação entre entidades
- Exemplo: "tornado preparation vs earthquake preparation"
- Peso: 15%

**Query 5: Domain Bridge**
- Detectar mudanças de domínio no histórico
- Buscar documentos que conectam domínios
- Exemplo: "disaster preparedness tornado earthquake general"
- Peso: 15%

**Pipeline de execução:**

1. **Multi-model retrieval** para cada query:
   - Sparse retrieval (BM25) para recall
   - Dense retrieval (Elser ou BGE-large) para semântica
   - Hybrid fusion (RRF - Reciprocal Rank Fusion)

2. **Aggregation**:
   - Cada query retorna top-K documentos (K=5-10)
   - Deduplicar por document_id mantendo max score
   - Score final = weighted sum baseado em tipo de query
   - Pool de ~20-30 documentos únicos

3. **Reranking**:
   - Cross-encoder com contexto completo (histórico + query standalone)
   - Considerar: relevância para pergunta atual E cobertura do histórico
   - Output: top-5 final

**Justificativa para essa abordagem:**
- Cobre gap identificado (context mismatch)
- Diversidade de queries aumenta recall
- Reranking com contexto completo melhora precision
- Weighted scoring permite calibração por tipo de pergunta

### 2.3 Context Window Management

**Problema:** Históricos longos excedem context length e diluem relevância

**Solução adaptativa por tamanho:**

- **Turnos 1-2**: Usar tudo (histórico curto)
- **Turnos 3-5**: Sliding window últimos 3 turnos + entidades-chave dos anteriores
- **Turnos 6+**: Sliding window + compression via summarization

**Técnica de compression:**
- Extrair: entidades mencionadas, tópicos discutidos, fatos estabelecidos
- Descartar: saudações, meta-conversação, redundâncias
- Preservar: informações referenciadas no turno atual

---

## 3. Estratégia Unificada: Generation (Subtask B)

### 3.1 Análise do Problema

**Baseline atual (do paper):**
- GPT-4o: RBllm = 0.76, RLf = 0.76
- Llama 3.1 405B: RBllm = 0.74, RLf = 0.75
- Reference (upper bound): RBllm = 0.95, RLf = 0.87

**Gap de ~20 pontos** entre modelos e referência

**Problema identificado nos dados:**
- Resposta gold balanceia extraction (dos docs) + synthesis (conhecimento geral)
- Menciona SMC Alert, CERT (não nos docs) mas contextualizados apropriadamente
- Comparação explícita entre earthquake prep e disaster prep geral

### 3.2 Proposta: Structured Prompt + Guardrails + Validation

**Fase 1: Pre-Generation Checks**

**Answerability Classification:**
- Features: retrieval scores, semantic similarity, term coverage, question type
- Thresholds calibrados por análise estatística do dataset
- Output: ANSWERABLE / PARTIALLY_ANSWERABLE / UNANSWERABLE

**Question Type Detection:**
- Follow-up (pede mais informação sobre tópico anterior)
- Clarification (usuário ou modelo pedindo esclarecimento)
- Comparison (pergunta sobre similaridades/diferenças)
- Troubleshooting (resolver problema específico)
- Estratégia de resposta ajustada por tipo

**Fase 2: Structured Prompt Design**

**Template com 5 seções claras:**

1. **Conversation Context**: Histórico formatado com speaker tags
2. **Retrieved Passages**: Docs numerados com metadata (título, score)
3. **Current Question**: Original + standalone version
4. **Instructions**: 4 princípios (Answerability, Faithfulness, Appropriateness, Completeness)
5. **Output Format**: Guidelines para estrutura da resposta

**Princípios detalhados:**

**Answerability:**
- Se ANSWERABLE: Responder completamente
- Se PARTIALLY_ANSWERABLE: Responder o que puder + disclaimer sobre gaps
- Se UNANSWERABLE: "I don't have enough information..." + opcionalmente pedir clarificação

**Faithfulness:**
- Base primária: documentos fornecidos
- Citações obrigatórias: formato [Doc X]
- Conhecimento geral permitido APENAS para contextualização, com distinção clara
- Exemplo: "Based on [Doc 1]... Generally speaking, [contexto geral]..."

**Appropriateness:**
- Responder diretamente à pergunta
- Para comparações: listar explicitamente similarities E differences
- Manter continuidade conversacional
- Ajustar tom/detalhamento ao question type

**Completeness:**
- Endereçar TODAS as partes da pergunta
- Fornecer detalhes suficientes sem verbosidade
- Antecipar follow-ups comuns (opcional)

**Fase 3: Post-Generation Validation**

**Faithfulness Check:**
- NLI model para verificar entailment
- Cada claim na resposta deve ter support nos docs
- Threshold: score > 0.7
- Se falha: regenerar com prompt mais restritivo

**Citation Verification:**
- Verificar que todas as afirmações factuais têm [Doc X]
- Se faltando mas conteúdo é faithful: adicionar citações automaticamente

**Completeness Check:**
- Parser para extrair sub-perguntas da query
- Verificar que resposta endereça cada uma
- Se incompleto: trigger para elaboração

### 3.3 Handling de Casos Especiais

**Comparison Questions** (tipo do exemplo):
- Estrutura de resposta: Similarities → Differences → Context
- Buscar ativamente informação sobre AMBAS entidades
- Exemplo: "Both X and Y recommend A. However, X also requires B, while Y focuses on C."

**Clarification Requests:**
- Reformular resposta anterior com outras palavras
- Adicionar exemplos ou detalhes
- Manter link explícito ao trecho sendo esclarecido

**User Corrections:**
- Acknowledgement educado: "Thank you for the correction..."
- Ajustar understanding
- Fornecer resposta revisada

---

## 4. Estratégia Unificada: Full RAG (Subtask C)

### 4.1 Pipeline Integrado

**Fluxo completo:**

1. **Query Rewriting**: Gerar 5 queries diversas
2. **Retrieval**: Multi-model hybrid search
3. **Quality Gate 1**: Verificar retrieval quality
   - Se avg_score < threshold: Trigger IDK path
   - Se answerability = UNANSWERABLE: Polite decline
4. **Adaptive Retrieval** (se necessário):
   - Coverage check: Docs cobrem contexto histórico?
   - Se insuficiente: Additional retrieval com context queries
5. **Reranking**: Top-5 final com cross-encoder
6. **Generation**: Com prompt estruturado
7. **Quality Gate 2**: Post-validation
   - Se faithfulness falha: Re-retrieve com queries refinadas
   - Se completeness falha: Generate elaboration
8. **Output**: Resposta final

### 4.2 Adaptive Retrieval Strategy

**Motivação:** Retrieval único pode falhar em capturar todo contexto necessário (visto no exemplo)

**Implementação:**

**Coverage Score:**
- Termos da query standalone presentes nos docs: 40%
- Entidades do histórico presentes nos docs: 30%
- Domain coverage (se há domain shift): 20%
- Semantic similarity threshold: 10%

**Decision tree:**
- Coverage > 0.75: Retrieval suficiente, prosseguir
- Coverage 0.50-0.75: Adicionar 2-3 docs com context queries
- Coverage < 0.50: Re-retrieve completo com queries reformuladas

**Context Queries** (para additional retrieval):
- Entidades do histórico não cobertas
- Tópicos mencionados mas não documentados
- Relações entre conceitos (domain bridges)

### 4.3 Error Recovery Mechanisms

**Cenário 1: Low Retrieval Scores**
- Se max_score < 7.0: Alta probabilidade de answerability issue
- Ação: Tentar query expansion + re-search
- Se ainda baixo: Trigger IDK response

**Cenário 2: Validation Failure**
- Se faithfulness check falha: Problema na geração
- Ação: Regenerar com temperatura mais baixa + prompt mais restritivo
- Limite: 2 tentativas, depois retornar disclaimer

**Cenário 3: Context Inconsistency**
- Detectar contradições entre histórico e docs atuais
- Ação: Explicitly address na resposta ("Earlier we discussed X, but current documents suggest Y...")

---

## 5. Answerability: Estratégia Profunda

### 5.1 Framework de Classificação

**Multi-factor scoring:**

**Feature 1: Retrieval Quality (35% weight)**
- avg_score dos 5 docs
- max_score
- score variance (baixa variance = docs similares, bom sinal)

**Feature 2: Semantic Overlap (30% weight)**
- Query embedding vs doc embeddings
- Cosine similarity
- Term coverage (keywords da query nos docs)

**Feature 3: Question Characteristics (20% weight)**
- Question type (clarifications mais fáceis de responder)
- Multi-turn type
- Presença de comparação/negação

**Feature 4: Historical Context (15% weight)**
- Entidades do histórico presentes nos docs
- Continuidade temática
- Domain consistency

**Thresholds calibrados:**
- Score > 0.75: ANSWERABLE (confiança alta)
- Score 0.50-0.75: PARTIALLY_ANSWERABLE (confiança média)
- Score < 0.50: UNANSWERABLE (confiança baixa)

### 5.2 Response Strategies por Label

**ANSWERABLE:**
- Generate full answer
- Use all 5 docs se relevantes
- Target: comprehensive response

**PARTIALLY_ANSWERABLE:**
- Generate partial answer
- Disclaimer explícito: "Based on available information, I can tell you X. However, I don't have specific details about Y."
- Opcionalmente: Suggest what info would help

**UNANSWERABLE:**
- Primary path: "I don't have sufficient information in the available documents to answer this question."
- Secondary path (se clarification possível): "To answer this, I would need to know more about [X]. Could you clarify [Y]?"

### 5.3 Análise Estatística para Calibração

**Dataset analysis necessária:**
- Distribuição de answerability labels
- Correlation entre retrieval scores e answerability
- Question types por answerability class
- Turn number vs answerability (later turns mais difíceis?)

**Calibração iterativa:**
1. Treinar classifier inicial no dev set
2. Analisar false positives (disse answerable mas não era)
3. Analisar false negatives (disse unanswerable mas era)
4. Ajustar thresholds para balance precision/recall
5. Meta: F1 > 0.80

---

## 6. Multi-Turn Context: Estratégias Avançadas

### 6.1 Intent Classification

**5 intents principais identificados:**

**1. Follow-up**
- Pattern: Pedido por mais informação relacionada
- Exemplo: "Can you tell me more about X?"
- Strategy: Expand on previous response, add details

**2. Clarification**
- Pattern: "I meant...", "No, I'm asking about...", "What I meant was..."
- Strategy: Rephrase understanding, confirm intent, answer revised question

**3. Comparison**
- Pattern: "Is it the same...", "How about...", "What about..."
- Strategy: Explicitly compare entities, list similarities/differences

**4. Correction**
- Pattern: User fornece informação corrigindo anterior
- Strategy: Acknowledge, adjust understanding, provide revised answer

**5. Elaboration Request**
- Pattern: "Can you explain more about...", "What does that mean..."
- Strategy: Deep dive no tópico específico da resposta anterior

### 6.2 Domain Shift Handling

**Problema:** Conversa pode mudar de domínio (tornado → earthquake no exemplo)

**Detection:**
- Named entity tracking por turno
- Topic modeling
- Keyword shift analysis

**Strategy quando detectado:**
1. Query deve incluir AMBOS domínios
2. Retrieval deve buscar: docs sobre novo domínio + docs de comparação/relação
3. Response deve: acknowledge shift + compare/contrast se apropriado
4. Exemplo: "While earlier we discussed tornado safety, for earthquakes the recommendations differ in [X] but are similar in [Y]..."

### 6.3 Entity Tracking

**Motivation:** Later turns referenciam entidades mencionadas anteriormente

**Implementation:**
- Manter entity registry por conversa
- Para cada turno: extract e update entities
- Types: Person, Location, Organization, Concept, Event
- Attributes: first_mention_turn, last_mention_turn, context

**Usage:**
- Query generation: Include relevant entities
- Answerability check: Verify entities are documented
- Response validation: Ensure entity consistency

---

## 7. Métricas e Avaliação

### 7.1 Métricas Oficiais da Competição

**Retrieval (Subtask A):**
- Recall@5, Recall@10
- nDCG@5, nDCG@10

**Generation (Subtasks B e C):**
- **RBalg**: Harmonic mean de BERT-Recall, RougeL, BERT-K-Precision
- **RBllm**: LLM-based judge (reference-based)
- **RLf**: RAGAS Faithfulness LLM judge

**Todas condicionadas em IDK judge** (primeiro determina se resposta contém answer)

**Human evaluation:** ~20 tasks para Subtask C

### 7.2 Targets Baseados em Baselines

**Subtask A (Retrieval):**
- Baseline: Elser nDCG@5 = 0.45
- **Target: nDCG@5 > 0.50** (+11% improvement)
- Stretch goal: nDCG@5 > 0.55

**Subtask B (Generation with Reference):**
- Baseline: GPT-4o RBllm = 0.76
- **Target: RBllm > 0.80** (+5% improvement)
- Faithfulness: RLf > 0.80

**Subtask C (Full RAG):**
- Baseline: GPT-4o RBllm = 0.76, RLf = 0.76
- **Target: RBllm > 0.78, RLf > 0.78**
- Critical: IDK precision > 0.85 (evitar false answerable)

### 7.3 Desenvolvimento Iterativo

**Fase 1: Baseline Implementation**
- Retrieval: Single query + BM25
- Generation: Simple prompt + GPT-4
- Evaluate on dev set
- Estabelecer baseline interno

**Fase 2: Core Improvements**
- Retrieval: Multiple queries + reranking
- Generation: Structured prompt + guardrails
- Answerability classifier
- Evaluate e compare vs baseline

**Fase 3: Advanced Features**
- Adaptive retrieval
- Post-validation
- Entity tracking
- Fine-tuning

**Fase 4: Optimization**
- Hyperparameter tuning
- Threshold calibration
- Error analysis em casos difíceis
- Ensemble strategies (se aplicável)

---

## 8. Análise de Casos Prioritários

### 8.1 Casos Difíceis Identificados

**1. Domain Shift (exemplo analisado)**
- Desafio: Retrieval não captura contexto de domínio anterior
- Solution: Domain bridge queries + entity tracking
- Success metric: Coverage score > 0.70 para ambos domínios

**2. Later Turns (Turn 5+)**
- Desafio: Context window management + non-standalone queries
- Solution: Sliding window + compression + robust query rewriting
- Success metric: nDCG@5 similar para later turns vs early turns

**3. Comparison Questions**
- Desafio: Requires docs sobre AMBAS entidades
- Solution: Comparison-specific queries + structured response
- Success metric: Response contém sections de similarity e difference

**4. Unanswerable**
- Desafio: Distinguir de low-quality retrieval vs genuinely unanswerable
- Solution: Multi-factor answerability classifier + adaptive retrieval
- Success metric: F1 > 0.80 no unanswerable detection

**5. Partial Answerability**
- Desafio: Balancear responder o possível vs admitir gaps
- Solution: Partial answer + explicit disclaimer
- Success metric: RBllm > 0.70 e IDK precision > 0.80

### 8.2 Error Analysis Framework

**Para cada erro no dev set:**

1. **Categorizar erro:**
   - Retrieval failure (docs não relevantes)
   - Generation failure (alucinação, off-topic)
   - Answerability misjudgment
   - Context misunderstanding

2. **Root cause analysis:**
   - Query rewriting inadequada?
   - Retrieval model limitação?
   - Prompt engineering issue?
   - Context window overflow?

3. **Propor fix específico:**
   - Ajustar query strategy
   - Tune reranking weights
   - Refine prompt
   - Adjust thresholds

4. **Validate fix:**
   - Testar em casos similares
   - Verificar que não regride outros casos
   - Update metrics

---

## 9. Diferenciadores Competitivos

### 9.1 Inovações Propostas

**1. Weighted Multi-Query Strategy**
- Não apenas multiple queries, mas weighted por tipo
- Calibração empírica dos weights
- Diferente de fusion simples

**2. Adaptive Retrieval com Coverage Check**
- Maioria dos sistemas: retrieval único
- Nossa abordagem: Verificar coverage e re-retrieve se necessário
- Trade-off latency vs quality

**3. Structured Prompt com Validation Loop**
- Não apenas prompt engineering
- Feedback loop: validation → regeneration
- Garante qualidade mínima

**4. Multi-Factor Answerability Classifier**
- Não apenas retrieval score
- 4 features independentes
- Calibrado especificamente para MTRAG

**5. Domain Shift Detection & Handling**
- Identificado como gap nos dados
- Solução específica implementada
- Potencial high impact em later turns

### 9.2 Trade-offs e Decisões de Design

**Latency vs Quality:**
- Multiple queries + reranking: +200-300ms
- Adaptive retrieval: +100-200ms extra quando triggered
- Decision: Priorizar quality, latency é secondary concern para competição

**Faithfulness vs Helpfulness:**
- Strict faithfulness pode gerar respostas incompletas
- Permitir contextual knowledge (como no gold example)
- Decision: Faithfulness primário, contexto geral permitido com distinção clara

**Recall vs Precision (Retrieval):**
- Multiple queries aumenta recall
- Reranking melhora precision
- Decision: Balance via top-K tuning (K=5 final)

**Simple vs Complex:**
- Pipeline complexo é mais difícil de debug
- Mas cada componente endereça problema específico
- Decision: Complexity justificada, mitigar com modular design

---

## 10. Plano de Execução

### 10.1 Cronograma (8-9 semanas)

**Semana 1: Setup & Baseline**
- Setup ambiente, carregar dataset completo
- Implementar retrieval baseline (BM25 + single query)
- Implementar generation baseline (simple prompt)
- Estabelecer metrics interno
- **Deliverable:** Baseline scores em dev set

**Semana 2: Análise Profunda**
- Análise exploratória completa do dataset
- Identificar casos difíceis
- Calibrar thresholds iniciais
- Error analysis do baseline
- **Deliverable:** Relatório de análise + casos prioritários

**Semana 3: Retrieval v1**
- Implementar query rewriting
- Testar multiple retrieval models (BM25, BGE, Elser)
- Implementar hybrid fusion
- **Deliverable:** Retrieval improvement sobre baseline

**Semana 4: Retrieval v2**
- Implementar multiple query strategy
- Implementar reranking
- Tuning de weights
- **Deliverable:** Target nDCG@5 > 0.50

**Semana 5: Generation v1**
- Desenvolver structured prompt
- Implementar answerability classifier
- Testar diferentes LLMs
- **Deliverable:** Generation improvement sobre baseline

**Semana 6: Generation v2**
- Implementar post-validation
- Refinamento de prompt por question type
- Error analysis e fixes
- **Deliverable:** Target RBllm > 0.80

**Semana 7: Integration (Subtask C)**
- Pipeline completo retrieval + generation
- Implementar adaptive retrieval
- Testing em end-to-end
- **Deliverable:** Full RAG funcionando

**Semana 8: Optimization**
- Hyperparameter tuning
- Threshold calibration
- Testing em casos difíceis
- **Deliverable:** Otimizado para test set

**Semana 9: Final Prep**
- Bug fixes
- Documentation
- Submission preparation
- Final testing
- **Deliverable:** Submission ready

### 10.2 Recursos Necessários

**Computacionais:**
- GPU para retrieval models (BGE, Elser)
- API access para LLMs (GPT-4, Claude, ou local Llama)
- Storage para dataset e embeddings cache

**Datasets:**
- MTRAG benchmark (110 conversas, 842 tasks)
- Test set (~200 tasks, fornecido durante eval)

**Libraries:**
- Retrieval: sentence-transformers, pyserini, faiss
- Generation: openai/anthropic APIs ou transformers
- Evaluation: ragas, bert-score, rouge
- NLI: transformers (DeBERTa-NLI)

**Modelos:**
- Retrieval: BGE-large, Elser (via Elastic)
- Reranking: cross-encoder/ms-marco-MiniLM
- Generation: GPT-4 / Claude Sonnet / Llama 3.1 70B+
- NLI validation: DeBERTa-v3-large-mnli

### 10.3 Risk Mitigation

**Risk 1: Retrieval não melhora suficiente**
- Mitigation: Testar múltiplos modelos early
- Fallback: Focus em reranking de alta qualidade

**Risk 2: Generation hallucination**
- Mitigation: Post-validation obrigatória
- Fallback: Temperature baixa + prompt ultra-restritivo

**Risk 3: Latency issues**
- Mitigation: Cache de embeddings, paralelização
- Fallback: Simplificar pipeline se necessário

**Risk 4: Test set muito diferente de dev**
- Mitigation: Robust error handling, não overfit em dev
- Fallback: Ensemble de múltiplas estratégias

---

## 11. Resumo Executivo

### 11.1 Proposta Core

**Retrieval:** Multiple weighted queries (5 tipos) + hybrid search + context-aware reranking
**Generation:** Structured prompt com 4 princípios + post-validation loop
**RAG:** Adaptive retrieval com coverage check + error recovery
**Answerability:** Multi-factor classifier calibrado empiricamente

### 11.2 Principais Inovações

1. Domain shift detection & handling
2. Weighted multi-query com tipos específicos
3. Coverage-based adaptive retrieval
4. Validation loop com regeneration
5. Entity tracking para later turns

### 11.3 Expected Outcomes

**Conservative (mínimo viável):**
- Subtask A: nDCG@5 = 0.50 (+11% vs Elser)
- Subtask B: RBllm = 0.80 (+5% vs GPT-4o)
- Subtask C: RBllm = 0.78 (+3% vs GPT-4o)

**Optimistic (se tudo funcionar):**
- Subtask A: nDCG@5 = 0.55
- Subtask B: RBllm = 0.85
- Subtask C: RBllm = 0.82, top-3 na leaderboard

### 11.4 Competitive Positioning

**Strengths:**
- Abordagem holística cobrindo todos os gaps identificados
- Inovações específicas para MTRAG challenges
- Balance entre complexity e robustness

**Weaknesses:**
- Pipeline complexo = mais pontos de falha
- Latency pode ser issue (mitigado por cache)
- Dependência de múltiplos modelos

**Key Success Factors:**
1. Query rewriting quality (crítico para retrieval)
2. Answerability precision (evitar false positives)
3. Faithfulness enforcement (crítico para RLf)
4. Error handling robustness (generalização para test)

---

## Conclusão

Esta estratégia integra:
- **Insights do paper**: Answerability, later turns, faithfulness
- **Análise dos dados**: Domain shift, context mismatch, score proximity
- **Propostas originais**: Multiple queries, reranking, guardrails
- **Refinamentos**: Adaptive retrieval, validation loop, entity tracking

A abordagem é **modular** (cada componente independente), **data-driven** (decisões baseadas em análise), e **competitiva** (targets realistas mas ambiciosos). O sucesso depende de execução cuidadosa e iteração contínua baseada em error analysis.
