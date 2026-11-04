# Tech Challenge · Fase 3 — Predição e Inteligência Analítica para Alfabetização no Brasil

Pós-graduação em IA para Cientistas de Dados (FIAP/Postech). Esta fase usa a **camada Gold da Fase 2**, enriquecida com
fontes externas, para construir um modelo supervisionado de risco educacional e transformar os resultados em inteligência para gestores.

## 1. Contexto do problema
A alfabetização na idade certa é um dos principais indicadores educacionais do país. Conhecer o resultado
de hoje não basta: gestores precisam **antecipar** quais municípios provavelmente não atingirão a meta,
entender o que pesa nesse risco e priorizar apoio.

## 2. Objetivo analítico
Prever, para cada município, se ele **atingirá ou não a meta de alfabetização de 2024**, usando apenas
informação disponível *antes* do resultado: desempenho de 2023 e contexto territorial, educacional e socioeconômico.
**A meta de 2024 não é uma variável do modelo** (ela define o alvo e carrega a política de metas de cada estado; veja a ablação na seção 7).
A classe positiva do modelo é **"em risco" (não atingiu a meta)**. Um segundo modelo, de regressão, estima o **nível** de
alfabetização para identificar os fatores associados a ele.

> **Sobre o "aluno" do enunciado:** a Gold da Fase 2 só contém agregados por município/ano. Por isso a unidade
> de análise é o **município**, não o aluno.
>
> **UF, região e meta ficam fora do modelo.** Queremos que o modelo aprenda com características do município, não com a identidade
> do estado nem com a regra de definição das metas. A UF permanece no dataset apenas para a validação por grupo e para relatórios.

## 3. Base utilizada
| Origem | Conteúdo | Situação |
|---|---|---|
| Gold Fase 2 · `gold_indicador_por_municipio` | % alfabetizados e proficiência por município/ano (2023–2024) | usada (features de 2023 e alvo de 2024) |
| Gold Fase 2 · `gold_meta_vs_resultado` | meta municipal | usada para **definir o alvo** (não é feature) |
| IBGE/SIDRA: população, área, densidade, PIB | contexto territorial/econômico | usada |
| INEP (Base dos Dados): Censo Escolar, alunos avaliados, IDEB/SAEB | contexto educacional | usada |
| Atlas do Desenvolvimento Humano (Censo 2010) | pobreza, renda, escolaridade, domicílio | usada (download manual) |
| FUNDEB: Nível Socioeconômico (NSE) 2023 | contexto socioeconômico | usada (download manual) |
| Diretório de municípios (Base dos Dados) | capital, região metropolitana | usada (download manual) |
| CadÚnico, PNAD por UF | contexto socioeconômico | **não obtidas.** A PNAD é estadual e codificaria a UF, de qualquer forma |

**Dataset final:** 5.232 municípios com meta e resultado em 2024 (46,1% em risco); 621 não têm medição em 2023.
O modelo recebe **68 colunas candidatas**, sendo 66 externas. Status de cada fonte em `reports/enriquecimento_status.csv`.

### Correção de um bug herdado da Fase 2
Em 2023 não há meta (100% nula), mas a Gold marca `atingiu_meta = False` para todos (`NaN >= 0` é falso).
Aqui o alvo só existe onde há meta **e** resultado; o restante é descartado, não rotulado como fracasso.

## 4. Etapas de modelagem
1. **EDA** (`notebooks/01_eda.ipynb`): distribuições, cobertura, território, correlações e hipóteses.
2. **Dataset temporal sem leakage** (`src/preprocessing/build_dataset.py`): features de 2023 + contexto → alvo em 2024 (resultado de 2024 abaixo da meta de 2024).
3. **Pipeline scikit-learn único** (`src/modeling/pipeline.py`): atributos derivados → imputação → escala/one-hot → filtro de correlação → modelo.
4. **Treino e validação** (`src/modeling/train.py`): hold-out, CV estratificada, busca de hiperparâmetros e validação por UF.
5. **Interpretabilidade** (`src/evaluation/`): importância nativa, por permutação e SHAP.
6. **Aplicação**: ranking de risco, clusters, projeção da meta de 2025 e modelo de fatores (`src/modeling/factors.py`).

### Como evitamos data leakage
- `percentual_alfabetizado`, `proficiencia_media` de 2024 e `gap_para_meta` **definem o alvo** e estão numa lista proibida; `verifica_leakage()` aborta o treino se aparecerem.
- O hold-out de teste (20%, estratificado) é separado **antes** de qualquer ajuste.
- Toda estatística aprendida (mediana, média/desvio, categorias, filtro de correlação) está **dentro do `Pipeline`**, logo é recalculada só com o treino de cada fold.
- O limiar de decisão é escolhido em predições *out-of-fold* do treino, nunca no teste.
- Fontes manuais usam o dado mais recente **até 2023**.
- A meta de 2024 (`meta_alvo`) fica fora das features (`config.USAR_META_NO_MODELO = False`).
- Testes unitários (`tests/test_pipeline.py`, 13 testes) cobrem alvo, leakage, categorias novas, NaN, exclusão de UF e a regra de seleção.

### Pré-processamento (dentro do pipeline)
Imputação por mediana **com indicador de ausência** · `log1p` em variáveis de escala (população, PIB, contagens) ·
`StandardScaler` · `OneHotEncoder(handle_unknown="ignore")` · atributos derivados: `pib_per_capita` (e `gap_ant`/`evolucao_req`/`sem_historico`, que só existem quando a meta está disponível) ·
**filtro de correlação** (remove colunas com |r| > 0,90).

### Medidas contra overfitting
Grades de hiperparâmetros restritivas (árvores rasas, folhas ≥ 40, subamostragem), regularização L2 forte na regressão logística,
filtro de correlação e seleção do modelo pela regra do 1 erro-padrão com menor gap treino→validação.
O gap de ROC-AUC treino→teste é **0,092**: era 0,138 na primeira versão (com UF) e 0,031 com a meta como variável. Sem a meta,
o sinal fica mais fraco e mais difuso, e o modelo escolhido (Gradient Boosting) tem mais capacidade de memorizar. Tratamos isso como limitação (seção 9).

## 5. Escolha do algoritmo
Foram comparados Dummy (piso), Regressão Logística, Random Forest e **Gradient Boosting**, com `RandomizedSearchCV`
(20 iterações, 5 folds estratificados, ROC-AUC) e CV feita apenas no conjunto de treino.

| Modelo | CV ROC-AUC | CV F1 (risco) | ROC-AUC treino | Gap treino→CV |
|---|---|---|---|---|
| **Gradient Boosting (escolhido)** | **0,720** ± 0,018 | **0,618** | 0,810 | 0,089 |
| Random Forest | 0,705 ± 0,007 | 0,575 | 0,766 | 0,061 |
| Regressão Logística | 0,699 ± 0,011 | 0,601 | 0,722 | 0,023 |
| Dummy | 0,500 | 0,000 | — | — |

**Regra de seleção:** 1 erro-padrão (margem de 0,008) e, entre os modelos dentro dela, o de menor gap treino→validação. Sem a meta, **só o
Gradient Boosting fica dentro da margem**: a diferença para a Regressão Logística (0,021 na CV e 0,04 no teste) é maior que o erro da validação.
Com a meta como variável, a Regressão Logística empatava com ele e era a escolhida; sem a meta, a relação deixa de ser quase linear e o modelo de árvores passa a compensar.

## 6. Métricas de avaliação (hold-out, 1.047 municípios)
| | Acurácia | Precisão (risco) | Recall (risco) | F1 (risco) | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Limiar 0,50 | 0,643 | 0,631 | 0,545 | 0,584 | **0,709** | 0,671 |
| Limiar 0,32 (via OOF) | 0,607 | 0,547 | **0,876** | **0,673** | 0,709 | 0,671 |
| Baseline Dummy | 0,539 | — | 0,000 | 0,000 | 0,500 | 0,461 |
| Baseline Reg. Logística | — | — | — | — | 0,670 | — |

O alvo é equilibrado (46% em risco), então a acurácia não engana, mas a métrica de maior interesse para o gestor é o **recall da classe de risco**.
O limiar 0,32 captura 88% dos municípios em risco (423 de 483) ao custo de mais falsos alarmes (351 de 564 que atingiram a meta).
A escolha do limiar é uma decisão de política: custo de uma visita de apoio *versus* custo de um município não atendido.

**Generalização**

| Verificação | ROC-AUC |
|---|---|
| Treino | 0,801 |
| Validação cruzada (treino) | 0,720 |
| Teste (hold-out) | **0,709** (gap treino→teste 0,092) |
| Validação por UF (`GroupKFold`, estados nunca vistos) | **0,568** |

Reexecuções dos scripts reproduzem os mesmos valores (sementes fixas).

## 7. Interpretação dos resultados
Importância por permutação (queda de ROC-AUC no teste ao embaralhar a variável) e SHAP concordam nas cinco primeiras:

| Variável | Queda (permutação) | \|SHAP\| médio |
|---|---|---|
| `ext_inep_ideb__nota_saeb_matematica` | 0,048 | 0,26 |
| `prof_ant` (proficiência 2023) | 0,031 | 0,21 |
| `ext_inep_ideb__taxa_aprovacao` | 0,024 | 0,23 |
| `ext_fundeb__nse` | 0,017 | 0,16 |
| `ext_atlas_idhm__taxa_ocupados_sem_carteira` | 0,015 | 0,13 |

Gráficos: `images/modelo_importancia_*.png`, `images/shap_summary.png`. Sem a meta, a importância se espalha: nenhuma variável domina,
e o **contexto educacional e socioeconômico (SAEB, aprovação, NSE, informalidade do trabalho) passa a liderar**.

### Ablação: por que a meta ficou de fora (`src/modeling/ablation.py`)
Mesma divisão, mesmo pipeline e mesma busca. ROC-AUC no teste (entre parênteses, validação por UF):

| Cenário | Regressão logística | Gradient Boosting |
|---|---|---|
| Referência **com** `meta_alvo` (não é o modelo final) | 0,729 (0,638) | 0,741 (0,599) |
| **Modelo final**: sem a meta, com `prof_ant` e `pct_ant` | 0,670 (0,579) | **0,709** (0,568) |
| Só contexto (sem meta e sem histórico de 2023) | 0,665 (0,578) | 0,688 (0,571) |
| Contexto sem IDEB/SAEB (só fatores socioeconômicos e estruturais) | 0,627 (0,526) | 0,656 (0,521) |

Leitura: **(i)** a meta custa ao modelo 0,03 de ROC-AUC (Gradient Boosting) a 0,06 (logística), e em troca ele deixa de aprender a política de metas
de cada estado; **(ii)** o histórico de 2023 agrega pouco (0,02) quando o SAEB está no modelo; **(iii)** o contexto sozinho prevê o risco acima do acaso
(0,63 a 0,69), mas **esse sinal quase não atravessa estados** (0,52 por UF sem SAEB), ou seja, parte dele é geografia, e não efeito socioeconômico
independente. Pequenas diferenças (±0,005) entre execuções do mesmo cenário vêm da ordem das colunas na amostragem do Gradient Boosting.

### Modelo de fatores (nível de alfabetização)
O risco depende de uma meta que é calibrada pelo ponto de partida do município; por isso, para responder "quais fatores impactam a alfabetização?",
modelamos o **`% alfabetizados em 2024`** (Gradient Boosting, `src/modeling/factors.py`), com três variantes:

| Variante | R² (CV) | R² (teste) | R² por UF | Maiores fatores (queda de R²) |
|---|---|---|---|---|
| Contexto completo | 0,492 | 0,487 | 0,017 | nota SAEB matemática (0,339), NSE (0,051), taxa de aprovação (0,046) |
| **Contexto sem IDEB/SAEB** | 0,367 | 0,356 | −0,213 | frequência escolar (0,112), analfabetismo 11–14 anos (0,100), NSE (0,061), analfabetismo 15+ (0,037), energia elétrica (0,035) |
| Contexto + histórico 2023 | 0,578 | 0,574 | 0,174 | — |

A nota do SAEB é **outro indicador de aprendizagem**, não um fator de contexto; por isso a variante sem IDEB/SAEB é a que melhor responde
"quais fatores impactam a alfabetização?". Imagens: `images/fatores_*.png`.

## 8. Insights encontrados
1. **Sem a meta, o contexto passa a pesar mais.** SAEB, taxa de aprovação, NSE do FUNDEB e informalidade do trabalho (Atlas) acompanham o histórico de 2023 entre as cinco principais variáveis. A hipótese H4 (contexto explica o risco além do desempenho passado) tem apoio parcial, com a ressalva do item 6.
2. **A taxa de risco varia de 8% (CE) a 92% (RS)**, e as metas médias vão de 36% (SE) a 79% (CE). Em RS a meta média é 72% e o resultado ficou em 54%; em CE a meta é 79% e o resultado, 90%. Parte do risco é **desenho de meta**, não necessariamente pior alfabetização.
3. **Sem a meta, o modelo subestima estados de metas rígidas:** em RS o risco médio previsto é 0,68 para 92% de municípios realmente em risco; em BA, 0,61 para 81%. Esse é o custo de não aprender a política de metas de cada estado.
4. **UFs de maior risco previsto:** RS (0,68), AP (0,65), PA (0,63), BA (0,61), AM (0,57). **Menor:** CE (0,21), MG e GO (0,32), MT e AL (0,37).
5. **Dois grupos de municípios** (K-Means, k=2, silhouette 0,289): grupo 1 (3.038 municípios; 46% Sudeste, 34% Sul) com IDHM 0,71, pobreza de 8% e ~69% de alfabetização; grupo 0 (2.194; 74% Nordeste, 14% Norte) com IDHM 0,59, pobreza de 41% e ~50–54%. A taxa de risco é parecida (44% vs 49%), pois as metas do grupo 0 são mais baixas. A estrutura é fraca; é uma **segmentação grosseira**.
6. **O sinal do contexto é em boa parte regional.** Sem SAEB e sem histórico, o ROC-AUC é 0,63 no teste e 0,53 entre estados: o contexto separa regiões mais do que municípios dentro de uma mesma região.
7. **Projeção 2025:** 649 de 5.352 municípios têm probabilidade acima de 70% de risco, usando o desempenho de 2024 e o contexto (**sem as metas de 2025**, que não entram no modelo). As maiores médias estaduais: RS (0,67), PA (0,67), AP (0,66), BA (0,62), AM (0,57). Deve ser lida como um **índice de vulnerabilidade**, não como a probabilidade exata de não cumprir a meta de 2025 (`reports/projecao_risco_meta_2025.csv`).
8. **Hipóteses:** H1 (gap negativo → mais risco) tem apoio parcial, avaliada na EDA (risco de 30% com gap > +10 p.p. contra 47–60% com gap ≤ 0, mas a relação não é monótona). **H3 (municípios sem histórico) não se confirmou:** 45,2% contra 46,2%. **H4 tem apoio parcial,** conforme os itens 1 e 6.
9. **Ranking nominal:** `reports/risco_municipios_2024.csv` (probabilidades *out-of-fold*, sem que o modelo tenha visto o rótulo do município).

## 9. Limitações do projeto
- **Poder preditivo moderado:** ROC-AUC de 0,709 no teste. Com limiar 0,32 ainda erra 60 municípios em risco e gera 351 falsos alarmes (limiar 0,50: 220 não detectados e 154 falsos alarmes).
- **Gap treino→teste de 0,092** (0,801 no treino): há algum overfitting residual no Gradient Boosting; a Regressão Logística tem menos (0,022 na CV), mas perde 0,04 de ROC-AUC no teste.
- **Generalização entre estados é fraca (ROC-AUC 0,568 por UF).** O modelo é mais útil para monitorar municípios do que para prever um estado novo. Retirar a meta tirou parte da política estadual, mas o contexto (Atlas, FUNDEB) ainda carrega geografia.
- **Sem a meta, o modelo estima vulnerabilidade, e não o cumprimento de uma meta específica:** a mesma probabilidade vale para metas diferentes, e a projeção de 2025 não usa as metas de 2025.
- **Apenas dois anos de dados.** Há um único par com alvo válido (2023→2024); não existe validação temporal verdadeira (treinar no passado, testar no futuro).
- **Fatores de contexto não se transferem entre estados** (R² por UF ≈ 0 ou negativo nas variantes de contexto), e o Atlas é do Censo 2010.
- **Fontes não usadas:** CadÚnico não foi obtido; PNAD por UF não entra, por codificar a UF.
- **Municípios pequenos:** percentuais extremos sugerem poucos alunos avaliados; `qtd_alunos_avaliados` tem 11,9% de ausências.
- **Clusters fracos** (silhouette 0,289).
- **Correlação, não causalidade:** SHAP, permutação e importâncias descrevem o modelo, não efeitos de intervenções.

## 10. Como reproduzir
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # preencha BD_BILLING_PROJECT_ID (projeto GCP em sandbox, sem billing)

# fontes externas: IBGE e INEP automáticas; Atlas, FUNDEB e diretório manuais (ver data/external/manual/README.md)
python -m src.preprocessing.external_sources

python -m src.preprocessing.build_dataset
python -m src.modeling.train
python -m src.evaluation.run_evaluation
python -m src.modeling.factors
python -m src.modeling.ablation
pytest -q
```
Sementes fixas (`RANDOM_STATE = 42`) garantem resultados idênticos entre execuções. Os notebooks são gerados por `scripts/make_notebooks.py`.

## 11. Aplicação prática para políticas públicas
- **Priorização:** usar `risco_municipios_2024.csv` e o índice de vulnerabilidade de 2025 para direcionar assistência técnica aos municípios de maior risco.
- **Monitoramento:** reexecutar o pipeline a cada nova edição do indicador e acompanhar a variação do risco.
- **Revisão de metas:** como o modelo não usa a meta, a distância entre o risco previsto e o risco observado (RS: 0,68 previsto contra 92% real) aponta onde a meta é mais rígida e merece auditoria antes de se concluir que há um problema de aprendizagem.
- **Prioridades de contexto:** frequência escolar, analfabetismo infantil e vulnerabilidade socioeconômica (NSE) são os fatores mais associados ao nível de alfabetização e orientam políticas de apoio.
- **Cuidado de uso:** o modelo é um instrumento de triagem, não de avaliação de gestores nem de alocação automática de recursos.

## 12. Possíveis evoluções futuras
- Incluir novas edições do indicador para uma validação temporal real.
- Obter CadÚnico e dados mais recentes que o Censo 2010.
- Calibração de probabilidades e curva de decisão com custos reais.
- Incluir XGBoost/LightGBM e modelos hierárquicos por UF.

## Estrutura do repositório
```
data/{raw,external,processed}   notebooks/   src/{preprocessing,modeling,evaluation,visualization}
reports/   images/   docs/   tests/   scripts/   models/   requirements.txt   README.md   .gitignore
```
