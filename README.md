# Tech Challenge · Fase 3 — Predição e Inteligência Analítica para Alfabetização no Brasil

Pós-graduação em IA para Cientistas de Dados (FIAP/Postech). Esta fase usa a **camada Gold da Fase 2**, enriquecida com
fontes externas, para construir um modelo supervisionado de risco educacional e transformar os resultados em inteligência para gestores.

## 1. Contexto do problema
A alfabetização na idade certa é um dos principais indicadores educacionais do país. Conhecer o resultado
de hoje não basta: gestores precisam **antecipar** quais municípios provavelmente não atingirão a meta,
entender o que pesa nesse risco e priorizar apoio.

## 2. Objetivo analítico
Prever, para cada município, se ele **atingirá ou não a meta de alfabetização de 2024**, usando apenas
informação disponível *antes* do resultado: desempenho de 2023, meta de 2024 e contexto territorial, educacional e socioeconômico.
A classe positiva do modelo é **"em risco" (não atingiu a meta)**. Um segundo modelo, de regressão, estima o **nível** de
alfabetização para identificar os fatores associados a ele.

> **Sobre o "aluno" do enunciado:** a Gold da Fase 2 só contém agregados por município/ano. Por isso a unidade
> de análise é o **município**, não o aluno.
>
> **UF e região ficam fora do modelo.** Queremos que o modelo aprenda com características do município, não com a identidade do estado.
> A UF permanece no dataset apenas para a validação por grupo e para relatórios.

## 3. Base utilizada
| Origem | Conteúdo | Situação |
|---|---|---|
| Gold Fase 2 · `gold_indicador_por_municipio` | % alfabetizados e proficiência por município/ano (2023–2024) | usada (features de 2023 e alvo de 2024) |
| Gold Fase 2 · `gold_meta_vs_resultado` | meta municipal, gap | usada (meta de 2024) |
| IBGE/SIDRA: população, área, densidade, PIB | contexto territorial/econômico | usada |
| INEP (Base dos Dados): Censo Escolar, alunos avaliados, IDEB/SAEB | contexto educacional | usada |
| Atlas do Desenvolvimento Humano (Censo 2010) | pobreza, renda, escolaridade, domicílio | usada (download manual) |
| FUNDEB: Nível Socioeconômico (NSE) 2023 | contexto socioeconômico | usada (download manual) |
| Diretório de municípios (Base dos Dados) | capital, região metropolitana | usada (download manual) |
| CadÚnico, PNAD por UF | contexto socioeconômico | **não obtidas.** A PNAD é estadual e codificaria a UF, de qualquer forma |

**Dataset final:** 5.232 municípios com meta e resultado em 2024 (46,1% em risco); 621 não têm medição em 2023.
O modelo recebe **69 colunas candidatas**, sendo 66 externas. Status de cada fonte em `reports/enriquecimento_status.csv`.

### Correção de um bug herdado da Fase 2
Em 2023 não há meta (100% nula), mas a Gold marca `atingiu_meta = False` para todos (`NaN >= 0` é falso).
Aqui o alvo só existe onde há meta **e** resultado; o restante é descartado, não rotulado como fracasso.

## 4. Etapas de modelagem
1. **EDA** (`notebooks/01_eda.ipynb`): distribuições, cobertura, território, correlações e hipóteses.
2. **Dataset temporal sem leakage** (`src/preprocessing/build_dataset.py`): features de 2023 + meta de 2024 → alvo em 2024.
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
- Testes unitários (`tests/test_pipeline.py`, 12 testes) cobrem alvo, leakage, categorias novas, NaN, exclusão de UF e a regra de seleção.

### Pré-processamento (dentro do pipeline)
Imputação por mediana **com indicador de ausência** · `log1p` em variáveis de escala (população, PIB, contagens) ·
`StandardScaler` · `OneHotEncoder(handle_unknown="ignore")` · atributos derivados: `gap_ant`, `evolucao_req`, `sem_historico`, `pib_per_capita` ·
**filtro de correlação** (remove colunas com |r| > 0,90).

### Medidas contra overfitting
Grades de hiperparâmetros restritivas (árvores rasas, folhas ≥ 40, subamostragem), regularização L2 forte na regressão logística,
filtro de correlação e seleção do modelo pela regra do 1 erro-padrão com menor gap treino→validação.
O gap de ROC-AUC treino→teste caiu de **0,138** (versão inicial, com UF) para **0,031**.

## 5. Escolha do algoritmo
Foram comparados Dummy (piso), **Regressão Logística**, Random Forest e Gradient Boosting, com `RandomizedSearchCV`
(20 iterações, 5 folds estratificados, ROC-AUC) e CV feita apenas no conjunto de treino.

| Modelo | CV ROC-AUC | CV F1 (risco) | ROC-AUC treino | Gap treino→CV |
|---|---|---|---|---|
| Gradient Boosting | **0,746** ± 0,018 | 0,637 | 0,825 | 0,080 |
| **Regressão Logística (escolhida)** | 0,742 ± 0,013 | **0,650** | 0,762 | **0,020** |
| Random Forest | 0,722 ± 0,010 | 0,594 | 0,783 | 0,062 |
| Dummy | 0,500 | 0,000 | — | — |

**Regra de seleção:** a vantagem do Gradient Boosting (0,004) é menor que o erro-padrão da validação (margem de 0,008).
Entre os modelos tecnicamente empatados escolhemos o de **menor gap treino→validação**. A decisão é nossa e declarada:
outra regra, como "maior ROC-AUC", escolheria o Gradient Boosting, com mais overfitting e menor interpretabilidade.

## 6. Métricas de avaliação (hold-out, 1.047 municípios)
| | Acurácia | Precisão (risco) | Recall (risco) | F1 (risco) | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Limiar 0,50 | 0,663 | 0,643 | 0,605 | 0,623 | **0,729** | 0,694 |
| Limiar 0,40 (via OOF) | 0,654 | 0,599 | **0,760** | **0,670** | 0,729 | 0,694 |
| Baseline Dummy | 0,539 | — | 0,000 | 0,000 | 0,500 | 0,461 |

O alvo é equilibrado (46% em risco), então a acurácia não engana, mas a métrica de maior interesse para o gestor é o **recall da classe de risco**.
O limiar 0,40 captura 76% dos municípios em risco (367 de 483) ao custo de mais falsos alarmes (246 de 564 que atingiram a meta).
A escolha do limiar é uma decisão de política: custo de uma visita de apoio *versus* custo de um município não atendido.

**Generalização**

| Verificação | ROC-AUC |
|---|---|
| Treino | 0,760 |
| Validação cruzada (treino) | 0,742 |
| Teste (hold-out) | **0,729** (gap treino→teste 0,031) |
| Validação por UF (`GroupKFold`, estados nunca vistos) | **0,638** |

Os resultados são reproduzíveis: uma reexecução independente dos scripts reproduziu os mesmos valores (diferença máxima de 3e-5).

## 7. Interpretação dos resultados
Importância por permutação (queda de ROC-AUC no teste ao embaralhar a variável) e SHAP concordam nas cinco primeiras:

| Variável | Queda (permutação) | |SHAP| médio |
|---|---|---|
| `meta_alvo` | 0,137 | 1,19 |
| `prof_ant` (proficiência 2023) | 0,097 | 0,58 |
| `ext_inep_ideb__nota_saeb_matematica` | 0,090 | 0,56 |
| `ext_fundeb__nse` | 0,059 | 0,54 |
| `ext_atlas_idhm__prop_pobreza_extrema` | 0,041 | abaixo de 0,40 (a 5ª em SHAP é `ext_inep_ideb__projecao`, 0,40) |

Gráficos: `images/modelo_importancia_*.png`, `images/shap_summary.png`.

### Modelo de fatores (nível de alfabetização)
O modelo de risco quase não "enxerga" o contexto socioeconômico, porque **a meta já é calibrada pelo ponto de partida do município**
(as correlações das fontes externas com o risco chegam a no máximo 0,18, mas chegam a 0,58 com o nível de alfabetização).
Por isso criamos um modelo de regressão do `% alfabetizados em 2024` (Gradient Boosting, `src/modeling/factors.py`), com três variantes:

| Variante | R² (CV) | R² (teste) | R² por UF | Maiores fatores (queda de R²) |
|---|---|---|---|---|
| Contexto completo | 0,492 | 0,487 | 0,017 | nota SAEB matemática (0,339), NSE (0,051), taxa de aprovação (0,046) |
| **Contexto sem IDEB/SAEB** | 0,367 | 0,356 | −0,213 | frequência escolar (0,112), analfabetismo 11–14 anos (0,100), NSE (0,061), analfabetismo 15+ (0,037), energia elétrica (0,035) |
| Contexto + histórico 2023 | 0,578 | 0,574 | 0,174 | — |

A nota do SAEB é **outro indicador de aprendizagem**, não um fator de contexto; por isso a variante sem IDEB/SAEB é a que melhor responde
"quais fatores impactam a alfabetização?". Imagens: `images/fatores_*.png`.

## 8. Insights encontrados
1. **Meta e histórico dominam o risco.** `meta_alvo` e `prof_ant` explicam mais que qualquer variável de contexto.
2. **O contexto socioeconômico importa para o nível, e aparece, em segundo plano, no risco.** NSE e pobreza extrema estão entre as 5 variáveis mais importantes do modelo de risco; frequência escolar, analfabetismo infantil e NSE lideram o modelo de fatores.
3. **A taxa de risco varia de 8% (CE) a 92% (RS).** Em RS, muitos municípios têm meta de 80% (o teto) com desempenho de ~70%. Parte do risco é **desenho de meta**, não necessariamente pior alfabetização.
4. **Sem a UF, o modelo subestima estados de metas rígidas:** em RS o risco médio previsto é 0,67 para 92% de municípios realmente em risco; em BA, 0,55 para 81%.
5. **UFs de maior risco previsto:** PA (0,68), RS (0,67), AP (0,66), RJ (0,64), AM (0,61). **Menor:** CE (0,21), MG (0,29), TO, AL, MT.
6. **Dois grupos de municípios** (K-Means, k=2, silhouette 0,288): grupo 0 (3.046 municípios; 46% Sudeste, 34% Sul) com IDHM 0,71, pobreza de 8% e ~69% de alfabetização; grupo 1 (2.186; 74% Nordeste, 14% Norte) com IDHM 0,59, pobreza de 41% e ~49–54%. A taxa de risco é parecida (44% vs 49%), pois as metas do grupo 1 são mais baixas. A estrutura é fraca; é uma **segmentação grosseira**.
7. **Projeção 2025:** 1.458 de 5.352 municípios têm probabilidade acima de 70% de não atingir a meta de 2025 (`reports/projecao_risco_meta_2025.csv`). As maiores médias estaduais: RS (0,77), PA (0,76), RJ e AP (0,71).
8. **Hipóteses:** H1 (gap negativo → mais risco) tem apoio parcial: risco de 30% com gap > +10 p.p. contra 47–60% com gap ≤ 0, mas a relação não é monótona. **H3 (municípios sem histórico) não se confirmou:** 45,2% contra 46,2%. **H4 (contexto socioeconômico) tem apoio parcial,** conforme os itens 2 e 6.
9. **Ranking nominal:** `reports/risco_municipios_2024.csv` (probabilidades *out-of-fold*, sem que o modelo tenha visto o rótulo do município).

## 9. Limitações do projeto
- **Generalização entre estados é moderada (ROC-AUC 0,638 por UF).** O modelo é mais útil para monitorar municípios do que para prever um estado novo, e a política de metas de cada estado continua influenciando o resultado via `meta_alvo`.
- **Apenas dois anos de dados.** Há um único par com alvo válido (2023→2024); não existe validação temporal verdadeira (treinar no passado, testar no futuro), e a projeção de 2025 assume que a relação aprendida se mantém.
- **O modelo de risco é modesto** (ROC-AUC 0,729): com limiar 0,40 ainda erra 116 municípios em risco e gera 246 falsos alarmes no teste.
- **Fatores de contexto não se transferem entre estados** (R² por UF ≈ 0 ou negativo nas variantes de contexto), e o Atlas é do Censo 2010.
- **Fontes não usadas:** CadÚnico não foi obtido; PNAD por UF não entra, por codificar a UF.
- **Municípios pequenos:** percentuais extremos sugerem poucos alunos avaliados; `qtd_alunos_avaliados` tem 11,9% de ausências.
- **Clusters fracos** (silhouette 0,288).
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
pytest -q
```
Sementes fixas (`RANDOM_STATE = 42`) garantem resultados idênticos entre execuções. Os notebooks são gerados por `scripts/make_notebooks.py`.

## 11. Aplicação prática para políticas públicas
- **Priorização:** usar `risco_municipios_2024.csv` e a projeção de 2025 para direcionar assistência técnica a municípios de alta probabilidade de risco.
- **Monitoramento:** reexecutar o pipeline a cada nova edição do indicador e acompanhar a variação do risco.
- **Revisão de metas:** UFs de risco extremo (RS) merecem uma auditoria de **como a meta foi calibrada** antes de se concluir que há um problema de aprendizagem.
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
