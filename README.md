# Tech Challenge · Fase 3 — Predição e Inteligência Analítica para Alfabetização no Brasil

Pós-graduação em IA para Cientistas de Dados (FIAP/Postech). Esta fase usa a **camada Gold da Fase 2** para
construir um modelo supervisionado de risco educacional e transformar os resultados em inteligência para gestores.

## 1. Contexto do problema
A alfabetização na idade certa é um dos principais indicadores educacionais do país. Conhecer o resultado
de hoje não basta: gestores precisam **antecipar** quais municípios provavelmente não atingirão a meta,
entender o que pesa nesse risco e priorizar apoio.

## 2. Objetivo analítico
Prever, para cada município, se ele **atingirá ou não a meta de alfabetização de 2024**, usando apenas
informação disponível *antes* do resultado: desempenho de 2023, meta de 2024 e contexto territorial.
A classe positiva do modelo é **"em risco" (não atingiu a meta)**.

> **Sobre o "aluno" do enunciado:** a Gold da Fase 2 só contém agregados por município/ano. Por isso a unidade
> de análise é o **município**, não o aluno.

## 3. Base utilizada
| Origem | Conteúdo | Uso |
|---|---|---|
| Gold Fase 2 · `gold_indicador_por_municipio` | % alfabetizados e proficiência por município/ano (2023–2024) | features (2023) e alvo (2024) |
| Gold Fase 2 · `gold_meta_vs_resultado` | meta municipal, gap, `atingiu_meta` | meta de 2024 |
| IBGE (SIDRA): população, área, densidade, PIB | contexto territorial/econômico | enriquecimento |
| INEP: Censo Escolar, nº de alunos avaliados, IDEB | contexto educacional | enriquecimento |
| Atlas do Desenvolvimento Humano, CadÚnico, FUNDEB, PNAD (UF) | contexto socioeconômico | enriquecimento (download manual) |

**Dados efetivos desta versão:** 5.232 municípios com meta e resultado em 2024 (46,1% em risco); 621 não têm medição em 2023.

> ⚠️ **Estado do enriquecimento.** Os resultados abaixo foram obtidos **somente com a Gold**. O código de coleta das
> fontes externas está escrito, mas **nunca foi executado com dados reais** (o ambiente de desenvolvimento
> não tinha acesso às APIs) e os nomes de tabelas/colunas da Base dos Dados e do SIDRA podem precisar de ajuste na primeira execução.
> Só os utilitários de código IBGE têm teste unitário. Rode a coleta (seção 10) e refaça o treino para medir o ganho real das fontes externas (hipótese H4).

### Correção de um bug herdado da Fase 2
Em 2023 não há meta (100% nula), mas a Gold marca `atingiu_meta = False` para todos (`NaN >= 0` é falso).
Aqui o alvo só existe onde há meta **e** resultado; o restante é descartado, não rotulado como fracasso.

## 4. Etapas de modelagem
1. **EDA** (`notebooks/01_eda.ipynb`): distribuições, cobertura, território, correlações e hipóteses.
2. **Dataset temporal sem leakage** (`src/preprocessing/build_dataset.py`): features de 2023 + meta de 2024 → alvo em 2024.
3. **Pipeline scikit-learn único** (`src/modeling/pipeline.py`): engenharia de atributos → imputação → escala/one-hot → modelo.
4. **Treino e validação** (`src/modeling/train.py`) com hold-out, CV estratificada, busca de hiperparâmetros e validação por UF.
5. **Interpretabilidade** (`src/evaluation/`): importância nativa, por permutação e SHAP.
6. **Aplicação**: ranking de risco, clusters regionais e projeção da meta seguinte.

### Como evitamos data leakage
- `percentual_alfabetizado`, `proficiencia_media` de 2024 e `gap_para_meta` **definem o alvo** e estão numa lista proibida; `verifica_leakage()` aborta o treino se aparecerem.
- O hold-out de teste (20%, estratificado) é separado **antes** de qualquer ajuste.
- Toda estatística aprendida (mediana, média/desvio, categorias, **média por UF**) está **dentro do `Pipeline`**, logo é recalculada só com o treino de cada fold.
- O limiar de decisão é escolhido em predições *out-of-fold* do treino, nunca no teste.
- Teste unitário garante que as estatísticas por UF vêm só do treino (`tests/test_pipeline.py`).

### Pré-processamento (dentro do pipeline)
Imputação por mediana **com indicador de ausência** (621 municípios sem histórico) · `log1p` em variáveis de escala (população, PIB, contagens) ·
`StandardScaler` · `OneHotEncoder(handle_unknown="ignore")` · atributos derivados: `gap_ant`, `evolucao_req`, `sem_historico`, `pct_ant_rel_uf`, `meta_alvo_rel_uf`.

## 5. Escolha do algoritmo
Foram comparados Dummy (piso), **Regressão Logística** (baseline interpretável), Random Forest e **Gradient Boosting**,
com `RandomizedSearchCV` (12 iterações, 5 folds, ROC-AUC). O Gradient Boosting teve a maior média em CV e foi escolhido.

| Modelo | CV ROC-AUC | CV F1 (risco) | Treino ROC-AUC |
|---|---|---|---|
| Gradient Boosting | **0,779** ± 0,012 | **0,654** | 0,830 |
| Random Forest | 0,775 ± 0,013 | 0,630 | 0,818 |
| Regressão Logística | 0,774 ± 0,014 | 0,645 | 0,783 |
| Dummy | 0,500 | 0,000 | — |

A vantagem do Gradient Boosting é **menor que o desvio entre folds**: a Regressão Logística é praticamente equivalente
e é muito mais simples de explicar. Em produção, essa seria uma opção legítima.

## 6. Métricas de avaliação (hold-out, 1.047 municípios)
| | Acurácia | Precisão (risco) | Recall (risco) | F1 (risco) | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Limiar 0,50 | 0,726 | 0,738 | 0,629 | 0,679 | **0,806** | 0,785 |
| Limiar 0,35 (via OOF) | 0,691 | 0,616 | **0,872** | **0,722** | 0,806 | 0,785 |
| Baseline Dummy | 0,539 | — | 0,000 | 0,000 | 0,500 | — |
| Baseline Reg. Logística | 0,724 | — | — | 0,681 | 0,804 | — |

O alvo é equilibrado (46% em risco), então a acurácia não engana, mas a métrica que importa ao gestor é o **recall da classe de risco**.
O limiar 0,35 captura 87% dos municípios em risco ao custo de mais falsos alarmes (262 de 564 que atingiram a meta);
a escolha do limiar é uma decisão de política (custo de uma visita de apoio *versus* custo de um município não atendido).

**Generalização.** Gap de ROC-AUC treino→teste de apenas 0,018: não há overfitting no sentido clássico.
**Mas** na validação por UF (`GroupKFold`, o modelo avalia estados que nunca viu), o ROC-AUC cai para **0,56**. Veja as limitações.

## 7. Interpretação dos resultados
Importância por permutação (queda de ROC-AUC no teste ao embaralhar a variável):

| Variável | Queda |
|---|---|
| `sigla_uf` | **0,248** |
| `meta_alvo` | 0,146 |
| `prof_ant` | 0,037 |
| `pct_ant` | 0,031 |
| `regiao` | 0,008 |

O SHAP (`images/shap_summary.png`) confirma: `meta_alvo_rel_uf`, indicadores de RS, MG e BA e `prof_ant` lideram.

## 8. Insights encontrados
1. **O território e a meta explicam mais que o desempenho.** A UF e a meta de 2024 concentram quase todo o poder preditivo; o desempenho passado vem depois.
2. **A taxa de risco varia de 8% (CE) a 92% (RS).** Em RS, muitos municípios têm meta de 80% (teto) com desempenho de ~70%; em CE, ocorre o contrário. Parte do risco é **desenho de meta**, não necessariamente pior alfabetização.
3. **Maior risco previsto:** RS (média 0,90), BA (0,80), PA (0,65), AM (0,59), RJ (0,57). **Menor:** CE, MG, GO, ES.
4. **Dois grupos de municípios** (K-Means, silhouette 0,42): o grupo 0 (3.033 municípios) tem ~74% de alfabetização e 39% de risco; o grupo 1 (2.199) tem ~43% e 56% de risco, com forte presença de Nordeste (50%) e Norte (14%). A separação k=2 é essencialmente "alto × baixo desempenho"; não encontramos subgrupos regionais finos.
5. **Hipóteses:** H1 (gap negativo → mais risco) tem apoio moderado: risco de 30% com gap > +10 p.p. contra 47–60% com gap negativo. **H3 (municípios sem histórico) não se confirmou**: 45% contra 46%. H2 (efeito da UF) confirmada. H4 (contexto socioeconômico) **ainda não testada**.
6. **Ranking nominal:** `reports/risco_municipios_2024.csv` (probabilidades *out-of-fold*, sem que o modelo tenha visto o rótulo do município).

## 9. Limitações do projeto
- **Validação por UF é fraca (ROC-AUC 0,56).** O modelo funciona para monitorar municípios **dentro das UFs conhecidas**; não generaliza para um estado novo, porque aprende o padrão de metas de cada estado.
- **Apenas dois anos de dados.** Existe um único par de anos com alvo válido (2023→2024); não há validação temporal verdadeira (treinar no passado, testar no futuro).
- **Sem fontes externas nesta versão** (ver aviso na seção 3): o efeito do contexto socioeconômico não foi medido.
- **Meta como feature.** Como as metas dependem do estado, o modelo pode estar aprendendo a regra de definição das metas, e não a causa da alfabetização.
- **Municípios pequenos:** percentuais de 0 a 100 sugerem poucos alunos avaliados; o número de alunos (`inep_alunos_avaliados`) ajuda a filtrar ruído, mas ainda não foi coletado.
- **Correlação, não causalidade:** SHAP e importância descrevem o modelo, não efeitos de intervenções.
- Os grupos e rankings são fotografias de 2024, não projeções causais.

## 10. Como reproduzir
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # preencha BD_BILLING_PROJECT_ID (projeto GCP em sandbox, sem billing)

# (opcional, recomendado) coletar fontes externas: IBGE e INEP automaticamente; Atlas/CadÚnico/FUNDEB/PNAD manualmente
python -m src.preprocessing.external_sources           # ver data/external/manual/README.md

python -m src.preprocessing.build_dataset
python -m src.modeling.train
python -m src.evaluation.run_evaluation
pytest -q
```
Sementes fixas (`RANDOM_STATE = 42`) garantem resultados idênticos entre execuções. A Gold usada (`data/raw/`) é pequena e fica versionada.

## 11. Aplicação prática para políticas públicas
- **Priorização:** usar `risco_municipios_2024.csv` para direcionar assistência técnica a municípios com alta probabilidade de não atingir a meta, **dentro da própria UF**.
- **Monitoramento:** reexecutar o pipeline a cada nova edição do indicador e acompanhar a variação do risco.
- **Revisão de metas:** UFs com risco extremo (RS) merecem uma auditoria de **como a meta foi calibrada** antes de se concluir que há um problema de aprendizagem.
- **Cuidado de uso:** o modelo é um instrumento de triagem, não de avaliação de gestores nem de alocação automática de recursos.

## 12. Possíveis evoluções futuras
- Rodar a coleta das fontes externas e testar H4 (ganho de ROC-AUC e SHAP socioeconômico).
- Incluir novas edições do indicador para uma validação temporal real e para prever 2025+ (`monta_projecao` já está pronto para usar as metas futuras).
- Modelar a **distância à meta** (regressão) além da classificação.
- Calibração de probabilidades e curva de decisão com custos reais.
- Incluir XGBoost/LightGBM e modelos hierárquicos por UF.

## Estrutura do repositório
```
data/{raw,external,processed}   notebooks/   src/{preprocessing,modeling,evaluation,visualization}
reports/   images/   docs/   tests/   scripts/   models/   requirements.txt   README.md   .gitignore
```
