"""Gera os notebooks (01_eda e 02_modelagem) a partir de células em texto."""
import nbformat as nbf
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / "notebooks"
SETUP = '''import sys, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
import pandas as pd, numpy as np
from IPython.display import Image, display
from src.config import *
from src.preprocessing.build_dataset import carrega_gold, constroi, colunas_features
from src.visualization import plots
pd.set_option("display.width", 160); pd.set_option("display.max_columns", 30)'''


def nb(cells, nome):
    n = nbf.v4.new_notebook()
    n.cells = [nbf.v4.new_markdown_cell(c) if t == "md" else nbf.v4.new_code_cell(c) for t, c in cells]
    n.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nbf.write(n, NB / nome)


eda = [
("md", "# 01 · Análise exploratória e entendimento do problema\n\n**Pergunta:** quais municípios têm risco de **não atingir a meta** de alfabetização, e por quê?\n\nA EDA apoia diretamente as decisões de modelagem: cada achado abaixo termina com a decisão que ele motiva."),
("code", SETUP),
("md", "## 1. A Gold da Fase 2: cobertura e qualidade"),
("code", '''ind, mv = carrega_gold()
print("indicador:", ind.shape, "| meta_vs_resultado:", mv.shape)
cob = mv.groupby("ano").agg(municipios=("id_municipio", "nunique"),
                            meta_nula=("meta_percentual", lambda s: s.isna().mean()),
                            atingiu_meta_True=("atingiu_meta", "sum"))
display(cob)'''),
("md", "**Achado 1 – bug herdado da Fase 2.** Em 2023 não existe meta (100% nula), mas `atingiu_meta` aparece como `False` para todos os municípios, porque `NaN >= 0` é falso. Esse rótulo é inválido.\n\n**Decisão:** o alvo só é definido onde há meta *e* resultado (2024). Linhas sem meta são descartadas, não rotuladas como fracasso."),
("code", '''t = ind.groupby("ano")[["percentual_alfabetizado", "proficiencia_media"]].describe().T
display(t.round(2))
print("nulos:", ind.isna().sum().to_dict())
display(Image(plots.distribuicao_alfabetizacao(ind)))'''),
("md", "## 2. Construção do alvo (sem leakage)\n\nDesenho temporal: **desempenho de 2023 + meta de 2024 → atingiu a meta em 2024?**  \nO `percentual_alfabetizado` e a proficiência de 2024 definem o alvo, então ficam **fora** das features."),
("code", '''df, usadas = constroi()
print(df.shape, "| fontes externas:", usadas or "nenhuma ainda")
print("taxa de risco (não atingiu):", round(df[TARGET].mean(), 3))
print("municípios sem histórico em 2023:", int(df["pct_ant"].isna().sum()))
print("features candidatas:", colunas_features(df))'''),
("md", "**Achado 2 – classes equilibradas** (≈46% em risco): acurácia não engana, mas reportamos F1, ROC-AUC e PR-AUC com foco no recall da classe de risco.  \n**Achado 3 – 621 municípios sem histórico:** imputação com mediana **mais um indicador de ausência**, dentro do Pipeline."),
("md", "## 3. Território: a UF explica muito do alvo"),
("code", '''display(Image(plots.risco_por_uf(df)))
display(Image(plots.alfabetizacao_por_regiao(df)))
display(df.groupby("regiao")[[TARGET, "pct_alvo", "meta_alvo"]].mean().round(3))'''),
("md", "**Achado 4 – a taxa de risco varia de ~8% a ~92% entre UFs.** Parte disso vem de **como as metas foram fixadas por estado**, não só do desempenho.  \n**Decisões:** (i) manter `sigla_uf`, mas criar variáveis *relativas à UF* (`pct_ant_rel_uf`, `meta_alvo_rel_uf`) dentro do Pipeline; (ii) tratar a validação **por UF** como teste de robustez, porque a validação aleatória é otimista."),
("md", "## 4. Relação entre anos e metas"),
("code", '''display(Image(plots.relacao_ant_alvo(df)))
num = ["pct_ant", "prof_ant", "meta_alvo", "pct_alvo", "prof_alvo", TARGET]
display(Image(plots.correlacoes(df, num)))
display(df[num].corr()[TARGET].round(3))'''),
("md", "**Achado 5 – o desempenho passado é preditivo, mas não decide tudo:** a correlação 2023↔2024 é ≈0,64.  \n**Hipóteses analíticas para a modelagem:**\n1. H1: municípios com `gap_ant = pct_ant − meta` muito negativo têm maior risco.\n2. H2: a UF (política de metas) tem poder preditivo próprio, além do desempenho.\n3. H3: municípios sem histórico se comportam de forma distinta (`sem_historico`).\n4. H4: contexto socioeconômico (IBGE, Atlas, CadÚnico) explica o risco além do desempenho passado — **a testar quando as fontes externas estiverem coletadas**."),
("code", '''df["gap_ant"] = df["pct_ant"] - df["meta_alvo"]
df["faixa_gap"] = pd.cut(df["gap_ant"], [-100, -20, -10, 0, 10, 100])
display(df.groupby("faixa_gap")[TARGET].agg(["mean", "count"]).round(3))   # H1
display(df.groupby(df["pct_ant"].isna())[TARGET].agg(["mean", "count"]).round(3))  # H3'''),
]

mod = [
("md", "# 02 · Modelagem, validação e interpretabilidade\n\nRode antes: `python -m src.preprocessing.build_dataset`, `python -m src.modeling.train` e `python -m src.evaluation.run_evaluation`."),
("code", SETUP + '''\nimport json
M = json.loads((REPORTS / "metricas_finais.json").read_text())
R = json.loads((REPORTS / "resumo_avaliacao.json").read_text())'''),
("md", "## 1. Comparação de modelos (validação cruzada estratificada, 5 folds, só no treino)"),
("code", '''cmp = pd.read_csv(REPORTS / "comparacao_modelos_cv.csv")
display(cmp.drop(columns="melhores_params").round(4))
print("modelo escolhido:", M["modelo"])'''),
("md", "O Gradient Boosting vence por margem pequena sobre a Regressão Logística. A diferença é menor que o desvio entre folds, então **o modelo simples é praticamente tão bom**, o que é um sinal de que o sinal está concentrado em poucas variáveis."),
("md", "## 2. Desempenho no hold-out de teste (separado antes de qualquer ajuste)"),
("code", '''tab = pd.DataFrame({"teste @0.5": M["teste_limiar_0.5"], f"teste @{M['limiar_otimo_oof']:.2f} (limiar via OOF)": M["teste_limiar_otimo"],
                    "treino @0.5": M["treino_limiar_0.5"]}).T
display(tab[["accuracy", "precision_risco", "recall_risco", "f1_risco", "roc_auc", "pr_auc", "brier"]].round(4))
display(pd.DataFrame(M["baselines_teste"]).T[["accuracy", "f1_risco", "roc_auc", "pr_auc"]].round(4))
display(Image(str(IMAGES / "modelo_roc_pr.png"))); display(Image(str(IMAGES / "modelo_matriz_confusao.png")))'''),
("md", "## 3. Generalização e overfitting"),
("code", '''print("gap ROC-AUC treino - teste:", round(M["gap_roc_auc_treino_vs_teste"], 4))
print("CV aleatória (treino):", cmp.iloc[0][["cv_roc_auc", "cv_f1"]].round(4).to_dict())
print("CV POR UF (estados não vistos):", {k: round(v, 4) for k, v in M["cv_por_uf_treino"].items()})'''),
("md", "**Leitura honesta:** o gap treino↔teste é pequeno (sem overfitting no sentido clássico), **mas o ROC-AUC cai para ≈0,56 quando o modelo é avaliado em UFs que nunca viu**. Ou seja: o modelo aprende em grande parte *o padrão de metas de cada estado*. Ele é útil para **monitorar municípios dentro de UFs conhecidas**, mas não deve ser usado para prever um estado novo."),
("md", "## 4. Interpretabilidade: Feature Importance, Permutation Importance e SHAP"),
("code", '''for f in ["modelo_importancia_nativa.png", "modelo_importancia_permutacao.png", "shap_summary.png"]:
    display(Image(str(IMAGES / f)))
display(pd.read_csv(REPORTS / "importancia_permutacao.csv").head(8).round(4))'''),
("md", "A **UF** e a **meta de 2024** concentram quase todo o poder preditivo; depois vêm proficiência e percentual de 2023. Regiões e variáveis derivadas contribuem pouco."),
("md", "## 5. Municípios de maior risco"),
("code", '''risco = pd.read_csv(REPORTS / "risco_municipios_2024.csv")
display(risco.head(15).round(3))
display(pd.read_csv(REPORTS / "risco_por_uf.csv").head(10))'''),
("md", "As probabilidades vêm de predição *out-of-fold*: nenhum município é previsto por um modelo que viu o próprio rótulo."),
("md", "## 6. Regiões com padrões semelhantes (K-Means)"),
("code", '''display(pd.read_csv(REPORTS / "silhouette_por_k.csv"))
display(pd.read_csv(REPORTS / "perfil_grupos.csv"))
display(Image(str(IMAGES / "cluster_pca.png"))); display(Image(str(IMAGES / "cluster_composicao_regiao.png")))'''),
("md", "## 7. Municípios que podem não atingir metas futuras"),
("code", '''print(R["projecao"])
p = REPORTS / "projecao_risco_meta_2025.csv"
if p.exists():
    display(pd.read_csv(p).head(15).round(3))'''),
]
nb(eda, "01_eda.ipynb")
nb(mod, "02_modelagem.ipynb")
print("notebooks gerados")
