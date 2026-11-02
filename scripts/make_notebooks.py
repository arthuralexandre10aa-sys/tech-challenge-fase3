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
("md", "## 3. Território e contexto socioeconômico"),
("code", '''display(Image(plots.risco_por_uf(df)))
display(Image(plots.alfabetizacao_por_regiao(df)))
display(df.groupby("regiao")[[TARGET, "pct_alvo", "meta_alvo"]].mean().round(3))'''),
("md", "**Achado 4 – a taxa de risco varia de ~8% (CE) a ~92% (RS) entre UFs.** Parte disso vem de **como as metas foram fixadas por estado**, não só do desempenho.  \n**Decisões:** (i) **UF e região não entram no modelo**: queremos que ele aprenda com características do município, não com a identidade do estado; (ii) `sigla_uf` fica no dataset apenas para a validação por grupo (`GroupKFold`), que mede a generalização a estados não vistos; (iii) fontes em nível estadual (PNAD/UF) também ficam de fora, porque identificariam a UF indiretamente."),
("md", "## 4. Fontes externas: o que elas dizem sobre o desempenho e sobre o risco"),
("code", '''ext = [c for c in df.columns if c.startswith("ext_")]
print(len(ext), "variáveis externas |", usadas)
print("nulos (maiores):"); display(df[ext].isna().mean().sort_values(ascending=False).head(5).round(3))
c_nivel = df[ext].corrwith(df["pct_alvo"]).sort_values(key=abs, ascending=False).head(10)
c_risco = df[ext].corrwith(df[TARGET]).sort_values(key=abs, ascending=False).head(10)
display(pd.concat({"corr com % alfabetizados 2024": c_nivel.reset_index(drop=True).to_frame().assign(variavel=c_nivel.index),
                   "corr com risco": c_risco.reset_index(drop=True).to_frame().assign(variavel=c_risco.index)}, axis=1).round(3))'''),
("md", "**Achado 5 – contexto explica o NÍVEL, mas quase não explica o RISCO.** As variáveis do IDEB/SAEB (r≈0,55–0,58), frequência escolar, atraso escolar e densidade domiciliar correlacionam com o percentual de alfabetizados; já com o risco de não atingir a meta a correlação máxima é ≈0,18. Isso acontece porque **as metas já são calibradas pelo ponto de partida do município**: um município pobre tem meta menor.  \n**Decisão:** além do modelo de risco, criamos um **modelo do nível de alfabetização** (`src/modeling/factors.py`) para responder à pergunta \"quais fatores mais impactam a alfabetização?\". Atenção: notas do SAEB são outro indicador de aprendizagem, então há uma variante **sem IDEB/SAEB** para isolar fatores de contexto."),
("md", "## 5. Relação entre anos e metas"),
("code", '''display(Image(plots.relacao_ant_alvo(df)))
num = ["pct_ant", "prof_ant", "meta_alvo", "pct_alvo", "prof_alvo", TARGET]
display(Image(plots.correlacoes(df, num)))
display(df[num].corr()[TARGET].round(3))'''),
("md", "**Achado 6 – o desempenho passado é preditivo, mas não decide tudo:** a correlação 2023↔2024 é ≈0,65.  \n**Hipóteses testadas:**\n1. **H1** (gap negativo → mais risco): apoio parcial, ver tabela abaixo.\n2. **H2** (UF tem poder próprio): confirmada na primeira versão; **a UF foi removida de propósito** e o modelo perde generalização entre estados (ver notebook 02).\n3. **H3** (sem histórico se comporta diferente): **não se confirmou** (45,2% vs 46,2%).\n4. **H4** (contexto socioeconômico explica o risco além do desempenho): **apoio parcial**: NSE do FUNDEB e pobreza extrema aparecem entre as 5 variáveis mais importantes, mas depois de `meta_alvo` e `prof_ant`."),
("code", '''df["gap_ant"] = df["pct_ant"] - df["meta_alvo"]
df["faixa_gap"] = pd.cut(df["gap_ant"], [-100, -20, -10, 0, 10, 100])
display(df.groupby("faixa_gap")[TARGET].agg(["mean", "count"]).round(3))   # H1
display(df.groupby(df["pct_ant"].isna())[TARGET].agg(["mean", "count"]).round(3))  # H3'''),
]

mod = [
("md", "# 02 · Modelagem, validação e interpretabilidade\n\nRode antes: `build_dataset`, `train`, `run_evaluation` e `factors` (veja o README)."),
("code", SETUP + '''\nimport json
M = json.loads((REPORTS / "metricas_finais.json").read_text())
R = json.loads((REPORTS / "resumo_avaliacao.json").read_text())
F = json.loads((REPORTS / "fatores_resumo.json").read_text())'''),
("md", "## 1. Comparação de modelos (validação cruzada estratificada, 5 folds, só no treino)"),
("code", '''cmp = pd.read_csv(REPORTS / "comparacao_modelos_cv.csv")
display(cmp.drop(columns="melhores_params").round(4))
print("modelo escolhido:", M["modelo"]); print(M["regra_selecao"])'''),
("md", "O Gradient Boosting tem a maior média de CV, mas a diferença para a Regressão Logística (≈0,004) é **menor que o erro-padrão** da validação. Pela regra do 1 erro-padrão, entre os modelos tecnicamente empatados escolhemos o de **menor distância treino→validação** (`gap_treino_cv`): a Regressão Logística (0,020, contra 0,080 do Gradient Boosting)."),
("md", "## 2. Desempenho no hold-out de teste (separado antes de qualquer ajuste)"),
("code", '''tab = pd.DataFrame({"teste @0.5": M["teste_limiar_0.5"], f"teste @{M['limiar_otimo_oof']:.2f} (limiar via OOF)": M["teste_limiar_otimo"],
                    "treino @0.5": M["treino_limiar_0.5"]}).T
display(tab[["accuracy", "precision_risco", "recall_risco", "f1_risco", "roc_auc", "pr_auc", "brier"]].round(4))
display(pd.DataFrame(M["baselines_teste"]).T[["accuracy", "f1_risco", "roc_auc", "pr_auc"]].round(4))
display(Image(str(IMAGES / "modelo_roc_pr.png"))); display(Image(str(IMAGES / "modelo_matriz_confusao.png")))'''),
("md", "## 3. Generalização e overfitting"),
("code", '''print("gap ROC-AUC treino - teste:", round(M["gap_roc_auc_treino_vs_teste"], 4))
print("CV aleatória (treino), modelo escolhido:", cmp[cmp.modelo == M["modelo"]][["cv_roc_auc", "cv_f1"]].round(4).to_dict("records")[0])
print("CV POR UF (estados não vistos):", {k: round(v, 4) for k, v in M["cv_por_uf_treino"].items()})'''),
("md", "**Leitura honesta:** o gap treino↔teste é pequeno (0,031; era 0,138 na primeira versão, com UF e grades amplas). Quando o modelo é avaliado em **UFs que nunca viu**, o ROC-AUC é **0,638**, bem acima do 0,56 que tínhamos com a Gold sozinha, mas ainda abaixo da validação aleatória (0,742). Parte do ganho vem das fontes externas (SAEB, NSE, pobreza); a diferença restante é a política de metas de cada estado, que o modelo não enxerga sem a UF."),
("md", "## 4. Interpretabilidade: Feature Importance, Permutation Importance e SHAP"),
("code", '''for f in ["modelo_importancia_nativa.png", "modelo_importancia_permutacao.png", "shap_summary.png"]:
    display(Image(str(IMAGES / f)))
display(pd.read_csv(REPORTS / "importancia_permutacao.csv").head(8).round(4))'''),
("md", "`meta_alvo` e `prof_ant` lideram nas três medidas. Em seguida vêm **nota de matemática do SAEB**, **NSE do FUNDEB**, **pobreza extrema** e **envelhecimento** (Atlas). O contexto socioeconômico aparece, mas como complemento ao histórico e à meta."),
("md", "## 5. Municípios de maior risco"),
("code", '''risco = pd.read_csv(REPORTS / "risco_municipios_2024.csv")
display(risco.head(15).round(3))
display(pd.read_csv(REPORTS / "risco_por_uf.csv").head(10))'''),
("md", "As probabilidades vêm de predição *out-of-fold*: nenhum município é previsto por um modelo que viu o próprio rótulo. **Atenção:** sem a UF, o modelo subestima estados com metas rígidas (ex.: RS tem 92% de municípios realmente em risco, e o risco médio previsto é 0,67)."),
("md", "## 6. Regiões com padrões semelhantes (K-Means)"),
("code", '''display(pd.read_csv(REPORTS / "silhouette_por_k.csv"))
perfil = pd.read_csv(REPORTS / "perfil_grupos.csv").set_index("grupo")
display(perfil[["n_municipios", "taxa_risco", "pct_ant", "pct_alvo", "meta_alvo", "ext_atlas_idhm__idhm",
                "ext_atlas_idhm__renda_pc", "ext_atlas_idhm__prop_pobreza", "ext_fundeb__nse",
                "ext_inep_ideb__nota_saeb_matematica"]].T.round(2))
display(Image(str(IMAGES / "cluster_pca.png"))); display(Image(str(IMAGES / "cluster_composicao_regiao.png")))'''),
("md", "Os grupos separam **contexto**, não risco: o grupo 0 (3.046 municípios, 46% Sudeste, 34% Sul) tem IDHM 0,71 e pobreza de 8%; o grupo 1 (2.186, 74% Nordeste) tem IDHM 0,59 e pobreza de 41%. A taxa de risco é parecida (44% vs 49%) porque as metas do grupo 1 são mais baixas. O silhouette (0,288) indica estrutura fraca: tratamos como **segmentação grosseira**, não como tipologia."),
("md", "## 7. Municípios que podem não atingir metas futuras (2025)"),
("code", '''print(R["projecao"])
proj = pd.read_csv(REPORTS / "projecao_risco_meta_2025.csv")
print("municípios com prob. > 70%:", int((proj.prob_risco_proxima_meta > 0.7).sum()), "de", len(proj))
display(proj.head(15).round(3))'''),
("md", "A projeção usa o desempenho de 2024 como base e as metas oficiais de 2025. Pressupõe que a relação aprendida em 2023→2024 se mantém: é um alerta, não uma previsão garantida."),
("md", "## 8. Fatores associados ao nível de alfabetização (regressão)"),
("code", '''comp = pd.read_csv(REPORTS / "fatores_comparacao.csv")
display(comp[["variante", "modelo", "n_variaveis", "r2_cv", "r2_por_uf", "r2_treino", "r2_teste", "mae_teste_pp"]].round(3))
for v in ["contexto", "contexto_sem_desempenho"]:
    print(v, "->", [(d["feature"].replace("ext_", ""), round(d["queda_r2"], 3)) for d in F[v]["top5_fatores"]])
for f in ["fatores_importancia_permutacao_contexto_sem_desempenho.png", "fatores_shap_summary_contexto_sem_desempenho.png",
          "fatores_importancia_permutacao_contexto.png"]:
    display(Image(str(IMAGES / f)))'''),
("md", "**Como ler:**\n- Com todo o contexto, a nota de matemática do SAEB explica sozinha 0,34 de R², mas é **outro indicador de aprendizagem**, então não é um \"fator\" no sentido de política.\n- **Sem IDEB/SAEB**, o contexto explica R²≈0,37 (CV): os fatores principais são **frequência escolar, analfabetismo de 11–14 anos e de 15+ (Atlas), NSE (FUNDEB) e infraestrutura domiciliar (energia)**.\n- Com o histórico de 2023 (`contexto_mais_historico`), R²≈0,58.\n- **Limite importante:** o R² por UF (treinando em alguns estados e testando em outros) é ≈0 ou negativo para as variantes de contexto: as relações valem *dentro* da amostra, mas não se transferem de forma estável entre estados. Os Atlas são do Censo 2010. **São associações, não causas.**"),
]
nb(eda, "01_eda.ipynb")
nb(mod, "02_modelagem.ipynb")
print("notebooks gerados")
