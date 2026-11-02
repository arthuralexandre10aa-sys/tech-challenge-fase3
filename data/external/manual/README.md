# Fontes baixadas manualmente

Coloque os arquivos **com o nome original** nesta pasta (o código reconhece os padrões abaixo).
Códigos de município de 6 ou 7 dígitos são aceitos; séries com vários anos usam o dado mais recente **até 2023**.

| Fonte | Arquivo aceito | Nível | O que usamos |
|---|---|---|---|
| Atlas do Desenvolvimento Humano (Base dos Dados, `mundo_onu_adh.municipio`) | `mundo_onu_adh_municipio.csv` ou `atlas_idhm.csv` | município | 41 de 230 colunas (IDHM, renda, pobreza, escolaridade, frequência escolar, saneamento) — dados do Censo 2010 |
| FUNDEB — Nível Socioeconômico (NSE) por ente federado, 2023 | `NSE_Entes_Federados_Fundeb2023.xlsx` ou `fundeb.xlsx` | município (linhas de UF são ignoradas) | `nse` (o ponderador é redundante: correlação −1,00 com o NSE) |
| Diretório de municípios (Base dos Dados, `br_bd_diretorios_brasil.municipio`) | `br_bd_diretorios_brasil_municipio.csv` ou `diretorio_municipios.csv` | município | `capital_uf`, `em_regiao_metropolitana` |
| Cadastro Único (opcional) | `cadunico.csv` / `.xlsx` | município | colunas numéricas |
| PNAD por UF (opcional) | `pnad_uf.csv` | UF (coluna `sigla_uf`) | **não entra no modelo** (constante por UF = "código da UF"); só para análise |

Por que o diretório entra só com dois sinalizadores: microrregião, mesorregião, centroide e Amazônia Legal
identificam o estado indiretamente, o que contraria a decisão de não usar UF no modelo.
O nome do município já vem na Gold da Fase 2, então a tabela não é necessária para isso.

Depois de colocar os arquivos:

```bash
python -m src.preprocessing.external_sources --only atlas_idhm fundeb diretorio_municipios
python -m src.preprocessing.build_dataset && python -m src.modeling.train && python -m src.evaluation.run_evaluation
```

Fontes ausentes não quebram o pipeline: o status fica em `reports/enriquecimento_status.csv`.
