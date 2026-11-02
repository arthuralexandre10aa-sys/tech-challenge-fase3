# Fontes que o grupo precisa baixar manualmente

Coloque os arquivos nesta pasta (CSV ou XLSX). O código padroniza os nomes das colunas
e usa a primeira coluna de código de município que encontrar
(`id_municipio`, `cod_ibge`, `codmun6`, `codmun7`, `codigo_municipio`...).
Códigos de 6 dígitos recebem o dígito verificador automaticamente.

| Arquivo | Fonte | Nível | Onde obter |
|---|---|---|---|
| `atlas_idhm.csv` | Atlas do Desenvolvimento Humano (IDHM, IDHM Educação, renda, longevidade) | município | atlasbrasil.org.br → Download |
| `cadunico.csv` | Cadastro Único (famílias cadastradas, % em extrema pobreza) | município | MDS/VIS DATA ou Base dos Dados |
| `fundeb.csv` | FUNDEB (matrículas ou repasses) | município | FNDE / Tesouro Transparente |
| `pnad_uf.csv` | PNAD Contínua (renda, escolaridade, informalidade) | **UF** (coluna `sigla_uf`) | SIDRA/IBGE |

Somente colunas **numéricas** entram no modelo. Depois de colocar os arquivos:

```bash
python -m src.preprocessing.external_sources --only atlas_idhm cadunico fundeb pnad_uf
python -m src.preprocessing.build_dataset && python -m src.modeling.train && python -m src.evaluation.run_evaluation
```

Fontes ausentes não quebram o pipeline: o status fica em `reports/enriquecimento_status.csv`.
