# Roteiro do vídeo executivo (até 5 min) — reunião com gestores

**Formato:** simular apresentação a secretários/gestores. Slides curtos, sem jargão. Sugestão de divisão entre os integrantes.

| Tempo | Conteúdo | Apoio visual |
|---|---|---|
| 0:00–0:40 | **O problema.** Saber o resultado de ontem não basta: precisamos antecipar quais municípios podem não atingir a meta de alfabetização. | número: 5.232 municípios, 46% em risco |
| 0:40–1:30 | **O que fizemos.** Juntamos os dados da Fase 2 com IBGE, INEP, Atlas e FUNDEB e treinamos um modelo que, com o desempenho de 2023, a meta de 2024 e o contexto do município, prevê quem fica abaixo da meta. **Não usamos o estado como informação.** | fluxo simples: dados → modelo → ranking |
| 1:30–2:30 | **Resultado.** O modelo ordena bem o risco (ROC-AUC 0,73) e, com o limiar ajustado, identifica 76% dos municípios que ficaram abaixo da meta. Escolhemos a regressão logística: é tão boa quanto modelos mais complexos e muito mais fácil de explicar. | `modelo_roc_pr.png` |
| 2:30–3:30 | **Insights.** A meta e o desempenho anterior pesam mais; depois vêm o nível socioeconômico (NSE), a pobreza extrema e a nota de matemática. No nível de alfabetização, frequência escolar e analfabetismo infantil lideram. RS tem 92% dos municípios em risco e CE só 8%, em parte pela **forma como as metas foram definidas**. | `shap_summary.png`, `fatores_importancia_permutacao_contexto_sem_desempenho.png`, `eda_risco_por_uf.png` |
| 3:30–4:20 | **Uso na política pública.** Triagem de municípios para apoio técnico (1.458 com alta probabilidade de não atingir a meta de 2025), monitoramento anual e **revisão da calibração das metas**. | tabela dos 10 municípios de maior risco |
| 4:20–5:00 | **Limites e próximos passos.** Em estados nunca vistos o ROC-AUC cai para 0,64; só dois anos de dados; os fatores são associações, não causas. Próximo: mais anos de dados e CadÚnico. | lista de 3 itens |

**Mensagem final:** "É uma ferramenta de triagem para priorizar apoio, não um veredito sobre gestões."
