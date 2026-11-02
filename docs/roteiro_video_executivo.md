# Roteiro do vídeo executivo (até 5 min) — reunião com gestores

**Formato:** simular apresentação a secretários/gestores. Slides curtos, sem jargão. Sugestão de divisão entre os integrantes.

| Tempo | Conteúdo | Apoio visual |
|---|---|---|
| 0:00–0:40 | **O problema.** Saber o resultado de ontem não basta: precisamos antecipar quais municípios podem não atingir a meta de alfabetização. | número: 5.232 municípios, 46% em risco |
| 0:40–1:30 | **O que fizemos.** Usamos os dados da Fase 2 e treinamos um modelo que, com o desempenho de 2023 e a meta de 2024, prevê quem fica abaixo da meta. | fluxo simples: dados → modelo → ranking |
| 1:30–2:30 | **Resultado.** O modelo acerta bem a ordenação (ROC-AUC 0,81) e, com o limiar ajustado, identifica 87% dos municípios em risco. | `modelo_roc_pr.png` |
| 2:30–3:30 | **Insights.** O território e a meta pesam mais que o desempenho passado. RS tem 92% dos municípios em risco e CE só 8%, em parte pela **forma como as metas foram definidas**. | `eda_risco_por_uf.png`, `shap_summary.png` |
| 3:30–4:20 | **Uso na política pública.** Triagem de municípios para apoio técnico, monitoramento anual e **revisão da calibração das metas**. | tabela dos 10 municípios de maior risco |
| 4:20–5:00 | **Limites e próximos passos.** Não generaliza para estados novos (ROC-AUC 0,56); só dois anos de dados; faltam as fontes socioeconômicas. Próximo: enriquecer e validar no tempo. | lista de 3 itens |

**Mensagem final:** "É uma ferramenta de triagem para priorizar apoio, não um veredito sobre gestões."
