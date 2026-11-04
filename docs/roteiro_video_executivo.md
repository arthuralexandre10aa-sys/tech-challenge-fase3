# Roteiro do vídeo executivo (até 5 min) — reunião com gestores

**Formato:** simular apresentação a secretários/gestores. Slides curtos, sem jargão. Sugestão de divisão entre os integrantes.
O deck correspondente está em `docs/` (`tech_challenge_fase3_video_executivo.pptx`) e traz este roteiro nas notas do apresentador.

| Tempo | Conteúdo | Apoio visual |
|---|---|---|
| 0:00–0:40 | **O problema.** Saber o resultado de ontem não basta: precisamos antecipar quais municípios podem não atingir a meta de alfabetização. Dos 5.232 municípios, 46% ficaram abaixo da meta em 2024, e a taxa varia de 8% (CE) a 92% (RS). | slide 2: risco, meta e resultado médio por estado |
| 0:40–1:30 | **O que fizemos.** Juntamos os dados da Fase 2 com IBGE, INEP, Atlas e FUNDEB e treinamos um modelo que, com o desempenho de 2023 e o contexto do município, prevê quem fica abaixo da meta. **Não usamos o estado nem a própria meta como informação.** | slide 3: dados → modelo → ranking |
| 1:30–2:30 | **Resultado.** O modelo ordena o risco com ROC-AUC 0,71 e, com o limiar ajustado, identifica 88% dos municípios que ficaram abaixo da meta, ao custo de alarmes falsos. Tirar a meta custou cerca de 0,03 de ROC-AUC e evita que o modelo aprenda a política de metas de cada estado. | slide 4: `modelo_roc_pr.png` |
| 2:30–3:30 | **Insights.** Lideram a nota de matemática do SAEB, a proficiência de 2023, a taxa de aprovação e o nível socioeconômico. No nível de alfabetização, frequência escolar e analfabetismo infantil lideram. O sinal do contexto é em parte geográfico: entre estados nunca vistos o ROC-AUC cai para 0,57. | slides 5 e 6: `shap_summary.png`, ablação, fatores de contexto |
| 3:30–4:20 | **Uso na política pública.** Triagem de municípios para apoio técnico (649 com índice de vulnerabilidade acima de 70% em 2025), monitoramento anual e **revisão das metas** onde o risco previsto fica abaixo do observado (RS). | slide 7: quatro usos |
| 4:20–5:00 | **Limites e próximos passos.** Só dois anos de dados; o modelo estima vulnerabilidade, não o cumprimento de uma meta específica; fatores são associações, não causas. Próximo: mais anos de dados e CadÚnico. | slide 8 |

**Mensagem final:** "É uma ferramenta de triagem para priorizar apoio, não um veredito sobre gestões."
