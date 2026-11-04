# Roteiro do vídeo executivo (até 5 min) — reunião com gestores

**Formato:** simular apresentação a secretários/gestores. Slides curtos, sem jargão. Sugestão de divisão entre os integrantes.
O deck correspondente é `docs/tech_challenge_fase3_video_executivo.pptx` (11 slides) e traz este roteiro nas notas do apresentador.
Os títulos dos slides 5 a 9 reproduzem as perguntas de negócio do enunciado.

| Tempo | Slide | Conteúdo |
|---|---|---|
| 0:00–0:15 | 1 · Capa | Apresentar o grupo: precisamos antecipar quais municípios podem não atingir a meta de alfabetização. |
| 0:15–0:45 | 2 · Quem ficará abaixo da meta? | 5.232 municípios, 46% abaixo da meta em 2024; risco de 8% (CE) a 92% (RS); metas médias de 36% a 79%. |
| 0:45–1:10 | 3 · Como o modelo funciona | Dados da Fase 2 + IBGE, INEP, Atlas e FUNDEB; Gradient Boosting com desempenho de 2023 e contexto; sem estado, região ou meta. |
| 1:10–1:40 | 4 · O que o modelo entrega | ROC-AUC 0,71; 88% dos municípios em risco identificados (limiar 0,32); 0,57 em estados nunca vistos; tirar a meta custou 0,03. |
| 1:40–2:10 | 5 · Quais variáveis possuem maior influência nos modelos? | SHAP: SAEB, proficiência de 2023, aprovação, NSE; ablação (0,74 com a meta, 0,71 sem, 0,66 só com fatores socioeconômicos). |
| 2:10–2:35 | 6 · Quais fatores mais impactam a alfabetização? | Sem SAEB: frequência escolar, analfabetismo de 11 a 14 anos, NSE, infraestrutura. Associações, não causas. |
| 2:35–3:05 | 7 · Quais municípios apresentam maior risco educacional? | Risco previsto × observado por UF; subestimação em RS e BA; cinco municípios de maior risco. |
| 3:05–3:30 | 8 · Quais regiões possuem padrões semelhantes? | Dois grupos (Nordeste/Norte × Sudeste/Sul); IDHM 0,59 × 0,71; separação grosseira. |
| 3:30–3:55 | 9 · Como prever municípios que podem não atingir metas futuras? | 649 de 5.352 com índice de vulnerabilidade acima de 70% em 2025; sem usar as metas de 2025. |
| 3:55–4:25 | 10 · Como apoia a política pública | Priorizar, monitorar, revisar metas, agir nas causas. |
| 4:25–4:55 | 11 · Limites e próximos passos | Dois anos de dados; 0,57 em estados novos; vulnerabilidade, não meta; próximos passos. |

**Mensagem final:** "É uma ferramenta de triagem para priorizar apoio, não um veredito sobre gestões."
Se o tempo estourar, o slide 3 (como o modelo funciona) pode ser falado em 15 segundos.
