# Decisões analíticas (log)

| # | Decisão | Motivo | Alternativa descartada |
|---|---|---|---|
| 1 | Unidade de análise = município | A Gold só tem agregados por município/ano | Modelo por aluno (não há microdados na Gold) |
| 2 | Alvo = não atingiu a meta em 2024 | Responde à pergunta de negócio sobre metas | Alfabetizado/não por limiar arbitrário |
| 3 | Descartar linhas sem meta | Bug da Fase 2 marcava 2023 inteiro como "não atingiu" | Manter `atingiu_meta` original |
| 4 | Features de 2023 + meta de 2024 | Só usa informação anterior ao resultado | Usar `percentual_alfabetizado` de 2024 (leakage) |
| 5 | Estatísticas por UF dentro do Pipeline | Evita vazamento do teste para o treino | Calcular médias por UF antes do split |
| 6 | Imputação por mediana + indicador | 621 municípios sem histórico | Remover esses municípios |
| 7 | Limiar escolhido em OOF do treino | O teste não pode influenciar nenhuma escolha | Otimizar o limiar no teste |
| 8 | ROC-AUC como critério de seleção | Alvo equilibrado e independência de limiar | Acurácia |
| 9 | Validação por UF como teste de robustez | Mede generalização a estados novos | Reportar só CV aleatória |
| 10 | Fontes sem acesso não derrubam o pipeline | Atlas/CadÚnico/FUNDEB/PNAD exigem download manual | Falhar a execução inteira |
