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
| 11 | UF e região fora do modelo (`USAR_UF_NO_MODELO = False`) | O modelo deve aprender com características do município, não com a identidade do estado; a UF explicava boa parte do alvo por desenho de metas | Manter `sigla_uf` (ROC-AUC maior, mas pouco generalizável) |
| 12 | Variáveis "relativas à UF" e fontes em nível estadual (PNAD/UF) também ficam fora | Codificam a UF indiretamente (constantes dentro do estado) | Usá-las e declarar que a UF continua no modelo |
| 13 | `sigla_uf` continua no dataset só para validação por grupo e relatórios | A validação por UF mede generalização a estados não vistos | Remover a coluna por completo |
| 14 | Fontes manuais: usar o dado mais recente até 2023 | Evita usar informação posterior ao período de treino | Usar o último ano disponível, qualquer que seja |
| 15 | Filtro de correlação (|r| > 0,90) dentro do Pipeline | Atlas traz colunas quase duplicadas e os indicadores de ausência se repetem | Manter todas as colunas |
| 16 | Grades de hiperparâmetros restritivas (árvores rasas, folhas ≥ 40, subamostragem) | Reduzir a memorização dos modelos de árvore | Grades amplas (maior AUC aparente, mais overfitting) |
| 17 | Seleção das K melhores variáveis foi testada e **descartada** | A busca escolheu `k="all"` nos três modelos e o AUC de teste caiu | Manter a etapa no Pipeline |
| 18 | Escolha do modelo: 1 erro-padrão + **menor gap treino→CV** | A primeira versão (ordem fixa de "simplicidade") escolheu o Random Forest, que tinha o MAIOR gap | Maior ROC-AUC de CV; ordem fixa de complexidade |
| 19 | Modelo complementar do **nível** de alfabetização (regressão) | O risco depende da meta, que absorve o efeito socioeconômico; a pergunta "quais fatores impactam a alfabetização?" pede o nível | Só classificação |
| 20 | Modelo de fatores com três variantes (`contexto`, `contexto_sem_desempenho`, `contexto_mais_historico`) | A nota de matemática do SAEB explica sozinha 0,34 de R² e é outro indicador de aprendizagem, não um fator de contexto | Reportar apenas a variante com todas as fontes |
| 21 | Regressão logística como modelo final de risco | Empatada com o Gradient Boosting dentro de 1 erro-padrão (ROC-AUC de CV 0,742 vs 0,746) e com gap treino→CV de 0,020 contra 0,080 | Gradient Boosting (maior AUC, mais overfitting) |
| 22 | Limiar de decisão 0,40 (escolhido em OOF do treino) como alternativa ao 0,50 | Prioriza o recall da classe de risco (0,76 contra 0,60 no teste) a custo de mais falsos alarmes | Manter 0,50 |
| 23 | PNAD (UF) e CadÚnico não usados | Arquivos não obtidos; PNAD é estadual e codificaria a UF | Simular as fontes |
| 24 | Clusters apresentados como segmentação grosseira | Silhouette 0,288 (k=2) e 0,10–0,14 para k≥3 | Nomear tipologias com mais grupos |
