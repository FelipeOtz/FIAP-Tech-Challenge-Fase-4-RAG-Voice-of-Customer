# Tech Challenge Fase 4

## Voice of Customer Intelligence com RAG

Projeto integrador da Fase 4 da pós-graduação. Transforma as avaliações escritas dos clientes do Brazilian E-Commerce Public Dataset by Olist em uma base de conhecimento consultável por perguntas em linguagem natural. Cada resposta é construída a partir de avaliações recuperadas da base e indica quais delas sustentam cada afirmação. Quando a base não tem avaliações relevantes para a pergunta, o sistema informa isso sem pedir uma resposta ao modelo de linguagem.

---

## Vídeo de Apresentação

Link: https://youtu.be/C4m2imcAt0Y

---

## Sumário

- [Vídeo de Apresentação](#vídeo-de-apresentação)
- [Contexto do Problema](#contexto-do-problema)
- [Objetivo da Solução](#objetivo-da-solução)
- [Fontes de Dados](#fontes-de-dados)
- [Achados da Exploração dos Dados](#achados-da-exploração-dos-dados)
- [Arquitetura da Solução](#arquitetura-da-solução)
- [Construção da Base de Conhecimento](#construção-da-base-de-conhecimento)
- [Pipeline RAG](#pipeline-rag)
- [Tratamento de Perguntas sem Evidência](#tratamento-de-perguntas-sem-evidência)
- [Interface](#interface)
- [Decisões Técnicas e Trade-offs](#decisões-técnicas-e-trade-offs)
- [Exemplos de Perguntas e Respostas](#exemplos-de-perguntas-e-respostas)
- [Estrutura do Repositório](#estrutura-do-repositório)
- [Tecnologias e Bibliotecas](#tecnologias-e-bibliotecas)
- [Limitações do Projeto](#limitações-do-projeto)
- [Possíveis Evoluções Futuras](#possíveis-evoluções-futuras)
- [Como Executar](#como-executar)
- [Replicabilidade](#replicabilidade)
- [Fonte dos Dados e Licença](#fonte-dos-dados-e-licença)

---

## Contexto do Problema

Um e-commerce recebe milhares de avaliações por mês. A nota de 1 a 5 mostra o nível de satisfação, mas o motivo de uma nota baixa, ou o que levou o cliente a recomendar a loja, está no texto do comentário. Ler dezenas de milhares de comentários à mão não escala, e um resumo feito por amostragem informal não deixa rastro de onde cada conclusão veio.

---

## Objetivo da Solução

Permitir que gestores e analistas façam perguntas como "quais reclamações sobre entrega aparecem com mais frequência?" e recebam uma resposta fundamentada em avaliações reais, acompanhada da lista das avaliações usadas. O edital avalia principalmente a capacidade de recuperar as avaliações certas e de responder de forma coerente com elas, incluindo reconhecer quando a base não tem informação suficiente.

---

## Fontes de Dados

Os cinco arquivos em `data/raw/` vêm do [Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce/data) e estão versionados no repositório, para que a reprodução não dependa de download.

| Arquivo | Tamanho | Uso |
|---|---|---|
| `olist_order_reviews_dataset.csv` | 14,5 MB | Fonte da base de conhecimento |
| `olist_orders_dataset.csv` | 17,7 MB | Usado para medir a viabilidade dos metadados de categoria e entrega (ver evoluções futuras) |
| `olist_order_items_dataset.csv` | 15,4 MB | Usado para medir a viabilidade dos metadados de categoria e entrega (ver evoluções futuras) |
| `olist_products_dataset.csv` | 2,4 MB | Usado para medir a viabilidade dos metadados de categoria e entrega (ver evoluções futuras) |
| `product_category_name_translation.csv` | 4 KB | Não usado |

---

## Achados da Exploração dos Dados

A investigação foi feita com `scripts/investigar_dados.py`, ferramenta manual que usa só a biblioteca padrão e que pode ser executada sem o ambiente virtual.

| Item | Valor medido em `olist_order_reviews_dataset.csv` |
|---|---|
| Registros | 99.224 (104.720 linhas físicas, porque há comentários com quebra de linha) |
| Encoding | UTF-8 válido, sem BOM, fim de linha CRLF, sem caracteres corrompidos |
| Com comentário (`review_comment_message`) | 40.950 (41,3%) |
| Com título (`review_comment_title`) | 11.566, sendo 1.737 só com título e sem comentário |
| Notas, todas as avaliações | 1: 11.424 · 2: 3.151 · 3: 8.179 · 4: 19.142 · 5: 57.328 |
| Notas, avaliações com comentário | 1: 8.744 · 2: 2.145 · 3: 3.556 · 4: 5.970 · 5: 20.535 |
| Período (`review_creation_date`) | 02/10/2016 a 31/08/2018 |
| `review_id` repetidos | 789 IDs em 1.603 linhas, todos com nota e texto idênticos e `order_id` diferente |
| Comentários com quebra de linha interna | 3.834 |
| Tamanho do comentário | mediana de 53 caracteres, 90% até 160, máximo de 208 |
| Comentários sem nenhuma letra | 159 (".", "10", "...", emoji solto) |
| Textos mais repetidos | "Muito bom" 305 vezes, "Bom" 215, "Ótimo" 158 |

A proporção de notas 1 sobe de 12% no total para 21% entre as avaliações com comentário. Das avaliações com nota 1, 77% têm comentário; das com nota 5, 36%. A base textual, portanto, tem mais reclamações do que a distribuição de notas sugere.

---

## Arquitetura da Solução

```
CONSTRUÇÃO DA BASE (src/base_conhecimento.py, roda uma vez, sem chave de API)

  olist_order_reviews_dataset.csv
        │  seleção: só avaliações com comentário, 1 por review_id, com ao menos uma letra
        │  limpeza: quebras de linha e espaços repetidos viram um espaço
        ▼
  40.482 documentos  (texto = "título. comentário", metadados: nota, data, ano_mes, order_id)
        │  paraphrase-multilingual-MiniLM-L12-v2 (local), vetores normalizados
        ▼
  Chroma persistido em chroma_db/  (coleção "avaliacoes", distância cosseno)


CONSULTA (src/pipeline_rag.py)

  Pergunta
     │  1. reescrita pelo Gemini (JSON: consulta, filtro de nota, modo), 1 tentativa;
     │     sem chave ou em falha, regras: remove meta-linguagem e decide filtro e modo por palavras
     │  2. filtro de nota: reclamação → notas 1 e 2, elogio → notas 4 e 5, misto → todas
     │  3. modo: pergunta de frequência busca 80 avaliações, pergunta direta busca 10
     ▼
  Embedding da consulta → busca no Chroma
     │  4. limiar: descarta avaliações com distância > 0,35
     │  5. nenhuma passou? → responde "sem base suficiente" SEM gerar resposta pelo LLM
     │  6. deduplicação: textos iguais viram uma evidência com contagem de repetições
     ▼
  Contexto numerado [E1], [E2] ... (nota, data, repetições, texto)
     │
     ├─ pergunta direta ──► Gemini (src/llm.py): resposta citando [E#]
     │
     └─ frequência ───────► Gemini devolve temas em JSON com os números das evidências
                            → o código soma as repetições e ordena os temas
     ▼
  Resposta + tabela de evidências (app.py, Streamlit)
```

A chamada ao modelo de linguagem fica isolada em `src/llm.py`. Trocar de provedor exige mexer só nesse arquivo.

---

## Construção da Base de Conhecimento

### Seleção dos dados

Entram só as avaliações com `review_comment_message` preenchido. As 58.274 linhas sem comentário têm apenas a nota e não trazem o motivo. As 1.737 avaliações que têm título e nenhum comentário também ficam de fora, porque os títulos isolados são "recomendo", "ótimo" ou "10".

### Duplicatas

Os 789 `review_id` repetidos são a mesma avaliação ligada a pedidos diferentes de uma mesma compra, sempre com nota e texto idênticos. Fica um documento por `review_id`, o que remove 309 linhas. Mantê-las contaria a mesma opinião mais de uma vez na recuperação.

### Limpeza e preparação dos textos

Quebras de linha e espaços repetidos viram um espaço. Caixa, acentos e emojis ficam como estão: o modelo de embeddings foi treinado com texto natural, e "NÃO RECEBI" em caixa alta carrega sentido. Os 159 comentários sem nenhuma letra (".", "10", "...") são descartados, porque não têm significado para o embedding.

Os comentários curtos, como "Bom" e "Ok", permanecem. São elogios reais e pesam nas perguntas sobre o que os clientes valorizam. A deduplicação feita na consulta impede que ocupem o contexto inteiro.

### Tratamento de valores ausentes

Nota e datas não têm valores vazios. Quando falta o título, o documento usa só o comentário. Quando falta o comentário, a avaliação fica fora da base, como descrito na seleção.

### Organização dos documentos

Cada avaliação vira um documento, sem chunking. O comentário mais longo tem 208 caracteres e cabe inteiro no limite do modelo, e assim cada evidência citada é uma avaliação completa, rastreável pelo `review_id`. Quando há título (9.741 documentos), o texto é `"{título}. {comentário}"`, porque o título costuma resumir a queixa ("Produto faltando peça").

Cada documento carrega como metadados `review_score`, `order_id`, `data_avaliacao`, `ano_mes` e `tem_titulo`. A nota habilita o filtro por satisfação. O período vai também como inteiro (`ano_mes`, por exemplo 201708) porque o Chroma só compara números em `$gte` e `$lte`.

### Criação dos embeddings

Os embeddings são gerados localmente com `paraphrase-multilingual-MiniLM-L12-v2`, um modelo multilíngue que roda em CPU. Gerar 40 mil embeddings por API consumiria a cota gratuita. Os vetores são normalizados, o que deixa a distância cosseno comparável entre consultas e sustenta o limiar de evidência.

### Armazenamento e indexação

O índice fica em um Chroma persistido em `chroma_db/`, com distância cosseno. Não precisa de servidor e aceita filtro por metadado. Reconstruir a base apaga e recria a coleção, e o script confere ao final que a coleção tem exatamente o número de documentos gerados.

O resultado são 40.482 documentos (nota 1: 8.632 · 2: 2.115 · 3: 3.518 · 4: 5.891 · 5: 20.326). A construção levou 2 min 37 s em um MacBook com Apple Silicon, incluindo o download do modelo na primeira execução, e o índice ocupa 124 MB.

---

## Pipeline RAG

### Reescrita da pergunta pelo Gemini

Com a chave configurada, a primeira etapa é uma chamada ao Gemini que converte a pergunta em três parâmetros, devolvidos em JSON: a consulta de busca, escrita como um cliente escreveria numa avaliação ("problema com os correios", "recebeu tudo certinho"), o filtro de nota (negativo, positivo ou todas) e o modo (direta ou frequência). A instrução manda devolver a pergunta sem adaptação quando ela não trata de compras, entregas, produtos, lojas ou atendimento, para que o limiar continue bloqueando perguntas fora do escopo. A temperatura dessa chamada é 0.

A chamada tem uma única tentativa. Sem chave, com resposta fora do formato ou com qualquer erro, inclusive limite de requisições, o pipeline usa as regras descritas a seguir e registra um aviso no log. A interface mostra se a consulta veio do Gemini ou das regras, e o filtro escolhido manualmente continua valendo sobre o filtro sugerido pelo modelo.

Para comparar, foram usadas as 5 perguntas inéditas da rodada 2 das regras (ver limitações) e 5 perguntas novas, escritas antes de qualquer execução e sem ajuste depois. O critério também foi fixado antes: boa quando as 3 primeiras avaliações tratam do assunto e o filtro é coerente; parcial quando 1 ou 2 tratam, ou o filtro está errado; fraca quando nenhuma trata.

| Pergunta | Regras: consulta, filtro | Regras | Gemini: consulta, filtro | Gemini |
|---|---|---|---|---|
| Os compradores ficaram felizes com a qualidade? | "compradores ficaram felizes qualidade", todas | fraca ("Feliz com a compra") | "qualidade do produto", 4 e 5 | boa |
| Do que o pessoal mais gosta? | "gosta", todas | fraca ("Gosto", "gostei") | "gosta qualidade bom atendimento", 4 e 5, frequência | boa |
| Alguém teve dificuldade para trocar ou devolver um produto? | pergunta inteira, todas | parcial | "dificuldade para trocar ou devolver um produto", 1 e 2 | parcial |
| Que tipo de queixa aparece sobre os Correios? | "tipo queixa aparece Correios", 1 e 2 | fraca (nenhuma cita os Correios) | "problema com os correios", 1 e 2, frequência | parcial (2 de 3 citam) |
| Os produtos chegam iguais às fotos do anúncio? | "produtos chegam iguais às fotos anúncio", todas | boa | "produto igual foto anúncio", todas | boa |
| Tem muita gente insatisfeita com o tamanho dos produtos? | "muita insatisfeita tamanho produtos", 1 e 2 | boa | "tamanho dos produtos", 1 e 2, frequência | boa |
| Os clientes recomendariam a loja para amigos? | "recomendariam loja para amigos", 4 e 5 | boa | "recomendo a loja", 4 e 5 | boa |
| O que deixa os compradores irritados no atendimento? | "deixa compradores irritados no atendimento", todas | parcial (filtro) | "atendimento ruim demorado mal educado", 1 e 2, frequência | boa |
| Quem deu nota máxima recebeu tudo certinho? | pergunta inteira, todas | parcial (filtro) | "recebeu tudo certinho", 4 e 5 | boa |
| Há relatos de cobrança indevida ou problema no pagamento? | "relatos cobrança indevida ou problema no pagamento", 1 e 2 | boa, mas só 2 de 10 no limiar | "cobrança indevida problema pagamento", 1 e 2 | boa, 9 de 10 no limiar |

As regras ficaram com 4 boas, 3 parciais e 3 fracas; a reescrita, com 8 boas e 2 parciais. As cinco primeiras perguntas tiveram, na rodada 2 das regras, 1 boa, 3 parciais e 1 fraca; com o critério fixado agora, mais estrito, a saída atual das regras conta 1 boa, 1 parcial e 3 fracas.

A calibração do limiar com a reescrita deu o mesmo resultado em 0,35: 14 de 14 perguntas dentro do escopo respondidas e 11 de 12 fora bloqueadas. A maior distância dentro do escopo caiu de 0,266 para 0,208, e o modelo devolveu sem adaptação todas as 12 perguntas fora do escopo, como a instrução pede.

### Transformação da pergunta por regras

Sem chave ou quando a reescrita falha, a transformação é feita por regras. Palavras de pergunta e de meta-linguagem afastam o embedding do texto das avaliações. Na calibração, "Quais são os principais problemas relatados pelos clientes em suas avaliações?" ficou a 0,393 da avaliação mais próxima, longe demais para passar no limiar. Sem essas palavras, a consulta vira "problemas" e a distância cai para 0,266.

Se a pergunta fica vazia depois da limpeza e há filtro de nota, a busca usa uma consulta neutra do sentimento ("reclamação problema" ou "elogio ótimo"). Sem filtro, usa a pergunta original.

### Filtro de nota

Com a reescrita, o filtro vem do Gemini. Nas regras, palavras como "reclamação", "queixa", "problema", "insatisfeito" e "defeito" aplicam o filtro de notas 1 e 2. "Elogio", "satisfeito", "recomendo" e "fala bem" aplicam notas 4 e 5. Quando a pergunta tem os dois sentidos, não há filtro. Esse é o uso de metadados do desafio adicional: em "O que os clientes relatam sobre problemas relacionados à entrega?", a busca sem filtro traria também elogios à entrega.

### Recuperação semântica e seleção

A consulta é convertida em embedding pelo mesmo modelo da base e comparada no Chroma. Só entram como evidência as avaliações com distância cosseno até 0,35. A calibração desse valor está na próxima seção.

Textos iguais depois de ignorar caixa e pontuação ("Muito bom", "muito bom!") viram uma evidência só, com o número de repetições. Isso abre espaço no contexto para avaliações diferentes e preserva a informação de frequência.

### Perguntas de frequência

Um top-10 traz alguns exemplos e não permite dizer o que é recorrente. Perguntas com "frequência", "principais", "recorrentes", "padrões" ou "mais" seguido de verbo buscam 80 avaliações. Até 40 textos distintos vão para o Gemini, que devolve em JSON os temas e os números das evidências de cada um.

A soma de avaliações por tema é feita pelo código, a partir das repetições reais. Na primeira versão o modelo fazia a conta e errou: informou 28 avaliações em um tema cujas evidências somavam 30. A resposta indica que os números se referem à amostra recuperada, e não à base inteira.

### Construção do contexto e geração da resposta

As evidências vão numeradas, no formato `[E1] (nota 1, 2018-03-02, repetida 9x) Demora na entrega`. A instrução do modelo exige citar as evidências de cada afirmação, proíbe conhecimento externo e pede que ele diga quando as evidências não respondem à pergunta. A temperatura é 0,2.

---

## Tratamento de Perguntas sem Evidência

O sistema tem duas barreiras. A primeira é o limiar de distância, aplicado antes da geração da resposta: se nenhuma avaliação recuperada fica a até 0,35, a resposta informa que não há base suficiente e o Gemini não é chamado para responder. Com a chave configurada, a pergunta já gastou uma chamada na reescrita. A segunda é a instrução do modelo, que cobre os casos em que algumas avaliações passam no limiar por coincidência de palavras sem responder à pergunta.

As três situações que o edital descreve ficam cobertas assim:

| Situação | Como o sistema trata | Exemplo |
|---|---|---|
| Não existem avaliações relevantes | Nenhuma avaliação passa no limiar; resposta padrão sem gerar resposta pelo LLM | Exemplo 5 |
| Pergunta fora do escopo da base | Na maioria dos casos, igual à anterior (11 de 12 na calibração); quando passa por coincidência de palavras, o modelo recusa | Exemplos 5 e 6 |
| Evidências recuperadas insuficientes | A instrução obriga o modelo a dizer quando as evidências não respondem ou respondem só em parte; nas perguntas de frequência, a resposta informa o tamanho da amostra usada | Exemplo 6 e a nota de amostra nos exemplos 2 e 3 |

A terceira situação depende do modelo seguir a instrução, sem uma regra no código que a garanta. O sistema não exige um número mínimo de evidências para responder.

O limiar foi calibrado com `scripts/calibrar_limiar.py`. O script mede a distância da avaliação mais próxima para 14 perguntas dentro do escopo, entre as do edital e variações, e 12 fora (capital da França, receita de bolo, cotação do dólar, faturamento da Olist e outras), aplicando a mesma transformação e o mesmo filtro do pipeline.

| Limiar | Dentro do escopo respondidas | Fora do escopo bloqueadas antes da geração |
|---|---|---|
| 0,25 | 13 de 14 | 11 de 12 |
| 0,30 | 14 de 14 | 11 de 12 |
| **0,35** | **14 de 14** | **11 de 12** |
| 0,40 | 14 de 14 | 8 de 12 |
| 0,45 | 14 de 14 | 7 de 12 |

A tabela vem do índice usado no desenvolvimento. Ao reconstruir a base em um clone limpo do repositório, as linhas até 0,35 se repetiram e as de 0,40 e 0,45 bloquearam uma pergunta a menos (7 e 6 de 12), pelo motivo descrito em [Replicabilidade](#replicabilidade).

A maior distância entre as perguntas dentro do escopo foi 0,266. Entre as de fora, excluindo o caso abaixo, a menor foi 0,356. Os limiares 0,30 e 0,35 têm o mesmo resultado, e 0,35 foi escolhido porque deixa passar mais evidências nas perguntas amplas: "principais problemas" recupera 15 de 80 avaliações com 0,35 e 4 com 0,30.

A pergunta fora do escopo que passa é "Qual o melhor time de futebol do Brasil?", que encontra a avaliação "Melhor do Brasil" a 0,132. Nenhum limiar separa esse caso sem bloquear perguntas legítimas. Nele atua a segunda barreira, e o Gemini respondeu que as avaliações não citam nenhum time (exemplo 6, abaixo).

---

## Interface

`app.py`, em Streamlit, tem um campo de pergunta, exemplos clicáveis na barra lateral e um seletor de filtro de nota com as opções automático, notas 1 e 2, notas 4 e 5 e todas as notas.

Depois de cada pergunta, um quadro informa o filtro aplicado e se ele foi automático ou escolhido manualmente, a consulta usada na busca e se ela veio do Gemini ou das regras, o modo (direta ou frequência) e quantas avaliações passaram no limiar. Abaixo vêm a resposta com as citações e a tabela de evidências, com referência, texto, nota, data, repetições, distância e `review_id`.

---

## Decisões Técnicas e Trade-offs

| Decisão | Alternativa considerada | Justificativa |
|---|---|---|
| Embeddings locais (MiniLM multilíngue) | Embeddings por API | 40 mil textos consumiriam a cota gratuita; o modelo local roda em CPU e a base é construída sem chave |
| Chroma persistido em disco | FAISS, banco vetorial em servidor | Dispensa servidor e tem filtro por metadado nativo |
| Uma avaliação por documento | Chunking | Nenhum comentário passa de 208 caracteres |
| Limiar de distância antes da geração | Confiar só na instrução do modelo | Pergunta fora do escopo não gasta a chamada de resposta e não depende de o modelo seguir a instrução |
| Reescrita da pergunta pelo Gemini, com regras como alternativa | Só regras | Na comparação com 10 perguntas, 8 boas contra 4; as regras mantêm a recuperação funcionando sem chave e quando a chamada falha |
| Uma única tentativa na reescrita | Repetir em erro | Uma falha cai nas regras sem gastar outra chamada da cota |
| Filtro de nota sugerido pelo Gemini, com troca manual na interface | Só palavras-chave | O modelo reconhece sentidos que a lista não cobre ("irritados", "nota máxima"); a troca manual corrige o que o modelo errar |
| Deduplicação com contagem | Manter textos repetidos no contexto | "Demora na entrega" aparece 9 vezes em uma consulta e ocuparia 9 posições do contexto |
| Temas em JSON e soma pelo código | Pedir ao LLM que conte | O modelo errou a soma no teste; o código usa as repetições reais |
| Gemini com no máximo 2 tentativas, só em erro 500 ou 503 | Repetir até conseguir | No nível gratuito, insistir em modelo indisponível ou cota esgotada gasta cota sem resposta |
| Chave em `.env` com `python-dotenv` | Variável de ambiente manual | `.env` fica fora do Git, e o programa explica como configurar quando a chave falta |

---

## Exemplos de Perguntas e Respostas

Execuções do pipeline final em 15/11/2026, pela linha de comando. As listas de evidências foram resumidas; a interface mostra todas.

**1. Investigação de um tema específico: "O que os clientes relatam sobre problemas relacionados à entrega?"**
Consulta `problemas entrega`, filtro notas 1 e 2, 10 de 10 dentro do limiar, 6 textos distintos.

> Com base nas avaliações fornecidas, os clientes relatam os seguintes problemas relacionados à entrega:
> - Atrasos e demora: os clientes mencionam "atraso na entrega" [E1], "demora a entrega" [E2], "demora na entrega" [E4] e classificam o problema como "TERRIVEL" [E6].
> - Falta de entrega: há relato de que o produto "nao entrega" [E3].
> - Erro no pedido: um cliente relata um "erro na entrega", afirmando que o pedido "nao chegou correto" e nem "completo" [E5].

Evidências: [E1] Atraso na entrega (nota 1, 2 repetições, distância 0,164) · [E2] Demora a entrega (0,170) · [E3] Nao entrega (0,174) · [E4] Demora na entrega (4 repetições, 0,176) · [E5] erro na entrega. a entrega nao chegou correta completa (0,210) · [E6] TERRIVEL ESTE PROBLEMA DA ENTREGA (0,220).

**2. Investigação de insatisfação: "Quais padrões podem ser identificados nas avaliações de clientes insatisfeitos?"**
Consulta `insatisfeitos`, filtro notas 1 e 2, modo frequência, 61 de 80 dentro do limiar.

> Temas recorrentes nas 50 avaliações usadas como evidência (40 textos distintos). É uma amostra recuperada, não uma contagem sobre a base inteira:
> - Insatisfação geral e atendimento ruim (25 avaliações): vários clientes expressam frustração generalizada, má qualidade da loja, falta de compromisso e atendimento péssimo.
> - Problemas com a entrega (15 avaliações): há reclamações sobre atrasos, demora, falta de entrega ou itens faltantes e incorretos.
> - Produtos com defeito ou má qualidade (9 avaliações): os clientes relatam produtos com defeitos, mal acabados ou de baixa qualidade.
> - Divergência de modelo (1 avaliação): um cliente relata que o produto recebido não corresponde ao modelo esperado.

Cada tema lista os números das evidências citadas. Entre elas: Péssimo (5 repetições), Não recebido (3), Produto com defeito. Produto com defeito desejo a troca, AINDA NÃO RECEBI A MERCADORIA.

**3. Exploração baseada em evidências: "Quais avaliações sustentam a conclusão de que existem problemas recorrentes relacionados ao prazo de entrega?"**
Consulta `conclusão problemas prazo entrega`, filtro notas 1 e 2, modo frequência, 14 de 80 dentro do limiar.

> Temas recorrentes nas 14 avaliações usadas como evidência (11 textos distintos):
> - Atrasos na entrega (13 avaliações): os clientes relatam que as entregas demoram, passam do prazo ou chegam depois do combinado.
> - Prazo longo (1 avaliação): há reclamações de que a loja possui um prazo de entrega muito longo.

Evidências: falta de comprometimento com o prazo · Entrga depois do prazo · prazo longo de entrega e não chegou no prazo · Passou do prazo · Ñ compriu seu prazo de entrega · Muita demora na entrega (3 repetições), entre outras.

**4. Busca por experiências positivas: "Os clientes elogiam a rapidez da entrega?"**
Consulta `rapidez entrega`, filtro notas 4 e 5, 10 de 10 dentro do limiar, 4 textos distintos. Execução de 16/11/2026.

> Sim, os clientes elogiam explicitamente a rapidez da entrega [E1], [E2], [E3], [E4]. As avaliações destacam a agilidade no recebimento dos produtos com notas altas (4 e 5) [E1], [E2], [E3], [E4].

Evidências: [E1] rápida entrega (nota 5, distância 0,014) · [E2] rapidez na entrega (nota 4, 0,021) · [E3] entrega rapida (nota 5, 5 repetições, 0,021) · [E4] entrega rápida (nota 5, 3 repetições, 0,021).

**5. Fora do escopo, bloqueada pelo limiar: "Qual é a capital da França?"**
Consulta `capital França`, 0 de 10 dentro do limiar. O Gemini não é chamado para responder. Este exemplo foi executado antes da reescrita; com ela, o modelo devolve a pergunta sem adaptação e o limiar bloqueia da mesma forma (calibração com reescrita).

> Não encontrei avaliações suficientemente relacionadas a essa pergunta na base. Ela pode estar fora do escopo das avaliações de clientes da Olist.

**6. Fora do escopo, recusada pelo modelo: "Qual o melhor time de futebol do Brasil?"**
3 de 10 dentro do limiar por coincidência de palavras ("Melhor do Brasil", "Toalha do maior campeão do Brasil maravilhosa").

> As avaliações fornecidas mencionam apenas que o produto é a "toalha do maior campeão do Brasil" [E2] e que o produto é o "melhor do Brasil" [E1], além de elogiar a empresa Lannister [E3]. No entanto, as evidências não citam o nome de nenhum time de futebol específico, portanto não é possível responder a esta pergunta com base apenas nos dados fornecidos.

Nomes de lojas como "Lannister" estão no texto original das avaliações. O dataset da Olist substituiu os nomes reais das lojas por nomes de casas de Game of Thrones.

---

## Estrutura do Repositório

```
├── app.py                      # interface Streamlit
├── data/raw/                   # CSVs da Olist (Kaggle)
├── scripts/
│   ├── investigar_dados.py     # investigação dos dados, ferramenta manual
│   └── calibrar_limiar.py      # calibração do limiar de distância, ferramenta manual
├── src/
│   ├── base_conhecimento.py    # seleção, limpeza, documentos, embeddings e índice
│   ├── pipeline_rag.py         # transformação, filtro, recuperação, limiar, deduplicação, contexto
│   └── llm.py                  # única integração com o Gemini
├── tests/test_pipeline_rag.py  # testes sem modelo, índice ou LLM
├── requirements.txt
└── .env.example
```

`chroma_db/` (o índice) e `.env` (a chave) são gerados localmente e ficam fora do Git.

---

## Tecnologias e Bibliotecas

| Biblioteca | Versão | Uso |
|---|---|---|
| `sentence-transformers` | 6.3.0 | Embeddings com `paraphrase-multilingual-MiniLM-L12-v2` |
| `chromadb` | 1.5.9 | Banco vetorial persistido com filtro por metadado |
| `google-genai` | 3.3.0 | Chamada ao Gemini, inclusive com saída estruturada em JSON |
| `python-dotenv` | 1.2.4 | Leitura da chave a partir do `.env` |
| `streamlit` | 1.67.0 | Interface |

A investigação dos dados e a construção dos documentos usam o módulo `csv` da biblioteca padrão.

---

## Limitações do Projeto

Cada pergunta custa duas chamadas ao Gemini, uma para a reescrita e outra para a resposta. O nível gratuito do `gemini-3.5-flash-lite` aceitou 15 requisições por minuto: na primeira calibração com reescrita, 26 perguntas seguidas estouraram esse limite e parte delas caiu nas regras, o que obrigou a espaçar as chamadas em 4,5 segundos no script. Na interface, isso limita o uso a cerca de 7 perguntas por minuto; acima disso, a reescrita falha e as regras assumem.

A reescrita também pode reduzir a amostra. Em "Quais são os principais problemas relatados pelos clientes?", o Gemini gerou "problemas relatados defeito insatisfação" e só 6 de 80 avaliações passaram no limiar, contra 15 com as regras; em "Quais padrões podem ser identificados nas avaliações de clientes insatisfeitos?", 24 de 80, contra 61. Em "Do que os clientes mais reclamam?", a consulta "defeito atraso problema" deixou passar 7 de 80, contra 2 com as regras. Consultas que juntam vários temas ficam longe de qualquer avaliação isolada.

As limitações abaixo valem para as regras, que continuam em uso sem chave ou quando a reescrita falha. A transformação por regras funciona por lista de palavras. Ela remove a meta-linguagem prevista, mas falha quando a pergunta usa outro vocabulário. Foram feitas duas rodadas de teste com perguntas escritas de forma diferente das usadas na calibração:

| Rodada | Perguntas | Resultado |
|---|---|---|
| 1, antes do ajuste | 5 coloquiais ("do que os clientes mais reclamam?", "o pessoal fala bem da entrega?") | 3 boas, 2 ruins |
| 1, depois do ajuste (as mesmas 5, já usadas no ajuste) | idem | 4 boas, 1 ruim |
| 2, inéditas, sem novo ajuste | 5 novas | 1 boa, 3 parciais, 1 fraca |

"Do que os clientes mais reclamam?" fica sem conteúdo depois da limpeza, e a consulta neutra "reclamação problema" deixa passar só 2 de 80 avaliações. "Do que o pessoal mais gosta?" vira "gosta" e recupera "Gosto" e "gostei", genéricos; além disso, "mais gosta", no singular, não ativa o modo frequência. "Os compradores ficaram felizes com a qualidade?" recupera satisfação com a compra em geral. "Alguém teve dificuldade para trocar ou devolver um produto?" recupera defeitos, com uma única menção a troca entre as três primeiras avaliações. "Que tipo de queixa aparece sobre os Correios?" ficava sem filtro de nota e misturava elogios; depois que "queixa" entrou nas palavras de reclamação, recebe o filtro de notas 1 e 2, mas só 5 de 10 avaliações passam no limiar e as três primeiras não citam os Correios. Em "Quais avaliações sustentam a conclusão...", a palavra "conclusão" sobra na consulta e só 14 de 80 avaliações passam no limiar.

Perguntas amplas recuperam poucas avaliações. "Quais são os principais problemas relatados pelos clientes?" deixa passar 15 de 80, porque "problemas" é uma consulta genérica. Os temas saem coerentes, mas a amostra é pequena. Perguntas sobre um tema específico funcionam melhor: "Quais reclamações relacionadas à entrega aparecem com maior frequência?" deixa passar 57 de 80.

A margem do limiar é estreita. Com 0,35, a pergunta fora do escopo mais próxima ficou a 0,356. A calibração usou 26 perguntas, e um conjunto maior pode indicar outro ponto de corte.

Os números por tema nas perguntas de frequência contam as avaliações recuperadas e enviadas ao modelo, até 40 textos distintos. Indicam temas recorrentes, sem medir quantas avaliações da base inteira tratam do assunto.

Muitos elogios são curtos e genéricos ("Ótima compra", "Muito bom"), e perguntas abertas sobre o que os clientes valorizam tendem a recuperá-los.

A qualidade foi verificada por inspeção das evidências e das respostas nos exemplos e nas rodadas descritas acima. Não há um conjunto de referência com métricas automáticas.

---

## Possíveis Evoluções Futuras

**Uma consulta por tema nas perguntas amplas.** A reescrita gera uma consulta só, e consultas que juntam vários temas recuperam poucas avaliações. Pedir ao modelo uma consulta por tema e buscar cada uma separadamente aumentaria a amostra das perguntas de frequência, ao custo de mais buscas por pergunta.

**Metadados de entrega e categoria.** Um join com `olist_orders_dataset.csv`, `olist_order_items_dataset.csv` e `olist_products_dataset.csv` permitiria perguntas como "o que reclamam os clientes cujo pedido atrasou?". Dos 40.482 documentos, 38.949 têm uma única categoria de produto, 443 têm várias e 1.090 não têm nenhuma; 4.227 pedidos chegaram depois da data estimada e 1.842 não têm data de entrega.

**Reranking.** Um cross-encoder reordenaria as avaliações recuperadas antes de montar o contexto.

**Avaliação automática.** Um conjunto de perguntas de referência, com métricas de recuperação e de fidelidade às evidências, substituiria a inspeção manual.

**Filtro por período na interface.** O metadado `ano_mes` já está no índice e não é usado na consulta.

---

## Como Executar

Requisitos: Python 3.12, a versão usada no desenvolvimento, e uma chave do Gemini gerada no [Google AI Studio](https://aistudio.google.com/), necessária só para a geração das respostas.

```bash
git clone https://github.com/FelipeOtz/FIAP-Tech-Challenge-Fase-4-RAG-Voice-of-Customer.git
cd FIAP-Tech-Challenge-Fase-4-RAG-Voice-of-Customer
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env
```

No Windows, troque `python3.12` por `py -3.12`, `.venv/bin/` por `.venv\Scripts\` e `cp` por `copy`, nesse comando e nos seguintes.

Preencha o `.env` com `GEMINI_API_KEY` e `GEMINI_MODEL`. O desenvolvimento usou `gemini-3.5-flash-lite`; confira no AI Studio os modelos disponíveis para a sua chave.

```bash
# 1. Construir a base (cerca de 3 minutos; não precisa da chave, mas a primeira
#    execução baixa o modelo de embeddings do Hugging Face e precisa de internet)
.venv/bin/python src/base_conhecimento.py

# 2. Interface
.venv/bin/streamlit run app.py

# Ou pela linha de comando
.venv/bin/python -m src.pipeline_rag "O que os clientes relatam sobre a entrega?"
```

Ferramentas e testes:

```bash
.venv/bin/python scripts/investigar_dados.py      # investigação dos dados (só biblioteca padrão)
.venv/bin/python -m scripts.calibrar_limiar       # calibração do limiar (não chama o LLM)
.venv/bin/python -m unittest discover tests       # 6 testes das partes determinísticas
```

Sem a chave, a construção da base, a calibração e a recuperação funcionam, com a transformação da pergunta por regras. Só a geração da resposta para, com uma mensagem explicando como configurar o `.env`. A calibração com a reescrita roda com `.venv/bin/python -m scripts.calibrar_limiar --llm`, com uma chamada por pergunta espaçada em 4,5 segundos.

---

## Replicabilidade

A seleção e a limpeza não têm componente aleatório: os mesmos CSVs geram sempre os mesmos 40.482 documentos, e a investigação dos dados reproduz os números deste documento. As versões das bibliotecas estão fixadas em `requirements.txt`.

A busca do Chroma usa HNSW, um índice de vizinhos aproximados cujo grafo é montado de forma diferente a cada construção. Uma busca pode, portanto, deixar de encontrar a avaliação mais próxima em um índice e encontrá-la em outro. O repositório foi clonado do zero e a base reconstruída para conferir: os 6 testes passaram, a coleção teve os mesmos 40.482 documentos e a calibração repetiu o resultado do limiar 0,35 (14 de 14 e 11 de 12). Nas linhas de 0,40 e 0,45, a pergunta "Qual o faturamento anual da Olist?" encontrou no índice novo a avaliação "nota mil pelo atendimento..." a 0,389, contra 0,510 no índice original, e os bloqueios caíram de 8 para 7 e de 7 para 6. As distâncias dos exemplos podem variar da mesma forma ao reconstruir a base.

A geração pelo Gemini usa temperatura 0,2, e o texto das respostas pode mudar entre execuções mesmo com as mesmas evidências. A reescrita usa temperatura 0, mas a consulta gerada também pode variar: na comparação, "Do que o pessoal mais gosta?" virou "gosta qualidade atendimento produto" numa execução e "gosta qualidade bom atendimento" na seguinte.

---

## Fonte dos Dados e Licença

Brazilian E-Commerce Public Dataset by Olist, disponível no [Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) sob a licença CC BY-NC-SA 4.0. Os arquivos em `data/raw/` são redistribuídos sem alteração, para fins acadêmicos e não comerciais.
