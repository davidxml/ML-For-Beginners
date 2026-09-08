# Técnicas de Aprendizagem Automática

O processo de construir, usar e manter modelos de aprendizagem automática e os dados que utilizam é um processo muito diferente de muitos outros fluxos de trabalho de desenvolvimento. Nesta lição, vamos desmistificar o processo e delinear as principais técnicas que precisa de conhecer. Você irá:

- Compreender os processos que sustentam a aprendizagem automática a um nível elevado.
- Explorar conceitos base como 'modelos', 'previsões' e 'dados de treino'.

## [Questionário pré-aula](https://ff-quizzes.netlify.app/en/ml/)

[![ML for beginners - Techniques of Machine Learning](https://img.youtube.com/vi/4NGM0U2ZSHU/0.jpg)](https://youtu.be/4NGM0U2ZSHU "ML for beginners - Techniques of Machine Learning")

> 🎥 Clique na imagem acima para um vídeo curto que percorre esta lição.

## Introdução

A um nível elevado, a arte de criar processos de aprendizagem automática (ML) é composta por vários passos:

1. **Decidir a questão**. A maioria dos processos ML começa por colocar uma questão que não pode ser respondida por um programa condicional simples ou motor baseado em regras. Estas questões muitas vezes giram em torno de previsões baseadas numa coleção de dados.
2. **Recolher e preparar dados**. Para conseguir responder à sua questão, precisa de dados. A qualidade e, por vezes, a quantidade dos seus dados determinarão o quão bem poderá responder à sua questão inicial. Visualizar os dados é um aspecto importante desta fase. Esta fase inclui também a divisão dos dados em grupos de treino e teste para construir um modelo.
3. **Escolher um método de treino**. Dependendo da sua questão e da natureza dos seus dados, precisa de escolher como deseja treinar um modelo para refletir melhor os seus dados e fazer previsões precisas sobre os mesmos. Esta é a parte do seu processo ML que requer especialização específica e, muitas vezes, um considerável número de experimentações.
4. **Treinar o modelo**. Usando os seus dados de treino, irá utilizar vários algoritmos para treinar um modelo a reconhecer padrões nos dados. O modelo pode aproveitar pesos internos que podem ser ajustados para privilegiar certas partes dos dados em relação a outras para construir um modelo melhor.
5. **Avaliar o modelo**. Usa dados nunca antes vistos (os seus dados de teste) do seu conjunto recolhido para ver como o modelo está a desempenhar-se.
6. **Ajuste de parâmetros**. Com base no desempenho do seu modelo, pode repetir o processo usando diferentes parâmetros, ou variáveis, que controlam o comportamento dos algoritmos usados para treinar o modelo.
7. **Prever**. Use novas entradas para testar a precisão do seu modelo.

## Que questão colocar

Os computadores são particularmente habilidosos a descobrir padrões ocultos nos dados. Esta utilidade é muito útil para investigadores que têm questões sobre um dado domínio que não podem ser facilmente respondidas através da criação de um motor de regras condicional. Dada uma tarefa atuarial, por exemplo, um cientista de dados pode ser capaz de construir regras feitas à mão em torno da mortalidade de fumadores vs não fumadores.

Quando muitas outras variáveis entram na equação, no entanto, um modelo ML pode revelar-se mais eficiente para prever taxas futuras de mortalidade com base no histórico de saúde passado. Um exemplo mais animador pode ser fazer previsões meteorológicas para o mês de abril numa dada localização com base em dados que incluem latitude, longitude, alterações climáticas, proximidade do oceano, padrões da corrente de jato e mais.

✅ Esta [apresentação de slides](https://www2.cisl.ucar.edu/sites/default/files/2021-10/0900%20June%2024%20Haupt_0.pdf) sobre modelos meteorológicos oferece uma perspetiva histórica para o uso de ML na análise meteorológica.

## Tarefas pré-construção

Antes de começar a construir o seu modelo, há várias tarefas que precisa de completar. Para testar a sua questão e formar uma hipótese com base nas previsões de um modelo, precisa de identificar e configurar vários elementos.

### Dados

Para poder responder à sua questão com algum grau de certeza, precisa de uma boa quantidade de dados do tipo certo. Há duas coisas que precisa de fazer neste ponto:

- **Recolher dados**. Tendo em conta a lição anterior sobre justiça na análise de dados, recolha os seus dados com cuidado. Esteja atento às fontes destes dados, a quaisquer preconceitos inerentes que possam ter, e documente a sua origem.
- **Preparar dados**. Existem vários passos no processo de preparação de dados. Pode precisar de consolidar dados e normalizá-los se vierem de fontes diversas. Pode melhorar a qualidade e quantidade dos dados através de vários métodos, como converter strings em números (como fazemos em [Clustering](../../5-Clustering/1-Visualize/README.md)). Pode também gerar novos dados, baseados nos originais (como fazemos em [Classificação](../../4-Classification/1-Introduction/README.md)). Pode limpar e editar os dados (como faremos antes da lição da [Aplicação Web](../../3-Web-App/README.md)). Finalmente, pode também precisar de os aleatorizar e baralhar, dependendo das suas técnicas de treino.

✅ Depois de recolher e processar os seus dados, tome um momento para ver se a sua forma permitirá que aborde a sua questão pretendida. Pode acontecer que os dados não desempenhem bem na tarefa dada, como descobrimos nas nossas lições de [Clustering](../../5-Clustering/1-Visualize/README.md)!

### Características e Alvo

Uma [característica](https://www.datasciencecentral.com/profiles/blogs/an-introduction-to-variable-and-feature-selection) é uma propriedade mensurável dos seus dados. Em muitos conjuntos de dados é expressa como o título de uma coluna como 'data', 'tamanho' ou 'cor'. A sua variável característica, geralmente representada como `X` em código, representa a variável de entrada que será usada para treinar um modelo.

Um alvo é aquilo que está a tentar prever. O alvo, geralmente representado como `y` em código, representa a resposta à questão que está a tentar colocar aos seus dados: em dezembro, que **cor** as abóboras serão mais baratas? em São Francisco, que bairros terão o melhor **preço** imobiliário? Por vezes o alvo também é referido como um atributo de etiqueta.

### Selecionar a sua variável característica

🎓 **Seleção de Características e Extração de Características** Como sabe qual variável escolher ao construir um modelo? Provavelmente passará por um processo de seleção de características ou extração de características para escolher as variáveis certas para o modelo mais performante. Contudo, não são a mesma coisa: "A extração de características cria novas características a partir de funções das características originais, enquanto a seleção de características retorna um subconjunto das características." ([fonte](https://wikipedia.org/wiki/Feature_selection))

### Visualize os seus dados

Um aspeto importante do kit de ferramentas do cientista de dados é o poder de visualizar dados usando várias excelentes bibliotecas como Seaborn ou MatPlotLib. Representar visualmente os seus dados pode permitir-lhe descobrir correlações ocultas que pode aproveitar. As suas visualizações também podem ajudar a descobrir enviesamentos ou dados desequilibrados (como descobrimos em [Classificação](../../4-Classification/2-Classifiers-1/README.md)).

### Divida o seu conjunto de dados

Antes do treino, precisa de dividir o seu conjunto de dados em duas ou mais partes de tamanho desigual que ainda representem bem os dados.

- **Treino**. Esta parte do conjunto de dados é ajustada ao seu modelo para o treinar. Este conjunto constitui a maioria do conjunto de dados original.
- **Teste**. Um conjunto de dados de teste é um grupo independente de dados, muitas vezes recolhidos do conjunto original, que usa para confirmar o desempenho do modelo construído.
- **Validação**. Um conjunto de validação é um grupo independente menor de exemplos que usa para ajustar os hiperparâmetros, ou arquitetura, do modelo para melhorar o modelo. Dependendo do tamanho dos seus dados e da questão que está a colocar, pode não precisar de construir este terceiro conjunto (como notamos em [Previsão de Séries Temporais](../../7-TimeSeries/1-Introduction/README.md)).

## Construção de um modelo

Usando os seus dados de treino, o seu objetivo é construir um modelo, ou uma representação estatística dos seus dados, usando vários algoritmos para **treiná-lo**. Treinar um modelo expõe-no a dados e permite que faça suposições sobre padrões percebidos que descobre, valida e aceita ou rejeita.

### Decida um método de treino

Dependendo da sua questão e da natureza dos seus dados, irá escolher um método para os treinar. Percorrendo a [documentação do Scikit-learn](https://scikit-learn.org/stable/user_guide.html) - que usamos neste curso - pode explorar várias formas de treinar um modelo. Dependendo da sua experiência, pode ter de tentar vários métodos diferentes para construir o melhor modelo. Provavelmente passará por um processo onde os cientistas de dados avaliam o desempenho de um modelo ao alimentá-lo com dados nunca vistos, verificando a precisão, enviesamento e outras questões que degradam a qualidade, e selecionando o método de treino mais apropriado para a tarefa em questão.

### Treinar um modelo

Armado com os seus dados de treino, está pronto para 'ajustar' o modelo. Vai notar que em muitas bibliotecas ML encontra o código 'model.fit' - é neste momento que envia a sua variável característica como um array de valores (normalmente 'X') e uma variável alvo (normalmente 'y').

### Avaliar o modelo

Uma vez concluído o processo de treino (pode levar muitas iterações, ou 'épocas', para treinar um modelo grande), poderá avaliar a qualidade do modelo usando dados de teste para medir o seu desempenho. Estes dados são um subconjunto dos dados originais que o modelo não analisou previamente. Pode imprimir uma tabela de métricas sobre a qualidade do seu modelo.

🎓 **Ajuste do modelo**

No contexto da aprendizagem automática, ajuste do modelo refere-se à precisão da função subjacente do modelo enquanto tenta analisar dados com os quais não está familiarizado.

🎓 **Subajuste** e **sobreajuste** são problemas comuns que degradam a qualidade do modelo, pois o modelo ajusta-se demasiado pouco ou demasiado. Isto faz com que o modelo faça previsões alinhadas demasiado estreitamente ou demasiado frouxamente com os dados de treino. Um modelo sobreajustado prevê os dados de treino muito bem porque aprendeu os detalhes e o ruído dos dados demasiado bem. Um modelo subajustado não é preciso pois não consegue analisar com precisão nem os seus dados de treino nem dados que ainda não 'viu'.

![overfitting model](../../../../translated_images/pt-PT/overfitting.1c132d92bfd93cb6.webp)
> Infografia por [Jen Looper](https://twitter.com/jenlooper)

## Ajuste de parâmetros

Uma vez concluído o treino inicial, observe a qualidade do modelo e considere melhorá-lo ajustando os seus 'hiperparâmetros'. Leia mais sobre o processo [na documentação](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters?WT.mc_id=academic-77952-leestott).

## Previsão

Este é o momento em que pode usar dados completamente novos para testar a precisão do seu modelo. Num cenário ML 'aplicado', onde está a construir recursos web para usar o modelo em produção, este processo pode envolver recolher a entrada do utilizador (um clique de botão, por exemplo) para definir uma variável e enviá-la para o modelo para inferência, ou avaliação.

Nestes módulos, irá descobrir como usar estes passos para preparar, construir, testar, avaliar e prever – todos os gestos de um cientista de dados e mais, enquanto avança na sua jornada para se tornar um engenheiro ML 'full stack'.

---

## 🚀Desafio

Desenhe um diagrama de fluxos refletindo os passos de um praticante de ML. Onde é que se vê agora no processo? Onde prevê que terá dificuldades? O que lhe parece fácil?

## [Questionário pós-aula](https://ff-quizzes.netlify.app/en/ml/)

## Revisão e Estudo Autónomo

Procure online entrevistas com cientistas de dados que discutam o seu trabalho diário. Aqui está [uma](https://www.youtube.com/watch?v=Z3IjgbbCEfs).

## Tarefa

[Entrevistar um cientista de dados](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Aviso Legal**:  
Este documento foi traduzido utilizando o serviço de tradução automática [Co-op Translator](https://github.com/Azure/co-op-translator). Embora nos esforcemos pela precisão, por favor tenha em conta que traduções automáticas podem conter erros ou imprecisões. O documento original na sua língua nativa deve ser considerado a fonte autorizada. Para informações críticas, recomenda-se tradução profissional humana. Não nos responsabilizamos por quaisquer mal-entendidos ou interpretações erradas resultantes do uso desta tradução.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->