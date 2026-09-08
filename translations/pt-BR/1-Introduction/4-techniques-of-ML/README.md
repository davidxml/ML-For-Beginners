# Técnicas de Aprendizado de Máquina

O processo de construir, usar e manter modelos de aprendizado de máquina e os dados que eles utilizam é muito diferente de muitos outros fluxos de trabalho de desenvolvimento. Nesta lição, vamos desmistificar o processo e destacar as principais técnicas que você precisa conhecer. Você irá:

- Entender os processos que fundamentam o aprendizado de máquina em um nível geral.
- Explorar conceitos básicos como 'modelos', 'previsões' e 'dados de treinamento'.

## [Quiz pré-aula](https://ff-quizzes.netlify.app/en/ml/)

[![ML for beginners - Techniques of Machine Learning](https://img.youtube.com/vi/4NGM0U2ZSHU/0.jpg)](https://youtu.be/4NGM0U2ZSHU "ML for beginners - Techniques of Machine Learning")

> 🎥 Clique na imagem acima para assistir a um vídeo curto explicando esta lição.

## Introdução

Em alto nível, a arte de criar processos de aprendizado de máquina (ML) é composta por várias etapas:

1. **Decidir a pergunta**. A maioria dos processos de ML começa fazendo uma pergunta que não pode ser respondida por um programa condicional simples ou um motor de regras. Essas perguntas geralmente giram em torno de previsões baseadas em uma coleção de dados.
2. **Coletar e preparar dados**. Para poder responder à sua pergunta, você precisa de dados. A qualidade e, às vezes, a quantidade dos seus dados determinarão quão bem você pode responder à sua pergunta inicial. Visualizar dados é um aspecto importante desta fase. Esta fase também inclui dividir os dados em um grupo de treinamento e um de teste para construir um modelo.
3. **Escolher um método de treinamento**. Dependendo da sua pergunta e da natureza dos seus dados, você precisa escolher como deseja treinar um modelo para melhor refletir seus dados e fazer previsões precisas. Esta é a parte do seu processo de ML que requer expertise específica e, frequentemente, uma quantidade considerável de experimentação.
4. **Treinar o modelo**. Usando seus dados de treinamento, você usará vários algoritmos para treinar um modelo a reconhecer padrões nos dados. O modelo pode usar pesos internos que podem ser ajustados para privilegiar certas partes dos dados em relação a outras para construir um modelo melhor.
5. **Avaliar o modelo**. Você usa dados nunca antes vistos (seus dados de teste) do seu conjunto coletado para ver como o modelo está performando.
6. **Ajuste de parâmetros**. Com base no desempenho do seu modelo, você pode refazer o processo usando diferentes parâmetros, ou variáveis, que controlam o comportamento dos algoritmos usados para treinar o modelo.
7. **Prever**. Use novas entradas para testar a precisão do seu modelo.

## Que pergunta fazer

Computadores são particularmente habilidosos em descobrir padrões ocultos nos dados. Essa utilidade é muito útil para pesquisadores que têm perguntas sobre um domínio específico que não podem ser facilmente respondidas criando um motor de regras baseado em condições. Dada uma tarefa atuarial, por exemplo, um cientista de dados pode ser capaz de construir regras manuais sobre a mortalidade de fumantes versus não fumantes.

Quando muitas outras variáveis são trazidas para a equação, no entanto, um modelo de ML pode se mostrar mais eficiente para prever futuras taxas de mortalidade com base no histórico de saúde passado. Um exemplo mais animador pode ser fazer previsões meteorológicas para o mês de abril em uma determinada localização com base em dados que incluem latitude, longitude, mudanças climáticas, proximidade do oceano, padrões da corrente de jato e mais.

✅ Este [slide deck](https://www2.cisl.ucar.edu/sites/default/files/2021-10/0900%20June%2024%20Haupt_0.pdf) sobre modelos meteorológicos oferece uma perspectiva histórica sobre o uso de ML na análise do tempo.  

## Tarefas pré-construção

Antes de começar a construir seu modelo, existem várias tarefas que você precisa completar. Para testar sua pergunta e formar uma hipótese com base nas previsões de um modelo, você precisa identificar e configurar vários elementos.

### Dados

Para poder responder à sua pergunta com qualquer tipo de certeza, você precisa de uma boa quantidade de dados do tipo certo. Há duas coisas que você precisa fazer neste ponto:

- **Coletar dados**. Mantendo em mente a lição anterior sobre justiça na análise de dados, colete seus dados com cuidado. Esteja ciente das fontes desses dados, quaisquer vieses inerentes que possam possuir, e documente sua origem.
- **Preparar dados**. Existem várias etapas no processo de preparação dos dados. Você pode precisar reunir dados e normalizá-los se eles vierem de fontes diversas. Você pode melhorar a qualidade e quantidade dos dados por vários métodos, tais como converter strings em números (como fazemos em [Clustering](../../5-Clustering/1-Visualize/README.md)). Você também pode gerar novos dados, baseados nos originais (como fazemos em [Classification](../../4-Classification/1-Introduction/README.md)). Você pode limpar e editar os dados (como faremos antes da lição [Web App](../../3-Web-App/README.md)). Finalmente, você também pode precisar randomizá-los e embaralhá-los, dependendo das suas técnicas de treinamento.

✅ Após coletar e processar seus dados, reserve um momento para ver se sua forma permitirá que você responda à sua pergunta pretendida. Pode ser que os dados não tenham um bom desempenho na tarefa proposta, como descobrimos em nossas lições de [Clustering](../../5-Clustering/1-Visualize/README.md)!

### Características e Alvo

Uma [característica](https://www.datasciencecentral.com/profiles/blogs/an-introduction-to-variable-and-feature-selection) é uma propriedade mensurável dos seus dados. Em muitos conjuntos de dados, é expressa como um cabeçalho de coluna como 'data', 'tamanho' ou 'cor'. Sua variável de característica, geralmente representada como `X` no código, representa a variável de entrada que será usada para treinar um modelo.

Um alvo é algo que você está tentando prever. O alvo, geralmente representado como `y` no código, representa a resposta à pergunta que você está tentando fazer sobre seus dados: em dezembro, qual **cor** de abóboras será a mais barata? em San Francisco, quais bairros terão o melhor **preço** imobiliário? Às vezes, o alvo também é referido como atributo de rótulo.

### Selecionando sua variável de característica

🎓 **Seleção de Características e Extração de Características** Como saber qual variável escolher ao construir um modelo? Você provavelmente passará por um processo de seleção ou extração de características para escolher as variáveis certas para o modelo mais performático. No entanto, elas não são a mesma coisa: "Extração de características cria novas características a partir de funções das características originais, enquanto seleção de características retorna um subconjunto das características." ([fonte](https://wikipedia.org/wiki/Feature_selection))

### Visualize seus dados

Um aspecto importante do kit de ferramentas do cientista de dados é a capacidade de visualizar dados usando diversas bibliotecas excelentes como Seaborn ou MatPlotLib. Representar seus dados visualmente pode permitir que você descubra correlações ocultas que pode aproveitar. Suas visualizações também podem ajudar a descobrir viés ou dados desequilibrados (como descobrimos em [Classification](../../4-Classification/2-Classifiers-1/README.md)).

### Divida seu conjunto de dados

Antes do treinamento, você precisa dividir seu conjunto de dados em duas ou mais partes de tamanho desigual que ainda representem bem os dados.

- **Treinamento**. Esta parte do conjunto de dados é ajustada ao seu modelo para treiná-lo. Este conjunto constitui a maior parte do conjunto de dados original.
- **Teste**. Um conjunto de dados de teste é um grupo independente de dados, frequentemente colhido do dado original, que você usa para confirmar o desempenho do modelo construído.
- **Validação**. Um conjunto de validação é um grupo independente menor de exemplos que você usa para ajustar os hiperparâmetros, ou a arquitetura, do modelo para melhorar o modelo. Dependendo do tamanho dos seus dados e da pergunta que está fazendo, você pode não precisar construir esse terceiro conjunto (como observamos em [Previsão de Séries Temporais](../../7-TimeSeries/1-Introduction/README.md)).

## Construindo um modelo

Usando seus dados de treinamento, seu objetivo é construir um modelo, ou uma representação estatística dos seus dados, usando vários algoritmos para **treiná-lo**. Treinar um modelo o expõe aos dados e permite que ele faça suposições sobre padrões percebidos que descobre, valida, aceita ou rejeita.

### Decida um método de treinamento

Dependendo da sua pergunta e da natureza dos seus dados, você escolherá um método para treiná-lo. Explorando a [documentação do Scikit-learn](https://scikit-learn.org/stable/user_guide.html) - que usamos neste curso - você pode explorar muitas formas de treinar um modelo. Dependendo da sua experiência, pode ser necessário tentar vários métodos diferentes para construir o melhor modelo. É provável que você passe por um processo no qual cientistas de dados avaliam o desempenho de um modelo alimentando-o com dados nunca vistos, verificando a precisão, o viés e outros problemas que degradam a qualidade, e selecionando o método de treinamento mais apropriado para a tarefa em questão.

### Treine um modelo

Armado com seus dados de treinamento, você está pronto para 'ajustá-lo' para criar um modelo. Você notará que em muitas bibliotecas de ML você encontrará o código 'model.fit' - é nesse momento que você envia sua variável característica como um array de valores (geralmente 'X') e uma variável alvo (geralmente 'y').

### Avalie o modelo

Uma vez que o processo de treinamento esteja completo (pode levar muitas iterações, ou 'épocas', para treinar um modelo grande), você poderá avaliar a qualidade do modelo usando dados de teste para medir seu desempenho. Esses dados são um subconjunto dos dados originais que o modelo não analisou anteriormente. Você pode imprimir uma tabela de métricas sobre a qualidade do seu modelo.

🎓 **Ajuste do modelo**

No contexto do aprendizado de máquina, ajuste do modelo refere-se à precisão da função subjacente do modelo enquanto ele tenta analisar dados com os quais não está familiarizado.

🎓 **Underfitting** e **overfitting** são problemas comuns que degradam a qualidade do modelo, pois o modelo se ajusta ou não adequadamente. Isso causa o modelo a fazer previsões alinhadas ou muito pouco alinhadas com seus dados de treinamento. Um modelo overfit prevê os dados de treinamento muito bem porque aprendeu muito bem os detalhes e o ruído dos dados. Um modelo underfit não é preciso, pois não consegue analisar precisamente seus dados de treinamento nem dados que ainda não viu.

![overfitting model](../../../../translated_images/pt-BR/overfitting.1c132d92bfd93cb6.webp)
> Infográfico por [Jen Looper](https://twitter.com/jenlooper)

## Ajuste de parâmetros

Uma vez que seu treinamento inicial esteja completo, observe a qualidade do modelo e considere melhorá-lo ajustando seus 'hiperparâmetros'. Leia mais sobre o processo [na documentação](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters?WT.mc_id=academic-77952-leestott).

## Previsão

Este é o momento em que você pode usar dados completamente novos para testar a precisão do seu modelo. Em um ambiente de ML 'aplicado', onde você está construindo ativos web para usar o modelo em produção, esse processo pode envolver coletar entradas do usuário (um clique de botão, por exemplo) para definir uma variável e enviá-la ao modelo para inferência ou avaliação.

Nessas lições, você descobrirá como usar essas etapas para preparar, construir, testar, avaliar e prever - todos os gestos de um cientista de dados e mais, à medida que avança na jornada para se tornar um engenheiro de ML 'full stack'.

---

## 🚀Desafio

Desenhe um fluxograma refletindo as etapas de um praticante de ML. Onde você se vê agora no processo? Onde você prevê encontrar dificuldade? O que parece fácil para você?

## [Quiz pós-aula](https://ff-quizzes.netlify.app/en/ml/)

## Revisão & Estudo Autônomo

Procure online por entrevistas com cientistas de dados que discutem seu trabalho diário. Aqui está [uma](https://www.youtube.com/watch?v=Z3IjgbbCEfs).

## Tarefa

[Entrevistar um cientista de dados](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Aviso Legal**:  
Este documento foi traduzido utilizando o serviço de tradução por IA [Co-op Translator](https://github.com/Azure/co-op-translator). Embora nos esforcemos para garantir a precisão, esteja ciente de que traduções automáticas podem conter erros ou imprecisões. O documento original em seu idioma nativo deve ser considerado a fonte autoritativa. Para informações críticas, recomenda-se tradução profissional feita por humano. Não nos responsabilizamos por quaisquer mal-entendidos ou interpretações incorretas decorrentes do uso desta tradução.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->