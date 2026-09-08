# Introdução ao aprendizado de máquina

## [Questionário pré-aula](https://ff-quizzes.netlify.app/en/ml/)

---

[![ML para iniciantes - Introdução ao Aprendizado de Máquina para Iniciantes](https://img.youtube.com/vi/6mSx_KJxcHI/0.jpg)](https://youtu.be/6mSx_KJxcHI "ML para iniciantes - Introdução ao Aprendizado de Máquina para Iniciantes")

> 🎥 Clique na imagem acima para um vídeo curto que aborda esta aula.

Bem-vindo a este curso sobre aprendizado de máquina clássico para iniciantes! Quer você seja completamente novo nesse assunto ou um praticante experiente de ML querendo se atualizar em uma área, estamos felizes por tê-lo conosco! Queremos criar um ponto de partida amigável para seu estudo de ML e ficaremos felizes em avaliar, responder e incorporar seu [feedback](https://github.com/microsoft/ML-For-Beginners/discussions).

[![Introdução ao ML](https://img.youtube.com/vi/h0e2HAPTGF4/0.jpg)](https://youtu.be/h0e2HAPTGF4 "Introdução ao ML")

> 🎥 Clique na imagem acima para um vídeo: John Guttag do MIT apresenta aprendizado de máquina

---
## Começando com aprendizado de máquina

Antes de começar com este currículo, você precisa ter seu computador configurado e pronto para executar notebooks localmente.

- **Configure sua máquina com estes vídeos**. Use os links seguintes para aprender [como instalar Python](https://youtu.be/CXZYvNRIAKM) em seu sistema e [configurar um editor de texto](https://youtu.be/EU8eayHWoZg) para desenvolvimento.
- **Aprenda Python**. Também é recomendado ter uma compreensão básica de [Python](https://docs.microsoft.com/learn/paths/python-language/?WT.mc_id=academic-77952-leestott), uma linguagem de programação útil para cientistas de dados que usamos neste curso.
- **Aprenda Node.js e JavaScript**. Também usamos JavaScript algumas vezes neste curso ao construir aplicativos web, então você precisará ter [node](https://nodejs.org) e [npm](https://www.npmjs.com/) instalados, assim como o [Visual Studio Code](https://code.visualstudio.com/) disponível para desenvolvimento em Python e JavaScript.
- **Crie uma conta no GitHub**. Como você nos encontrou aqui no [GitHub](https://github.com), talvez já tenha uma conta, mas se não, crie uma e depois faça um fork deste currículo para usar por conta própria. (Fique à vontade para nos dar uma estrela também 😊)
- **Explore o Scikit-learn**. Familiarize-se com o [Scikit-learn](https://scikit-learn.org/stable/user_guide.html), um conjunto de bibliotecas de ML que usamos como referência nestas lições.

---
## O que é aprendizado de máquina?

O termo "aprendizado de máquina" é um dos termos mais populares e frequentemente usados hoje em dia. Existe uma possibilidade considerável de que você tenha ouvido esse termo ao menos uma vez se tem alguma familiaridade com tecnologia, independentemente do domínio em que trabalha. A mecânica do aprendizado de máquina, entretanto, é um mistério para a maioria das pessoas. Para um iniciante em aprendizado de máquina, o assunto às vezes pode parecer assustador. Por isso, é importante entender o que realmente é aprendizado de máquina e aprender sobre ele passo a passo, por meio de exemplos práticos.

---
## A curva do hype

![curva do hype do ml](../../../../translated_images/pt-BR/hype.07183d711a17aafe.webp)

> O Google Trends mostra a recente "curva do hype" do termo "aprendizado de máquina"

---
## Um universo misterioso

Vivemos em um universo cheio de mistérios fascinantes. Grandes cientistas como Stephen Hawking, Albert Einstein e muitos outros dedicaram suas vidas à busca por informações significativas que desvendaram os mistérios do mundo ao nosso redor. Esta é a condição humana do aprendizado: uma criança aprende coisas novas e descobre a estrutura do seu mundo ano após ano enquanto cresce até a idade adulta.

---
## O cérebro da criança

O cérebro e os sentidos de uma criança percebem os fatos ao seu redor e gradualmente aprendem os padrões ocultos da vida que ajudam a criança a criar regras lógicas para identificar padrões aprendidos. O processo de aprendizado do cérebro humano torna os humanos a criatura viva mais sofisticada deste mundo. Aprender continuamente descobrindo padrões ocultos e depois inovando sobre esses padrões nos permite melhorar cada vez mais ao longo da vida. Essa capacidade de aprendizado e evolução está relacionada a um conceito chamado [plasticidade cerebral](https://www.simplypsychology.org/brain-plasticity.html). Superficialmente, podemos traçar algumas similaridades motivadoras entre o processo de aprendizado do cérebro humano e os conceitos de aprendizado de máquina.

---
## O cérebro humano

O [cérebro humano](https://www.livescience.com/29365-human-brain.html) percebe coisas do mundo real, processa as informações percebidas, toma decisões racionais e executa certas ações baseadas nas circunstâncias. Isso é o que chamamos de comportamento inteligente. Quando programamos uma réplica do processo comportamental inteligente em uma máquina, chamamos isso de inteligência artificial (IA).

---
## Alguns termos

Embora os termos possam ser confundidos, aprendizado de máquina (ML) é um subconjunto importante da inteligência artificial. **O ML está preocupado com o uso de algoritmos especializados para descobrir informações significativas e encontrar padrões ocultos a partir dos dados percebidos para corroborar o processo de tomada de decisão racional**.

---
## IA, ML, Deep Learning

![IA, ML, deep learning, ciência de dados](../../../../translated_images/pt-BR/ai-ml-ds.537ea441b124ebf6.webp)

> Diagrama mostrando as relações entre IA, ML, deep learning e ciência de dados. Infográfico por [Jen Looper](https://twitter.com/jenlooper) inspirado por [este gráfico](https://softwareengineering.stackexchange.com/questions/366996/distinction-between-ai-ml-neural-networks-deep-learning-and-data-mining)

---
## Conceitos a cobrir

Neste currículo, abordaremos apenas os conceitos principais de aprendizado de máquina que um iniciante deve conhecer. Cobrimos o que chamamos de "aprendizado de máquina clássico", principalmente usando o Scikit-learn, uma excelente biblioteca usada por muitos estudantes para aprender o básico. Para entender conceitos mais amplos de inteligência artificial ou deep learning, um forte conhecimento fundamental de aprendizado de máquina é indispensável, e por isso gostaríamos de oferecê-lo aqui.

---
## Neste curso você vai aprender:

- conceitos centrais do aprendizado de máquina
- a história do ML
- ML e equidade
- técnicas de ML para regressão
- técnicas de ML para classificação
- técnicas de ML para agrupamento (clustering)
- técnicas de ML para processamento de linguagem natural
- técnicas de ML para previsão de séries temporais
- aprendizado por reforço
- aplicações reais para ML

---
## O que não vamos cobrir

- deep learning
- redes neurais
- IA

Para oferecer uma melhor experiência de aprendizado, evitaremos as complexidades das redes neurais, "deep learning" – construção de modelos com múltiplas camadas usando redes neurais – e IA, que abordaremos em um currículo diferente. Também ofereceremos um currículo futuro de ciência de dados para focar nesse aspecto desse campo maior.

---
## Por que estudar aprendizado de máquina?

Do ponto de vista dos sistemas, aprendizado de máquina é definido como a criação de sistemas automatizados que podem aprender padrões ocultos a partir de dados para ajudar na tomada de decisões inteligentes.

Essa motivação é vagamente inspirada por como o cérebro humano aprende certas coisas com base nos dados que percebe do mundo exterior.

✅ Pense por um minuto por que uma empresa gostaria de tentar usar estratégias de aprendizado de máquina em vez de criar um motor baseado em regras codificadas manualmente.

---
## Por que a qualidade dos dados é importante

Dados de alta qualidade melhoram o desempenho do modelo. Dados pobres ou ruidosos podem levar a previsões imprecisas, mesmo usando algoritmos avançados de aprendizado de máquina.

---
## Aplicações do aprendizado de máquina

As aplicações do aprendizado de máquina estão agora praticamente em toda parte e são tão onipresentes quanto os dados que circulam em nossas sociedades, gerados por nossos smartphones, dispositivos conectados e outros sistemas. Considerando o imenso potencial de algoritmos de aprendizado de máquina de última geração, pesquisadores vêm explorando sua capacidade de resolver problemas da vida real multidimensionais e multidisciplinares com ótimos resultados positivos.

---
## Exemplos de ML aplicado

**Você pode usar aprendizado de máquina de várias maneiras**:

- Para prever a probabilidade de uma doença a partir do histórico médico ou relatórios de um paciente.
- Para aproveitar dados meteorológicos para prever eventos do tempo.
- Para entender o sentimento de um texto.
- Para detectar notícias falsas e impedir a disseminação de propaganda.

Finanças, economia, ciências da terra, exploração espacial, engenharia biomédica, ciência cognitiva e até campos das humanidades adaptaram o aprendizado de máquina para resolver problemas árduos e pesados em processamento de dados de seus domínios.

---
## Conclusão

O aprendizado de máquina automatiza o processo de descoberta de padrões encontrando informações significativas a partir de dados do mundo real ou gerados. Ele se provou extremamente valioso em negócios, saúde e aplicações financeiras, entre outras.

No futuro próximo, entender o básico de aprendizado de máquina será obrigatório para pessoas de qualquer área devido à sua ampla adoção.

---
# 🚀 Desafio

Esboce, no papel ou usando um aplicativo online como o [Excalidraw](https://excalidraw.com/), seu entendimento das diferenças entre IA, ML, deep learning e ciência de dados. Adicione algumas ideias de problemas que cada uma dessas técnicas é boa para resolver.

# [Questionário pós-aula](https://ff-quizzes.netlify.app/en/ml/)

---
# Revisão e Autoestudo

Para aprender mais sobre como trabalhar com algoritmos de ML na nuvem, siga este [Caminho de Aprendizagem](https://docs.microsoft.com/learn/paths/create-no-code-predictive-models-azure-machine-learning/?WT.mc_id=academic-77952-leestott).

Faça um [Caminho de Aprendizagem](https://docs.microsoft.com/learn/modules/introduction-to-machine-learning/?WT.mc_id=academic-77952-leestott) sobre os conceitos básicos de ML.

---
# Tarefa

[Comece a trabalhar](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Aviso Legal**:
Este documento foi traduzido usando o serviço de tradução por IA [Co-op Translator](https://github.com/Azure/co-op-translator). Embora nos esforcemos pela precisão, por favor, esteja ciente de que traduções automatizadas podem conter erros ou imprecisões. O documento original em seu idioma nativo deve ser considerado a fonte autorizada. Para informações críticas, recomenda-se tradução profissional humana. Não nos responsabilizamos por quaisquer mal-entendidos ou interpretações incorretas decorrentes do uso desta tradução.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->