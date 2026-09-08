# Introduzione al machine learning

## [Quiz pre-lezione](https://ff-quizzes.netlify.app/en/ml/)

---

[![ML per principianti - Introduzione al Machine Learning per principianti](https://img.youtube.com/vi/6mSx_KJxcHI/0.jpg)](https://youtu.be/6mSx_KJxcHI "ML per principianti - Introduzione al Machine Learning per principianti")

> 🎥 Clicca sull'immagine sopra per un breve video che descrive questa lezione.

Benvenuto a questo corso sul machine learning classico per principianti! Che tu sia completamente nuovo in questo argomento, o un praticante esperto di ML che vuole ripassare un settore, siamo felici di averti con noi! Vogliamo creare un punto di partenza amichevole per il tuo studio del ML e saremmo felici di valutare, rispondere e incorporare i tuoi [commenti](https://github.com/microsoft/ML-For-Beginners/discussions).

[![Introduzione al ML](https://img.youtube.com/vi/h0e2HAPTGF4/0.jpg)](https://youtu.be/h0e2HAPTGF4 "Introduzione al ML")

> 🎥 Clicca sull'immagine sopra per un video: John Guttag del MIT introduce il machine learning

---
## Iniziare con il machine learning

Prima di iniziare con questo curriculum, devi avere il tuo computer configurato e pronto per eseguire notebook localmente.

- **Configura la tua macchina con questi video**. Usa i seguenti link per imparare [come installare Python](https://youtu.be/CXZYvNRIAKM) nel tuo sistema e [impostare un editor di testo](https://youtu.be/EU8eayHWoZg) per lo sviluppo.
- **Impara Python**. È anche consigliato avere una conoscenza base di [Python](https://docs.microsoft.com/learn/paths/python-language/?WT.mc_id=academic-77952-leestott), un linguaggio di programmazione utile per i data scientist che usiamo in questo corso.
- **Impara Node.js e JavaScript**. Usiamo anche JavaScript alcune volte in questo corso nello sviluppo di app web, quindi dovrai avere installati [node](https://nodejs.org) e [npm](https://www.npmjs.com/) e avere a disposizione [Visual Studio Code](https://code.visualstudio.com/) sia per lo sviluppo in Python che in JavaScript.
- **Crea un account GitHub**. Poiché ci hai trovato qui su [GitHub](https://github.com), potresti già avere un account, ma se no, creane uno e poi fai un fork di questo curriculum per usarlo da te. (Sentiti libero di lasciarci una stella, anche 😊)
- **Esplora Scikit-learn**. Familiarizza con [Scikit-learn](https://scikit-learn.org/stable/user_guide.html), un insieme di librerie ML che usiamo come riferimento in queste lezioni.

---
## Cos'è il machine learning?

Il termine 'machine learning' è uno dei termini più popolari e frequentemente usati oggi. C'è una seria possibilità che tu abbia sentito questo termine almeno una volta se hai qualche familiarità con la tecnologia, indipendentemente dal dominio in cui lavori. Le meccaniche del machine learning, tuttavia, sono un mistero per la maggior parte delle persone. Per un principiante di machine learning, l'argomento può a volte sembrare schiacciante. Perciò, è importante capire cosa sia realmente il machine learning e impararlo passo dopo passo, attraverso esempi pratici.

---
## La curva dell'entusiasmo

![curva hype ml](../../../../translated_images/it/hype.07183d711a17aafe.webp)

> Google Trends mostra la recente 'curva dell'entusiasmo' del termine 'machine learning'

---
## Un universo misterioso

Viviamo in un universo pieno di affascinanti misteri. Grandi scienziati come Stephen Hawking, Albert Einstein e molti altri hanno dedicato la loro vita alla ricerca di informazioni significative che svelino i misteri del mondo intorno a noi. Questa è la condizione umana dell'apprendimento: un bambino umano impara cose nuove e scopre la struttura del suo mondo anno dopo anno mentre cresce fino all'età adulta.

---
## Il cervello del bambino

Il cervello e i sensi di un bambino percepiscono i fatti del loro ambiente e gradualmente imparano i modelli nascosti della vita che aiutano il bambino a creare regole logiche per identificare i modelli appresi. Il processo di apprendimento del cervello umano rende gli esseri umani la creatura vivente più sofisticata di questo mondo. Imparare continuamente scoprendo modelli nascosti e poi innovando su quei modelli ci rende migliori e migliori durante tutta la nostra vita. Questa capacità di apprendimento e la capacità di evoluzione sono legate a un concetto chiamato [plasticità cerebrale](https://www.simplypsychology.org/brain-plasticity.html). In superficie, possiamo tracciare alcune similitudini motivazionali tra il processo di apprendimento del cervello umano e i concetti di machine learning.

---
## Il cervello umano

Il [cervello umano](https://www.livescience.com/29365-human-brain.html) percepisce cose dal mondo reale, elabora le informazioni percepite, prende decisioni razionali e compie determinate azioni basate sulle circostanze. Questo è ciò che chiamiamo comportamento intelligente. Quando programmiamo una versione simile del processo comportamentale intelligente in una macchina, si chiama intelligenza artificiale (AI).

---
## Alcuna terminologia

Sebbene i termini possano essere confusi, il machine learning (ML) è un sottoinsieme importante dell'intelligenza artificiale. **Il ML si occupa di utilizzare algoritmi specializzati per scoprire informazioni significative e trovare modelli nascosti dai dati percepiti per corroborare il processo decisionale razionale**.

---
## AI, ML, Deep Learning

![AI, ML, deep learning, data science](../../../../translated_images/it/ai-ml-ds.537ea441b124ebf6.webp)

> Un diagramma che mostra le relazioni tra AI, ML, deep learning e data science. Infografica di [Jen Looper](https://twitter.com/jenlooper) ispirata da [questa grafica](https://softwareengineering.stackexchange.com/questions/366996/distinction-between-ai-ml-neural-networks-deep-learning-and-data-mining)

---
## Concetti da trattare

In questo curriculum, copriremo solo i concetti fondamentali del machine learning che un principiante deve conoscere. Tratteremo ciò che chiamiamo 'machine learning classico' principalmente usando Scikit-learn, un'ottima libreria usata da molti studenti per apprendere le basi. Per comprendere concetti più ampi di intelligenza artificiale o deep learning, una solida conoscenza fondamentale del machine learning è indispensabile, e quindi vogliamo offrirla qui.

---
## In questo corso imparerai:

- i concetti fondamentali del machine learning
- la storia del ML
- ML e equità
- tecniche ML di regressione
- tecniche ML di classificazione
- tecniche ML di clustering
- tecniche ML di elaborazione del linguaggio naturale
- tecniche ML di previsione di serie temporali
- apprendimento per rinforzo
- applicazioni reali del ML

---
## Cosa non tratteremo

- deep learning
- reti neurali
- AI

Per offrire una migliore esperienza di apprendimento, eviteremo le complessità delle reti neurali, del 'deep learning' - costruzione di modelli a molti strati usando reti neurali - e dell'AI, che tratteremo in un curriculum differente. Offriremo anche un prossimo curriculum di data science per focalizzarci su quell'aspetto di questo campo più ampio.

---
## Perché studiare il machine learning?

Il machine learning, da una prospettiva sistemica, è definito come la creazione di sistemi automatizzati che possono apprendere modelli nascosti dai dati per aiutare a prendere decisioni intelligenti.

Questa motivazione è liberamente ispirata a come il cervello umano impara certe cose basandosi sui dati percepiti dal mondo esterno.

✅ Pensa per un attimo al motivo per cui un'azienda vorrebbe usare strategie di machine learning piuttosto che creare un motore basato su regole hard-coded.

---
## Perché la qualità dei dati è importante

Dati di alta qualità migliorano le prestazioni del modello. Dati scadenti o rumorosi possono portare a previsioni inaccurate, anche usando algoritmi avanzati di machine learning.

---
## Applicazioni del machine learning

Le applicazioni del machine learning sono ora praticamente ovunque, e sono ubiquitarie come i dati che fluiscono nelle nostre società, generati dai nostri smartphone, dispositivi connessi e altri sistemi. Considerando l'immenso potenziale degli algoritmi di machine learning all'avanguardia, i ricercatori hanno esplorato la loro capacità di risolvere problemi reali multidimensionali e multidisciplinari con ottimi risultati positivi.

---
## Esempi di ML applicato

**Puoi usare il machine learning in molti modi**:

- Per prevedere la probabilità di malattia dalla storia medica o dai referti di un paziente.
- Per sfruttare i dati meteo per prevedere eventi meteorologici.
- Per capire il sentimento di un testo.
- Per rilevare fake news e fermare la diffusione della propaganda.

Finanza, economia, scienze della terra, esplorazione spaziale, ingegneria biomedica, scienze cognitive e persino campi delle scienze umane hanno adattato il machine learning per risolvere i complessi problemi di elaborazione dati dei loro domini.

---
## Conclusione

Il machine learning automatizza il processo di scoperta dei modelli trovando intuizioni significative dai dati reali o generati. Si è dimostrato altamente prezioso in ambito aziendale, sanitario e finanziario, tra gli altri.

Nel prossimo futuro, comprendere le basi del machine learning sarà indispensabile per persone di qualsiasi settore a causa della sua diffusione.

---
# 🚀 Sfida

Disegna, su carta o usando un'app online come [Excalidraw](https://excalidraw.com/), la tua comprensione delle differenze tra AI, ML, deep learning e data science. Aggiungi alcune idee su problemi che ciascuna di queste tecniche è adatta a risolvere.

# [Quiz post-lezione](https://ff-quizzes.netlify.app/en/ml/)

---
# Revisione & Autoapprendimento

Per imparare di più su come lavorare con algoritmi ML nel cloud, segui questo [Percorso di apprendimento](https://docs.microsoft.com/learn/paths/create-no-code-predictive-models-azure-machine-learning/?WT.mc_id=academic-77952-leestott).

Segui un [Percorso di apprendimento](https://docs.microsoft.com/learn/modules/introduction-to-machine-learning/?WT.mc_id=academic-77952-leestott) sulle basi del ML.

---
# Compito

[Inizia subito](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Disclaimer**:
Questo documento è stato tradotto utilizzando il servizio di traduzione AI [Co-op Translator](https://github.com/Azure/co-op-translator). Sebbene ci impegniamo per garantire la precisione, si prega di notare che le traduzioni automatizzate possono contenere errori o imprecisioni. Il documento originale nella sua lingua nativa deve essere considerato la fonte autorevole. Per informazioni critiche, si raccomanda una traduzione professionale effettuata da un essere umano. Non siamo responsabili per eventuali malintesi o interpretazioni errate derivanti dall’uso di questa traduzione.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->