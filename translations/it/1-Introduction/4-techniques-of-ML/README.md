# Tecniche di Machine Learning

Il processo di costruzione, utilizzo e manutenzione di modelli di machine learning e dei dati che utilizzano è un processo molto diverso da molti altri flussi di lavoro di sviluppo. In questa lezione, demistificheremo il processo e delineeremo le principali tecniche che devi conoscere. Tu:

- Comprenderai i processi alla base del machine learning a un livello alto.
- Esplorerai concetti di base come 'modelli', 'predizioni' e 'dati di addestramento'.

## [Quiz pre-lezione](https://ff-quizzes.netlify.app/en/ml/)

[![ML for beginners - Techniques of Machine Learning](https://img.youtube.com/vi/4NGM0U2ZSHU/0.jpg)](https://youtu.be/4NGM0U2ZSHU "ML for beginners - Techniques of Machine Learning")

> 🎥 Clicca sull'immagine sopra per un breve video che spiega questa lezione.

## Introduzione

A un livello alto, l'arte di creare processi di machine learning (ML) è composta da diversi passi:

1. **Decidi la domanda**. La maggior parte dei processi di ML inizia con una domanda che non può essere risposta da un semplice programma condizionale o da un motore basato su regole. Queste domande spesso ruotano attorno a predizioni basate su una raccolta di dati.
2. **Raccogli e prepara i dati**. Per poter rispondere alla tua domanda, hai bisogno di dati. La qualità e, a volte, la quantità dei tuoi dati determineranno quanto bene potrai rispondere alla tua domanda iniziale. Visualizzare i dati è un aspetto importante di questa fase. Questa fase include anche la suddivisione dei dati in un gruppo di addestramento e uno di test per costruire un modello.
3. **Scegli un metodo di addestramento**. A seconda della tua domanda e della natura dei tuoi dati, devi scegliere come vuoi addestrare un modello per riflettere al meglio i tuoi dati e fare predizioni accurate. Questa è la parte del tuo processo di ML che richiede competenze specifiche e, spesso, un considerevole numero di sperimentazioni.
4. **Addestra il modello**. Usando i tuoi dati di addestramento, utilizzerai vari algoritmi per addestrare un modello a riconoscere pattern nei dati. Il modello potrebbe sfruttare pesi interni che possono essere regolati per privilegiare certe parti dei dati rispetto ad altre per costruire un modello migliore.
5. **Valuta il modello**. Utilizzi dati che non ha mai visto prima (i dati di test) dal tuo set raccolto per vedere come il modello sta performando.
6. **Ottimizzazione dei parametri**. Basandoti sulle prestazioni del tuo modello, puoi ripetere il processo utilizzando parametri diversi, o variabili, che controllano il comportamento degli algoritmi usati per addestrare il modello.
7. **Predici**. Usa nuovi input per testare la precisione del tuo modello.

## Quale domanda porre

I computer sono particolarmente abili nel scoprire pattern nascosti nei dati. Questa utilità è molto utile per i ricercatori che hanno domande su un determinato dominio che non possono essere facilmente risposte creando un motore basato su regole condizionali. Dato un compito attuariale, per esempio, uno scienziato dei dati potrebbe essere in grado di costruire regole artigianali riguardo alla mortalità di fumatori vs non fumatori.

Quando molte altre variabili sono portate nell'equazione, tuttavia, un modello ML potrebbe rivelarsi più efficiente per predire i tassi di mortalità futuri basati sulla storia sanitaria passata. Un esempio più allegro potrebbe essere fare previsioni meteorologiche per il mese di aprile in una data località basandosi su dati che includono latitudine, longitudine, cambiamento climatico, prossimità all'oceano, schemi del getto d'aria e altro.

✅ Questa [presentazione](https://www2.cisl.ucar.edu/sites/default/files/2021-10/0900%20June%2024%20Haupt_0.pdf) sui modelli meteo offre una prospettiva storica sull'uso del ML nell'analisi meteorologica.  

## Compiti preliminari alla costruzione

Prima di iniziare a costruire il tuo modello, ci sono diversi compiti che devi completare. Per testare la tua domanda e formare un'ipotesi basata sulle predizioni di un modello, devi identificare e configurare diversi elementi.

### Dati

Per poter rispondere alla tua domanda con una certa certezza, hai bisogno di una buona quantità di dati del tipo giusto. Ci sono due cose che devi fare a questo punto:

- **Raccogliere dati**. Tenendo a mente la lezione precedente sull'equità nell'analisi dei dati, raccogli i tuoi dati con cura. Sii consapevole delle fonti di questi dati, di eventuali bias intrinseci e documenta la loro origine.
- **Preparare i dati**. Ci sono diversi passi nel processo di preparazione dei dati. Potresti dover raccogliere dati e normalizzarli se provengono da fonti diverse. Puoi migliorare la qualità e la quantità dei dati tramite vari metodi come convertire stringhe in numeri (come facciamo in [Clustering](../../5-Clustering/1-Visualize/README.md)). Potresti anche generare nuovi dati, basati su quelli originali (come facciamo in [Classification](../../4-Classification/1-Introduction/README.md)). Puoi pulire e modificare i dati (come faremo prima della lezione [Web App](../../3-Web-App/README.md)). Infine, potresti anche doverli randomizzare e mescolarli, a seconda delle tecniche di addestramento.

✅ Dopo aver raccolto e processato i tuoi dati, prenditi un momento per vedere se la loro forma ti permetterà di affrontare la tua domanda prevista. Potrebbe risultare che i dati non si comportino bene nel compito assegnato, come scopriamo nelle nostre lezioni di [Clustering](../../5-Clustering/1-Visualize/README.md)!

### Caratteristiche e Obiettivo

Una [feature](https://www.datasciencecentral.com/profiles/blogs/an-introduction-to-variable-and-feature-selection) è una proprietà misurabile dei tuoi dati. In molti dataset è espressa come intestazione di colonna come 'data', 'dimensione' o 'colore'. La tua variabile feature, di solito rappresentata come `X` nel codice, rappresenta la variabile di input che sarà usata per addestrare un modello.

Un obiettivo è ciò che stai cercando di prevedere. L'obiettivo, di solito rappresentato come `y` nel codice, rappresenta la risposta alla domanda che stai cercando di porre ai tuoi dati: a dicembre, quali **colori** di zucche saranno i più economici? A San Francisco, quali quartieri avranno il miglior **prezzo** immobiliare? A volte l'obiettivo è anche chiamato attributo etichetta.

### Selezionare la variabile feature

🎓 **Selezione e estrazione delle feature** Come si fa a sapere quale variabile scegliere quando si costruisce un modello? Probabilmente passerai attraverso un processo di selezione o estrazione delle feature per scegliere le variabili giuste per il modello più performante. Tuttavia, non sono la stessa cosa: "L'estrazione delle feature crea nuove feature da funzioni delle feature originali, mentre la selezione delle feature restituisce un sottoinsieme delle feature." ([fonte](https://wikipedia.org/wiki/Feature_selection))

### Visualizza i tuoi dati

Un aspetto importante del toolkit dello scienziato dei dati è il potere di visualizzare i dati usando diverse ottime librerie come Seaborn o MatPlotLib. Rappresentare visivamente i tuoi dati potrebbe permetterti di scoprire correlazioni nascoste che puoi sfruttare. Le tue visualizzazioni potrebbero anche aiutarti a scoprire bias o dati sbilanciati (come scopriamo in [Classification](../../4-Classification/2-Classifiers-1/README.md)).

### Suddividi il tuo dataset

Prima dell'addestramento, devi suddividere il tuo dataset in due o più parti di dimensioni disuguali che rappresentino comunque bene i dati.

- **Addestramento**. Questa parte del dataset viene adattata al tuo modello per addestrarlo. Questo set costituisce la maggioranza del dataset originale.
- **Test**. Un dataset di test è un gruppo indipendente di dati, spesso raccolti dal dato originale, che usi per confermare le prestazioni del modello costruito.
- **Validazione**. Un set di validazione è un gruppo indipendente più piccolo di esempi che usi per ottimizzare gli iperparametri del modello, o la sua architettura, per migliorarlo. A seconda delle dimensioni dei tuoi dati e della domanda che stai ponendo, potresti non aver bisogno di costruire questo terzo set (come notiamo in [Time Series Forecasting](../../7-TimeSeries/1-Introduction/README.md)).

## Costruire un modello

Usando i tuoi dati di addestramento, il tuo obiettivo è costruire un modello, o una rappresentazione statistica dei tuoi dati, usando vari algoritmi per **addestrarlo**. Addestrare un modello lo espone ai dati e gli permette di fare assunzioni sui pattern percepiti che scopre, convalida e accetta o rifiuta.

### Decidi un metodo di addestramento

A seconda della tua domanda e della natura dei tuoi dati, scegli un metodo per addestrarlo. Sfogliando la [documentazione di Scikit-learn](https://scikit-learn.org/stable/user_guide.html) - che usiamo in questo corso - puoi esplorare molti modi di addestrare un modello. A seconda della tua esperienza, potresti dover provare diversi metodi per costruire il miglior modello. Probabilmente passerai attraverso un processo in cui gli scienziati dei dati valutano la prestazione di un modello alimentandolo con dati non visti prima, controllando l'accuratezza, il bias e altri problemi che degradano la qualità, e selezionando il metodo di addestramento più appropriato per il compito.

### Addestra un modello

Munito dei tuoi dati di addestramento, sei pronto a 'adattarlo' per creare un modello. Noterai che in molte librerie ML trovi il codice 'model.fit' - è in questo momento che mandi la tua variabile feature come un array di valori (di solito 'X') e una variabile obiettivo (di solito 'y').

### Valuta il modello

Una volta completato il processo di addestramento (può richiedere molte iterazioni, o 'epoche', per addestrare un modello grande), sarai in grado di valutare la qualità del modello usando dati di test per misurarne le prestazioni. Questi dati sono un sottoinsieme dei dati originali che il modello non ha analizzato in precedenza. Puoi stampare una tabella di metriche sulla qualità del tuo modello.

🎓 **Model fitting**

Nel contesto del machine learning, il model fitting si riferisce all'accuratezza della funzione sottostante del modello mentre tenta di analizzare dati con cui non è familiare.

🎓 **Underfitting** e **overfitting** sono problemi comuni che degradano la qualità del modello, poiché il modello si adatta o troppo poco o troppo bene. Questo fa sì che il modello faccia predizioni troppo strettamente o troppo liberamente allineate con i dati di addestramento. Un modello overfit predice troppo bene i dati di addestramento perché ha imparato troppo bene i dettagli e il rumore dei dati. Un modello underfit non è accurato perché non può né analizzare accuratamente i dati di addestramento né i dati che non ha ancora 'visto'.

![overfitting model](../../../../translated_images/it/overfitting.1c132d92bfd93cb6.webp)
> Infografica di [Jen Looper](https://twitter.com/jenlooper)

## Ottimizzazione dei parametri

Una volta completato il primo addestramento, osserva la qualità del modello e considera di migliorarlo modificando i suoi 'iperparametri'. Leggi di più sul processo [nella documentazione](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters?WT.mc_id=academic-77952-leestott).

## Predizione

Questo è il momento in cui puoi usare dati completamente nuovi per testare la precisione del tuo modello. In un contesto ML 'applicato', in cui costruisci asset web per usare il modello in produzione, questo processo potrebbe coinvolgere la raccolta di input dall'utente (ad esempio la pressione di un pulsante) per impostare una variabile e inviarla al modello per inferenza, o valutazione.

In queste lezioni, scoprirai come usare questi passaggi per preparare, costruire, testare, valutare e predire - tutti i gesti di uno scienziato dei dati e altro ancora, mentre avanzi nel tuo percorso per diventare un ingegnere ML 'full stack'.

---

## 🚀Sfida

Disegna un diagramma di flusso che riflette i passaggi di un praticante ML. Dove ti vedi attualmente nel processo? Dove prevedi di incontrare difficoltà? Cosa ti sembra facile?

## [Quiz post-lezione](https://ff-quizzes.netlify.app/en/ml/)

## Revisione e Autoapprendimento

Cerca online interviste con scienziati dei dati che discutono del loro lavoro quotidiano. Eccone [una](https://www.youtube.com/watch?v=Z3IjgbbCEfs).

## Compito

[Intervista uno scienziato dei dati](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Disclaimer**:  
Questo documento è stato tradotto utilizzando il servizio di traduzione automatica [Co-op Translator](https://github.com/Azure/co-op-translator). Sebbene ci impegniamo per l’accuratezza, si prega di notare che le traduzioni automatiche possono contenere errori o inesattezze. Il documento originale nella sua lingua nativa deve essere considerato la fonte autorevole. Per informazioni critiche si raccomanda la traduzione professionale umana. Non siamo responsabili per eventuali malintesi o interpretazioni errate derivanti dall’uso di questa traduzione.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->