# Technieken van Machine Learning

Het proces van het bouwen, gebruiken en onderhouden van machine learning-modellen en de data die ze gebruiken is een heel ander proces dan veel andere ontwikkelingsworkflows. In deze les zullen we het proces ontrafelen en de belangrijkste technieken schetsen die je moet kennen. Je zult:

- Het proces achter machine learning op een hoog niveau begrijpen.
- Basisbegrippen zoals 'modellen', 'voorspellingen' en 'trainingsdata' verkennen.

## [Voorleesspelquiz](https://ff-quizzes.netlify.app/en/ml/)

[![ML voor beginners - Technieken van Machine Learning](https://img.youtube.com/vi/4NGM0U2ZSHU/0.jpg)](https://youtu.be/4NGM0U2ZSHU "ML voor beginners - Technieken van Machine Learning")

> 🎥 Klik op de afbeelding hierboven voor een korte video waarin deze les wordt behandeld.

## Introductie

Op een hoog niveau bestaat het vakmanschap van het creëren van machine learning (ML)-processen uit een aantal stappen:

1. **Bepaal de vraag**. De meeste ML-processen beginnen met het stellen van een vraag die niet beantwoord kan worden met een eenvoudig conditioneel programma of op regels gebaseerd systeem. Deze vragen draaien vaak om voorspellingen op basis van een verzameling data.
2. **Verzamel en bereid data voor**. Om je vraag te kunnen beantwoorden, heb je data nodig. De kwaliteit en soms ook de hoeveelheid van je data bepaalt hoe goed je je oorspronkelijke vraag kunt beantwoorden. Het visualiseren van data is een belangrijk onderdeel van deze fase. Deze fase omvat ook het splitsen van de data in een trainings- en testgroep om een model te bouwen.
3. **Kies een trainingsmethode**. Afhankelijk van je vraag en de aard van je data, moet je kiezen hoe je een model wilt trainen om je data het best te weerspiegelen en nauwkeurige voorspellingen te maken. Dit is het gedeelte van je ML-proces dat specifieke expertise vereist en vaak een aanzienlijke hoeveelheid experimenteren.
4. **Train het model**. Met je trainingsdata gebruik je verschillende algoritmen om een model te trainen dat patronen in de data herkent. Het model kan interne gewichten gebruiken die kunnen worden aangepast om bepaalde delen van de data voorrang te geven om zo een beter model te bouwen.
5. **Evalueer het model**. Je gebruikt nooit eerder geziene data (je testdata) uit je verzamelde set om te zien hoe het model presteert.
6. **Parameterafstemming**. Op basis van de prestaties van je model kun je het proces opnieuw doorlopen met andere parameters, of variabelen, die het gedrag van de algoritmen die het model trainen aansturen.
7. **Voorspel**. Gebruik nieuwe invoer om de nauwkeurigheid van je model te testen.

## Welke vraag te stellen

Computers zijn bijzonder goed in het ontdekken van verborgen patronen in data. Deze bruikbaarheid is heel handig voor onderzoekers die vragen hebben over een bepaald domein die niet makkelijk beantwoord kunnen worden door een op voorwaardelijke regels gebaseerd systeem te maken. Bij een actuariele taak, bijvoorbeeld, kan een data scientist handgemaakte regels construeren rond de sterfte van rokers versus niet-rokers.

Wanneer veel andere variabelen in de vergelijking worden gebracht, kan een ML-model echter efficiënter blijken te zijn bij het voorspellen van toekomstige sterftecijfers op basis van eerdere gezondheidsgeschiedenis. Een vrolijker voorbeeld is het maken van weersvoorspellingen voor de maand april op een gegeven locatie op basis van data die latitude, longitude, klimaatverandering, nabijheid van de oceaan, patronen van de straalstroom en meer bevat.

✅ Deze [slide deck](https://www2.cisl.ucar.edu/sites/default/files/2021-10/0900%20June%2024%20Haupt_0.pdf) over weermodellen biedt een historisch perspectief op het gebruik van ML in weeranalyse.  

## Taken vóór het bouwen

Voordat je begint met het bouwen van je model, moet je verschillende taken voltooien. Om je vraag te testen en een hypothese te vormen op basis van de voorspellingen van een model, moet je verschillende elementen identificeren en configureren.

### Data

Om je vraag met enige zekerheid te kunnen beantwoorden, heb je voldoende data van het juiste type nodig. Er zijn twee dingen die je op dit punt moet doen:

- **Verzamel data**. Houd rekening met de eerdere les over eerlijkheid in data-analyse en verzamel je data zorgvuldig. Wees je bewust van de bronnen van deze data, eventuele inherente vooroordelen die het kan bevatten, en documenteer de oorsprong.
- **Bereid data voor**. Er zijn verschillende stappen in het voorbereidingsproces van data. Je moet mogelijk data samenvoegen en normaliseren als het uit diverse bronnen komt. Je kunt de kwaliteit en kwantiteit van de data verbeteren door verschillende methoden zoals het omzetten van strings naar getallen (zoals we doen in [Clustering](../../5-Clustering/1-Visualize/README.md)). Je kunt ook nieuwe data genereren op basis van de originele (zoals we doen in [Classificatie](../../4-Classification/1-Introduction/README.md)). Je kunt de data schoonmaken en bewerken (zoals we zullen doen voorafgaand aan de [Web App](../../3-Web-App/README.md) les). Ten slotte moet je het mogelijk randomizen en schudden, afhankelijk van je trainingsmethoden.

✅ Na het verzamelen en verwerken van je data, neem even de tijd om te controleren of de vorm ervan je in staat stelt je bedoelde vraag te beantwoorden. Het kan zijn dat de data niet goed presteert voor jouw gegeven taak, zoals we ontdekken in onze [Clustering](../../5-Clustering/1-Visualize/README.md) lessen!

### Features en target

Een [feature](https://www.datasciencecentral.com/profiles/blogs/an-introduction-to-variable-and-feature-selection) is een meetbare eigenschap van je data. In veel datasets wordt het uitgedrukt als een kolomkop zoals 'datum', 'grootte' of 'kleur'. Je featurevariabele, meestal weergegeven als `X` in code, vertegenwoordigt de invoervariabele die gebruikt zal worden om een model te trainen.

Een target is iets dat je probeert te voorspellen. Target, meestal weergegeven als `y` in code, is het antwoord op de vraag die je aan je data stelt: in december, welke **kleur** pompoenen zullen het goedkoopst zijn? In San Francisco, welke buurten zullen de beste vastgoed**prijs** hebben? Soms wordt target ook wel een labelattribuut genoemd.

### Je featurevariabele kiezen

🎓 **Feature selectie en feature extractie** Hoe weet je welke variabele je moet kiezen bij het bouwen van een model? Je zult waarschijnlijk door een proces van feature selectie of feature extractie gaan om de juiste variabelen te kiezen voor het best presterende model. Ze zijn echter niet hetzelfde: "Feature extractie creëert nieuwe features uit functies van de originele features, terwijl feature selectie een subset van de features teruggeeft." ([bron](https://wikipedia.org/wiki/Feature_selection))

### Visualiseer je data

Een belangrijk aspect van het gereedschap van de data scientist is de mogelijkheid om data te visualiseren met behulp van verschillende uitstekende bibliotheken zoals Seaborn of MatPlotLib. Het visueel weergeven van je data kan je in staat stellen verborgen correlaties te ontdekken die je kunt benutten. Je visualisaties kunnen je ook helpen vooroordelen of onevenwichtige data te ontdekken (zoals we ontdekken in [Classificatie](../../4-Classification/2-Classifiers-1/README.md)).

### Splits je dataset

Voor het trainen moet je je dataset splitsen in twee of meer ongelijkwaardige delen die de data toch goed representeren.

- **Training**. Dit deel van de dataset is geschikt voor je model om het te trainen. Deze set vormt het grootste deel van de originele dataset.
- **Testen**. Een testdataset is een onafhankelijke groep data, vaak verzameld uit de originele data, die je gebruikt om de prestaties van het gebouwde model te bevestigen.
- **Valideren**. Een validatieset is een kleinere onafhankelijke groep voorbeelden die je gebruikt om de hyperparameters, of architectuur, van het model af te stemmen om het te verbeteren. Afhankelijk van de omvang van je data en de vraag die je stelt heb je dit derde deel mogelijk niet nodig (zoals we opmerken in [Time Series Forecasting](../../7-TimeSeries/1-Introduction/README.md)).

## Een model bouwen

Met behulp van je trainingsdata is het je doel om een model te bouwen, of een statistische weergave van je data, met behulp van verschillende algoritmes om het te **trainen**. Het trainen van een model stelt het bloot aan data en stelt het in staat aannames te maken over waargenomen patronen die het ontdekt, valideert en accepteert of verwerpt.

### Kies een trainingsmethode

Afhankelijk van je vraag en de aard van je data kies je een methode om het te trainen. Door de documentatie van [Scikit-learn](https://scikit-learn.org/stable/user_guide.html) - die we in deze cursus gebruiken - te doorlopen, kun je vele manieren verkennen om een model te trainen. Afhankelijk van je ervaring moet je mogelijk verschillende methoden proberen om het beste model te bouwen. Je zult waarschijnlijk een proces doorlopen waarbij data scientists de prestaties van een model evalueren door het onzichtbare data te voeren, te controleren op nauwkeurigheid, vooringenomenheid en andere kwaliteitsverminderende problemen, en de meest geschikte trainingsmethode voor de taak te kiezen.

### Train een model

Met je trainingsdata ben je klaar om het te 'fitten' om een model te creëren. Je zult merken dat in veel ML-bibliotheken de code 'model.fit' voorkomt - dit is het moment dat je je featurevariabele als een array van waarden (meestal 'X') en een targetvariabele (meestal 'y') verstuurt.

### Evalueer het model

Zodra het trainingsproces voltooid is (het kan vele iteraties, of 'epochs', duren om een groot model te trainen), kun je de kwaliteit van het model evalueren door testdata te gebruiken om de prestaties te beoordelen. Deze data is een subset van de oorspronkelijke data die het model nog niet eerder heeft geanalyseerd. Je kunt een tabel met metrics over de kwaliteit van je model printen.

🎓 **Model fitting**

In de context van machine learning verwijst model fitting naar de nauwkeurigheid van de onderliggende functie van het model terwijl het probeert data te analyseren waar het niet mee vertrouwd is.

🎓 **Onderfitting** en **overfitting** zijn veelvoorkomende problemen die de kwaliteit van het model verminderen, doordat het model het niet goed genoeg of juist te goed past. Dit veroorzaakt dat het model voorspellingen maakt die te nauw aansluiten bij of juist te los zijn ten opzichte van de trainingsdata. Een overfit model voorspelt trainingsdata te goed omdat het de details en ruis van de data te goed heeft geleerd. Een underfit model is niet nauwkeurig omdat het noch zijn trainingsdata, noch data die het nog niet heeft 'gezien', nauwkeurig kan analyseren.

![overfitting model](../../../../translated_images/nl/overfitting.1c132d92bfd93cb6.webp)
> Infographic door [Jen Looper](https://twitter.com/jenlooper)

## Parameterafstemming

Zodra je initiële training voltooid is, observeer je de kwaliteit van het model en overweeg je het te verbeteren door de 'hyperparameters' aan te passen. Lees meer over dit proces [in de documentatie](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters?WT.mc_id=academic-77952-leestott).

## Voorspelling

Dit is het moment waarop je compleet nieuwe data kunt gebruiken om de nauwkeurigheid van je model te testen. In een 'toegepaste' ML-omgeving, waar je webassets bouwt om het model in productie te gebruiken, kan dit proces het verzamelen van gebruikersinvoer inhouden (bijvoorbeeld een druk op een knop) om een variabele in te stellen en naar het model te sturen voor inferentie, of evaluatie.

In deze lessen ontdek je hoe je deze stappen gebruikt om voor te bereiden, bouwen, testen, evalueren en voorspellen - alle handelingen van een data scientist en meer, terwijl je vordert in je reis om een 'full stack' ML-engineer te worden.

---

## 🚀Uitdaging

Teken een stroomdiagram dat de stappen van een ML-praktijker weergeeft. Waar zie je jezelf nu in het proces? Waar verwacht je moeilijkheden? Wat lijkt je makkelijk?

## [Nabespreking quiz](https://ff-quizzes.netlify.app/en/ml/)

## Review & Zelfstudie

Zoek online naar interviews met data scientists die hun dagelijkse werk bespreken. Hier is [één](https://www.youtube.com/watch?v=Z3IjgbbCEfs).

## Opdracht

[Interview een data scientist](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Disclaimer**:  
Dit document is vertaald met behulp van de AI-vertalingsdienst [Co-op Translator](https://github.com/Azure/co-op-translator). Hoewel we streven naar nauwkeurigheid, dient u er rekening mee te houden dat automatische vertalingen fouten of onnauwkeurigheden kunnen bevatten. Het originele document in de oorspronkelijke taal moet als de gezaghebbende bron worden beschouwd. Voor kritieke informatie wordt professionele menselijke vertaling aanbevolen. Wij zijn niet aansprakelijk voor enige misverstanden of verkeerde interpretaties die voortvloeien uit het gebruik van deze vertaling.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->