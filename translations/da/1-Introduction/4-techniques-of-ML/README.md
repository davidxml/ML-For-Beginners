# Teknikker inden for maskinlæring

Processen med at bygge, bruge og vedligeholde maskinlæringsmodeller og de data, de bruger, er en meget anderledes proces end mange andre udviklingsarbejdsflows. I denne lektion vil vi afmystificere processen og skitsere de vigtigste teknikker, du skal kende. Du vil:

- Forstå processerne, der ligger til grund for maskinlæring på et overordnet niveau.
- Udforske grundlæggende begreber som 'modeller', 'forudsigelser' og 'træningsdata'.

## [Quiz før forelæsning](https://ff-quizzes.netlify.app/en/ml/)

[![ML for beginners - Techniques of Machine Learning](https://img.youtube.com/vi/4NGM0U2ZSHU/0.jpg)](https://youtu.be/4NGM0U2ZSHU "ML for beginners - Techniques of Machine Learning")

> 🎥 Klik på billedet ovenfor for en kort video, der gennemgår denne lektion.

## Introduktion

På et højt niveau består håndværket ved oprettelse af maskinlæringsprocesser (ML) af en række trin:

1. **Beslut dig for spørgsmålet**. De fleste ML-processer starter med at stille et spørgsmål, som ikke kan besvares af et simpelt betinget program eller en regelbaseret motor. Disse spørgsmål drejer sig ofte om forudsigelser baseret på en samling data.
2. **Indsaml og forbered data**. For at kunne besvare dit spørgsmål har du brug for data. Kvaliteten og nogle gange også mængden af dine data vil afgøre, hvor godt du kan besvare dit oprindelige spørgsmål. Visualisering af data er en vigtig del af denne fase. Denne fase inkluderer også opdeling af data i en trænings- og testgruppe til at bygge en model.
3. **Vælg en træningsmetode**. Afhængig af dit spørgsmål og karakteren af dine data skal du vælge, hvordan du vil træne en model for bedst at afspejle dine data og lave nøjagtige forudsigelser. Dette er den del af din ML-proces, der kræver specifik ekspertise og ofte en betydelig mængde eksperimenter.
4. **Træn modellen**. Med dine træningsdata bruger du forskellige algoritmer til at træne en model til at genkende mønstre i dataene. Modellen kan benytte interne vægte, som kan justeres for at favorisere visse dele af dataene frem for andre for at bygge en bedre model.
5. **Evaluér modellen**. Du bruger data, som modellen ikke har set før (dine testdata) fra dit samlede datasæt for at se, hvordan modellen præsterer.
6. **Parametertuning**. Baseret på modellens præstation kan du gentage processen ved at bruge forskellige parametre eller variable, som styrer algoritmernes adfærd, der træner modellen.
7. **Forudsig**. Brug nye input til at teste din modells nøjagtighed.

## Hvilket spørgsmål man skal stille

Computere er særligt dygtige til at opdage skjulte mønstre i data. Denne egenskab er meget nyttig for forskere, som har spørgsmål om et givet domæne, som ikke let kan besvares ved at oprette en betingelsesbaseret regelsystem. Givet en aktuaropgave kan en dataforsker for eksempel konstruere håndlavede regler omkring dødeligheden for rygere kontra ikke-rygere.

Når mange andre variable bringes ind i ligningen, kan en ML-model imidlertid vise sig mere effektiv til at forudsige fremtidige dødelighedsrater baseret på tidligere sygehistorik. Et mere positivt eksempel kan være at lave vejrudsigter for april måned på et givet sted baseret på data, der inkluderer breddegrad, længdegrad, klimaforandringer, nærhed til havet, jetstrømmenes mønstre og mere.

✅ Dette [slide deck](https://www2.cisl.ucar.edu/sites/default/files/2021-10/0900%20June%2024%20Haupt_0.pdf) om vejrmæssige modeller giver et historisk perspektiv på brug af ML i vejr-analyse.

## Opgaver før modellering

Før du går i gang med at bygge din model, er der flere opgaver, du skal gennemføre. For at teste dit spørgsmål og danne en hypotese baseret på en modells forudsigelser, skal du identificere og konfigurere flere elementer.

### Data

For at kunne besvare dit spørgsmål med en vis sikkerhed har du brug for en god mængde data af den rette type. Der er to ting, du skal gøre på dette tidspunkt:

- **Indsaml data**. Med tanke på den tidligere lektion om fairness i dataanalyse, skal du indsamle dine data med omhu. Vær opmærksom på kilderne til disse data, eventuelle iboende skævheder og dokumentér oprindelsen.
- **Forbered data**. Der er flere trin i dataforberedelsesprocessen. Du kan have behov for at samle data og normalisere dem, hvis de kommer fra forskellige kilder. Du kan forbedre datakvaliteten og -mængden gennem forskellige metoder som konvertering af tekststrenge til tal (som vi gør i [Clustering](../../5-Clustering/1-Visualize/README.md)). Du kan også generere nye data baseret på de originale (som vi gør i [Classification](../../4-Classification/1-Introduction/README.md)). Du kan rense og redigere dataene (som vi vil gøre før [Web App](../../3-Web-App/README.md)-lektionen). Endelig kan du også have behov for at tilfældiggøre og blande dataene, afhængigt af dine træningsteknikker.

✅ Efter at have indsamlet og behandlet dine data, tag et øjeblik til at se, om deres form vil gøre det muligt for dig at adressere dit tiltænkte spørgsmål. Det kan være, at dataene ikke klarer sig godt i din givne opgave, som vi opdager i vores [Clustering](../../5-Clustering/1-Visualize/README.md)-lektioner!

### Features og mål

En [feature](https://www.datasciencecentral.com/profiles/blogs/an-introduction-to-variable-and-feature-selection) er en målbar egenskab ved dine data. I mange datasæt udtrykkes det som en kolonneoverskrift som 'dato', 'størrelse' eller 'farve'. Din feature-variabel, som oftest repræsenteres som `X` i kode, repræsenterer inputvariablen, der bruges til at træne en model.

Et mål er noget, du prøver at forudsige. Mål, som oftest repræsenteres som `y` i kode, repræsenterer svaret på det spørgsmål, du forsøger at stille til dine data: i december, hvilken **farve** græskar vil være billigst? i San Francisco, hvilke nabolag vil have den bedste ejendoms-**pris**? Nogle gange kaldes mål også for et label-attribut.

### Valg af din feature-variabel

🎓 **Feature Selection og Feature Extraction** Hvordan ved du, hvilken variabel du skal vælge, når du bygger en model? Du går sandsynligvis igennem en proces med feature selection eller feature extraction for at vælge de rigtige variable til den mest præsterende model. Det er dog ikke det samme: "Feature extraction skaber nye features ud fra funktioner af de oprindelige features, mens feature selection returnerer et underudvalg af features." ([kilde](https://wikipedia.org/wiki/Feature_selection))

### Visualisér dine data

En vigtig del af dataforskerens værktøjskasse er evnen til at visualisere data ved hjælp af flere fremragende biblioteker som Seaborn eller MatPlotLib. At repræsentere dine data visuelt kan give dig mulighed for at opdage skjulte sammenhænge, som du kan udnytte. Dine visualiseringer kan også hjælpe dig med at opdage skævhed eller ubalancerede data (som vi opdager i [Classification](../../4-Classification/2-Classifiers-1/README.md)).

### Del dit datasæt

Inden træning skal du opdele dit datasæt i to eller flere dele af ulige størrelse, som stadig repræsenterer dataene godt.

- **Træning**. Denne del af datasættet bruges til at træne din model. Denne del udgør størstedelen af det oprindelige datasæt.
- **Test**. Et testdatasæt er en uafhængig gruppe af data, ofte udtaget fra det oprindelige datasæt, som du bruger til at bekræfte bygningens models præstation.
- **Validering**. Et valideringssæt er en mindre uafhængig gruppe af eksempler, som du bruger til at finjustere modellens hyperparametre eller arkitektur for at forbedre modellen. Afhængigt af størrelsen på dine data og spørgsmålet, du stiller, behøver du måske ikke bygge dette tredje sæt (som vi noterer i [Time Series Forecasting](../../7-TimeSeries/1-Introduction/README.md)).

## Bygning af en model

Ved hjælp af dine træningsdata er dit mål at bygge en model eller en statistisk repræsentation af dine data ved hjælp af forskellige algoritmer til at **træne** den. Træning af en model udsætter den for data og tillader den at lave antagelser om opfattede mønstre, den opdager, validerer og accepterer eller afviser.

### Beslut dig for en træningsmetode

Afhængigt af dit spørgsmål og karakteren af dine data vælger du en metode til at træne den. Hvis du går igennem [Scikit-learn's dokumentation](https://scikit-learn.org/stable/user_guide.html) - som vi bruger i dette kursus - kan du udforske mange måder at træne en model på. Afhængigt af din erfaring kan du være nødt til at prøve flere forskellige metoder for at bygge den bedste model. Du vil sandsynligvis gennemgå en proces, hvor dataforskere evaluerer en modells præstation ved at fodre den med uanset data, tjekke for nøjagtighed, bias og andre problemstillinger, der nedbryder kvalitet, og vælge den mest passende træningsmetode til den givne opgave.

### Træn en model

Bevæbnet med dine træningsdata er du klar til at 'fitte' den for at skabe en model. Du vil bemærke, at du i mange ML-biblioteker finder koden 'model.fit' - det er på dette tidspunkt, at du sender din feature-variabel som en værdiarray (normalt 'X') og en målvariabel (normalt 'y').

### Evaluér modellen

Når træningsprocessen er færdig (det kan tage mange iterationer, eller 'epochs', at træne en stor model), vil du kunne evaluere modellens kvalitet ved at bruge testdata til at måle dens præstation. Disse data er en delmængde af de oprindelige data, som modellen ikke tidligere har analyseret. Du kan udskrive en tabel med metrikker om modellens kvalitet.

🎓 **Model fitting**

I forbindelse med maskinlæring refererer model fitting til nøjagtigheden af modellens underliggende funktion, mens den forsøger at analysere data, den ikke er bekendt med.

🎓 **Underfitting** og **overfitting** er almindelige problemer, der reducerer modellens kvalitet, fordi modellen enten passer ikke godt nok eller for godt. Det får modellen til at lave forudsigelser, som enten er for tæt forbundet med eller for løst forbundet med træningsdataene. En overfit model forudsiger træningsdata for godt, fordi den har lært detaljerne og støjen i dataene for godt. En underfit model er ikke præcis, da den hverken kan analysere sine træningsdata korrekt eller data, den ikke har 'set' endnu.

![overfitting model](../../../../translated_images/da/overfitting.1c132d92bfd93cb6.webp)
> Infografik af [Jen Looper](https://twitter.com/jenlooper)

## Parametertuning

Når din indledende træning er færdig, kan du se på modellens kvalitet og overveje at forbedre den ved at justere dens 'hyperparametre'. Læs mere om processen [i dokumentationen](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters?WT.mc_id=academic-77952-leestott).

## Forudsigelse

Dette er øjeblikket, hvor du kan bruge helt nye data til at teste din modells nøjagtighed. I en 'anvendt' ML-sammenhæng, hvor du bygger webressourcer til at bruge modellen i produktion, kan denne proces involvere at indsamle brugerinput (for eksempel et tryk på en knap) for at sætte en variabel og sende den til modellen for inferens eller evaluering.

I disse lektioner vil du opdage, hvordan du bruger disse trin til at forberede, bygge, teste, evaluere og forudsige – alle gestusserne af en dataforsker og mere, efterhånden som du skrider frem i din rejse mod at blive en 'full stack' ML-ingeniør.

---

## 🚀Udfordring

Tegn et flowdiagram, der afspejler trinnene for en ML-udøver. Hvor ser du dig selv lige nu i processen? Hvor forudser du, at du vil finde vanskeligheder? Hvad virker let for dig?

## [Quiz efter forelæsning](https://ff-quizzes.netlify.app/en/ml/)

## Gennemgang & Selvstudie

Søg online efter interviews med dataforskere, der diskuterer deres daglige arbejde. Her er [et](https://www.youtube.com/watch?v=Z3IjgbbCEfs).

## Opgave

[Interview en dataforsker](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Ansvarsfraskrivelse**:  
Dette dokument er blevet oversat ved hjælp af AI-oversættelsestjenesten [Co-op Translator](https://github.com/Azure/co-op-translator). Selvom vi bestræber os på nøjagtighed, skal du være opmærksom på, at automatiserede oversættelser kan indeholde fejl eller unøjagtigheder. Det originale dokument på dets oprindelige sprog betragtes som den autoritative kilde. For kritisk information anbefales professionel menneskelig oversættelse. Vi påtager os intet ansvar for misforståelser eller fejltolkninger, der opstår som følge af brugen af denne oversættelse.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->