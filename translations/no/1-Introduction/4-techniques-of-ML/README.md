# Teknikker for maskinlæring

Prosessen med å bygge, bruke og vedlikeholde maskinlæringsmodeller og dataene de bruker er en svært annerledes prosess enn mange andre utviklingsarbeidsflyter. I denne leksjonen vil vi avmystifisere prosessen og skissere de viktigste teknikkene du trenger å kjenne til. Du vil:

- Forstå prosessene som ligger til grunn for maskinlæring på et overordnet nivå.
- Utforske grunnleggende konsepter som 'modeller', 'prediksjoner' og 'treningsdata'.

## [Pre-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

[![ML for beginners - Techniques of Machine Learning](https://img.youtube.com/vi/4NGM0U2ZSHU/0.jpg)](https://youtu.be/4NGM0U2ZSHU "ML for beginners - Techniques of Machine Learning")

> 🎥 Klikk på bildet over for en kort video som går gjennom denne leksjonen.

## Introduksjon

På et overordnet nivå består kunstverket av å lage maskinlæringsprosesser (ML) av flere trinn:

1. **Bestem spørsmålet**. De fleste ML-prosesser starter med å stille et spørsmål som ikke kan besvares av et enkelt betinget program eller et regelbasert system. Disse spørsmålene dreier seg ofte om prediksjoner basert på en samling data.
2. **Samle inn og forberede data**. For å kunne svare på spørsmålet ditt trenger du data. Kvaliteten og av og til mengden av dataene dine vil avgjøre hvor godt du kan svare på det opprinnelige spørsmålet. Visualisering av data er en viktig del av denne fasen. Denne fasen inkluderer også å dele dataene i en trenings- og testgruppe for å bygge en modell.
3. **Velg en treningsmetode**. Avhengig av spørsmålet ditt og hva slags data du har, må du bestemme hvordan du vil trene en modell som best reflekterer dataene dine og gir nøyaktige prediksjoner. Dette er den delen av ML-prosessen som krever spesifikk ekspertise og ofte en betydelig mengde eksperimentering.
4. **Tren modellen**. Ved å bruke treningsdataene dine, vil du bruke ulike algoritmer til å trene en modell til å gjenkjenne mønstre i dataene. Modellen kan bruke interne vekter som kan justeres for å prioritere visse deler av dataene fremfor andre for å bygge en bedre modell.
5. **Evaluer modellen**. Du bruker data som aldri før er sett (testdataene dine) fra det innsamlede datasettet for å se hvordan modellen presterer.
6. **Parameterjustering**. Basert på modellens prestasjon kan du gjøre prosessen på nytt ved å bruke forskjellige parametere eller variabler som styrer adferden til algoritmene som brukes for å trene modellen.
7. **Predikere**. Bruk nye innganger for å teste nøyaktigheten til modellen din.

## Hvilket spørsmål skal stilles

Datamaskiner er spesielt dyktige til å oppdage skjulte mønstre i data. Denne nytten er svært hjelpsom for forskere som har spørsmål om et gitt domene som ikke lett lar seg besvare ved å lage et regelbasert betinget system. Gitt en aktuaroppgave, for eksempel, kan en dataforsker konstruere håndlagde regler rundt dødeligheten til røykere vs ikke-røykere.

Når mange andre variabler tas med i regnestykket, kan imidlertid en ML-modell vise seg å være mer effektiv for å forutsi fremtidige dødelighetsrater basert på tidligere helsehistorikk. Et mer lystigere eksempel kan være å lage værvarsler for april måned på et gitt sted basert på data som inkluderer breddegrad, lengdegrad, klimaendringer, nærhet til havet, mønstre i jetstrømmen og mer.

✅ Denne [slide-decken](https://www2.cisl.ucar.edu/sites/default/files/2021-10/0900%20June%2024%20Haupt_0.pdf) om værmodeller gir et historisk perspektiv på bruk av ML i væranalyse.  

## Oppgaver før bygging

Før du begynner å bygge modellen din, er det flere oppgaver du må fullføre. For å teste spørsmålet ditt og danne en hypotese basert på modellens prediksjoner må du identifisere og konfigurere flere elementer.

### Data

For å kunne svare på spørsmålet ditt med en viss sikkerhet, trenger du en god mengde data av riktig type. Det er to ting du må gjøre på dette tidspunktet:

- **Samle inn data**. Med tanke på forrige leksjon om rettferdighet i dataanalyse, samle inn dataene dine med omhu. Vær oppmerksom på kildene til disse dataene, eventuelle iboende skjevheter de kan ha, og dokumenter opprinnelsen.
- **Forbered data**. Det er flere steg i dataforberedelsesprosessen. Du kan trenge å sammenstille data og normalisere dem hvis de kommer fra ulike kilder. Du kan forbedre datakvaliteten og -mengden gjennom ulike metoder som å konvertere tekststrenger til tall (som vi gjør i [Clustering](../../5-Clustering/1-Visualize/README.md)). Du kan også generere nye data basert på originalene (som vi gjør i [Classification](../../4-Classification/1-Introduction/README.md)). Du kan rense og redigere data (som vi vil gjøre før [Web App](../../3-Web-App/README.md)-leksjonen). Til slutt kan det også være nødvendig å randomisere og stokke dataene, avhengig av treningsmetodene dine.

✅ Etter å ha samlet og behandlet dataene dine, ta et øyeblikk for å se om formen på dataene tillater deg å adressere det tiltenkte spørsmålet. Det kan hende datamaterialet ikke fungerer godt til oppgaven, som vi oppdager i våre [Clustering](../../5-Clustering/1-Visualize/README.md)-leksjoner!

### Egenskaper og målvariabel

En [feature](https://www.datasciencecentral.com/profiles/blogs/an-introduction-to-variable-and-feature-selection) er en målbar egenskap ved dataene dine. I mange datasett uttrykkes det som en kolonneoverskrift som 'dato', 'størrelse' eller 'farge'. Egenskapsvariabelen din, vanligvis representert som `X` i kode, er inngangsvariabelen som brukes for å trene en modell.

Et mål er noe du prøver å forutsi. Målvariabelen, vanligvis representert som `y` i kode, er svaret på spørsmålet du stiller om dataene dine: i desember, hvilken **farge** vil gresskarene være billigst? I San Francisco, hvilke bydeler vil ha den beste eiendoms**prisen**? Målet blir noen ganger også kalt en etikettattributt.

### Velge egenskapsvariabelen din

🎓 **Feature Selection og Feature Extraction** Hvordan vet du hvilken variabel du skal velge når du bygger en modell? Du vil sannsynligvis gå gjennom en prosess med funksjonsutvalg eller funksjonsekstraksjon for å velge de riktige variablene for den mest effektive modellen. De er imidlertid ikke det samme: "Feature extraction skaper nye funksjoner fra funksjoner av de opprinnelige funksjonene, mens feature selection returnerer et delsett av funksjonene." ([kilde](https://wikipedia.org/wiki/Feature_selection))

### Visualiser dataene dine

En viktig del av verktøykassen til en dataforsker er muligheten til å visualisere data ved hjelp av flere utmerkede biblioteker som Seaborn eller MatPlotLib. Å representere dataene dine visuelt kan gi deg mulighet til å avdekke skjulte korrelasjoner som du kan dra nytte av. Visualiseringene dine kan også hjelpe deg med å oppdage skjevheter eller ubalanserte data (som vi oppdager i [Classification](../../4-Classification/2-Classifiers-1/README.md)).

### Del datasettet ditt

Før treningen må du dele datasettet ditt i to eller flere deler av ulik størrelse som fortsatt representerer dataene godt.

- **Trening**. Denne delen av datasettet tilpasses modellen din for å trene den. Dette settet utgjør mesteparten av det opprinnelige datasettet.
- **Testing**. Et testdatasett er en uavhengig gruppe data, ofte hentet fra de opprinnelige dataene, som du bruker for å bekrefte ytelsen til den bygde modellen.
- **Validering**. Et valideringssett er en mindre uavhengig gruppe eksempler som du bruker for å justere modellens hyperparametere eller arkitektur for å forbedre modellen. Avhengig av datastørrelsen og spørsmålet du stiller, kan det hende du ikke trenger å bygge dette tredje settet (som vi bemerker i [Time Series Forecasting](../../7-TimeSeries/1-Introduction/README.md)).

## Bygge en modell

Ved å bruke treningsdataene dine, er målet ditt å bygge en modell, eller en statistisk representasjon av dataene dine, ved å bruke ulike algoritmer for å **trene** den. Å trene en modell eksponerer den for data og lar den gjøre antakelser om oppfattede mønstre den oppdager, validerer og godtar eller forkaster.

### Velg en treningsmetode

Avhengig av spørsmålet ditt og dataenes natur, vil du velge en metode for å trene den. Når du går gjennom [Scikit-learns dokumentasjon](https://scikit-learn.org/stable/user_guide.html) – som vi bruker i dette kurset – kan du utforske mange måter å trene en modell på. Avhengig av erfaringen din kan det hende du må prøve flere ulike metoder for å bygge den beste modellen. Du vil sannsynligvis gå gjennom en prosess hvor dataforskere evaluerer ytelsen til en modell ved å mate den med ukjente data, sjekke for nøyaktighet, skjevhet og andre kvalitetsreduserende problemer, og velge den mest passende treningsmetoden for oppgaven som skal utføres.

### Tren en modell

Bevæpnet med treningsdataene dine er du klar til å 'tilpasse' den for å lage en modell. Du vil legge merke til at i mange ML-biblioteker finner du koden 'model.fit' – det er på dette tidspunktet du sender inn egenskapsvariabelen som en matrise av verdier (vanligvis 'X') og målvariabelen (vanligvis 'y').

### Evaluer modellen

Når treningsprosessen er fullført (det kan ta mange iterasjoner, eller 'epoker', å trene en stor modell), vil du kunne evaluere modellens kvalitet ved å bruke testdata for å måle ytelsen. Disse dataene er et delsett av de opprinnelige dataene som modellen ikke har analysert tidligere. Du kan skrive ut en tabell med målinger om modellens kvalitet.

🎓 **Modelltilpasning**

I sammenheng med maskinlæring refererer modelltilpasning til nøyaktigheten til modellens underliggende funksjon mens den forsøker å analysere data den ikke kjenner til.

🎓 **Underfitting** og **overfitting** er vanlige problemer som forringer modellens kvalitet, da modellen enten passer for dårlig eller for godt. Dette gjør at modellen gjør prediksjoner som enten er for nært knyttet eller for løst knyttet til treningsdataene. En overfit modell predikerer treningsdataene for godt fordi den har lært detaljene og støyen i dataene for godt. En underfit modell er ikke nøyaktig fordi den verken kan analysere treningsdataene sine nøyaktig eller data den ikke har 'sett' før.

![overfitting model](../../../../translated_images/no/overfitting.1c132d92bfd93cb6.webp)
> Infografikk av [Jen Looper](https://twitter.com/jenlooper)

## Parameterjustering

Når den innledende treningen er fullført, observer modellens kvalitet og vurder å forbedre den ved å finjustere dens 'hyperparametere'. Les mer om prosessen [i dokumentasjonen](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters?WT.mc_id=academic-77952-leestott).

## Prediksjon

Dette er øyeblikket hvor du kan bruke helt nye data for å teste modellens nøyaktighet. I et 'anvendt' ML-miljø, hvor du bygger webressurser for å bruke modellen i produksjon, kan denne prosessen innebære å samle inn brukerinngang (for eksempel et knappetrykk) for å sette en variabel og sende den til modellen for inferens eller evaluering.

I disse leksjonene vil du oppdage hvordan du bruker disse trinnene for å forberede, bygge, teste, evaluere og predikere – alle bevegelsene til en dataforsker og mer, etter hvert som du utvikler deg til å bli en 'full stack' ML-ingeniør.

---

## 🚀Utfordring

Lag et flytskjema som gjenspeiler trinnene til en ML-utøver. Hvor ser du deg selv akkurat nå i prosessen? Hvor tror du du vil møte vanskeligheter? Hva virker lett for deg?

## [Post-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## Gjennomgang & Selvstudium

Søk på nettet etter intervjuer med dataforskere som diskuterer deres daglige arbeid. Her er [et](https://www.youtube.com/watch?v=Z3IjgbbCEfs).

## Oppgave

[Intervju en dataforsker](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Ansvarsfraskrivelse**:  
Dette dokumentet er oversatt ved hjelp av AI-oversettingstjenesten [Co-op Translator](https://github.com/Azure/co-op-translator). Selv om vi streber etter nøyaktighet, vennligst vær oppmerksom på at automatiserte oversettelser kan inneholde feil eller unøyaktigheter. Det originale dokumentet på dets opprinnelige språk skal betraktes som den autoritative kilden. For kritisk informasjon anbefales profesjonell menneskelig oversettelse. Vi er ikke ansvarlige for eventuelle misforståelser eller feiltolkninger som oppstår ved bruk av denne oversettelsen.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->