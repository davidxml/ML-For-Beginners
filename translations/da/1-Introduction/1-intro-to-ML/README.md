# Introduktion til maskinlæring

## [Quiz før lektionen](https://ff-quizzes.netlify.app/en/ml/)

---

[![ML for beginners - Introduction to Machine Learning for Beginners](https://img.youtube.com/vi/6mSx_KJxcHI/0.jpg)](https://youtu.be/6mSx_KJxcHI "ML for beginners - Introduction to Machine Learning for Beginners")

> 🎥 Klik på billedet ovenfor for en kort video, der gennemgår denne lektion.

Velkommen til dette kursus om klassisk maskinlæring for begyndere! Uanset om du er helt ny på dette emne, eller en erfaren ML-praktiker, der ønsker at friske et område op, er vi glade for at have dig med! Vi ønsker at skabe et venligt afsæt for din ML-studier og vil gerne evaluere, besvare og inkorporere din [feedback](https://github.com/microsoft/ML-For-Beginners/discussions).

[![Introduction to ML](https://img.youtube.com/vi/h0e2HAPTGF4/0.jpg)](https://youtu.be/h0e2HAPTGF4 "Introduction to ML")

> 🎥 Klik på billedet ovenfor for en video: MIT's John Guttag introducerer maskinlæring

---
## Kom godt i gang med maskinlæring

Før du starter med dette pensum, skal din computer være opsat og klar til at køre notebooks lokalt.

- **Konfigurer din maskine med disse videoer**. Brug følgende links til at lære [hvordan man installerer Python](https://youtu.be/CXZYvNRIAKM) på dit system og [opsæt en teksteditor](https://youtu.be/EU8eayHWoZg) til udvikling.
- **Lær Python**. Det anbefales også at have en grundlæggende forståelse af [Python](https://docs.microsoft.com/learn/paths/python-language/?WT.mc_id=academic-77952-leestott), et programmeringssprog nyttigt for dataloger, som vi bruger i dette kursus.
- **Lær Node.js og JavaScript**. Vi bruger også JavaScript et par gange i dette kursus, når vi bygger webapps, så du skal have [node](https://nodejs.org) og [npm](https://www.npmjs.com/) installeret, samt have [Visual Studio Code](https://code.visualstudio.com/) tilgængeligt til både Python- og JavaScript-udvikling.
- **Opret en GitHub-konto**. Da du fandt os her på [GitHub](https://github.com), har du måske allerede en konto, men hvis ikke, opret en og lav derefter en fork af dette pensum til brug på egen hånd. (Du er også velkommen til at give os en stjerne 😊)
- **Udforsk Scikit-learn**. Sæt dig ind i [Scikit-learn](https://scikit-learn.org/stable/user_guide.html), et sæt ML-biblioteker, som vi refererer til i disse lektioner.

---
## Hvad er maskinlæring?

Udtrykket 'maskinlæring' er et af de mest populære og ofte anvendte begreber i dag. Der er en ikke-triviel mulighed for, at du har hørt dette begreb mindst én gang, hvis du har nogen form for kendskab til teknologi, uanset hvilket område du arbejder indenfor. Mekanikken bag maskinlæring er dog et mysterium for de fleste. For en maskinlæringsbegynder kan emnet nogle gange føles overvældende. Derfor er det vigtigt at forstå, hvad maskinlæring faktisk er, og at lære om det trin for trin gennem praktiske eksempler.

---
## Hype-kurven

![ml hype curve](../../../../translated_images/da/hype.07183d711a17aafe.webp)

> Google Trends viser den seneste 'hypekurve' for udtrykket 'maskinlæring'

---
## Et mystisk univers

Vi lever i et univers fuldt af fascinerende mysterier. Store videnskabsfolk som Stephen Hawking, Albert Einstein og mange flere har viet deres liv til at søge meningsfuld information, der afdækker mysterierne i verden omkring os. Dette er den menneskelige lærings tilstand: et menneskebarn lærer nye ting og afdækker strukturen i sin verden år for år, mens det vokser op til voksen.

---
## Barnets hjerne

Et barns hjerne og sanser opfatter fakta fra omgivelserne og lærer gradvist livets skjulte mønstre, som hjælper barnet med at skabe logiske regler til at identificere lærte mønstre. Læringsprocessen i den menneskelige hjerne gør mennesker til den mest sofistikerede levende skabning i denne verden. At lære kontinuerligt ved at opdage skjulte mønstre og derefter innovere ud fra disse mønstre gør det muligt for os at gøre os selv bedre og bedre gennem livet. Denne læringsevne og udviklende kapacitet relaterer til et begreb kaldet [hjernes plasticitet](https://www.simplypsychology.org/brain-plasticity.html). Overfladisk kan vi drage nogle motiverende ligheder mellem den menneskelige hjernes læringsproces og konceptet maskinlæring.

---
## Den menneskelige hjerne

Den [menneskelige hjerne](https://www.livescience.com/29365-human-brain.html) opfatter ting fra den virkelige verden, bearbejder den opfattede information, træffer rationelle beslutninger og udfører visse handlinger baseret på omstændighederne. Det er det, vi kalder at opføre sig intelligent. Når vi programmerer en efterligning af denne intelligente adfærdsproces til en maskine, kaldes det kunstig intelligens (AI).

---
## Nogle begreber

Selvom begreberne kan forveksles, er maskinlæring (ML) en vigtig delmængde af kunstig intelligens. **ML handler om at bruge specialiserede algoritmer til at afdække meningsfuld information og finde skjulte mønstre fra opfattede data for at understøtte den rationelle beslutningsproces**.

---
## AI, ML, Deep Learning

![AI, ML, deep learning, data science](../../../../translated_images/da/ai-ml-ds.537ea441b124ebf6.webp)

> Et diagram, der viser sammenhængen mellem AI, ML, deep learning og data science. Infografik af [Jen Looper](https://twitter.com/jenlooper) inspireret af [denne grafik](https://softwareengineering.stackexchange.com/questions/366996/distinction-between-ai-ml-neural-networks-deep-learning-and-data-mining)

---
## Begreber vi vil dække

I dette pensum vil vi kun dække de grundlæggende begreber inden for maskinlæring, som en begynder skal kende. Vi dækker det, vi kalder 'klassisk maskinlæring', primært ved brug af Scikit-learn, et fremragende bibliotek, som mange studerende bruger til at lære det grundlæggende. For at forstå bredere begreber inden for kunstig intelligens eller deep learning er en stærk grundlæggende viden om maskinlæring uundværlig, og derfor vil vi tilbyde den her.

---
## I dette kursus vil du lære:

- grundlæggende begreber inden for maskinlæring
- historien om ML
- ML og fairness
- regressions ML-teknikker
- klassifikations ML-teknikker
- clustering ML-teknikker
- naturlig sprogbehandling ML-teknikker
- tidsserieprognose ML-teknikker
- forstærkningslæring
- virkelige anvendelser af ML

---
## Hvad vi ikke vil dække

- deep learning
- neurale netværk
- AI

For at give en bedre læringsoplevelse vil vi undgå kompleksiteterne ved neurale netværk, 'deep learning' - mange-lags modelbygning ved hjælp af neurale netværk - og AI, som vi vil diskutere i et andet pensum. Vi vil også tilbyde et kommende data science-pensum til at fokusere på den del af dette større felt.

---
## Hvorfor studere maskinlæring?

Maskinlæring defineres fra et systemperspektiv som skabelsen af automatiserede systemer, der kan lære skjulte mønstre fra data for at hjælpe med at træffe intelligente beslutninger.

Denne motivation er løst inspireret af, hvordan den menneskelige hjerne lærer visse ting baseret på de data, den opfatter fra den ydre verden.

✅ Tænk et øjeblik over, hvorfor en virksomhed skulle ønske at bruge maskinlæringsstrategier fremfor at lave en hårdkodet, regelsbaseret motor.

---
## Hvorfor datakvalitet betyder noget

Data af høj kvalitet forbedrer modellernes præstation. Dårlige eller støjende data kan føre til unøjagtige forudsigelser, selv ved brug af avancerede maskinlæringsalgoritmer.

---
## Anvendelser af maskinlæring

Anvendelser af maskinlæring findes nu næsten overalt og er lige så allestedsnærværende som de data, der flyder rundt i vores samfund, genereret af vores smartphones, forbundne enheder og andre systemer. Givet det enorme potentiale i moderne maskinlæringsalgoritmer har forskere udforsket deres evne til at løse multidimensionelle og tværfaglige virkelige problemer med gode positive resultater.

---
## Eksempler på anvendt ML

**Du kan bruge maskinlæring på mange måder**:

- Til at forudsige sandsynligheden for sygdom ud fra en patients medicinske historie eller rapporter.
- Til at udnytte vejrdata til at forudsige vejrbegivenheder.
- Til at forstå sentimentet i en tekst.
- Til at opdage falske nyheder for at stoppe spredningen af propaganda.

Finance, økonomi, jordvidenskab, rumforskning, biomedicinsk ingeniørkunst, kognitiv videnskab og endda humaniora har tilpasset maskinlæring til at løse de krævende databehandlingsproblemer i deres områder.

---
## Konklusion

Maskinlæring automatiserer processen med mønstergenkendelse ved at finde meningsfulde indsigter fra virkelige eller genererede data. Det har vist sig at være yderst værdifuldt i forretning, sundhed og finansielle anvendelser, blandt andre.

I den nærmeste fremtid vil det at forstå grundlæggende maskinlæring være et must for folk fra alle områder på grund af dens udbredte anvendelse.

---
# 🚀 Udfordring

Skitsér på papir eller ved hjælp af en online-app som [Excalidraw](https://excalidraw.com/) din forståelse af forskellene mellem AI, ML, deep learning og data science. Tilføj nogle ideer om problemer, som hver af disse teknikker er gode til at løse.

# [Quiz efter lektionen](https://ff-quizzes.netlify.app/en/ml/)

---
# Gennemgang & Selvstudium

For at lære mere om, hvordan du kan arbejde med ML-algoritmer i skyen, følg denne [Learning Path](https://docs.microsoft.com/learn/paths/create-no-code-predictive-models-azure-machine-learning/?WT.mc_id=academic-77952-leestott).

Tag en [Learning Path](https://docs.microsoft.com/learn/modules/introduction-to-machine-learning/?WT.mc_id=academic-77952-leestott) om det grundlæggende inden for ML.

---
# Opgave

[Kom i gang](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Ansvarsfraskrivelse**:
Dette dokument er blevet oversat ved hjælp af AI-oversættelsestjenesten [Co-op Translator](https://github.com/Azure/co-op-translator). Selvom vi bestræber os på nøjagtighed, skal du være opmærksom på, at automatiserede oversættelser kan indeholde fejl eller unøjagtigheder. Det originale dokument på dets oprindelige sprog bør betragtes som den autoritative kilde. For kritisk information anbefales professionel menneskelig oversættelse. Vi påtager os intet ansvar for misforståelser eller fejltolkninger, der opstår som følge af brugen af denne oversættelse.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->