# Introduksjon til maskinlæring

## [Quiz før forelesning](https://ff-quizzes.netlify.app/en/ml/)

---

[![ML for beginners - Introduction to Machine Learning for Beginners](https://img.youtube.com/vi/6mSx_KJxcHI/0.jpg)](https://youtu.be/6mSx_KJxcHI "ML for beginners - Introduction to Machine Learning for Beginners")

> 🎥 Klikk på bildet ovenfor for en kort video som går gjennom denne leksjonen.

Velkommen til dette kurset om klassisk maskinlæring for nybegynnere! Enten du er helt ny på dette området, eller en erfaren ML-praktiker som ønsker å friske opp kunnskapen, er vi glade for at du vil bli med oss! Vi ønsker å skape et vennlig utgangspunkt for din ML-studier og vil gjerne evaluere, svare på, og inkorporere din [tilbakemelding](https://github.com/microsoft/ML-For-Beginners/discussions).

[![Introduction to ML](https://img.youtube.com/vi/h0e2HAPTGF4/0.jpg)](https://youtu.be/h0e2HAPTGF4 "Introduction to ML")

> 🎥 Klikk på bildet ovenfor for en video: MITs John Guttag introduserer maskinlæring

---
## Komme i gang med maskinlæring

Før du starter med dette pensumet, må du ha datamaskinen din satt opp og klar til å kjøre notebooks lokalt.

- **Konfigurer maskinen din med disse videoene**. Bruk følgende lenker for å lære [hvordan installere Python](https://youtu.be/CXZYvNRIAKM) på systemet ditt og [sette opp en teksteditor](https://youtu.be/EU8eayHWoZg) for utvikling.
- **Lær Python**. Det anbefales også å ha en grunnleggende forståelse av [Python](https://docs.microsoft.com/learn/paths/python-language/?WT.mc_id=academic-77952-leestott), et programmeringsspråk som er nyttig for dataforskere, og som vi bruker i dette kurset.
- **Lær Node.js og JavaScript**. Vi bruker også JavaScript noen ganger i dette kurset når vi bygger webapper, så du må ha [node](https://nodejs.org) og [npm](https://www.npmjs.com/) installert, samt [Visual Studio Code](https://code.visualstudio.com/) tilgjengelig for både Python- og JavaScript-utvikling.
- **Opprett en GitHub-konto**. Siden du fant oss her på [GitHub](https://github.com), har du kanskje allerede en konto, men hvis ikke, opprett en og fork dette pensumet for å bruke det selv. (Gi oss gjerne en stjerne også 😊)
- **Utforsk Scikit-learn**. Gjør deg kjent med [Scikit-learn](https://scikit-learn.org/stable/user_guide.html), et sett med ML-biblioteker som vi refererer til i disse leksjonene.

---
## Hva er maskinlæring?

Begrepet 'maskinlæring' er et av de mest populære og ofte brukte uttrykkene i dag. Det er ganske sannsynlig at du har hørt dette begrepet minst én gang hvis du har en viss kjennskap til teknologi, uansett hvilket felt du jobber innenfor. Mekanikken bak maskinlæring er imidlertid et mysterium for de fleste. For en nybegynner kan temaet noen ganger føles overveldende. Derfor er det viktig å forstå hva maskinlæring faktisk er, og lære det steg for steg gjennom praktiske eksempler.

---
## Hype-kurven

![ml hype curve](../../../../translated_images/no/hype.07183d711a17aafe.webp)

> Google Trends viser den nylige 'hype-kurven' for begrepet 'maskinlæring'

---
## Et mystisk univers

Vi lever i et univers fullt av fascinerende mysterier. Store forskere som Stephen Hawking, Albert Einstein og mange flere har viet livene sine til å lete etter meningsfull informasjon som avdekker mysteriene i verden rundt oss. Dette er menneskets natur når det gjelder læring: et menneskebarn lærer nye ting og avdekker strukturen i sin verden år for år mens det vokser opp til voksen alder.

---
## Barnets hjerne

Et barns hjerne og sanser oppfatter fakta om omgivelsene og lærer gradvis de skjulte mønstrene i livet, noe som hjelper barnet å lage logiske regler for å identifisere lærte mønstre. Læreprosessen i den menneskelige hjernen gjør mennesker til de mest sofistikerte levende skapningene i denne verden. Å lære kontinuerlig ved å oppdage skjulte mønstre og deretter innovere på dem gjør at vi kan forbedre oss selv gjennom hele livet. Denne læreevnen og utviklende kapasiteten er knyttet til et begrep kalt [hjerneplastisitet](https://www.simplypsychology.org/brain-plasticity.html). Overfladisk kan vi trekke noen motiverende likheter mellom menneskehjernens læringsprosess og begrepene i maskinlæring.

---
## Den menneskelige hjernen

[Den menneskelige hjernen](https://www.livescience.com/29365-human-brain.html) oppfatter ting fra virkeligheten, prosesserer den oppfattede informasjonen, tar rasjonelle beslutninger og utfører visse handlinger basert på omstendighetene. Dette kaller vi å opptre intelligent. Når vi programmerer en kopi av den intelligente atferdsprosessen til en maskin, kalles det kunstig intelligens (AI).

---
## Noe terminologi

Selv om begrepene kan forveksles, er maskinlæring (ML) en viktig underkategori av kunstig intelligens. **ML handler om å bruke spesialiserte algoritmer for å avdekke meningsfull informasjon og finne skjulte mønstre i oppfattet data for å støtte den rasjonelle beslutningsprosessen**.

---
## AI, ML, dyp læring

![AI, ML, deep learning, data science](../../../../translated_images/no/ai-ml-ds.537ea441b124ebf6.webp)

> Et diagram som viser forholdet mellom AI, ML, dyp læring og datavitenskap. Infografikk av [Jen Looper](https://twitter.com/jenlooper) inspirert av [denne grafikken](https://softwareengineering.stackexchange.com/questions/366996/distinction-between-ai-ml-neural-networks-deep-learning-and-data-mining)

---
## Begreper som skal dekkes

I dette pensumet skal vi dekke bare kjernebegrepene i maskinlæring som en nybegynner må kjenne til. Vi dekker det vi kaller 'klassisk maskinlæring' primært ved bruk av Scikit-learn, et utmerket bibliotek som mange studenter bruker for å lære det grunnleggende. For å forstå bredere begreper innen kunstig intelligens eller dyp læring, er det helt nødvendig med en sterk grunnleggende kunnskap i maskinlæring, og det ønsker vi å tilby her.

---
## I dette kurset vil du lære:

- kjernebegrepene i maskinlæring
- historien til ML
- ML og rettferdighet
- regresjonsmetoder i ML
- klassifiseringsmetoder i ML
- klyngemetoder i ML
- naturlig språkbearbeiding i ML
- tidsserieprognoser i ML
- forsterkende læring
- virkelige anvendelser av ML

---
## Hva vi ikke vil dekke

- dyp læring
- nevrale nettverk
- AI

For å gi en bedre læringsopplevelse vil vi unngå kompleksiteten i nevrale nettverk, 'dyp læring' – flerlags modellbygging ved bruk av nevrale nettverk – og AI, som vi vil diskutere i et annet pensum. Vi vil også tilby et kommende datavitenskaps-pensum for å fokusere på det aspektet av dette større feltet.

---
## Hvorfor studere maskinlæring?

Maskinlæring, sett fra et systemperspektiv, defineres som skapelsen av automatiserte systemer som kan lære skjulte mønstre fra data for å hjelpe til med å ta intelligente beslutninger.

Denne motivasjonen er løst inspirert av hvordan den menneskelige hjernen lærer visse ting basert på data den oppfatter fra den ytre verden.

✅ Tenk et øyeblikk på hvorfor en bedrift ville prøve å bruke maskinlæringsstrategier i stedet for å lage en regler-basert motor med hardkodede regler.

---
## Hvorfor datakvalitet er viktig

Data av høy kvalitet forbedrer modellens ytelse. Dårlige eller støyende data kan føre til unøyaktige prediksjoner, selv når man bruker avanserte maskinlæringsalgoritmer.

---
## Anvendelser av maskinlæring

Anvendelser av maskinlæring finnes nå nesten overalt, og er like allestedsnærværende som dataene som flyter rundt i samfunnet vårt, generert av smarttelefoner, tilkoblede enheter og andre systemer. Med tanke på det enorme potensialet til moderne maskinlæringsalgoritmer, har forskere utforsket deres evne til å løse multidimensjonale og flerfaglige virkelige problemer med gode, positive resultater.

---
## Eksempler på anvendt ML

**Du kan bruke maskinlæring på mange måter**:

- For å forutsi sannsynligheten for sykdom basert på en pasients medisinske historie eller rapporter.
- For å bruke værdata til å forutsi værhendelser.
- For å forstå sentimentet i en tekst.
- For å oppdage falske nyheter for å stoppe spredning av propaganda.

Finans, økonomi, jordvitenskap, romforskning, biomedisinsk ingeniørfag, kognitiv vitenskap og til og med humaniora har tilpasset maskinlæring for å løse de utfordrende, datatunge problemene i sine felter.

---
## Konklusjon

Maskinlæring automatiserer prosessen med å oppdage mønstre ved å finne meningsfulle innsikter fra virkelige eller genererte data. Det har vist seg å være svært verdifullt innen virksomhet, helse og finans, blant annet.

I nær fremtid vil det være nødvendig for folk i alle felt å forstå det grunnleggende om maskinlæring på grunn av den brede adopsjonen.

---
# 🚀 Utfordring

Skissér, på papir eller ved bruk av en nettapp som [Excalidraw](https://excalidraw.com/), din forståelse av forskjellene mellom AI, ML, dyp læring og datavitenskap. Legg til noen tanker om hvilke problemer hver av disse teknikkene er gode til å løse.

# [Quiz etter forelesning](https://ff-quizzes.netlify.app/en/ml/)

---
# Gjennomgang & Selvstudium

For å lære mer om hvordan du kan jobbe med ML-algoritmer i skyen, følg denne [læringsveien](https://docs.microsoft.com/learn/paths/create-no-code-predictive-models-azure-machine-learning/?WT.mc_id=academic-77952-leestott).

Ta en [læringsvei](https://docs.microsoft.com/learn/modules/introduction-to-machine-learning/?WT.mc_id=academic-77952-leestott) om det grunnleggende i ML.

---
# Oppgave

[Kom i gang](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Ansvarsfraskrivelse**:
Dette dokumentet er oversatt ved hjelp av AI-oversettelsestjenesten [Co-op Translator](https://github.com/Azure/co-op-translator). Selv om vi streber etter nøyaktighet, vær oppmerksom på at automatiske oversettelser kan inneholde feil eller unøyaktigheter. Det opprinnelige dokumentet på originalspråket skal betraktes som den autoritative kilden. For kritisk informasjon anbefales profesjonell menneskelig oversettelse. Vi er ikke ansvarlige for eventuelle misforståelser eller feiltolkninger som oppstår ved bruk av denne oversettelsen.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->