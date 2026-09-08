# Introduktion till maskininlärning

## [Quiz före lektionen](https://ff-quizzes.netlify.app/en/ml/)

---

[![ML for beginners - Introduction to Machine Learning for Beginners](https://img.youtube.com/vi/6mSx_KJxcHI/0.jpg)](https://youtu.be/6mSx_KJxcHI "ML for beginners - Introduction to Machine Learning for Beginners")

> 🎥 Klicka på bilden ovan för en kort video som går igenom denna lektion.

Välkommen till denna kurs om klassisk maskininlärning för nybörjare! Oavsett om du är helt ny inom detta ämne, eller en erfaren ML-utövare som vill fräscha upp ett område, är vi glada att ha dig med oss! Vi vill skapa en vänlig startpunkt för dina ML-studier och är glada att utvärdera, svara på och införliva din [feedback](https://github.com/microsoft/ML-For-Beginners/discussions).

[![Introduction to ML](https://img.youtube.com/vi/h0e2HAPTGF4/0.jpg)](https://youtu.be/h0e2HAPTGF4 "Introduction to ML")

> 🎥 Klicka på bilden ovan för en video: MIT:s John Guttag introducerar maskininlärning

---
## Kom igång med maskininlärning

Innan du börjar med detta utbildningsprogram behöver du ha din dator uppsatt och klar för att köra notebooks lokalt.

- **Konfigurera din maskin med dessa videor**. Använd följande länkar för att lära dig [hur man installerar Python](https://youtu.be/CXZYvNRIAKM) i ditt system och [sätter upp en textredigerare](https://youtu.be/EU8eayHWoZg) för utveckling.
- **Lär dig Python**. Det rekommenderas också att ha en grundläggande förståelse för [Python](https://docs.microsoft.com/learn/paths/python-language/?WT.mc_id=academic-77952-leestott), ett programmeringsspråk användbart för dataforskare som vi använder i denna kurs.
- **Lär dig Node.js och JavaScript**. Vi använder också JavaScript ett par gånger i denna kurs när vi bygger webbappar, så du behöver ha [node](https://nodejs.org) och [npm](https://www.npmjs.com/) installerade, samt [Visual Studio Code](https://code.visualstudio.com/) tillgängligt för både Python- och JavaScript-utveckling.
- **Skapa ett GitHub-konto**. Eftersom du hittade oss här på [GitHub](https://github.com) kanske du redan har ett konto, men annars, skapa ett och gör en fork av detta läroprogram för att använda själv. (Känn dig fri att ge oss en stjärna också 😊)
- **Utforska Scikit-learn**. Bekanta dig med [Scikit-learn](https://scikit-learn.org/stable/user_guide.html), en uppsättning ML-bibliotek som vi refererar till i dessa lektioner.

---
## Vad är maskininlärning?

Termen 'maskininlärning' är en av de mest populära och frekvent använda termerna idag. Det är en icke-trivial möjlighet att du har hört detta ord minst en gång om du har någon form av bekantskap med teknik, oavsett vilket område du arbetar inom. Mekaniken bakom maskininlärning är dock en gåta för de flesta. För en nybörjare inom maskininlärning kan ämnet ibland kännas överväldigande. Därför är det viktigt att förstå vad maskininlärning egentligen är och att lära sig om det steg för steg, genom praktiska exempel.

---
## Hypekurvan

![ml hype curve](../../../../translated_images/sv/hype.07183d711a17aafe.webp)

> Google Trends visar den senaste 'hypekurvan' för termen 'maskininlärning'

---
## Ett mystiskt universum

Vi lever i ett universum fullt av fascinerande mysterier. Stora vetenskapsmän såsom Stephen Hawking, Albert Einstein och många fler har ägnat sina liv åt att söka meningsfull information som avslöjar mysterierna i världen omkring oss. Detta är människans villkor för lärande: ett barn lär sig nya saker och upptäcker strukturen i sin värld år efter år när hen växer upp till vuxen.

---
## Barnets hjärna

Ett barns hjärna och sinnen uppfattar fakta om sin omgivning och lär sig gradvis livets dolda mönster som hjälper barnet att skapa logiska regler för att identifiera inlärda mönster. Mänskliga hjärnans inlärningsprocess gör människor till den mest sofistikerade levande varelsen på denna värld. Att ständigt lära genom att upptäcka dolda mönster och sedan förnya dessa mönster gör att vi kan göra oss själva bättre och bättre under hela vår livstid. Denna inlärningskapacitet och utvecklande förmåga är kopplad till ett begrepp som kallas [hjärnplasticitet](https://www.simplypsychology.org/brain-plasticity.html). Ytligt kan vi dra vissa motiverande likheter mellan den mänskliga hjärnans inlärningsprocess och koncepten bakom maskininlärning.

---
## Den mänskliga hjärnan

Den [mänskliga hjärnan](https://www.livescience.com/29365-human-brain.html) uppfattar saker från den verkliga världen, bearbetar den uppfattade informationen, fattar rationella beslut och utför vissa handlingar baserade på omständigheter. Detta är vad vi kallar att bete sig intelligent. När vi programmerar en avbild av denna intelligenta beteendeprocess till en maskin kallas det artificiell intelligens (AI).

---
## Vissa termer

Även om termerna kan blandas ihop är maskininlärning (ML) en viktig delmängd av artificiell intelligens. **ML handlar om att använda specialiserade algoritmer för att avslöja meningsfull information och hitta dolda mönster från uppfattade data för att styrka den rationella beslutsprocessen**.

---
## AI, ML, djupinlärning

![AI, ML, deep learning, data science](../../../../translated_images/sv/ai-ml-ds.537ea441b124ebf6.webp)

> En diagram som visar relationerna mellan AI, ML, djupinlärning och data science. Infografik av [Jen Looper](https://twitter.com/jenlooper) inspirerad av [den här grafiken](https://softwareengineering.stackexchange.com/questions/366996/distinction-between-ai-ml-neural-networks-deep-learning-and-data-mining)

---
## Begrepp att täcka

I detta utbildningsprogram kommer vi endast att täcka de grundläggande begreppen i maskininlärning som en nybörjare måste känna till. Vi täcker vad vi kallar 'klassisk maskininlärning' främst med hjälp av Scikit-learn, ett utmärkt bibliotek som många studenter använder för att lära sig grunderna. För att förstå bredare begrepp som artificiell intelligens eller djupinlärning är en stark grundkunskap i maskininlärning oumbärlig, och därför vill vi erbjuda den här.

---
## I denna kurs kommer du att lära dig:

- grundläggande begrepp inom maskininlärning
- historien om ML
- ML och rättvisa
- regressionsbaserade ML-tekniker
- klassificeringsbaserade ML-tekniker
- klustringsbaserade ML-tekniker
- ML-tekniker för naturlig språkbehandling
- ML-tekniker för tidsserieprognoser
- förstärkningsinlärning
- verkliga tillämpningar av ML

---
## Vad vi inte kommer att täcka

- djupinlärning
- neurala nätverk
- AI

För att skapa en bättre lärandeupplevelse kommer vi att undvika komplexiteten i neurala nätverk, 'djupinlärning' – flerskiktsmodelluppbyggnad med neurala nätverk – och AI, vilket vi kommer att diskutera i ett annat utbildningsprogram. Vi kommer också att erbjuda ett kommande data science-utbildningsprogram som fokuserar på den aspekten av detta större område.

---
## Varför studera maskininlärning?

Maskininlärning definieras, ur ett systemperspektiv, som skapandet av automatiserade system som kan lära sig dolda mönster från data för att hjälpa till att fatta intelligenta beslut.

Denna motivation är löst inspirerad av hur den mänskliga hjärnan lär sig vissa saker baserat på data den uppfattar från omvärlden.

✅ Tänk en stund på varför ett företag skulle vilja försöka använda maskininlärningsstrategier istället för att skapa en hårdkodad regelbaserad motor.

---
## Varför datakvalitet är viktig

Högkvalitativ data förbättrar modellens prestanda. Dålig eller brusig data kan leda till felaktiga förutsägelser, även när avancerade maskininlärningsalgoritmer används.

---
## Tillämpningar av maskininlärning

Tillämpningar av maskininlärning är nu nästan överallt och lika allmänt förekommande som den data som flödar runt i våra samhällen, genererad av våra smarta telefoner, uppkopplade enheter och andra system. Med tanke på den enorma potentialen i toppmoderna maskininlärningsalgoritmer har forskare utforskat deras förmåga att lösa multidimensionella och multidisciplinära verkliga problem med stora positiva resultat.

---
## Exempel på tillämpad ML

**Du kan använda maskininlärning på många sätt**:

- För att förutsäga sannolikheten för sjukdom utifrån en patients sjukdomshistoria eller rapporter.
- För att utnyttja väderdata för att förutsäga väderhändelser.
- För att förstå känslan i en text.
- För att upptäcka fejkade nyheter för att stoppa spridningen av propaganda.

Finans, ekonomi, jordvetenskap, rymdforskning, biomedicinsk teknik, kognitiv vetenskap och till och med humaniora har anpassat maskininlärning för att lösa de tunga, dataintensiva problem som finns inom deras områden.

---
## Slutsats

Maskininlärning automatiserar processen att upptäcka mönster genom att hitta meningsfulla insikter från verkliga eller genererade data. Det har visat sig vara mycket värdefullt inom affärer, hälsa och finans, bland annat.

Framöver kommer det att vara ett måste för personer i alla domäner att förstå grunderna i maskininlärning på grund av dess utbredda användning.

---
# 🚀 Utmaning

Skissa, på papper eller med en online-app som [Excalidraw](https://excalidraw.com/), din förståelse av skillnaderna mellan AI, ML, djupinlärning och data science. Lägg till några idéer om problem som varje teknik är bra på att lösa.

# [Quiz efter lektionen](https://ff-quizzes.netlify.app/en/ml/)

---
# Repetition & Självstudier

För att lära dig mer om hur du kan arbeta med ML-algoritmer i molnet, följ denna [Learning Path](https://docs.microsoft.com/learn/paths/create-no-code-predictive-models-azure-machine-learning/?WT.mc_id=academic-77952-leestott).

Gå en [Learning Path](https://docs.microsoft.com/learn/modules/introduction-to-machine-learning/?WT.mc_id=academic-77952-leestott) om grunderna i ML.

---
# Uppgift

[Kom igång](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Ansvarsfriskrivning**:
Detta dokument har översatts med hjälp av AI-översättningstjänsten [Co-op Translator](https://github.com/Azure/co-op-translator). Även om vi strävar efter noggrannhet, var vänlig notera att automatiska översättningar kan innehålla fel eller brister. Det ursprungliga dokumentet på dess modersmål bör betraktas som den auktoritativa källan. För kritisk information rekommenderas professionell mänsklig översättning. Vi ansvarar inte för några missförstånd eller feltolkningar som uppstår till följd av användningen av denna översättning.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->