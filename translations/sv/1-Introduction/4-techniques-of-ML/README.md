# Tekniker för maskininlärning

Processen att bygga, använda och underhålla maskininlärningsmodeller och den data de använder är en mycket annorlunda process jämfört med många andra utvecklingsarbetsflöden. I denna lektion kommer vi att avmystifiera processen och skissera de viktigaste teknikerna du behöver känna till. Du kommer att:

- Förstå processerna bakom maskininlärning på en övergripande nivå.
- Utforska grundläggande begrepp såsom "modeller", "förutsägelser" och "träningsdata".

## [Quiz före föreläsning](https://ff-quizzes.netlify.app/en/ml/)

[![ML for beginners - Techniques of Machine Learning](https://img.youtube.com/vi/4NGM0U2ZSHU/0.jpg)](https://youtu.be/4NGM0U2ZSHU "ML for beginners - Techniques of Machine Learning")

> 🎥 Klicka på bilden ovan för en kort video som går igenom denna lektion.

## Introduktion

På en övergripande nivå består hantverket att skapa maskininlärningsprocesser (ML) av ett antal steg:

1. **Bestäm frågan**. De flesta ML-processer börjar med att ställa en fråga som inte kan besvaras med ett enkelt villkorsbaserat program eller regelbaserat system. Dessa frågor handlar ofta om förutsägelser baserade på en samling data.
2. **Samla in och förbered data**. För att kunna besvara din fråga behöver du data. Kvaliteten och ibland även mängden av din data avgör hur väl du kan besvara din ursprungliga fråga. Att visualisera data är en viktig del av denna fas. Denna fas inkluderar också att dela upp data i tränings- och testgrupper för att bygga en modell.
3. **Välj en träningsmetod**. Beroende på din fråga och din datas natur behöver du välja hur du vill träna en modell för att bäst spegla din data och göra noggranna förutsägelser utifrån den. Detta är den del av din ML-process som kräver specifik expertis och ofta en betydande mängd experimenterande.
4. **Träna modellen**. Med hjälp av din träningsdata kommer du att använda olika algoritmer för att träna en modell att känna igen mönster i datan. Modellen kan använda interna vikter som kan justeras för att prioritera vissa delar av datan över andra för att bygga en bättre modell.
5. **Utvärdera modellen**. Du använder tidigare aldrig sedd data (din testdata) från din samling för att se hur modellen presterar.
6. **Parameterjustering**. Baserat på modellens prestanda kan du upprepa processen med olika parametrar, eller variabler, som styr algoritmernas beteende vid träningen av modellen.
7. **Förutsägelse**. Använd nya indata för att testa modellens noggrannhet.

## Vilken fråga att ställa

Datorer är särskilt skickliga på att upptäcka dolda mönster i data. Denna förmåga är mycket användbar för forskare som har frågor om ett visst område som inte lätt kan besvaras genom att skapa ett villkorsbaserat regelverk. Vid en aktuarieuppgift, till exempel, kan en data scientist kanske konstruera handgjorda regler kring dödligheten för rökare jämfört med icke-rökare.

När många andra variabler inkluderas i ekvationen kan dock en ML-modell visa sig vara mer effektiv för att förutsäga framtida dödlighet baserat på tidigare hälsodata. Ett mer muntert exempel kan vara att göra väderprognoser för april månad på en given plats baserat på data som inkluderar latitud, longitud, klimatförändringar, närhet till havet, jetströmmens mönster med mera.

✅ Denna [presentation](https://www2.cisl.ucar.edu/sites/default/files/2021-10/0900%20June%2024%20Haupt_0.pdf) om vädermodeller erbjuder ett historiskt perspektiv på användning av ML i väderanalys.  

## Förberedande uppgifter

Innan du börjar bygga din modell finns flera uppgifter du behöver slutföra. För att testa din fråga och forma en hypotes baserat på modellens förutsägelser behöver du identifiera och konfigurera flera element.

### Data

För att kunna besvara din fråga med någon form av säkerhet behöver du en stor mängd data av rätt typ. Det finns två saker du behöver göra vid detta tillfälle:

- **Samla in data**. Med tanke på föregående lektion om rättvisa i dataanalys, samla in din data med omsorg. Var medveten om var datan kommer ifrån, eventuella inneboende bias och dokumentera dess ursprung.
- **Förbered data**. Det finns flera steg i dataförberedelseprocessen. Du kan behöva sammanställa data och normalisera den om den kommer från olika källor. Du kan förbättra datans kvalitet och kvantitet genom olika metoder, såsom att omvandla strängar till siffror (som i [Klustring](../../5-Clustering/1-Visualize/README.md)). Du kan också generera ny data baserat på den ursprungliga (som i [Klassificering](../../4-Classification/1-Introduction/README.md)). Du kan rengöra och redigera datan (vilket vi gör innan lektionen om [Web App](../../3-Web-App/README.md)). Slutligen kan du behöva slumpa och blanda den, beroende på dina träningsmetoder.

✅ Efter att ha samlat in och bearbetat din data, ta en stund för att se om dess form kommer att tillåta dig att behandla din avsedda fråga. Det kan vara så att datan inte fungerar bra för din givna uppgift, som vi upptäcker i våra [Klustrings](../../5-Clustering/1-Visualize/README.md) lektioner!

### Egenskaper och mål

En [feature](https://www.datasciencecentral.com/profiles/blogs/an-introduction-to-variable-and-feature-selection) är en mätbar egenskap hos din data. I många dataset uttrycks den som en kolumnrubrik som 'datum', 'storlek' eller 'färg'. Din feature-variabel, vanligtvis representerad som `X` i kod, är indatavariabeln som används för att träna en modell.

Ett mål är det du försöker förutsäga. Målvariabeln, vanligtvis representerad som `y` i kod, representerar svaret på frågan du ställer till din data: i december, vilken **färg** på pumpor kommer att vara billigast? i San Francisco, vilka bostadsområden kommer att ha det bästa fastighetspriset? Ibland kallas målvariabeln också för en etikettattribut.

### Välja din feature-variabel

🎓 **Feature selection och feature extraction** Hur vet man vilken variabel man ska välja när man bygger en modell? Du kommer troligen att gå igenom en process för feature selection eller feature extraction för att välja rätt variabler för den mest presterande modellen. De är dock inte samma sak: "Feature extraction skapar nya egenskaper från funktioner hos de ursprungliga egenskaperna, medan feature selection returnerar en delmängd av egenskaperna." ([källa](https://wikipedia.org/wiki/Feature_selection))

### Visualisera din data

En viktig del i data scientistens verktygslåda är möjligheten att visualisera data med hjälp av flera utmärkta bibliotek som Seaborn eller MatPlotLib. Att representera din data visuellt kan hjälpa dig att upptäcka dolda korrelationer som du kan utnyttja. Dina visualiseringar kan också hjälpa dig att upptäcka bias eller obalanserad data (som vi upptäcker i [Klassificering](../../4-Classification/2-Classifiers-1/README.md)).

### Dela upp din dataset

Innan träning behöver du dela upp ditt dataset i två eller flera delar av ojämn storlek som ändå representerar datan väl.

- **Träning**. Denna del av datasetet används för att träna modellen. Detta är den största delen av originaldatasetet.
- **Testning**. Ett testdataset är en oberoende grupp data, ofta hämtad från originaldatan, som används för att bekräfta prestandan hos den byggda modellen.
- **Validering**. Ett valideringsdataset är en mindre oberoende grupp exempel som du använder för att finjustera modellens hyperparametrar, eller arkitektur, för att förbättra modellen. Beroende på din datas storlek och frågeställning kan du behöva eller inte behöva skapa denna tredje grupp (som vi noterar i [Tidsserieförutsägelse](../../7-TimeSeries/1-Introduction/README.md)).

## Bygga en modell

Med hjälp av din träningsdata är målet att bygga en modell, eller en statistisk representation av din data, genom att använda olika algoritmer för att **träna** den. Att träna en modell innebär att exponera den för data och låta den göra antaganden om uppfattade mönster den upptäcker, validerar och accepterar eller avvisar.

### Välj en träningsmetod

Beroende på din fråga och din datas natur kommer du att välja en metod för träning. Genomgång av [Scikit-learns dokumentation](https://scikit-learn.org/stable/user_guide.html) – som vi använder i denna kurs – ger möjlighet att utforska många sätt att träna en modell. Beroende på din erfarenhet kan du behöva prova flera metoder för att bygga den bästa modellen. Du kommer sannolikt att genomgå en process där data scientists utvärderar modellens prestanda genom att mata in tidigare aldrig sedd data, kontrollera noggrannhet, bias och andra kvalitetsnedbrytande faktorer, och välja den mest lämpliga träningsmetoden för uppgiften.

### Träna en modell

Med din träningsdata redo kan du 'fit'-träna modellen. Du kommer att märka att i många ML-bibliotek finns koden 'model.fit' - det är just vid detta tillfälle du skickar in din feature-variabel som en matris av värden (vanligtvis 'X') och en målvariabel (vanligtvis 'y').

### Utvärdera modellen

När träningsprocessen är klar (det kan ta många iterationer, eller 'epochs', att träna en stor modell) kan du utvärdera modellens kvalitet genom att använda testdata för att mäta dess prestanda. Denna data är ett underutdrag av originaldatan som modellen tidigare inte analyserat. Du kan skriva ut en tabell med mätvärden som beskriver din modells kvalitet.

🎓 **Model fitting**

Inom maskininlärning avser model fitting hur noggrant modellens underliggande funktion analyserar data som den inte tidigare är bekant med.

🎓 **Underfitting** och **overfitting** är vanliga problem som försämrar modellens kvalitet, då modellen passar antingen för dåligt eller för bra. Detta gör att modellen förutsäger antingen alltför strikt eller alltför löst i förhållande till sin träningsdata. En överanpassad modell förutsäger träningsdata för bra eftersom den lärt sig datadetaljer och brus alltför väl. En underanpassad modell är inte noggrann eftersom den varken kan analysera sin träningsdata korrekt eller data den inte tidigare 'sett'.

![overfitting model](../../../../translated_images/sv/overfitting.1c132d92bfd93cb6.webp)
> Infografik av [Jen Looper](https://twitter.com/jenlooper)

## Parameterjustering

När din initiala träning är klar, observera modellens kvalitet och överväg att förbättra den genom att justera dess 'hyperparametrar'. Läs mer om processen [i dokumentationen](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters?WT.mc_id=academic-77952-leestott).

## Förutsägelse

Detta är ögonblicket när du kan använda helt nya data för att testa din modells noggrannhet. I en "tillämpad" ML-miljö, där du bygger webbresurser för att använda modellen i produktion, kan denna process innebära att samla in användarinput (till exempel ett knapptryck) för att sätta en variabel och skicka den till modellen för inferens, eller utvärdering.

I dessa lektioner kommer du att upptäcka hur du använder dessa steg för att förbereda, bygga, testa, utvärdera och förutsäga – alla steg i en data scientists arbete och mer, när du fortsätter din resa för att bli en "fullstack" ML-ingenjör.

---

## 🚀Utmaning

Rita ett flödesschema som speglar stegen en ML-utövare går igenom. Var ser du dig själv just nu i processen? Var tror du att du kommer att stöta på svårigheter? Vad verkar enkelt för dig?

## [Quiz efter föreläsning](https://ff-quizzes.netlify.app/en/ml/)

## Repetition & Självstudier

Sök på nätet efter intervjuer med data scientists som diskuterar sitt dagliga arbete. Här är [en](https://www.youtube.com/watch?v=Z3IjgbbCEfs).

## Uppgift

[Intervjua en data scientist](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Ansvarsfriskrivning**:  
Detta dokument har översatts med hjälp av AI-översättningstjänsten [Co-op Translator](https://github.com/Azure/co-op-translator). Även om vi strävar efter noggrannhet, var vänlig observera att automatiska översättningar kan innehålla fel eller brister. Det ursprungliga dokumentet på dess modersmål bör betraktas som den auktoritativa källan. För kritisk information rekommenderas professionell mänsklig översättning. Vi ansvarar inte för några missförstånd eller feltolkningar som uppstår till följd av användningen av denna översättning.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->