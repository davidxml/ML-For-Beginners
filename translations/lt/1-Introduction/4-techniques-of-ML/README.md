# Mašininio mokymosi metodai

Proceso, kuriuo kuriami, naudojami ir palaikomi mašininio mokymosi modeliai bei duomenys, yra labai skirtingas nuo daugelio kitų kūrimo darbo eigų. Šioje pamokoje mes atskleisime šio proceso esmę ir išdėstysime pagrindines technikas, kurias jums reikia žinoti. Jūs:

- Suprasite mašininio mokymosi pagrindinius procesus aukštu lygiu.
- Išnagrinėsite pagrindines sąvokas, tokias kaip 'modeliai', 'prognozės' ir 'mokymo duomenys'.

## [Priešpaskaitos testas](https://ff-quizzes.netlify.app/en/ml/)

[![ML for beginners - Techniques of Machine Learning](https://img.youtube.com/vi/4NGM0U2ZSHU/0.jpg)](https://youtu.be/4NGM0U2ZSHU "ML for beginners - Techniques of Machine Learning")

> 🎥 Paspauskite paveikslėlį aukščiau, kad pamatytumėte trumpą šios pamokos vaizdo įrašą.

## Įvadas

Aukštu lygiu mašininio mokymosi (ML) procesų kūrimo amatas susideda iš kelių žingsnių:

1. **Nustatyti klausimą**. Dauguma ML procesų prasideda uždavus klausimą, kurio negalima atsakyti paprastu sąlyginiu programu arba taisyklių varikliu. Šie klausimai dažnai susiję su prognozėmis, pagrįstomis duomenų rinkiniu.
2. **Surinkti ir paruošti duomenis**. Norint atsakyti į klausimą, reikia duomenų. Duomenų kokybė ir kartais kiekis lemia, kaip gerai galite atsakyti į pirminį klausimą. Duomenų vizualizavimas yra svarbi šios fazės dalis. Šiame etape taip pat atliekamas duomenų padalinimas į mokymo ir testavimo grupes modelio kūrimui.
3. **Pasirinkti mokymo metodą**. Priklausomai nuo jūsų klausimo ir duomenų pobūdžio, turite pasirinkti, kaip norite apmokyti modelį, kad geriausiai atspindėtų duomenis ir leistų tiksliai prognozuoti. Tai ML proceso dalis, kuri reikalauja specifinių žinių ir dažnai nemažai eksperimentų.
4. **Apkrova modelį**. Naudodami mokymo duomenis, jūs įvairiais algoritmais apmokysite modelį atpažinti duomenų struktūras. Modelis gali naudoti vidinius svorius, kuriuos galima koreguoti, kad tam tikros duomenų dalys būtų svarbesnės, siekiant sukurti geresnį modelį.
5. **Įvertinti modelį**. Naudojate dar nematytus duomenis (testavimo duomenis) iš savo surinkto rinkinio, kad pamatytumėte, kaip modelis veikia.
6. **Parametrų suderinimas**. Remdamiesi modelio veikimu, galite procesą pakartoti su skirtingais parametrais arba kintamaisiais, kurie kontroliuoja algoritmų veikimą mokant modelį.
7. **Prognozė**. Naudokite naujus įvesties duomenis savo modelio tikslumui išbandyti.

## Koks klausimas turi būti užduotas

Kompiuteriai ypač gerai atranda paslėptas tendencijas duomenyse. Ši nauda ypač svarbi tyrėjams, turintiems klausimų apie tam tikrą sritį, kurių neįmanoma lengvai atsakyti sukuriant sąlyginiu pagrindu veikiantį taisyklių variklį. Pavyzdžiui, aktuaro užduotyje duomenų mokslininkas gali sukurti rankomis aprašytas taisykles apie rūkalių ir nerūkančiųjų mirtingumo skirtumus.

Tačiau įtraukus daugybę kitų kintamųjų, ML modelis gali būti efektyvesnis prognozuojant būsimus mirtingumo rodiklius, remiantis praeities sveikatos istorija. Džiugesnis pavyzdys būtų orų prognozės balandžio mėnesiui tam tikroje vietovėje, remiantis duomenimis apie platumą, ilgumą, klimato kaitą, artumą prie vandenyno, srovių modelius ir kt.

✅ Ši [skaidrių prezentacija](https://www2.cisl.ucar.edu/sites/default/files/2021-10/0900%20June%2024%20Haupt_0.pdf) apie orų modelius pateikia istorinių ML naudojimo orų analizėje perspektyvą.  

## Užduotys prieš modelio kūrimą

Prieš pradėdami kurti modelį, turite atlikti keletą užduočių. Kad galėtumėte išbandyti klausimą ir suformuluoti hipotezę, pagrįstą modelio prognozėmis, turite nustatyti ir sukonfigūruoti kelis elementus.

### Duomenys

Kad galėtumėte užtikrintai atsakyti į savo klausimą, jums reikia tinkamo tipo ir pakankamai daug duomenų. Šiuo metu reikia atlikti dvi užduotis:

- **Surinkti duomenis**. Atsižvelgiant į ankstesnę pamoką apie duomenų analizės sąžiningumą, duomenis rinkite atsargiai. Žinokite šių duomenų šaltinius, jų galimus šališkumus ir dokumentuokite jų kilmę.
- **Paruošti duomenis**. Duomenų paruošimo procese yra kelios užduotys. Gali prireikti surinkti duomenis ir juos normalizuoti, jei jie gaunami iš įvairių šaltinių. Duomenų kokybę ir kiekį galite pagerinti įvairiais metodais, pavyzdžiui, konvertuodami tekstines eilutes į skaičius (kaip darome [Klasterizacijoje](../../5-Clustering/1-Visualize/README.md)). Taip pat galite generuoti naujus duomenis, remdamiesi originaliais (kaip darome [Klasifikacijoje](../../4-Classification/1-Introduction/README.md)). Galite valyti ir redaguoti duomenis (kaip darysime prieš [Internetinę programėlę](../../3-Web-App/README.md)). Galiausiai, priklausomai nuo mokymo metodų, gali prireikti duomenis atsitiktinai permaišyti ir sukrauti.

✅ Surinkę ir apdoroję duomenis, patikrinkite, ar jų forma leistų jums spręsti ketinamą klausimą. Duomenys gali būti netinkami užduočiai, kaip mes atrandame mūsų [Klasterizacijos](../../5-Clustering/1-Visualize/README.md) pamokose!

### Savybės ir tikslas

[Savybė](https://www.datasciencecentral.com/profiles/blogs/an-introduction-to-variable-and-feature-selection) yra matuojama jūsų duomenų savybė. Daugelyje duomenų rinkinių ji išreiškiama kaip stulpelio pavadinimas, pvz., 'data', 'dydis' arba 'spalva'. Jūsų savybės kintamasis, dažnai žymimas kaip `X` kode, reiškia įvesties kintamąjį, kuris bus naudojamas modelio mokymui.

Tikslas yra tai, ką bandote prognozuoti. Tikslas, dažnai žymimas kaip `y` kode, yra atsakymas į klausimą, kurį bandote užduoti savo duomenims: gruodį kokios **spalvos** moliūgai bus pigiausi? San Franciske, kokie rajonai turės geriausias nekilnojamojo turto **kainas**? Kartais tikslas vadinamas žymos atributu.

### Savybės kintamojo pasirinkimas

🎓 **Savybės atranka ir savybės išgavimas** Kaip žinote, kurį kintamąjį rinktis kuriant modelį? Tikriausiai praeisite per savybės atrankos arba savybės išgavimo procesą, norėdami pasirinkti tinkamus kintamuosius geriausiam modeliui. Tačiau jie nėra tas pats: "Savybės išgavimas sukuria naujas savybes iš originalių savybių funkcijų, o savybės atranka grąžina savybių posetį." ([šaltinis](https://wikipedia.org/wiki/Feature_selection))

### Vizualizuokite savo duomenis

Svarbi duomenų mokslininko įrankių dalis yra gebėjimas vizualizuoti duomenis naudojant kelias puikias bibliotekas, tokias kaip Seaborn arba MatPlotLib. Duomenų vizualizacija gali padėti atskleisti paslėptus koreliacijų ryšius, kuriuos galite pasinaudoti. Taip pat jūsų vaizdai gali padėti atskleisti šališkumą arba disbalansuotą duomenų pasiskirstymą (kaip sužinome [Klasifikacijoje](../../4-Classification/2-Classifiers-1/README.md)).

### Padalinkite savo duomenų rinkinį

Prieš mokymą reikia padalyti duomenų rinkinį į du ar daugiau nevienodo dydžio dalių, kurios vis tiek gerai atspindi duomenis.

- **Mokymas**. Ši dalis skirta modelio mokymui. Ji sudaro daugumą originalaus duomenų rinkinio.
- **Testavimas**. Testavimo duomenų rinkinys yra nepriklausoma duomenų grupė, dažnai gaunama iš originalių duomenų, kurią naudojate patvirtinti sukurto modelio veikimą.
- **Validavimas**. Validavimo rinkinys yra mažesnė nepriklausoma pavyzdžių grupė, kurią naudojate modelio hiperparametrams arba architektūrai derinti, siekiant pagerinti modelį. Priklausomai nuo duomenų dydžio ir keliamo klausimo, jums gali neprireikti kurti šio trečio rinkinio (kaip pastebima [Laiko eilių prognozavime](../../7-TimeSeries/1-Introduction/README.md)).

## Modelio kūrimas

Naudodami mokymo duomenis, jūsų tikslas yra sukurti modelį arba statistinį duomenų atvaizdą, naudodami įvairius algoritmus jį **apmokyti**. Mokymas leidžia modeliui įgyti duomenų ir daryti prielaidas apie aptiktas, patvirtintas arba atmestas tendencijas.

### Pasirinkite mokymo metodą

Priklausomai nuo jūsų klausimo ir duomenų pobūdžio, pasirinksite mokymo metodą. Peržiūrėję [Scikit-learn dokumentaciją](https://scikit-learn.org/stable/user_guide.html) – kuria naudositės šiame kurse – galėsite išbandyti daug skirtingų būdų modelio mokymui. Priklausomai nuo patirties, gali tekti išbandyti keletą skirtingų būdų geriausiam modeliui sukurti. Dažnai duomenų mokslininkai vertina modelio veikimą, pateikdami jam nematytus duomenis, tikrindami tikslumą, šališkumą ir kitus kokybę mažinančius veiksnius, bei pasirenka tinkamiausią mokymo metodą.

### Apmokykite modelį

Turėdami mokymo duomenis, galite juos pritaikyti ir sukurti modelį. Daugelio ML bibliotekų kode rasite 'model.fit' – būtent šiuo metu siunčiate savo savybės kintamąjį kaip verčių masyvą (dažniausiai 'X') ir tikslo kintamąjį (dažniausiai 'y').

### Įvertinkite modelį

Kai mokymas baigtas (tai gali užtrukti daug iteracijų arba 'epochų' mokant didelį modelį), galėsite įvertinti modelio kokybę naudodami testavimo duomenis. Šie duomenys yra dalis originalių duomenų, kurių modelis anksčiau neanalizavo. Galite atspausdinti lentelę su savo modelio kokybės metrikomis.

🎓 **Modelio pritaikymas**

Mašininio mokymosi kontekste modelio pritaikymas reiškia modelio pagrindinės funkcijos tikslumą, kai modelis bando analizuoti jam nežinomus duomenis.

🎓 **Permaitinimas** (overfitting) ir **nepasmaitinimas** (underfitting) yra dažnos problemos, kurios mažina modelio kokybę, kai modelis pritaikomas nepakankamai arba per daug. Tai sukelia, kad modelio prognozės yra per daug artimos arba pernelyg nutolusios nuo mokymo duomenų. Permaitinamas modelis per gerai prognozuoja mokymo duomenis, nes per daug gerai išmoko jų detales ir triukšmą. Nepasmaitintas modelis nėra tikslus, nes nei tiksliai analizuoja mokymo duomenis, nei duomenis, kurių dar nematė.

![overfitting model](../../../../translated_images/lt/overfitting.1c132d92bfd93cb6.webp)
> Infografika: [Jen Looper](https://twitter.com/jenlooper)

## Parametrų suderinimas

Baigus pradinį mokymą, stebėkite modelio kokybę ir apsvarstykite galimybę ją pagerinti koreguodami 'hiperparametrus'. Daugiau apie šį procesą skaitykite [dokumentacijoje](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters?WT.mc_id=academic-77952-leestott).

## Prognozė

Tai momentas, kai galite naudoti visiškai naujus duomenis modelio tikslumui patikrinti. 'Taikomojo' ML scenarijuje, kai kuriate internetines priemones modelio naudojimui gamyboje, šis procesas gali apimti vartotojo įvesties surinkimą (pvz., mygtuko paspaudimą), kad nustatyti kintamąjį ir išsiųsti jį modeliui inferencijai arba įvertinimui.

Šiose pamokose jūs sužinosite, kaip naudoti šiuos žingsnius modelio paruošimui, kūrimui, testavimui, vertinimui ir prognozavimui – visa tai yra duomenų mokslininko gestai ir dar daugiau, kai progresuosite link 'pilno paketo' ML inžinieriaus profesijos.

---

## 🚀Iššūkis

Nupieškite srautų schemą, atspindinčią ML praktiko žingsnius. Kurioje proceso dalyje dabar save matote? Kur manote, kad susidursite su sunkumais? Kas jums atrodo paprasta?

## [Po paskaitos testas](https://ff-quizzes.netlify.app/en/ml/)

## Apžvalga ir savarankiškas mokymasis

Internete ieškokite interviu su duomenų mokslininkais, kuriuose jie aptaria savo kasdienį darbą. Štai [vienas](https://www.youtube.com/watch?v=Z3IjgbbCEfs).

## Namų darbai

[Interviu su duomenų mokslininku](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Atsakomybės apribojimas**:  
Šis dokumentas buvo išverstas naudojant dirbtinio intelekto vertimo paslaugą [Co-op Translator](https://github.com/Azure/co-op-translator). Nors siekiame tikslumo, atkreipkite dėmesį, kad automatizuoti vertimai gali turėti klaidų ar netikslumų. Pirminis dokumentas jo gimtąja kalba turi būti laikomas autoritetingu šaltiniu. Dėl svarbios informacijos rekomenduojama naudotis profesionalų žmogaus vertimu. Mes neatsakome už bet kokius nesusipratimus ar klaidingas interpretacijas, kylančias naudojant šį vertimą.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->