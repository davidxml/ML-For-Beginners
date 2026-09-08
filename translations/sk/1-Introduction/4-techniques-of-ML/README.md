# Techniky strojového učenia

Proces tvorby, používania a údržby modelov strojového učenia a dát, ktoré používajú, je veľmi odlišný od mnohých iných vývojových pracovných tokov. V tejto lekcii demystifikujeme tento proces a načrtneme hlavné techniky, ktoré potrebujete poznať. Naučíte sa:

- Pochopiť procesy, ktoré stoja za strojovým učením na vysokej úrovni.
- Preskúmať základné pojmy ako „modely“, „predikcie“ a „tréningové dáta“.

## [Prednáškový kvíz](https://ff-quizzes.netlify.app/en/ml/)

[![ML for beginners - Techniques of Machine Learning](https://img.youtube.com/vi/4NGM0U2ZSHU/0.jpg)](https://youtu.be/4NGM0U2ZSHU "ML for beginners - Techniques of Machine Learning")

> 🎥 Kliknite na obrázok vyššie pre krátke video k tejto lekcii.

## Úvod

Na vysokej úrovni je remeslo tvorby procesov strojového učenia (ML) zložené z niekoľkých krokov:

1. **Rozhodnite sa o otázke**. Väčšina procesov ML začína otázkou, na ktorú nemožno odpovedať jednoduchým podmieneným programom alebo pravidlovým motorom. Tieto otázky sa často týkajú predikcií založených na zbierke dát.
2. **Zhromaždite a pripravte dáta**. Aby ste mohli odpovedať na svoju otázku, potrebujete dáta. Kvalita a niekedy aj množstvo vašich dát určí, ako dobre dokážete zodpovedať vašu pôvodnú otázku. Vizualizácia dát je dôležitým aspektom tejto fázy. Táto fáza tiež zahŕňa rozdelenie dát na tréningovú a testovaciu skupinu na vytvorenie modelu.
3. **Vyberte metódu trénovania**. V závislosti od vašej otázky a povahy vašich dát potrebujete zvoliť, ako chcete model trénovať, aby najlepšie odrážal vaše dáta a umožnil presné predikcie. Táto časť vášho ML procesu vyžaduje špecifickú odbornosť a často aj značné množstvo experimentovania.
4. **Natrénujte model**. S použitím tréningových dát použijete rôzne algoritmy na trénovanie modelu, ktorý rozpozná vzory v dátach. Model môže využívať vnútorné váhy, ktoré možno upravovať na zvýhodnenie určitých častí dát, aby sa vytvoril lepší model.
5. **Vyhodnoťte model**. Použijete nikdy predtým nevidené dáta (vaše testovacie dáta) zo zberu údajov, aby ste zistili, ako model funguje.
6. **Ladenie parametrov**. Na základe výkonnosti vášho modelu môžete proces zopakovať s rôznymi parametrami, alebo premennými, ktoré riadia správanie algoritmov používaných na trénovanie modelu.
7. **Predikcia**. Použite nové vstupy na otestovanie presnosti vášho modelu.

## Akú otázku sa pýtať

Počítače sú obzvlášť zdatné v objavovaní skrytých vzorov v dátach. Táto schopnosť je veľmi užitočná pre výskumníkov, ktorí majú otázky o danej oblasti, na ktoré sa nedá ľahko odpovedať vytvorením pravidlového motoru založeného na podmienkach. Napríklad pri aktuárskej úlohe by dátový vedec mohol zostaviť ručne tvorené pravidlá týkajúce sa úmrtnosti fajčiarov a nefajčiarov.

Keď však do rovnice zapojíte mnohé ďalšie premenné, model ML môže byť efektívnejší pri predpovedaní budúcich úmrtnostných mier na základe minulých zdravotných údajov. Veselším príkladom môže byť predpoveď počasia na mesiac apríl v určitej lokalite na základe dát, ktoré zahŕňajú zemepisnú šírku, dĺžku, klimatické zmeny, blízkosť oceánu, vzory prúdenia jet streamu a ďalšie.

✅ Táto [prezentačná prezentácia](https://www2.cisl.ucar.edu/sites/default/files/2021-10/0900%20June%2024%20Haupt_0.pdf) o modeloch počasia ponúka historický pohľad na použitie ML v analýze počasia.  

## Úlohy pred stavbou

Pred začatím tvorby modelu je potrebné splniť niekoľko úloh. Aby ste otestovali svoju otázku a vytvorili hypotézu na základe predikcií modelu, musíte identifikovať a nakonfigurovať niekoľko prvkov.

### Dáta

Aby ste mohli otázku zodpovedať s nejakou istotou, potrebujete veľké množstvo správnych dát. V tomto bode treba urobiť dve veci:

- **Zber dát**. V duchu predchádzajúcej lekcie o spravodlivosti v analýze dát ich zozbierajte starostlivo. Buďte si vedomí zdrojov týchto dát, akýchkoľvek vrodených predsudkov a zdokumentujte ich pôvod.
- **Príprava dát**. Proces prípravy dát zahŕňa niekoľko krokov. Možno budete musieť zlúčiť dáta a normalizovať ich, ak pochádzajú z rôznych zdrojov. Kvalitu a množstvo dát môžete vylepšiť rôznymi metódami, ako je konverzia reťazcov na čísla (ako robíme v [Zhlukovaní](../../5-Clustering/1-Visualize/README.md)). Môžete tiež generovať nové dáta na základe pôvodných (ako robíme v [Klasifikácii](../../4-Classification/1-Introduction/README.md)). Dáta môžete vyčistiť a upraviť (ako budeme robiť pred lekciou [Webová aplikácia](../../3-Web-App/README.md)). Nakoniec ich možno bude potrebné náhodne zamiešať, v závislosti od vašich tréningových techník.

✅ Po zozbieraní a spracovaní dát si urobte chvíľu na posúdenie, či ich tvar umožní odpovedať na zamýšľanú otázku. Môže sa stať, že dáta nebudú vo vašej konkretnej úlohe dobre fungovať, čo zistíme v našich lekciách o [Zhlukovaní](../../5-Clustering/1-Visualize/README.md)!

### Prvky a cieľ

[Prvok](https://www.datasciencecentral.com/profiles/blogs/an-introduction-to-variable-and-feature-selection) je merateľná vlastnosť vášho dátového súboru. Vo väčšine datasetov sa vyjadruje ako názov stĺpca ako 'dátum', 'veľkosť' alebo 'farba'. Vaša premenná prvku, zvyčajne reprezentovaná ako `X` v kóde, predstavuje vstupnú premennú, ktorá sa použije na trénovanie modelu.

Cieľ je vec, ktorú sa snažíte predpovedať. Cieľ, zvyčajne reprezentovaný ako `y` v kóde, predstavuje odpoveď na otázku, ktorú kladiete svojim dátam: v decembri, aká **farba** tekvíc bude najlacnejšia? v San Franciscu, ktoré štvrte budú mať najlepšiu cenu nehnuteľností (real estate) **cenu**? Niekedy sa cieľ označuje aj ako atribút štítka (label).

### Výber premenných prvkov

🎓 **Výber prvkov a extrakcia prvkov** Ako viete, ktorú premennú vybrať pri tvorbe modelu? Pravdepodobne prejdete proces výberu alebo extrakcie prvkov, aby ste zvolili správne premenné pre najvýkonnejší model. Nie sú to však to isté: „Extrakcia prvkov vytvára nové prvky z funkcií pôvodných prvkov, zatiaľ čo výber prvkov vracia podmnožinu prvkov.“ ([zdroj](https://wikipedia.org/wiki/Feature_selection))

### Vizualizujte svoje dáta

Dôležitým aspektom nástroja dátového vedca je schopnosť vizualizovať dáta pomocou niekoľkých vynikajúcich knižníc, ako sú Seaborn alebo MatPlotLib. Vizualizácia dát vám môže umožniť objaviť skryté korelácie, ktoré môžete využiť. Vaše vizualizácie vám môžu tiež pomôcť odhaliť zaujatosti alebo nevyvážené dáta (čo zistíme v [Klasifikácii](../../4-Classification/2-Classifiers-1/README.md)).

### Rozdeľte svoj dataset

Pred trénovaním musíte rozdeliť svoj dataset na dve alebo viac nerovnakých častí, ktoré stále dobre reprezentujú dáta.

- **Tréning**. Táto časť datasetu sa použije na trénovanie modelu. Táto skupina predstavuje väčšinu pôvodného datasetu.
- **Testovanie**. Testovacie dáta sú nezávislá skupina dát, často získavaná z pôvodných dát, ktorú používate na potvrdenie funkčnosti vybudovaného modelu.
- **Validácia**. Validačná skupina je menšia nezávislá skupina príkladov, ktorú používate na ladenie hyperparametrov modelu, alebo architektúry, na zlepšenie modelu. V závislosti od veľkosti vašich dát a otázky, ktorú kladiete, nemusíte potrebovať vytvoriť túto tretiu skupinu (ako si všímame v [Predikcii časových radov](../../7-TimeSeries/1-Introduction/README.md)).

## Tvorba modelu

Použitím svojich tréningových dát je vaším cieľom zostaviť model, alebo štatistickú reprezentáciu vašich dát, pomocou rôznych algoritmov na jeho **tréning**. Trénovanie modelu znamená vystaviť ho dátam a umožniť mu robiť predpoklady o vnímaných vzoroch, ktoré objaví, overí a prijme alebo odmietne.

### Rozhodnúť sa o metóde trénovania

V závislosti od vašej otázky a povahy vašich dát si vyberiete metódu na trénovanie. Prechádzaním [dokumentácie Scikit-learn](https://scikit-learn.org/stable/user_guide.html) – ktorú používame v tomto kurze – môžete preskúmať mnohé spôsoby trénovania modelu. V závislosti od vašich skúseností možno budete musieť vyskúšať niekoľko rôznych metód, aby ste vybudovali najlepší model. Pravdepodobne prejdete procesom, kedy dátoví vedci vyhodnocujú výkonnosť modelu na základe testovania na nevidených dátach, kontrolujú presnosť, zaujatosti a iné problémy znižujúce kvalitu a vyberajú najvhodnejšiu metódu trénovania pre danú úlohu.

### Natrénujte model

S tréningovými dátami ste pripravení „natrénovať“ model. V mnohých knižniciach ML nájdete kód „model.fit“ – práve v tomto momente posielate svoju premennú prvkov ako pole hodnôt (zvyčajne 'X') a cieľovú premennú (zvyčajne 'y').

### Vyhodnoťte model

Keď je proces trénovania dokončený (môže to trvať mnoho iterácií, alebo „epoch“, na trénovanie veľkého modelu), budete môcť vyhodnotiť kvalitu modelu použitím testovacích dát na zmeranie jeho výkonnosti. Tieto dáta sú podmnožinou pôvodných dát, ktoré model doposiaľ neanalyzoval. Môžete vypísať tabuľku metrík o kvalite vášho modelu.

🎓 **Fitting modelu**

V kontexte strojového učenia fitting modelu znamená presnosť základnej funkcie modelu pri analýze dát, s ktorými nie je oboznámený.

🎓 **Podtrénovanie** a **pretrénovanie** sú bežné problémy, ktoré znižujú kvalitu modelu, pretože model sa trénuje buď nedostatočne alebo príliš dôkladne. To spôsobuje, že model robí predikcie buď príliš presne zhodné alebo príliš voľné v porovnaní s tréningovými dátami. Pretrénovaný model predikuje tréningové dáta príliš dobre, pretože sa naučil detaily a šum v dátach príliš dôkladne. Podtrénovaný model nie je presný, pretože nedokáže ani presne analyzovať svoje tréningové dáta ani dáta, ktoré ešte nevidel.

![overfitting model](../../../../translated_images/sk/overfitting.1c132d92bfd93cb6.webp)
> Infografika od [Jen Looper](https://twitter.com/jenlooper)

## Ladenie parametrov

Keď je vaše počiatočné trénovanie dokončené, pozorujte kvalitu modelu a zvážte jeho zlepšenie úpravou „hyperparametrov“. Viac o tomto procese si môžete prečítať [v dokumentácii](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters?WT.mc_id=academic-77952-leestott).

## Predikcia

Toto je moment, kedy môžete použiť úplne nové dáta na otestovanie presnosti vášho modelu. V „aplikovanom“ prostredí ML, kde budujete webové zdroje na použitie modelu v produkcii, môže tento proces zahŕňať získavanie vstupov od používateľa (napríklad stlačením tlačidla) na nastavenie premennej a odoslanie jej do modelu na vyvodenie záveru, alebo vyhodnotenie.

V týchto lekciách objavíte, ako používať tieto kroky na prípravu, tvorbu, testovanie, vyhodnocovanie a predikciu – všetky gestá dátového vedca a viac, ako napredujete na svojej ceste stať sa „full stack“ ML inžinierom.

---

## 🚀Výzva

Nakreslite diagram znázorňujúci kroky praktikanta ML. Kde sa v procese práve vidíte? Kde predpokladáte, že budete mať ťažkosti? Čo sa vám zdá jednoduché?

## [Poprednáškový kvíz](https://ff-quizzes.netlify.app/en/ml/)

## Prehľad a samostatné štúdium

Vyhľadajte si online rozhovory s dátovými vedcami, ktorí rozprávajú o svojej každodennej práci. Tu je [jeden](https://www.youtube.com/watch?v=Z3IjgbbCEfs).

## Zadanie

[Rozhovor s dátovým vedcom](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Zrieknutie sa zodpovednosti**:  
Tento dokument bol preložený pomocou AI prekladateľskej služby [Co-op Translator](https://github.com/Azure/co-op-translator). Aj keď sa snažíme o presnosť, berte prosím na vedomie, že automatizované preklady môžu obsahovať chyby alebo nepresnosti. Pôvodný dokument v jeho rodnom jazyku by mal byť považovaný za autoritatívny zdroj. Pre kritické informácie sa odporúča profesionálny ľudský preklad. Nenesieme zodpovednosť za akékoľvek nedorozumenia alebo nesprávne interpretácie vyplývajúce z použitia tohto prekladu.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->