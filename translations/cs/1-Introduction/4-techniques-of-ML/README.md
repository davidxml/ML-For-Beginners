# Techniky strojového učení

Proces vytváření, používání a udržování modelů strojového učení a dat, která používají, je velmi odlišný od mnoha jiných vývojových pracovních postupů. V této lekci si tento proces objasníme a nastíníme hlavní techniky, které potřebujete znát. Naučíte se:

- Pochopit procesy, které stojí za strojovým učením na vysoké úrovni.
- Prozkoumat základní pojmy jako „modely“, „predikce“ a „tréninková data“.

## [Kvíz před přednáškou](https://ff-quizzes.netlify.app/en/ml/)

[![ML for beginners - Techniques of Machine Learning](https://img.youtube.com/vi/4NGM0U2ZSHU/0.jpg)](https://youtu.be/4NGM0U2ZSHU "ML for beginners - Techniques of Machine Learning")

> 🎥 Klikněte na obrázek výše pro krátké video, které prochází touto lekcí.

## Úvod

Na vysoké úrovni je umění vytváření procesů strojového učení (ML) složeno z několika kroků:

1. **Rozhodněte se o otázce**. Většina ML procesů začíná položením otázky, na kterou nelze odpovědět jednoduchým podmíněným programem nebo pravidlovým enginem. Tyto otázky se často točí kolem predikcí založených na sbírce dat.
2. **Sběr a příprava dat**. Abyste mohli odpovědět na svou otázku, potřebujete data. Kvalita a někdy i množství dat určí, jak dobře dokážete zodpovědět vaši původní otázku. Vizualizace dat je důležitou součástí této fáze. Tato fáze také zahrnuje rozdělení dat na tréninkovou a testovací skupinu pro vytvoření modelu.
3. **Výběr metody tréninku**. Podle vaší otázky a povahy dat je potřeba zvolit, jak chcete trénovat model tak, aby co nejlépe odrážel vaše data a umožnil přesné predikce. Toto je část ML procesu, která vyžaduje specifickou odbornost a často značné množství experimentování.
4. **Trénování modelu**. Pomocí tréninkových dat použijete různé algoritmy k natrénování modelu, který rozeznává vzory v datech. Model může využívat vnitřní váhy, které lze nastavit tak, aby upřednostňoval určité části dat před jinými a vytvořil tak lepší model.
5. **Hodnocení modelu**. Použijete dříve neviděná data (testovací data) z vašeho souboru, abyste zjistili, jak model funguje.
6. **Ladění parametrů**. Na základě výkonnosti modelu můžete celý proces zopakovat s různými parametry nebo proměnnými, které ovládají chování algoritmů použitých k tréninku modelu.
7. **Predikce**. Použijte nové vstupy k otestování přesnosti modelu.

## Jakou otázku položit

Počítače jsou zvlášť schopné objevovat skryté vzory v datech. Tato užitečnost je velmi přínosná pro výzkumníky, kteří mají otázky o dané oblasti, jež nelze snadno zodpovědět vytvořením podmíněného pravidlového enginu. Například datový vědec může pro aktuárské úkoly vytvořit ručně vyrobená pravidla o úmrtnosti kuřáků a nekuřáků.

Pokud však do rovnice přidáme mnoho dalších proměnných, může být model ML efektivnější pro předpověď budoucích úmrtnostních sazeb na základě minulé zdravotní historie. Veselější příklad může být předpověď počasí na měsíc duben v dané lokalitě na základě dat obsahujících zeměpisnou šířku, délku, změnu klimatu, blízkost oceánu, vzory proudění jet streamu a další.

✅ Tato [prezentace](https://www2.cisl.ucar.edu/sites/default/files/2021-10/0900%20June%2024%20Haupt_0.pdf) o modelech počasí nabízí historický pohled na použití ML v analýze počasí.  

## Úkoly před samotným vytvářením

Než začnete stavět svůj model, musíte splnit několik úkolů. Abyste mohli otestovat svou otázku a vytvořit hypotézu na základě predikcí modelu, je potřeba identifikovat a nakonfigurovat několik prvků.

### Data

Abyste mohli otázku odpovědět s jistotou, potřebujete dostatečné množství dat správného typu. V tomto bodě je třeba udělat dvě věci:

- **Sbírejte data**. Mějte na paměti předchozí lekci o spravedlnosti v analýze dat a sbírejte data pečlivě. Buďte si vědomi zdrojů dat, jakýchkoli inherentních zkreslení a dokumentujte jejich původ.
- **Připravte data**. Existuje několik kroků v procesu přípravy dat. Můžete potřebovat data sloučit a normalizovat, pokud pocházejí z různých zdrojů. Kvalitu a množství dat můžete vylepšit různými metodami, například převodem řetězců na čísla (jak děláme v [Clustering](../../5-Clustering/1-Visualize/README.md)). Můžete také generovat nová data na základě originálních (jak děláme v [Classification](../../4-Classification/1-Introduction/README.md)). Data můžete čistit a upravovat (jak uděláme před lekcí [Web App](../../3-Web-App/README.md)). Nakonec můžete data náhodně promíchat podle použitých tréninkových technik.

✅ Po sběru a zpracování dat si udělejte chvíli čas a zjistěte, zda jejich struktura umožní odpovědět na zamýšlenou otázku. Může se stát, že data nebudou pro daný úkol vhodná, jak zjistíme v lekcích o [Clusteringu](../../5-Clustering/1-Visualize/README.md)!

### Vlastnosti a cíl

[Feature](https://www.datasciencecentral.com/profiles/blogs/an-introduction-to-variable-and-feature-selection) (vlastnost) je měřitelná charakteristika dat. Většinou je vyjádřena jako nadpis sloupce, například „datum“, „velikost“ nebo „barva“. Vaše proměnná vlastnosti, obvykle označovaná jako `X` v kódu, představuje vstupní proměnnou použitou k natrénování modelu.

Cíl je věc, kterou se snažíte předpovědět. Cíl, obvykle označovaný jako `y` v kódu, představuje odpověď na otázku, kterou svým datům kladete: v prosinci, jakou **barvu** budou mít nejlevnější dýně? V San Francisku, která čtvrť bude mít nejlepší cenu nemovitostí **cena**? Cíl se někdy také nazývá štítek nebo label atribut.

### Výběr proměnné vlastnosti

🎓 **Výběr a extrakce vlastností** Jak vybrat správnou proměnnou při budování modelu? Pravděpodobně projdete procesem výběru vlastností nebo extrakce vlastností, abyste vybrali správné proměnné pro co nejlepší model. Nejde však o totéž: „Extrakce vlastností vytváří nové vlastnosti z funkcí originálních vlastností, zatímco výběr vlastností vybírá podmnožinu vlastností.“ ([zdroj](https://wikipedia.org/wiki/Feature_selection))

### Vizualizujte svá data

Důležitou součástí nástrojů datového vědce je schopnost vizualizovat data pomocí několika vynikajících knihoven, jako jsou Seaborn nebo MatPlotLib. Vizualizace dat vám může umožnit objevit skryté korelace, které můžete využít. Vaše vizualizace vám také může pomoci odhalit zkreslení nebo nevyvážená data (jak zjistíme v lekci o [Classification](../../4-Classification/2-Classifiers-1/README.md)).

### Rozdělte svůj dataset

Před tréninkem je třeba dataset rozdělit na dvě nebo více částí různé velikosti, které stále dobře reprezentují data.

- **Tréninková sada**. Tato část datasetu se používá k přizpůsobení modelu a jeho tréninku. Tato sada tvoří většinu původních dat.
- **Testovací sada**. Testovací data jsou nezávislá skupina dat, často vybraná z původních dat, kterou používáte ke kontrole výkonnosti natrénovaného modelu.
- **Validace**. Validace je menší nezávislá skupina příkladů, kterou používáte ke ladění hyperparametrů nebo architektury modelu, abyste model vylepšili. Podle velikosti vašich dat a otázky, kterou chcete zodpovědět, možná nebudete potřebovat tuto třetí sadu (jak uvádíme v [Time Series Forecasting](../../7-TimeSeries/1-Introduction/README.md)).

## Vytváření modelu

Pomocí tréninkových dat je vaším cílem vytvořit model nebo statistické zastoupení dat pomocí různých algoritmů k jeho **tréninku**. Trénování modelu jej vystavuje datům a umožňuje mu činit předpoklady o vnímaných vzorech, které objeví, ověří a přijme nebo odmítne.

### Rozhodněte se pro metodu tréninku

Podle otázky a povahy dat zvolíte vhodnou metodu pro trénování. Prozkoumáním [dokumentace Scikit-learn](https://scikit-learn.org/stable/user_guide.html) – kterou v tomto kurzu používáme – můžete objevovat mnoho způsobů, jak model trénovat. Podle zkušeností možná budete muset vyzkoušet několik různých metod, abyste vytvořili nejlepší model. Pravděpodobně se vydáte procesem, kdy datoví vědci hodnotí výkonnost modelu tak, že mu dávají neviděná data, kontrolují přesnost, zkreslení a další problémy s kvalitou a vybírají nejvhodnější metodu tréninku pro daný úkol.

### Natrénujte model

Vybaveni tréninkovými daty jste připraveni 'přizpůsobit' model. V mnoha knihovnách strojového učení narazíte na kód 'model.fit' – právě tehdy posíláte vstupní proměnnou jako pole hodnot (obvykle 'X') a cílovou proměnnou (obvykle 'y').

### Vyhodnoťte model

Jakmile je trénink kompletní (může to trvat mnoho iterací, nebo „epoch“, než se natrénuje velký model), můžete vyhodnotit kvalitu modelu použitím testovacích dat k měření jeho výkonnosti. Tato data jsou podmnožinou původních dat, která model předtím neviděl. Můžete si vypsat tabulku metrik kvality modelu.

🎓 **Přizpůsobení modelu**

V kontextu strojového učení znamená přizpůsobení modelu přesnost základní funkce modelu při pokusu analyzovat data, která nezná.

🎓 **Podtrénování** a **přetrénování** jsou běžné problémy snižující kvalitu modelu, kdy je model buď „málo přizpůsobený“ nebo „příliš přizpůsobený“. To způsobuje, že model dělá predikce, které jsou buď příliš přesné, nebo příliš volné vůči tréninkovým datům. Přetrénovaný model předpovídá tréninková data příliš dobře, protože detailně poznal data a šum. Podtrénovaný model není přesný, protože nedokáže správně analyzovat ani tréninková ani nová data, která dříve „neviděl“.

![overfitting model](../../../../translated_images/cs/overfitting.1c132d92bfd93cb6.webp)
> Infografika od [Jen Looper](https://twitter.com/jenlooper)

## Ladění parametrů

Jakmile dokončíte počáteční trénink, zhodnoťte kvalitu modelu a zvažte jeho zlepšení laděním jeho „hyperparametrů“. Více o procesu si můžete přečíst [v dokumentaci](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters?WT.mc_id=academic-77952-leestott).

## Predikce

Toto je okamžik, kdy můžete použít zcela nová data, abyste otestovali přesnost modelu. V „aplikačním“ prostředí ML, kdy vytváříte webové nástroje k nasazení modelu do produkce, může tento proces zahrnovat sběr uživatelského vstupu (například stisk tlačítka) k nastavení proměnné a odeslání do modelu pro odhad nebo vyhodnocení.

V těchto lekcích zjistíte, jak používat tyto kroky k přípravě, vytváření, testování, hodnocení a predikci – všechny úkony datového vědce a více, jak postupujete na cestě stát se „full stack“ ML inženýrem.

---

## 🚀Výzva

Nakreslete diagram znázorňující kroky praktikanta ML. Kde se nyní v procesu vidíte? Kde předpokládáte, že narazíte na potíže? Co vám přijde snadné?

## [Kvíz po přednášce](https://ff-quizzes.netlify.app/en/ml/)

## Revize & Samostudium

Vyhledejte online rozhovory s datovými vědci, kteří popisují svou každodenní práci. Zde je [jeden](https://www.youtube.com/watch?v=Z3IjgbbCEfs).

## Zadání

[Interview s datovým vědcem](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Prohlášení o vyloučení odpovědnosti**:  
Tento dokument byl přeložen pomocí AI překladatelské služby [Co-op Translator](https://github.com/Azure/co-op-translator). I když usilujeme o přesnost, mějte na paměti, že automatické překlady mohou obsahovat chyby nebo nepřesnosti. Původní dokument v jeho mateřském jazyce by měl být považován za závazný zdroj. Pro důležité informace se doporučuje profesionální lidský překlad. Nejsme odpovědní za jakékoli nedorozumění nebo chybné výklady vzniklé použitím tohoto překladu.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->