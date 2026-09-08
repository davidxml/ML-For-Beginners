# Gépi tanulás technikái

A gépi tanulási modellek és az általuk használt adatok felépítésének, használatának és karbantartásának folyamata nagyon eltér sok más fejlesztési munkafolyamattól. Ebben a leckében ezt a folyamatot megértjük, és áttekintjük a legfontosabb technikákat, amelyeket ismerned kell. Meg fogod érteni:

- A gépi tanulást alátámasztó folyamatokat magas szinten.
- Alapfogalmakat, például „modellek”, „előrejelzések” és „képzési adatok”.

## [Előadás előtti kvíz](https://ff-quizzes.netlify.app/en/ml/)

[![Gépi tanulás kezdőknek - Gépi tanulás technikái](https://img.youtube.com/vi/4NGM0U2ZSHU/0.jpg)](https://youtu.be/4NGM0U2ZSHU "Gépi tanulás kezdőknek - Gépi tanulás technikái")

> 🎥 Kattints a fenti képre egy rövid videó megtekintéséhez, amely bemutatja ezt a leckét.

## Bevezetés

Magas szinten a gépi tanulási (ML) folyamatok megalkotásának művészete több lépésből áll:

1. **Döntsd el a kérdést**. A legtöbb ML folyamat egy olyan kérdéssel kezdődik, amelyre nem lehet egyszerű feltételes programmal vagy szabályalapú motorral választ adni. Ezek a kérdések gyakran adatok alapján történő előrejelzésekhez kapcsolódnak.
2. **Gyűjtsd össze és készítsd elő az adatokat**. Ahhoz, hogy válaszolni tudj a kérdésedre, adatokra van szükséged. Az adatok minősége és néha mennyisége fogja meghatározni, hogy milyen jól tudod megválaszolni az eredeti kérdést. Az adatvizualizáció fontos része ennek a fázisnak. Ez a szakasz az adatok képzési és tesztelési csoportokra való felosztását is magában foglalja a modell felépítéséhez.
3. **Válassz képzési módszert**. A kérdésedtől és az adatok természetétől függően ki kell választanod, hogy hogyan szeretnél egy modellt tanítani, hogy az a legjobban tükrözze az adatokat és pontos előrejelzéseket adjon. Ez a gépi tanulási folyamat azon része, amely különleges szakértelmet igényel, és gyakran jelentős kísérletezést.
4. **Tanítsd meg a modellt**. A képzési adataid felhasználásával különböző algoritmusok segítségével tanítasz egy modellt, hogy felismerje az adatmintázatokat. A modell belső súlyokat alkalmazhat, amelyeket beállíthatsz, hogy bizonyos részeit az adatoknak előnyben részesítse a jobb modell érdekében.
5. **Értékeld ki a modellt**. Soha nem látott adatokkal (tesztelési adatokkal) ellenőrzöd, hogyan teljesít a modell.
6. **Paraméterhangolás**. A modell teljesítménye alapján újra végrehajthatod a folyamatot eltérő paraméterekkel, vagyis változókkal, amelyek irányítják az algoritmusok viselkedését a modell képzése során.
7. **Előrejelzés**. Új bemenetek használata a modell pontosságának tesztelésére.

## Milyen kérdést tegyünk fel

A számítógépek különösen jók az adatokban rejtett minták felfedezésében. Ez a képesség nagyon hasznos a kutatók számára, akik olyan kérdéseket tesznek fel adott területen, amelyekre nem könnyű válaszolni feltételes szabálymotor készítésével. Egy aktuáriusi feladat esetén például egy adattudós kézzel készített szabályokat dolgozhat ki a dohányzók és nem dohányzók halálozási arányára.

Amikor azonban sok más változót is figyelembe vesznek, egy ML modell hatékonyabb lehet a jövőbeli halálozási arány előrejelzésében a múltbeli egészségi adatok alapján. Egy derűsebb példa lehet az áprilisi időjárás előrejelzése egy adott területen, figyelembe véve az adatokat, például a szélességi és hosszúsági fokot, az éghajlatváltozást, a tenger közelségét, a sugáráramlat mintázatait és még sok mást.

✅ Ez az [időjárási modellekről szóló diákkészlet](https://www2.cisl.ucar.edu/sites/default/files/2021-10/0900%20June%2024%20Haupt_0.pdf) történelmi perspektívát kínál a gépi tanulás időjárási elemzésben való alkalmazásához.  

## Előkészületek

Mielőtt elkezdenéd a modell építését, több feladatot el kell végezned. A kérdés teszteléséhez, illetve a modell előrejelzésein alapuló hipotézis kialakításához azonosítanod és konfigurálnod kell több elemet.

### Adatok

Ahhoz, hogy bármilyen bizonyossággal válaszolni tudj a kérdésedre, megfelelő mennyiségű és megfelelő típusú adatra van szükséged. Ebben a pontban két dolgot kell tenned:

- **Adatgyűjtés**. A korábbi adatelemzési igazságosságról szóló leckére emlékezve, gyűjtsd az adatokat gondosan. Legyél tudatában az adatok forrásának, esetleges torzításainak, és dokumentáld azok eredetét.
- **Adatelőkészítés**. Az adatok előkészítése több lépést foglal magában. Összeszítheted és normalizálhatod az adatokat, ha azok különböző forrásból származnak. Javíthatod az adatok minőségét és mennyiségét különböző módszerekkel, például sztringek számokká konvertálásával (ahogy a [Klaszterezés](../../5-Clustering/1-Visualize/README.md) leckében tesszük). Új adatokat is generálhatsz az eredeti alapján (ahogy a [Klasszifikáció](../../4-Classification/1-Introduction/README.md) leckében is). Az adatokat tisztíthatod és szerkesztheted (ahogyan a [Web App](../../3-Web-App/README.md) leckéhez előkészítjük). Végül lehet, hogy véletlenszerűvé kell tenned, és át kell keverned az adatot a választott képzési technikáidtól függően.

✅ Az adatok összegyűjtése és feldolgozása után szánj egy pillanatot arra, hogy megnézd, az adatok formája megfelel-e a szándékolt kérdés megválaszolásához. Lehet, hogy az adat nem fog jól teljesíteni az adott feladatban, ahogy azt a [Klaszterezés](../../5-Clustering/1-Visualize/README.md) leckéinkben tapasztaljuk!

### Jellemzők és cél

A [jellemző](https://www.datasciencecentral.com/profiles/blogs/an-introduction-to-variable-and-feature-selection) az adataid mérhető tulajdonsága. Sok adatkészlet esetén oszlopfejlécként jelenik meg, például 'dátum', 'méret' vagy 'szín'. A jellemző változó, amelyet általában `X`-ként jelölünk a kódban, az a bemeneti változó, amelyet a modell betanításához használnak.

A cél az, amit megpróbálsz előrejelezni. A célt, amelyet általában `y`-ként jelölnek a kódban, a kérdésed megválaszolására használt válasznak tekintjük: decemberben milyen **színű** lesz a legolcsóbb tök? San Franciscóban melyik városrészeknek lesz a legjobb ingatlan **ára**? Néha a célt címkeattribútumként is emlegetik.

### Válaszd ki a jellemző változót

🎓 **Jellemző kiválasztás és jellemző kinyerés** Hogyan tudod, mely változót válaszd a modell építésénél? Valószínűleg végigjársz egy jellemzőkiválasztási vagy jellemzőkinyerési folyamatot, hogy a legmegfelelőbb változókat válaszd ki a legjobban teljesítő modellhez. Ezek azonban nem ugyanazok: „A jellemzőkinyerés új jellemzőket hoz létre az eredeti jellemzők függvényeiből, míg a jellemzőkiválasztás a jellemzők egy részhalmazát adja vissza.” ([forrás](https://wikipedia.org/wiki/Feature_selection))

### Vizualizáld az adatokat

Az adattudós eszköztárának fontos eleme az adatok vizualizációjának képessége, amely számos kiváló könyvtár segítségével elérhető, például Seaborn vagy MatPlotLib. Az adatok vizuális megjelenítése lehetővé teheti rejtett korrelációk felfedezését, amelyeket kihasználhatsz. Vizualizációid segíthetnek a torzítás vagy kiegyensúlyozatlan adatok felismerésében is (ahogyan azt a [Klasszifikáció](../../4-Classification/2-Classifiers-1/README.md) leckében látjuk).

### Oszd fel az adatkészletet

A képzés előtt az adatokat két vagy több, méretben eltérő, de mégis jól reprezentatív részre kell bontani.

- **Képzés**. Ez az adatkészlet azon része, amelyen a modellt tanítod. Ez az eredeti adatkészlet többségét alkotja.
- **Tesztelés**. A tesztadat-készlet független adatcsoport, gyakran az eredeti adatokból gyűjtve, amellyel a létrehozott modell teljesítményét ellenőrzöd.
- **Érvényesítés**. Az érvényesítő halmaz egy kisebb, független példa csoport, amellyel a modell hiperparamétereit vagy architektúráját hangolod a modell javítása érdekében. Az adatmérettől és a kérdéstől függően előfordulhat, hogy nincs szükség erre a harmadik halmazra (ahogy azt az [Idősor előrejelzés](../../7-TimeSeries/1-Introduction/README.md) leckében megjegyezzük).

## Modell építése

A képzési adathalmaz felhasználásával célod egy olyan modell, vagy statisztikai reprezentáció felépítése az adatokról, amelyet különböző algoritmusok segítségével **tanítasz**. A modell tanítása kitenni az adatnak a modellt, amely így feltételezéseket tehet az általa felfedezett, ellenőrzött és elfogadott mintákról.

### Válassz képzési módszert

A kérdésedtől és az adatok természetétől függően választhatsz egy módszert a tanításhoz. A [Scikit-learn dokumentációjában](https://scikit-learn.org/stable/user_guide.html) – amelyet ebben a kurzusban használunk – számos módon megvizsgálhatod a modell tréningjét. A tapasztalatodtól függően lehet, hogy több különböző módszert ki kell próbálnod a legjobb modell felépítéséhez. Valószínűleg végig fogsz menni egy folyamaton, amelyben adatkutatók különféle, nem látott adatok betáplálásával értékelik a modell teljesítményét, ellenőrzik pontosságát, torzítását és más minőséget rontó problémákat, majd kiválasztják a feladathoz legmegfelelőbb képzési módszert.

### Taníts meg egy modellt

Az adatokkal felvértezve készen állsz arra, hogy „illeszd” (fit) a modellt. Sok ML könyvtárban találkozhatsz a 'model.fit' kóddal – itt küldöd be a jellemző változót értéktömbként (általában 'X') és a célváltozót (általában 'y').

### Értékeld ki a modellt

Amikor a képzési folyamat befejeződik (egy nagy modellt sok iteráció, vagy „epocha” szükséges), értékelheted a modell minőségét tesztadatokkal, hogy megmérd a teljesítményét. Ez az adat az eredeti adatok egy olyan részhalmaza, amit a modell korábban nem elemzett. Ki tudsz nyomtatni egy táblázatot a modell minőségi mérőszámairól.

🎓 **Modellillesztés**

A gépi tanulás kontextusában a modellillesztés azt jelenti, hogy a modell alapvető függvénye mennyire pontos, amikor olyan adatokat próbál elemezni, amelyekhez nem szokott hozzá.

🎓 **Alulmérés** és **túlillesztés** gyakori problémák, amelyek rontják a modell minőségét, mivel a modell vagy nem illeszkedik eléggé, vagy túl jól. Ez azt eredményezi, hogy a modell vagy túl szorosan, vagy túl lazán igazodik a képzési adathoz. Egy túlillesztett modell túl jól előrejelez a képzési adatokon, mert túl jól megtanulta az adatok részleteit és zaját. Egy alulmért modell pontatlan, mivel sem a képzési adatokat, sem azokat az adatokat nem tudja pontosan elemezni, amelyeket még nem látott.

![túlillesztett modell](../../../../translated_images/hu/overfitting.1c132d92bfd93cb6.webp)
> Infografika: [Jen Looper](https://twitter.com/jenlooper)

## Paraméterhangolás

Miután az első képzés befejeződött, figyeld meg a modell minőségét, és fontold meg a 'hiperparaméterek' finomhangolását a modell javítása érdekében. Olvass többet a folyamatról [a dokumentációban](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters?WT.mc_id=academic-77952-leestott).

## Előrejelzés

Ez az a pillanat, amikor teljesen új adatokat használsz a modell pontosságának tesztelésére. Egy „alkalmazott” ML környezetben, ahol webes eszközöket építesz fel a modell éles használatához, ez a folyamat magában foglalhatja a felhasználói bemenet összegyűjtését (például egy gomb megnyomását), hogy egy változót állíts tele, amelyet a modellhez továbbítasz következtetéshez vagy értékeléshez.

Ezekben a leckékben felfedezed, hogyan használhatod ezeket a lépéseket az adatok előkészítésére, modellépítésre, tesztelésre, értékelésre és előrejelzésre – mindezt az adattudós gesztusai szerint, és még többet, ahogy haladsz az úton, hogy „full stack” ML mérnökké válj.

---

## 🚀Kihívás

Rajzolj egy folyamatábrát, amely bemutatja a gépi tanulással foglalkozó szakember lépéseit. Hol látod magad most a folyamatban? Hol vársz nehézségeket? Mi tűnik könnyűnek számodra?

## [Előadás utáni kvíz](https://ff-quizzes.netlify.app/en/ml/)

## Áttekintés és önálló tanulás

Keress online interjúkat olyan adattudósokkal, akik a napi munkájukról beszélnek. Itt van [egy](https://www.youtube.com/watch?v=Z3IjgbbCEfs).

## Feladat

[Interjú egy adattudóssal](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Felelősségkizárás**:
Ezt a dokumentumot az AI fordítási szolgáltatás, a [Co-op Translator](https://github.com/Azure/co-op-translator) segítségével fordítottuk. Bár a pontosságra törekszünk, kérjük, vegye figyelembe, hogy az automatikus fordítások hibákat vagy pontatlanságokat tartalmazhatnak. Az eredeti dokumentum az anyanyelvén tekintendő hiteles forrásnak. Kritikus információk esetén javasolt a professzionális emberi fordítás igénybevétele. Nem vállalunk felelősséget a fordítás használatából eredő félreértésekért vagy téves értelmezésekért.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->