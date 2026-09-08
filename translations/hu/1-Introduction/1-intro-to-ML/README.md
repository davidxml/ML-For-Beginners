# Bevezetés a gépi tanulásba

## [Előadás előtti kvíz](https://ff-quizzes.netlify.app/en/ml/)

---

[![ML kezdőknek - Bevezetés a gépi tanulásba kezdőknek](https://img.youtube.com/vi/6mSx_KJxcHI/0.jpg)](https://youtu.be/6mSx_KJxcHI "ML kezdőknek - Bevezetés a gépi tanulásba kezdőknek")

> 🎥 Kattints a fenti képre egy rövid videó megtekintéséhez, amely ezen az órán dolgozik.

Üdvözlünk ezen a klasszikus gépi tanulással foglalkozó kezdő tanfolyamon! Akár teljesen új vagy a témában, akár tapasztalt ML-gyakorló, aki egy területet szeretne felfrissíteni, örülünk, hogy csatlakoztál hozzánk! Barátságos kiindulópontot szeretnénk teremteni a gépi tanulás tanulmányozásához, és örömmel értékeljük, válaszolunk, valamint beépítjük az [visszajelzésedet](https://github.com/microsoft/ML-For-Beginners/discussions).

[![Bevezetés a gépi tanulásba](https://img.youtube.com/vi/h0e2HAPTGF4/0.jpg)](https://youtu.be/h0e2HAPTGF4 "Bevezetés a gépi tanulásba")

> 🎥 Kattints a fenti képre egy videó megtekintéséhez: John Guttag az MIT-ről bemutatja a gépi tanulást

---
## Gépi tanulás kezdőknek

Mielőtt elkezdenéd ezt a tananyagot, be kell állítanod a számítógépedet, hogy képes legyen helyben futtatni a jegyzetfüzeteket.

- **Állítsd be a gépedet ezekkel a videókkal**. Használd az alábbi linkeket, hogy megtudd, [hogyan telepítsd a Pythont](https://youtu.be/CXZYvNRIAKM) a rendszeredre, és hogyan [állíts be egy szövegszerkesztőt](https://youtu.be/EU8eayHWoZg) fejlesztéshez.
- **Tanuld meg a Pythont**. Ajánlott alapszinten ismerni a [Pythont](https://docs.microsoft.com/learn/paths/python-language/?WT.mc_id=academic-77952-leestott), egy programozási nyelvet, amely hasznos az adattudósok számára, és amelyet ezen a tanfolyamon is használunk.
- **Tanuld meg a Node.js-t és a JavaScriptet**. Néhányszor JavaScriptet is használunk majd a tanfolyamon webalkalmazások építésekor, ezért szükséged lesz a [node](https://nodejs.org) és [npm](https://www.npmjs.com/) telepítésére, valamint a [Visual Studio Code](https://code.visualstudio.com/) használatára Python és JavaScript fejlesztéshez.
- **Hozz létre GitHub fiókot**. Mivel itt találtál ránk a [GitHub-on](https://github.com), lehet, hogy már van fiókod, de ha nincs, hozz létre egyet, majd forkold ezt a tananyagot, hogy a saját kedved szerint használd. (Nyugodtan adj nekünk egy csillagot is 😊)
- **Ismerkedj meg a Scikit-learnnel**. Ismerd meg a [Scikit-learn](https://scikit-learn.org/stable/user_guide.html) könyvtárat, egy olyan ML könyvtárkészletet, amelyet ezekben a leckékben hivatkozunk.

---
## Mi a gépi tanulás?

A "gépi tanulás" kifejezés manapság az egyik legnépszerűbb és leggyakrabban használt fogalom. Nem kis valószínűséggel hallottad már ezt a kifejezést legalább egyszer, ha valamennyire ismered a technológiát, bármilyen területen dolgozol is. A gépi tanulás mechanikája azonban a legtöbb ember számára titokzatos. Egy gépi tanulásban kezdő számára a téma néha túlnyomónak tűnhet. Ezért fontos megérteni, pontosan mi is a gépi tanulás, és lépésről lépésre, gyakorlati példákon keresztül tanulni róla.

---
## A hisztériahullám

![ml hype curve](../../../../translated_images/hu/hype.07183d711a17aafe.webp)

> A Google Trends bemutatja a 'gépi tanulás' kifejezés legutóbbi 'hisztériahullámát' (hype curve)

---
## Egy titokzatos univerzum

Egy érdekes titkokkal teli univerzumban élünk. Nagy tudósok, mint Stephen Hawking, Albert Einstein és sokan mások az életüket a világ titkainak feltárására szentelték. Ez az emberi tanulás állapota: egy gyerek folyamatosan új ismereteket tanul és évente fedezi fel a világának szerkezetét, miközben felnőtté válik.

---
## A gyermek agya

Egy gyermek agya és érzékelései észlelik a környezetük tényeket, és fokozatosan megtanulják az élet rejtett mintázatait, amelyek segítenek logikai szabályokat alkotni a megtanult minták azonosításához. Az emberi agy tanulási folyamata teszi az embert a világ legösszetettebb élőlényévé. A rejtett minták felfedezésével és azokon való innovációval folyamatosan jobbá tehetjük magunkat élethosszig tartó tanulás révén. Ez a tanulási képesség és fejlődő képesség kapcsolódik egy [agyplaszticitásnak](https://www.simplypsychology.org/brain-plasticity.html) nevezett fogalomhoz. Felszínesen motivációs hasonlóságokat találhatunk az emberi agy tanulási folyamata és a gépi tanulás fogalmai között.

---
## Az emberi agy

Az [emberi agy](https://www.livescience.com/29365-human-brain.html) érzékeli a valós világ dolgait, feldolgozza az észlelt információt, racionális döntéseket hoz és bizonyos cselekvéseket hajt végre a körülmények alapján. Ezt nevezzük intelligens viselkedésnek. Amikor egy gépre programozunk egy olyan utánozható intelligens viselkedési folyamatot, ezt mesterséges intelligenciának (AI) nevezzük.

---
## Néhány terminológia

Bár ezek a kifejezések összekeverhetőek, a gépi tanulás (ML) az AI fontos része. **Az ML specializált algoritmusokat használ arra, hogy értelmes információt tárjon fel és rejtett mintákat találjon az észlelt adatokból, hogy alátámassza a racionális döntéshozatalt**.

---
## AI, ML, Mélytanulás

![AI, ML, deep learning, data science](../../../../translated_images/hu/ai-ml-ds.537ea441b124ebf6.webp)

> Egy ábra, amely bemutatja az AI, ML, mélytanulás és adattudomány közti kapcsolatokat. Infografika készítője [Jen Looper](https://twitter.com/jenlooper), inspirálódva [ebben a grafikában](https://softwareengineering.stackexchange.com/questions/366996/distinction-between-ai-ml-neural-networks-deep-learning-and-data-mining)

---
## Megbeszélendő fogalmak

Ebben a tananyagban csak a gépi tanulás alapvető fogalmait fogjuk tárgyalni, amelyeket egy kezdőnek tudnia kell. Elsősorban a "klasszikus gépi tanulással" foglalkozunk, főként a Scikit-learn használatával, egy nagyszerű könyvtárral, amelyet sok tanuló használ az alapok elsajátításához. Az általánosabb mesterséges intelligencia vagy mélytanulás megértéséhez elengedhetetlen egy erős alapvető gépi tanulási ismeretanyag, amelyet itt kínálunk.

---
## Ebben a tanfolyamban megtanulod:

- a gépi tanulás alapfogalmait
- az ML történetét
- az ML és a méltányosság kérdését
- regressziós ML technikákat
- osztályozási ML technikákat
- klaszterezési ML technikákat
- természetes nyelvfeldolgozási ML technikákat
- idősor előrejelzési ML technikákat
- megerősítéses tanulást
- valós világban használatos ML alkalmazásokat

---
## Amit nem tárgyalunk

- mélytanulás
- neurális hálózatok
- AI

A jobb tanulási élmény érdekében elkerüljük a neurális hálózatok, a "mélytanulás" – azaz többrétegű neurális hálózatokkal való modellezés – és az AI bonyodalmait, amelyeket más tananyagban tárgyalunk majd. Hamarosan egy adattudományi tananyagot is kínálunk majd, amely erre a témakörre koncentrál.

---
## Miért tanuljunk gépi tanulást?

Rendszerszempontból a gépi tanulás az automatizált rendszerek létrehozását jelenti, amelyek képesek rejtett mintákat tanulni az adatokból, hogy intelligens döntéseket segítsenek hozni.

Ez az indíttatás lazán inspirált az emberi agy azon képessége által, hogy bizonyos dolgokat megtanul a külvilágból érkező adatok alapján.

✅ Gondolkozz egy percig, miért szeretne egy vállalkozás inkább gépi tanulási stratégiákat alkalmazni, ahelyett, hogy keménykódolt szabályalapú rendszert hozna létre.

---
## Miért számít az adatminőség?

A magas minőségű adatok javítják a modell teljesítményét. A rossz vagy zajos adatok pontatlan előrejelzésekhez vezethetnek, még fejlett gépi tanulási algoritmusok használata esetén is.

---
## Gépi tanulás alkalmazásai

A gépi tanulás alkalmazásai szinte mindenhol megtalálhatók, olyan elterjedtek, mint a társadalmunkban áramló adatok, amelyeket okostelefonjaink, kapcsolódó eszközeink és más rendszereink generálnak. Tekintve a legkorszerűbb gépi tanulási algoritmusok hatalmas potenciálját, a kutatók számos terepen játszanak a képességeivel, hogy többdimenziós, több tudományterületet érintő való életbeli problémákat oldjanak meg nagy sikerekkel.

---
## Alkalmazott ML példák

**Sokféleképpen használható a gépi tanulás**:

- Előrejelezni a betegség valószínűségét egy beteg orvosi története vagy jelentései alapján.
- Időjárási adatok kihasználásával megjósolni az időjárási eseményeket.
- Megérteni egy szöveg hangulatát.
- Hamis hírek felismerése a propaganda terjedésének megállításához.

A pénzügy, közgazdaságtan, földtudomány, űrkutatás, biomedikai mérnökség, kognitív tudomány, sőt még a bölcsészettudományok is alkalmazzák a gépi tanulást, hogy megoldják saját területük nehéz, adatfeldolgozási feladatait.

---
## Összefoglalás

A gépi tanulás automatizálja a mintakeresési folyamatot azzal, hogy jelentős betekintést nyújt valós vagy generált adatokból. Bizonyította értékességét az üzleti, egészségügyi és pénzügyi alkalmazások között.

A közeljövőben a gépi tanulás alapjainak megértése elengedhetetlen lesz bármely területről származó emberek számára a széleskörű terjedés miatt.

---
# 🚀 Kihívás

Rajzold le papíron vagy egy online alkalmazásban, például [Excalidraw](https://excalidraw.com/), az AI, ML, mélytanulás és adattudomány közti különbségekről alkotott elképzelésedet. Adj hozzá néhány ötletet azokról a problémákról, amelyeket ezek a technikák jól oldanak meg.

# [Előadás utáni kvíz](https://ff-quizzes.netlify.app/en/ml/)

---
# Áttekintés és önálló tanulás

Azt is megtudhatod, hogyan dolgozhatsz ML algoritmusokkal a felhőben, ha ezt a [tanulási utat](https://docs.microsoft.com/learn/paths/create-no-code-predictive-models-azure-machine-learning/?WT.mc_id=academic-77952-leestott) követed.

Készíts egy [tanulási utat](https://docs.microsoft.com/learn/modules/introduction-to-machine-learning/?WT.mc_id=academic-77952-leestott) a gépi tanulás alapjairól.

---
# Feladat

[Állítsd be és futtass](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Jogi nyilatkozat**:
Ez a dokumentum az AI fordítási szolgáltatás, a [Co-op Translator](https://github.com/Azure/co-op-translator) segítségével készült. Bár az pontosságra törekszünk, kérjük, vegye figyelembe, hogy az automatikus fordítások hibákat vagy pontatlanságokat tartalmazhatnak. Az eredeti dokumentum az anyanyelvén tekintendő hiteles forrásnak. Fontos információk esetén professzionális emberi fordítást javasolunk. Nem vállalunk felelősséget semmilyen félreértésért vagy téves értelmezésért, amely ebből a fordításból ered.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->