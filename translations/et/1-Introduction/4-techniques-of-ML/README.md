# Masinõppe tehnikad

Masinõppemudelite loomise, kasutamise ja hooldamise protsess ning andmed, mida nad kasutavad, on väga erinev paljudest teistest arendusvoogudest. Selles õppetükis demüstifitseerime protsessi ja anname ülevaate peamistest tehnikatest, mida on vaja teada. Sa:

- Mõistad masinõppe alusprotsesse ülevaatlikult.
- Uurite põhimõisteid nagu „mudelid“, „ennustused“ ja „õppematerjalid“.

## [Eelõppe viktoriin](https://ff-quizzes.netlify.app/en/ml/)

[![ML algajatele - Masinõppe tehnikad](https://img.youtube.com/vi/4NGM0U2ZSHU/0.jpg)](https://youtu.be/4NGM0U2ZSHU "ML algajatele - Masinõppe tehnikad")

> 🎥 Klõpsa ülaloleval pildil, et vaadata lühikest videot, mis läbib selle õppetüki.

## Sissejuhatus

Kõrgemal tasemel koosneb masinõppe (ML) protsesside loomine mitmest etapist:

1. **Küsimuse valimine**. Enamik ML protsesse algab küsimusest, millele ei saa vastata lihtsa tingimusprogrammi või reeglitel põhineva mootoriga. Need küsimused keerlevad sageli andmekogumi põhjal tehtavate ennustuste ümber.
2. **Andmete kogumine ja ettevalmistamine**. Küsimusele vastamiseks on vaja andmeid. Andmete kvaliteet ja mõnikord ka hulk määravad, kui hästi saate oma algset küsimust vastata. Andmete visualiseerimine on selle etapi oluline osa. Käesolev etapp hõlmab ka andmete jagamist treening- ja testimisrühmadeks mudeli ehitamiseks.
3. **Õppemeetodi valimine**. Sõltuvalt küsimusest ja andmete olemusest peate valima, kuidas mudelit treenida, et kõige paremini andmeid kajastada ja teha täpseid ennustusi. See on ML protsessi osa, mis nõuab eriteadmisi ja sageli ka rohkelt katsetamist.
4. **Mudeli treenimine**. Kasutades treeningandmeid, kasutate erinevaid algoritme, et treenida mudelit, mis tunneb ära mustreid andmetes. Mudel võib kasutada sisemisi kaalusid, mida saab kohandada, et anda teatud andmeosadele eelistust parema mudeli loomiseks.
5. **Mudeli hindamine**. Kasutate esmakordselt nägemata andmeid (testandmed) oma kogumist, et näha, kuidas mudel toimib.
6. **Parameetrite häälestamine**. Tuginedes mudeli jõudlusele, saate protsessi läbi teha uuesti, kasutades erinevaid parameetreid või muutujaid, mis kontrollivad algoritmide käitumist mudeli treenimiseks.
7. **Ennustamine**. Kasutage uusi sisendeid, et testida mudeli täpsust.

## Millist küsimust küsida

Arvutid on eriti osavad varjatud mustrite avastamisel andmetes. See omadus on väga kasulik teadlastele, kellel on küsimusi antud valdkonna kohta, millele ei saa lihtsalt tingimusel põhineva reeglimootoriga vastata. Näiteks aktuaariülesandes võib andmeteadlane koostada käsitsi loodud reegleid suitsetajate ja mitte-suitsetajate suremuse kohta.

Kui võetakse arvesse palju teisi muutujaid, võib masinõppemudel olla efektiivsem tulevaste suremusmäärade ennustamiseks varasema terviseandmete põhjal. Rõõmsam näide võib olla aprilli kuu ilmaennustuse tegemine antud asukoha kohta, põhinedes andmetel, mis sisaldavad laiust, pikkust, kliimamuutust, lähedust ookeanile, jetivoolu mustreid ja palju muud.

✅ See [slaidikomplekt](https://www2.cisl.ucar.edu/sites/default/files/2021-10/0900%20June%2024%20Haupt_0.pdf) ilma mudelite kohta pakub ajaloolist perspektiivi masinõppe kasutamise kohta ilmaanalüüsis.  

## Enne ehitamise ülesanded

Enne mudeli ehitamise alustamist tuleb sooritada mitu ülesannet. Selleks, et testida oma küsimust ja vormida hüpoteesi mudeli ennustuste põhjal, peate kindlaks tegema ja seadistama mitmed elemendid.

### Andmed

Et küsimusele kindlusega vastata, on vaja head kogust andmeid õiges formaadis. Sel hetkel tuleb teha kaks asja:

- **Koguge andmed**. Pidades meeles eelmist õppetundi õiglusest andmeanalüüsis, koguge andmed hoolikalt. Olge teadlik andmete allikatest, võimalike kallutatuste olemasolust ja dokumenteerige selle päritolu.
- **Valmistage andmed ette**. Andmete ettevalmistamise protsessis on mitu sammu. Võite vajada andmete kogumist ja normaliseerimist, kui need pärinevad erinevatest allikatest. Andmete kvaliteeti ja hulka saab parandada erinevate meetoditega, näiteks konverteerides stringid numbriteks (nagu teeme [klastris](../../5-Clustering/1-Visualize/README.md)). Samuti võite genereerida uusi andmeid, lähtudes algsetest andmetest (nagu teeme [klassifitseerimises](../../4-Classification/1-Introduction/README.md)). Andmeid saab puhastada ja redigeerida (nagu teeme enne [veebirakenduse](../../3-Web-App/README.md) õpetust). Lõpuks võib olla vajalik andmete juhuslikustamine ja segamine, sõltuvalt teie õppetehnikatest.

✅ Pärast andmete kogumist ja töötlemist võtke hetk, et hinnata, kas andmete kuju võimaldab teil esitatud küsimusele vastata. Võib-olla ei sobi andmed teie ülesande jaoks hästi, nagu avastame meie [klastrite](../../5-Clustering/1-Visualize/README.md) õppetundides!

### Tunnused ja sihtmärk

[Tunnus](https://www.datasciencecentral.com/profiles/blogs/an-introduction-to-variable-and-feature-selection) on andmete mõõdetav omadus. Paljudes andmekogumites väljendub see veerupäisena nagu „kuupäev“, „suurus“ või „värv“. Teie tunnuse muutuja, tavaliselt koodis tähistatud `X`-ga, esindab sisendmuutujat, mida kasutatakse mudeli treenimiseks.

Sihtmärk on see, mida püüate ennustada. Sihtmärk, tavaliselt tähistatud koodis `y`-ga, esindab küsimusele, mida esitate oma andmetele, vastust: detsembris, mis värvi kõrvitsad on odavaimad? San Franciscos, millistel linnaosadel on parim kinnisvara hind? Mõnikord nimetatakse sihtmärki ka sildi atribuudi nimega.

### Tunnuse muutuja valimine

🎓 **Tunnuse valik ja tunnuse ekstraheerimine** Kuidas valida muutuja mudeli ehitamisel? Tõenäoliselt läbite tunnuse valiku või tunnuse ekstraheerimise protsessi, et valida kõige sobivamad muutujad parima jõudlusega mudeli jaoks. Need pole siiski sama asi: "Tunnuse ekstraheerimine loob uusi tunnuseid originaaltunnuste funktsioonidest, samas kui tunnuse valik tagastab tunnuste alamhulga." ([allikas](https://wikipedia.org/wiki/Feature_selection))

### Visualiseeri oma andmeid

Andmeteadlase tööriistakasti oluline aspekt on võime visualiseerida andmeid, kasutades mitmeid suurepäraseid raamistikke nagu Seaborn või MatPlotLib. Andmete visuaalne esitamine võib aidata teil avastada varjatud korrelatsioone, mida saate ära kasutada. Teie visualiseeringud võivad aidata ka avastada kallutatust või ebatasakaalustatud andmeid (nagu avastame [klassifitseerimises](../../4-Classification/2-Classifiers-1/README.md)).

### Jagage oma andmekogum

Enne treenimist peate jagama oma andmekogu kahte või enamasse ebavõrdsesse ossa, mis siiski andmeid hästi esindavad.

- **Treening**. See osa andmekogumist sobitub mudeliga selle treenimiseks. See komplekt moodustab enamikku algsest andmekogumist.
- **Testimine**. Testandmekogu on sõltumatu andmed rühm, sageli kogutud algsetest andmetest, mida kasutatakse ehitatud mudeli jõudluse kinnitamiseks.
- **Kinnitamine**. Kinnitamise komplekt on väiksem sõltumatu näidiste kogum, mida kasutatakse mudeli hüperparameetrite või arhitektuuri häälestamiseks mudeli parandamiseks. Sõltuvalt andmete suurusest ja küsimusest, mida esitate, ei pruugi olla vaja seda kolmandat komplekti luua (nagu märgime [aegrea ennustamises](../../7-TimeSeries/1-Introduction/README.md)).

## Mudeli ehitamine

Kasutades treeningandmeid, on eesmärgiks ehitada mudel ehk statistiline andmete esitlus, kasutades erinevaid algoritme selle **treenimiseks**. Mudeli treenimine eksponeerib seda andmetele ja võimaldab tal teha oletusi avastatud mustrite kohta, kinnitada neid ja vastu võtta või tagasi lükata.

### Õppemeetodi valimine

Sõltuvalt küsimusest ja andmete olemusest valite, kuidas mudelit treenida. Läbides [Scikit-learni dokumentatsiooni](https://scikit-learn.org/stable/user_guide.html) – mida selles kursuses kasutame – saate uurida mitmeid viise, kuidas mudelit treenida. Oma kogemuse põhjal võib teil olla vaja proovida mitut erinevat meetodit parima mudeli ehitamiseks. Tõenäoliselt läbite protsessi, kus andmeteadlased hindavad mudeli jõudlust, pakkudes sellele nägemata andmeid, kontrollides täpsust, kallutatust ja muid kvaliteeti halvendavaid tegureid ning valides ülesande jaoks kõige sobivama õppemeetodi.

### Mudeli treenimine

Varustatuna treeningandmetega olete valmis mudelit „sobitama“. Paljuski leiab paljudes ML raamistikus koodi 'model.fit' – just sel hetkel saadate oma tunnuse muutuja väärtuste massiivi (tavaliselt 'X') ja sihtmuutuja (tavaliselt 'y').

### Mudeli hindamine

Kui treenimisprotsess on lõpule jõudnud (suurte mudelite treenimine võib võtta mitu iteratsiooni ehk "epohhi"), saate mudeli kvaliteeti hinnata, kasutades testandmeid tema jõudluse mõõtmiseks. Need andmed on alamhulk algsetest andmetest, mida mudel pole varem analüüsinud. Saate printida mudeli kvaliteedi mõõdikute tabeli.

🎓 **Mudeli sobitamine**

Masinõppe kontekstis viitab mudeli sobitamine mudeli aluseks oleva funktsiooni täpsusele, kui ta proovib analüüsida andmeid, millega tal veel kogemusi pole.

🎓 **Alasobitamine** ja **ülesobitamine** on levinud probleemid, mis vähendavad mudeli kvaliteeti, kui mudel kas ei sobitu piisavalt hästi või sobitub liiga täpselt. Selline olukord põhjustab, et mudel teeb ennustusi kas liiga rangelt või liiga vabalt sobitudes oma treeningandmetega. Ülesobitatud mudel ennustab treeningandmeid liiga hästi, kuna on õppinud andmete detailid ja müra liiga põhjalikult. Alasobitatud mudel ei ole täpne, kuna ei suuda täpselt analüüsida ei oma treeningandmeid ega ka varem "nägemata" andmeid.

![ülesobitatud mudel](../../../../translated_images/et/overfitting.1c132d92bfd93cb6.webp)
> Infograafika autor: [Jen Looper](https://twitter.com/jenlooper)

## Parameetrite häälestamine

Kui esmane treening on tehtud, jälgige mudeli kvaliteeti ja kaalutlege selle parandamist, häälestades selle „hüperparameetreid“. Selle protsessi kohta lugege rohkem [dokumentatsioonist](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters?WT.mc_id=academic-77952-leestott).

## Ennustamine

See on hetk, kus saate kasutada täiesti uusi andmeid, et testida mudeli täpsust. Rakendusliku ML keskkonnas, kus ehitate veebivarasid mudeli kasutamiseks tootmises, võib see protsess hõlmata kasutajasisendi kogumist (näiteks nupu vajutuse) muutuja seadmiseks ja mudelile edastamiseks inferentsiks ehk hinnanguks.

Nendes õppetundides avastate, kuidas neid samme kasutada ettevalmistamiseks, ehitamiseks, testimiseks, hindamiseks ja ennustamiseks – kõik need on andmeteadlase žestid ja palju muud – kui te arendate end ‘täispaketi’ masinõppe inseneriks.

---

## 🚀Väljakutse

Joonista vooskeem, mis peegeldab masinõppe praktikandi samme. Kus sa hetkel protsessis seisad? Kus arvad, et võivad tekkida raskused? Mis tundub sulle kerge?

## [Pärastloengu viktoriin](https://ff-quizzes.netlify.app/en/ml/)

## Kordamine ja iseseisev õppimine

Otsi internetist intervjuusid andmeteadlastega, kes räägivad oma igapäevatööst. Siin on [üks](https://www.youtube.com/watch?v=Z3IjgbbCEfs).

## Kodune ülesanne

[Intervjueeri andmeteadlast](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Vastutusest loobumine**:  
See dokument on tõlgitud AI tõlketeenuse [Co-op Translator](https://github.com/Azure/co-op-translator) abil. Kuigi püüame täpsust, palun pange tähele, et automatiseeritud tõlked võivad sisaldada vigu või ebatäpsusi. Originaaldokument oma emakeeles tuleks pidada autoriteetseks allikaks. Olulise teabe puhul soovitatakse professionaalset inimtõlget. Me ei vastuta selle tõlke kasutamisest tulenevate arusaamatuste ega valesti tõlgendamise eest.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->