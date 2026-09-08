# Įvadas į mašininį mokymąsi

## [Priešpaskaitinė viktorina](https://ff-quizzes.netlify.app/en/ml/)

---

[![ML for beginners - Introduction to Machine Learning for Beginners](https://img.youtube.com/vi/6mSx_KJxcHI/0.jpg)](https://youtu.be/6mSx_KJxcHI "ML for beginners - Introduction to Machine Learning for Beginners")

> 🎥 Spustelėkite aukščiau esantį vaizdą, kad pamatytumėte trumpą vaizdo įrašą, apžvelgiantį šią pamoką.

Sveiki atvykę į šį klasikinių mašininio mokymosi kursą pradedantiesiems! Nesvarbu, ar jūs visiškai naujas šioje srityje, ar patyręs ML praktikas, norintis pasipraktikuoti tam tikrą sritį, mes džiaugiamės, kad prisijungėte! Mes norime sukurti draugišką starto vietą jūsų ML studijoms ir būtų malonu įvertinti, atsakyti į ir įtraukti jūsų [atsiliepimus](https://github.com/microsoft/ML-For-Beginners/discussions).

[![Introduction to ML](https://img.youtube.com/vi/h0e2HAPTGF4/0.jpg)](https://youtu.be/h0e2HAPTGF4 "Introduction to ML")

> 🎥 Spustelėkite aukščiau esantį vaizdą, kad pamatytumėte vaizdo įrašą: MIT's John Guttag pristato mašininį mokymąsi

---
## Pradžia su mašininiu mokymusi

Prieš pradedant šią programą, reikia turėti paruoštą kompiuterį, kuris galėtų vykdyti užrašų knygeles vietoje.

- **Sužinokite, kaip sukonfigūruoti savo mašiną pagal šiuos vaizdo įrašus.** Naudokite šias nuorodas, norėdami sužinoti [kaip įdiegti Python](https://youtu.be/CXZYvNRIAKM) sistemoje ir [kaip nustatyti teksto redaktorių](https://youtu.be/EU8eayHWoZg) kūrimui.
- **Išmokite Python.** Taip pat rekomenduojama turėti pagrindines žinias apie [Python](https://docs.microsoft.com/learn/paths/python-language/?WT.mc_id=academic-77952-leestott), tai programavimo kalba, naudinga duomenų mokslininkams ir kurią naudodami dirbsime šiame kurse.
- **Išmokite Node.js ir JavaScript.** Šiame kurse taip pat keletą kartų naudosime JavaScript, kuriant žiniatinklio programas, todėl reikės turėti įdiegtus [node](https://nodejs.org) ir [npm](https://www.npmjs.com/), taip pat [Visual Studio Code](https://code.visualstudio.com/), skirtą Python ir JavaScript kūrimui.
- **Sukurkite GitHub paskyrą.** Kadangi radote mus čia, [GitHub](https://github.com), galbūt jau turite paskyrą, bet jei ne, sukurkite ją ir tada susikurkite šios programos forką naudoti savo tikslams. (Taip pat galite mums uždegti žvaigždutę 😊)
- **Susipažinkite su Scikit-learn.** Pažinkite [Scikit-learn](https://scikit-learn.org/stable/user_guide.html), ML bibliotekų rinkinį, kurį naudosime šioms pamokoms.

---
## Kas yra mašininis mokymasis?

Terminas „mašininis mokymasis“ yra vienas iš populiariausių ir dažniausiai naudojamų šiandien. Neabejotinai yra didelė tikimybė, kad šį terminą bent kartą esate girdėjęs, jei turite tam tikrą pažinimą apie technologijas, nesvarbu, kurioje srityje dirbate. Tačiau mašininio mokymosi mechanizmai daugumai žmonių yra paslaptis. Mašininio mokymosi pradedančiajam šis dalykas kartais gali pasirodyti pribloškiantis. Todėl svarbu suprasti, kas iš tikrųjų yra mašininis mokymasis, ir mokytis po žingsnelį, praktiškai.

---
## Populiarumo kreivė

![ml hype curve](../../../../translated_images/lt/hype.07183d711a17aafe.webp)

> Google Trends rodo neseną termino „mašininis mokymasis“ „populiarumo kreivę“

---
## Paslaptinga visata

Mes gyvename visatoje, kupinoje įdomių paslapčių. Didieji mokslininkai, tokie kaip Stephen Hawking, Albert Einstein ir daugelis kitų, skyrė savo gyvenimus reikšmingos informacijos paieškai, atveriančiai mus supančio pasaulio paslaptis. Tai yra žmogaus mokymosi sąlyga: žmogaus vaikas mokosi naujų dalykų ir metų metais atranda savo pasaulio struktūrą, kol subręsta.

---
## Vaiko smegenys

Vaiko smegenys ir jutimai suvokia aplinkos faktus ir palaipsniui įsisavina gyvenimo paslėptas taisykles, kurios padeda vaikui sukurti loginius dėsnius, identifikuojančius išmoktas taisykles. Žmogaus smegenų mokymosi procesas daro žmogų šio pasaulio sudėtingiausiu gyvuoju padaru. Nuolatinis mokymasis atrandant paslėptas taisykles ir jas tobulinant leidžia mums vis gerėti visą gyvenimą. Šis mokymosi gebėjimas ir evoliucinis potencialas susijęs su sąvoka, vadinama [smegenų plastika](https://www.simplypsychology.org/brain-plasticity.html). Paviršutiniškai galima rasti keletą motyvacinių panašumų tarp žmogaus smegenų mokymosi proceso ir mašininio mokymosi koncepcijų.

---
## Žmogaus smegenys

[Žmogaus smegenys](https://www.livescience.com/29365-human-brain.html) suvokia dalykus iš realaus pasaulio, apdoroja gautą informaciją, priima racionalius sprendimus ir atlieka tam tikrus veiksmus, atsižvelgdamos į aplinkybes. Tai vadiname protingu elgesiu. Kai mes programuojame imituoti šį protingą elgesio procesą mašinoje, tai vadinama dirbtiniu intelektu (AI).

---
## Terminai

Nors terminai kartais painiojami, mašininis mokymasis (ML) yra svarbi dirbtinio intelekto dalis. **ML susijęs su specialių algoritmų naudojimu siekiant atrasti reikšmingą informaciją ir paslėptas taisykles iš duomenų, patvirtinančių racionalaus sprendimų priėmimo procesą**.

---
## AI, ML, gilusis mokymasis

![AI, ML, deep learning, data science](../../../../translated_images/lt/ai-ml-ds.537ea441b124ebf6.webp)

> Diagrama, rodanti AI, ML, giliojo mokymosi ir duomenų mokslo ryšius. Infografiką sukūrė [Jen Looper](https://twitter.com/jenlooper), įkvėpta [šios grafikos](https://softwareengineering.stackexchange.com/questions/366996/distinction-between-ai-ml-neural-networks-deep-learning-and-data-mining)

---
## Sąvokos, kurias aptarsime

Šiame programoje mes aprėpsime tik pagrindines mašininio mokymosi sąvokas, kurių pradedantiesiems būtina išmokti. Mes daugiausia kalbėsime apie „klasikinį mašininį mokymąsi“, naudodamiesi puikia biblioteka Scikit-learn, kurią daugelis studentų naudoja mokydamiesi pagrindų. Norint suprasti platesnes dirbtinio intelekto ar giliojo mokymosi koncepcijas, būtinos tvirtos mašininio mokymosi žinios, kurias mes norime čia suteikti.

---
## Šiame kurse išmoksite:

- pagrindines mašininio mokymosi sąvokas
- ML istoriją
- ML ir sąžiningumo aspektus
- regresijos ML metodus
- klasifikacijos ML metodus
- klasterizacijos ML metodus
- natūralios kalbos apdorojimo ML metodus
- laiko eilučių prognozavimo ML metodus
- stiprinamąjį mokymąsi
- praktines ML taikymo sritis

---
## Ko neaptarsime

- giliojo mokymosi
- dirbtinių neuroninių tinklų
- AI

Gerinti mokymosi patirtį vengsime neuroninių tinklų kompleksų, „giliojo mokymosi“ – daugiasluoksnio modelių kūrimo naudojant neuroninius tinklus – bei AI temų, kurios bus aptariamos kitoje programoje. Taip pat artimiausiu metu pasiūlysime duomenų mokslo programą, skirtą šiai sričiai.

---
## Kodėl verta studijuoti mašininį mokymąsi?

Mašininis mokymasis, žiūrint iš sistemų perspektyvos, apibrėžiamas kaip automatizuotų sistemų kūrimas, kurios gali išmokti paslėptas taisykles iš duomenų, padėdamos priimti protingus sprendimus.

Ši motyvacija yra laisvai įkvėpta žmogaus smegenų mokymosi principo, kai smegenys mokosi dalykų iš išorinio pasaulio suvoktos informacijos.

✅ Pagalvokite akimirką, kodėl verslas norėtų naudoti mašininio mokymosi strategijas, o ne kurti taisyklių variklį pagal iš anksto apibrėžtas taisykles.

---
## Kodėl svarbi duomenų kokybė

Aukštos kokybės duomenys gerina modelio našumą. Prasti arba triukšmingi duomenys gali sukelti netikslius prognozes net naudojant pažangiausius mašininio mokymosi algoritmus.

---
## Mašininio mokymosi taikymai

Mašininio mokymosi taikymai dabar yra beveik visur ir yra tokie įprasti, kaip duomenys, kurie plaukia mūsų visuomenėse, generuojami mūsų išmaniuosiuose telefonuose, prijungtuose įrenginiuose ir kituose sistemose. Atsižvelgiant į pažangių mašininio mokymosi algoritmų didžiulį potencialą, mokslininkai tyrinėja jų galimybes spręsti daugiasluoksnes ir daugdisciplinines realaus gyvenimo problemas su puikiais teigiamais rezultatais.

---
## Mašininio mokymosi taikymo pavyzdžiai

**Mašininį mokymąsi galite naudoti įvairiais būdais**:

- Nuspėti ligos tikimybę iš paciento medicininės istorijos ar įrašų.
- Naudotis orų duomenimis orų įvykiams prognozuoti.
- Suprasti teksto nuotaiką.
- Aptikti netikras žinias ir sustabdyti propagandos plitimą.

Finansai, ekonomika, žemės moksla, kosmoso tyrimai, biomedicinos inžinerija, pažinimo mokslas ir net humanitariniai mokslai pritaikė mašininį mokymąsi spręsti sudėtingas, duomenų apdorojimo reikalaujančias savo srities problemas.

---
## Išvada

Mašininis mokymasis automatizuoja modelių atradimo procesą, ieškodamas prasmingų įžvalgų iš realių ar generuotų duomenų. Jis pasitvirtino esąs itin vertingas verslo, sveikatos ir finansų taikymuose bei kituose.

Artimoje ateityje pagrindinių mašininio mokymosi žinių supratimas bus būtinas bet kurios srities žmonėms dėl plačiai paplitusio taikymo.

---
# 🚀 Iššūkis

Nupieškite ant popieriaus arba naudodamiesi internetine programa, pvz., [Excalidraw](https://excalidraw.com/), kaip suprantate skirtumus tarp AI, ML, giliojo mokymosi ir duomenų mokslo. Pridėkite idėjų, kokias problemas kiekviena iš šių technikų gerai sprendžia.

# [Po paskaitos viktorina](https://ff-quizzes.netlify.app/en/ml/)

---
# Peržiūra ir savarankiškas mokymasis

Norėdami sužinoti daugiau apie tai, kaip dirbti su ML algoritmais debesyje, sekite šią [mokymosi programą](https://docs.microsoft.com/learn/paths/create-no-code-predictive-models-azure-machine-learning/?WT.mc_id=academic-77952-leestott).

Išmokite pagrindus naudodamiesi šia [mokymosi programa](https://docs.microsoft.com/learn/modules/introduction-to-machine-learning/?WT.mc_id=academic-77952-leestott).

---
# Užduotis

[Pradėkite ir paleiskite](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Atsakomybės apribojimas**:
Šis dokumentas buvo išverstas naudojant dirbtinio intelekto vertimo paslaugą [Co-op Translator](https://github.com/Azure/co-op-translator). Nors siekiame tikslumo, prašome atkreipti dėmesį, kad automatiniai vertimai gali turėti klaidų ar netikslumų. Originalus dokumentas jo gimtąja kalba laikomas autoritetingu šaltiniu. Svarbiai informacijai rekomenduojama naudoti profesionalų žmogiškąjį vertimą. Mes neatsakome už jokius nesusipratimus ar neteisingą interpretaciją, kilusią naudojantis šiuo vertimu.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->