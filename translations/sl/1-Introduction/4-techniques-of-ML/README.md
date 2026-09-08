# Tehnike strojnega učenja

Postopek gradnje, uporabe in vzdrževanja modelov strojnega učenja ter podatkov, ki jih uporabljajo, je zelo drugačen od mnogih drugih razvojnih delovnih tokov. V tej lekciji bomo razjasnili proces in opisali glavne tehnike, ki jih morate poznati. Naučili se boste:

- Razumeti procese, ki stojijo za strojnim učenjem na visoki ravni.
- Raziskati osnovne pojme, kot so 'modeli', 'napovedi' in 'učni podatki'.

## [Predpredavanje kviz](https://ff-quizzes.netlify.app/en/ml/)

[![ML for beginners - Techniques of Machine Learning](https://img.youtube.com/vi/4NGM0U2ZSHU/0.jpg)](https://youtu.be/4NGM0U2ZSHU "ML for beginners - Techniques of Machine Learning")

> 🎥 Kliknite na sliko zgoraj za kratek video, ki pokriva to lekcijo.

## Uvod

Na visoki ravni je obrt ustvarjanja procesov strojnega učenja (ML) sestavljena iz več korakov:

1. **Odločite se za vprašanje**. Večina procesov ML se začne z zastavljanjem vprašanja, na katerega ni mogoče odgovoriti z enostavnim pogojnim programom ali motorjem, ki temelji na pravilih. Ta vprašanja se pogosto nanašajo na napovedi na podlagi zbirke podatkov.
2. **Zberite in pripravite podatke**. Da bi lahko odgovorili na svoje vprašanje, potrebujete podatke. Kvaliteta in včasih tudi količina podatkov določata, kako dobro lahko odgovorite na začetno vprašanje. Vizualizacija podatkov je pomemben del te faze. Ta faza vključuje tudi razdelitev podatkov na učno in testno skupino za izdelavo modela.
3. **Izberite metodo učenja**. Glede na vaše vprašanje in naravo vaših podatkov morate izbrati, kako boste trenirali model, da bo najbolje odražal vaše podatke in omogočal natančne napovedi. To je del vašega ML procesa, ki zahteva posebno strokovno znanje in pogosto precej eksperimentiranja.
4. **Naučite model**. Z uporabo učnih podatkov boste uporabili različne algoritme za učenje modela, da prepozna vzorce v podatkih. Model lahko uporablja notranje uteži, ki jih je mogoče prilagoditi, da nekaterim delom podatkov dajejo prednost pred drugimi za boljši model.
5. **Ocenite model**. Uporabite nikoli prej nevidene podatke (testne podatke) iz zbranega nabora, da preverite, kako model deluje.
6. **Nastavitev parametrov**. Glede na uspešnost modela lahko postopek ponovite z drugačnimi parametri ali spremenljivkami, ki nadzorujejo vedenje algoritmov za učenje modela.
7. **Napovedovanje**. Uporabite nove vhodne podatke za testiranje natančnosti modela.

## Katero vprašanje zastaviti

Računalniki so še posebej vešči odkrivanja skritih vzorcev v podatkih. Ta uporabnost je zelo koristna za raziskovalce, ki imajo vprašanja o določenem področju, na katera ni enostavno odgovoriti z ustvarjanjem pogojnega motorja pravil. Na primer, pri aktuarijskih nalogah bi lahko podatkovni znanstvenik izdelal ročno izdelana pravila o smrtnosti kadilcev in nekadilcev.

Ko pa v enačbo vključimo še številne druge spremenljivke, se lahko izkaže, da je model ML učinkovitejši za napovedovanje prihodnjih stopenj smrtnosti na podlagi pretekle zdravstvene zgodovine. Bolj vesel primer je lahko izdelava vremenskih napovedi za mesec april v določeni lokaciji na podlagi podatkov, ki vključujejo geografsko širino, dolžino, podnebne spremembe, bližino oceana, vzorce jetnih tokov in še več.

✅ Ta [predstavitev](https://www2.cisl.ucar.edu/sites/default/files/2021-10/0900%20June%2024%20Haupt_0.pdf) o vremenskih modelih ponuja zgodovinski pogled na uporabo ML pri analizi vremena.

## Naloge pred gradnjo

Preden začnete graditi svoj model, morate opraviti več nalog. Da bi preizkusili svoje vprašanje in oblikovali hipotezo na podlagi napovedi modela, morate identificirati in konfigurirati več elementov.

### Podatki

Da lahko z gotovostjo odgovorite na svoje vprašanje, potrebujete dovolj podatkov ustrezne vrste. Na tej točki morate narediti dve stvari:

- **Zbirajte podatke**. Glede na prejšnjo lekcijo o pravičnosti pri analizi podatkov zbirajte podatke previdno. Zavedajte se virov teh podatkov, morebitnih pristranskosti, ki jih vsebujejo, in dokumentirajte njihov izvor.
- **Pripravite podatke**. Priprava podatkov vključuje več korakov. Morda boste morali podatke združiti in jih normalizirati, če prihajajo iz različnih virov. Izboljšate lahko kakovost in količino podatkov na različne načine, na primer s pretvarjanjem nizov v številke (kot smo naredili v [Gropljenju](../../5-Clustering/1-Visualize/README.md)). Lahko tudi ustvarite nove podatke na podlagi izvirnih (kot delamo v [Klasifikaciji](../../4-Classification/1-Introduction/README.md)). Podatke lahko tudi počistite in uredite (kot bomo naredili pred lekcijo [Web App](../../3-Web-App/README.md)). Nazadnje jih boste morda morali naključno premešati glede na vaše učne tehnike.

✅ Po zbiranju in obdelavi podatkov si vzemite trenutek in preverite, ali vam njihova oblika omogoča nasloviti zastavljeno vprašanje. Lahko se zgodi, da podatki za vaš podani nalog ne bodo delovali najbolje, kar bomo odkrili v lekcijah [Gropljenja](../../5-Clustering/1-Visualize/README.md)!

### Značilnosti in cilj

[Značilnost](https://www.datasciencecentral.com/profiles/blogs/an-introduction-to-variable-and-feature-selection) je merljiva lastnost vaših podatkov. V mnogih zbirkah podatkov je izražena kot naslov stolpca, kot sta 'datum', 'velikost' ali 'barva'. Vaša spremenljivka značilnosti, običajno označena kot `X` v kodi, predstavlja vhodno spremenljivko, ki jo bomo uporabili za učenje modela.

Cilj je stvar, ki jo poskušate napovedati. Cilj, običajno označen kot `y` v kodi, predstavlja odgovor na vprašanje, ki ga želite zastaviti podatkom: v decembru, katere **barve** bodo buče najcenejše? v San Franciscu, kateri predeli bodo imeli najboljšo **ceno** nepremičnin? Cilj se včasih imenuje tudi atribut oznake.

### Izbira spremenljivke značilnosti

🎓 **Izbor značilnosti in ekstrakcija značilnosti** Kako veste, katero spremenljivko izbrati pri gradnji modela? Verjetno boste opravili postopek izbire značilnosti ali ekstrakcije značilnosti, da boste izbrali prave spremenljivke za najbolj zmogljiv model. Vendar to ni isto: "Ekstrakcija značilnosti ustvari nove značilnosti iz funkcij izvirnih značilnosti, medtem ko izbor značilnosti vrne podmnožico značilnosti." ([vir](https://wikipedia.org/wiki/Feature_selection))

### Vizualizirajte podatke

Pomemben del orodjarne podatkovnega znanstvenika je moč vizualizirati podatke z uporabo več odličnih knjižnic, kot sta Seaborn ali MatPlotLib. Vizualna predstavitev podatkov vam lahko pomaga odkriti skrite korelacije, ki jih lahko uporabite. Vizualizacije vam lahko pomagajo tudi odkriti pristranskost ali neuravnotežene podatke (kot bomo videli v [Klasifikaciji](../../4-Classification/2-Classifiers-1/README.md)).

### Razdelite svojo podatkovno zbirko

Pred učenjem morate podatke razdeliti na dva ali več delov neenake velikosti, ki pa še vedno dobro predstavljajo podatke.

- **Učenje**. Ta del podatkovnega nabora je namenjen treningu vašega modela. Ta nabor predstavlja večino izvirnega podatkovnega nabora.
- **Testiranje**. Testni podatkovni nabor je neodvisna skupina podatkov, pogosto zbranih iz izvirnih podatkov, ki jih uporabite za potrditev uspešnosti zgrajenega modela.
- **Validacija**. Validacijski nabor je manjša neodvisna skupina primerov, ki jo uporabite za nastavitev hiperparametrov ali arhitekture modela za izboljšanje le-tega. Glede na velikost vaših podatkov in zastavljeno vprašanje morda ta tretji nabor ne bo potreben (kot ugotavljamo v [Napovedovanju časovnih vrst](../../7-TimeSeries/1-Introduction/README.md)).

## Gradnja modela

Z uporabo učnih podatkov je vaš cilj zgraditi model ali statistični prikaz vaših podatkov z uporabo različnih algoritmov za **učenje**. Učenje modela ga izpostavi podatkom in mu omogoča, da naredi domneve o zaznanih vzorcih, ki jih odkrije, potrdi ter sprejme ali zavrne.

### Odločite se za metodo učenja

Glede na vaše vprašanje in naravo podatkov boste izbrali metodo za učenje modela. Pregled [dokumentacije Scikit-learn](https://scikit-learn.org/stable/user_guide.html) – ki jo uporabljamo v tem kurzu – vam omogoča raziskati mnogo načinov treniranja modela. Glede na vaše izkušnje boste morda morali preizkusiti več različnih metod, da boste zgradili najboljši model. Verjetno boste šli skozi postopek, kjer podatkovni znanstveniki ocenjujejo učinkovitost modela z uporabo neznanih podatkov, preverjajo natančnost, pristranskosti in druge težave, ki zmanjšujejo kakovost, ter izberejo najbolj primerno metodo učenja za dano nalogo.

### Naučite model

Z učnimi podatki ste pripravljeni, da jih 'prilagodite' za ustvarjanje modela. Opazili boste, da v mnogih ML knjižnicah najdete ukaz 'model.fit' – prav v tem trenutku pošljete svojo spremenljivko značilnosti kot polje vrednosti (običajno 'X') in ciljno spremenljivko (običajno 'y').

### Ocenite model

Ko je postopek učenja zaključen (lahko traja več ponovitev ali 'epoh', da se nauči velik model), boste lahko ocenili kakovost modela z uporabo testnih podatkov za merjenje njegove zmogljivosti. Ti podatki so podmnožica izvirnih podatkov, ki jih model prej ni analiziral. Lahko si izpišete tabelo metrik kakovosti modela.

🎓 **Prilagajanje modela**

V kontekstu strojnega učenja se prilagajanje modela nanaša na natančnost funkcije modela, ki skuša analizirati podatke, s katerimi ni seznanjen.

🎓 **Podučenje** in **preučenje** sta pogosti težavi, ki zmanjšujeta kakovost modela, saj se model ali ni dovolj dobro prilagodi ali pa se preveč prilega. To povzroči, da model naredi napovedi, ki so preveč natančne ali preveč ohlapne glede na učne podatke. Preučen model preveč dobro napoveduje učne podatke, ker je preveč dobro naučil podrobnosti in šum podatkov. Podučen model ni natančen, saj ne more natančno analizirati ne učnih ne še nevidenih podatkov.

![overfitting model](../../../../translated_images/sl/overfitting.1c132d92bfd93cb6.webp)
> Infografika avtorice [Jen Looper](https://twitter.com/jenlooper)

## Nastavitev parametrov

Ko je začetno učenje končano, opazujte kakovost modela in razmislite o izboljšavah z nastavitvijo njegovih 'hiperparametrov'. Več o tem procesu preberite [v dokumentaciji](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters?WT.mc_id=academic-77952-leestott).

## Napovedovanje

To je trenutek, ko lahko uporabite popolnoma nove podatke za testiranje natančnosti modela. V 'uporabnem' okolju ML, kjer gradite spletne aplikacije za uporabo modela v produkciji, ta proces lahko vključuje zbiranje uporabniškega vnosa (na primer pritisk na gumb) za nastavitev spremenljivke in pošiljanje modelu v namen sklepanja ali ocene.

V teh lekcijah boste odkrili, kako uporabljati te korake za pripravo, gradnjo, testiranje, ocenjevanje in napovedovanje – vse geste podatkovnega znanstvenika in še več, ko boste napredovali v svoji poti, da postanete 'full stack' ML inženir.

---

## 🚀Izziv

Narišite potek postopka, ki prikazuje korake praktikanta ML. Kje vidite sebe zdaj v procesu? Kje pričakujete težave? Kaj vam zdi lahko?

## [Po-predavanje kviz](https://ff-quizzes.netlify.app/en/ml/)

## Pregled in samostojno učenje

Poiščite na spletu intervjuje s podatkovnimi znanstveniki, ki opisujejo svoje vsakdanje delo. Tukaj je [eden](https://www.youtube.com/watch?v=Z3IjgbbCEfs).

## Domača naloga

[Intervju s podatkovnim znanstvenikom](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Omejitev odgovornosti**:  
Ta dokument je bil preveden z uporabo AI prevajalske storitve [Co-op Translator](https://github.com/Azure/co-op-translator). Čeprav si prizadevamo za natančnost, vas prosimo, da upoštevate, da lahko samodejni prevodi vsebujejo napake ali netočnosti. Izvirni dokument v njegovem matičnem jeziku je treba obravnavati kot avtoritativni vir. Za kritične informacije priporočamo profesionalni človeški prevod. Za morebitna nesporazume ali napačne interpretacije, ki izhajajo iz uporabe tega prevoda, ne odgovarjamo.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->