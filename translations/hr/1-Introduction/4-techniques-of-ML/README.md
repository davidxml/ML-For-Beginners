# Tehnike strojnog učenja

Proces izgradnje, korištenja i održavanja modela strojnog učenja i podataka koje oni koriste vrlo je različit od mnogih drugih razvojnih tijekova. U ovoj lekciji razjasnit ćemo proces i izložiti glavne tehnike koje trebate znati. Naučit ćete:

- Razumjeti procese koji stoje iza strojnog učenja na visokoj razini.
- Istražiti osnovne pojmove kao što su 'modeli', 'predviđanja' i 'podaci za treniranje'.

## [Kviz prije predavanja](https://ff-quizzes.netlify.app/en/ml/)

[![ML for beginners - Techniques of Machine Learning](https://img.youtube.com/vi/4NGM0U2ZSHU/0.jpg)](https://youtu.be/4NGM0U2ZSHU "ML for beginners - Techniques of Machine Learning")

> 🎥 Kliknite na sliku gore za kratki video koji obrađuje ovu lekciju.

## Uvod

Na visokoj razini, vještina stvaranja procesa strojnog učenja (ML) sastoji se od niza koraka:

1. **Odlučite o pitanju**. Većina ML procesa započinje pitanjem koje se ne može odgovoriti jednostavnim uvjetnim programom ili motorom temeljenim na pravilima. Ta pitanja često se tiču predviđanja na temelju skupa podataka.
2. **Prikupite i pripremite podatke**. Da biste mogli odgovoriti na svoje pitanje, trebate podatke. Kvaliteta i ponekad količina podataka odredit će koliko dobro možete odgovoriti na početno pitanje. Vizualizacija podataka važan je aspekt ove faze. Ova faza također uključuje podjelu podataka u skup za treniranje i testiranje za izgradnju modela.
3. **Odaberite metodu treniranja**. Ovisno o vašem pitanju i prirodi podataka, morate odabrati kako želite trenirati model da najbolje odražava vaše podatke i da daje točna predviđanja. Ovo je dio ML procesa koji zahtijeva specifičnu stručnost i često značajnu količinu eksperimentiranja.
4. **Trenirajte model**. Koristeći svoje podatke za treniranje, upotrijebit ćete različite algoritme za treniranje modela kako bi prepoznao obrasce u podacima. Model može koristiti unutarnje težine koje se mogu prilagoditi kako bi se privilegirali određeni dijelovi podataka u odnosu na druge za izgradnju boljeg modela.
5. **Evaluirajte model**. Koristite podatke koje model prije nije vidio (vaše testne podatke) iz prikupljenog skupa kako biste vidjeli kako model radi.
6. **Podesite parametre**. Na temelju performansi modela možete ponoviti proces koristeći različite parametre ili varijable koje kontroliraju ponašanje algoritama korištenih za treniranje modela.
7. **Predviđajte**. Koristite nove ulaze za testiranje točnosti modela.

## Koje pitanje postaviti

Računala su posebno vješta u otkrivanju skrivenih obrazaca u podacima. Ova korisnost vrlo je korisna istraživačima koji imaju pitanja o određenom području na koja se ne može lako odgovoriti stvaranjem sustava pravila temeljenih na uvjetima. Na primjer, za aktuarijski zadatak, znanstvenik podataka mogao bi sastaviti ručno izrađena pravila o smrtnosti pušača naspram nepušača.

Međutim, kad se u jednadžbu uključi mnogo drugih varijabli, ML model može se pokazati učinkovitijim za predviđanje budućih stopa smrtnosti na temelju prijašnje zdravstvene povijesti. Veseliji primjer može biti predviđanje vremena za mjesec travanj na određenom mjestu temeljem podataka koji uključuju širinu, dužinu, klimatske promjene, blizinu oceana, obrasce jet stream-a i još mnogo toga.

✅ Ova [prezentacija](https://www2.cisl.ucar.edu/sites/default/files/2021-10/0900%20June%2024%20Haupt_0.pdf) o vremenskim modelima nudi povijesnu perspektivu korištenja ML u analizi vremena.

## Zadaci prije izgradnje

Prije nego što započnete s izgradnjom modela, postoje nekoliko zadataka koje morate dovršiti. Da biste testirali svoje pitanje i oblikovali hipotezu na temelju predviđanja modela, trebate identificirati i konfigurirati nekoliko elemenata.

### Podaci

Da biste mogli odgovoriti na svoje pitanje s bilo kakvom sigurnošću, trebate dobru količinu podataka odgovarajućeg tipa. Trenutno morate učiniti dvije stvari:

- **Prikupite podatke**. Imajući na umu prethodnu lekciju o pravednosti u analizi podataka, pažljivo prikupite svoje podatke. Budite svjesni izvora tih podataka, bilo kojih unutarnjih pristranosti koje mogu imati i dokumentirajte njihov izvor.
- **Pripremite podatke**. Postoji nekoliko koraka u procesu pripreme podataka. Možda ćete trebati objediniti podatke i normalizirati ih ako dolaze iz različitih izvora. Možete poboljšati kvalitetu i količinu podataka raznim metodama, poput pretvaranja nizova u brojeve (kao u [Klasifikaciji](../../5-Clustering/1-Visualize/README.md)). Možete također generirati nove podatke, temeljene na originalima (kao u [Klasifikaciji](../../4-Classification/1-Introduction/README.md)). Možete očistiti i urediti podatke (kao što ćemo učiniti prije lekcije [Web aplikacije](../../3-Web-App/README.md)). Na kraju, možda ćete ih također trebati nasumično posložiti i promiješati, ovisno o vašoj tehnici treniranja.

✅ Nakon prikupljanja i obrade podataka, odvojite trenutak da provjerite hoće li njihov oblik omogućiti odgovor na vaše postavljeno pitanje. Možda se pokaže da podaci neće dobro funkcionirati u zadatku, kao što otkrivamo u našim lekcijama o [Klasifikaciji](../../5-Clustering/1-Visualize/README.md)!

### Značajke i cilj

[Značajka](https://www.datasciencecentral.com/profiles/blogs/an-introduction-to-variable-and-feature-selection) je mjerljiva svojstva vaših podataka. U mnogim skupovima podataka izražena je kao naslov stupca poput 'datum', 'veličina' ili 'boja'. Vaša značajka varijabla, obično prikazana kao `X` u kodu, predstavlja ulaznu varijablu koja će se koristiti za treniranje modela.

Cilj je stvar koju pokušavate predvidjeti. Cilj, obično označen kao `y` u kodu, predstavlja odgovor na pitanje koje postavljate svojim podacima: u prosincu, koja će boja bundeva biti najjeftinija? U San Franciscu, koje će četvrti imati najbolju cijenu nekretnina? Cilj se ponekad naziva i atributom oznake.

### Odabir varijable značajke

🎓 **Odabir značajke i izdvajanje značajki** Kako znate koju varijablu odabrati prilikom izrade modela? Vjerojatno ćete proći kroz proces odabira značajki ili izdvajanja značajki kako biste odabrali prave varijable za najuspješniji model. Međutim, to nisu iste stvari: "Izdvajanje značajki stvara nove značajke iz funkcija izvornih značajki, dok odabir značajki vraća podskup značajki." ([izvor](https://wikipedia.org/wiki/Feature_selection))

### Vizualizirajte svoje podatke

Važan aspekt alata znanstvenika podataka je moć vizualizacije podataka koristeći nekoliko izvrsnih biblioteka poput Seaborn ili MatPlotLib. Vizualno predstavljanje podataka može vam omogućiti otkrivanje skrivenih korelacija koje možete iskoristiti. Vaši prikazi također vam mogu pomoći otkriti pristranost ili neuravnotežene podatke (kao što otkrivamo u [Klasifikaciji](../../4-Classification/2-Classifiers-1/README.md)).

### Podijelite svoj skup podataka

Prije treniranja, trebate podijeliti svoj skup podataka u dva ili više dijelova različite veličine koji još uvijek dobro predstavljaju podatke.

- **Trening**. Ovaj dio skupa podataka koristi se za treniranje modela. Ovaj skup čini većinu izvornog skupa podataka.
- **Testiranje**. Skup podataka za testiranje je neovisna skupina podataka, često prikupljena iz izvornog skupa, koju koristite za potvrdu performansi izgrađenog modela.
- **Validacija**. Skup za validaciju je manja neovisna skupina primjera koju koristite za podešavanje hiperparametara modela ili arhitekture da biste poboljšali model. Ovisno o veličini podataka i pitanju koje postavljate, možda nećete trebati stvarati ovaj treći skup (kao što navodimo u [Predviđanju vremenskih serija](../../7-TimeSeries/1-Introduction/README.md)).

## Izgradnja modela

Koristeći svoje podatke za treniranje, cilj vam je izgraditi model ili statistički prikaz podataka, koristeći razne algoritme za **treniranje**. Trening modela izlaže ga podacima i omogućuje mu da donosi pretpostavke o otkrivenim obrascima koje provjerava i prihvaća ili odbacuje.

### Odlučite o metodi treniranja

Ovisno o vašem pitanju i prirodi podataka, odabrat ćete metodu za treniranje. Pregledavajući [Scikit-learn dokumentaciju](https://scikit-learn.org/stable/user_guide.html) – koju koristimo u ovom tečaju – možete istražiti mnoge načine treniranja modela. Ovisno o iskustvu, možda ćete morati isprobati nekoliko različitih metoda da biste izgradili najbolji model. Vjerojatno ćete proći proces u kojem znanstvenici podataka ocjenjuju performanse modela tako da mu daju neviđene podatke, provjeravaju točnost, pristranost i druge probleme koji umanjuju kvalitetu, te odabiru najprikladniju metodu treniranja za zadatak.

### Trenirajte model

Naoružani podacima za treniranje, spremni ste na 'fit' modela. Primijetit ćete da mnoge ML biblioteke koriste kod 'model.fit' – upravo u toj fazi šaljete svoju varijablu značajke kao niz vrijednosti (obično 'X') i ciljnu varijablu (obično 'y').

### Evaluirajte model

Nakon što je proces treniranja završen (može trajati mnogo iteracija ili 'epoha' za treniranje velikog modela), moći ćete ocijeniti kvalitetu modela koristeći testne podatke za procjenu njegove izvedbe. Ti podaci su podskup izvornih podataka koje model ranije nije analizirao. Možete ispisati tablicu metrike o kvaliteti vašeg modela.

🎓 **Prilagođavanje modela**

U kontekstu strojnog učenja, prilagođavanje modela odnosi se na točnost funkcije modela dok pokušava analizirati podatke s kojima nije upoznat.

🎓 **Nedovoljno prilagođavanje** i **preveliko prilagođavanje** česti su problemi koji smanjuju kvalitetu modela, jer model prilagođava podatke ili nedovoljno ili previše. To uzrokuje da model predviđa preblizu ili previše slobodno u odnosu na svoje podatke za treniranje. Model koji je previše prilagođen predviđa podatke za treniranje previše točno jer je naučio detalje i šum podataka previše dobro. Model koji je nedovoljno prilagođen nije točan jer ne može ni precizno analizirati podatke za treniranje ni podatke koje još nije 'vidio'.

![overfitting model](../../../../translated_images/hr/overfitting.1c132d92bfd93cb6.webp)
> Infografika od [Jen Looper](https://twitter.com/jenlooper)

## Podešavanje parametara

Kad je vaše početno treniranje završeno, promatrajte kvalitetu modela i razmislite o njegovom poboljšanju podešavanjem njegovih 'hiperparametara'. Više o procesu pročitajte [u dokumentaciji](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters?WT.mc_id=academic-77952-leestott).

## Predviđanje

Ovo je trenutak kada možete koristiti potpuno nove podatke za testiranje točnosti vašeg modela. U 'primijenjenom' ML okruženju, gdje gradite web resurse za korištenje modela u produkciji, ovaj proces može uključivati prikupljanje korisničkog unosa (primjerice pritiskom na gumb) kako biste postavili varijablu i poslali je modelu na izvođenje zaključivanja ili evaluacije.

U ovim lekcijama otkrit ćete kako koristiti ove korake za pripremu, izgradnju, testiranje, evaluaciju i predviđanje – sve geste znanstvenika podataka i više, dok napredujete na putu da postanete 'full stack' ML inženjer.

---

## 🚀Izazov

Nacrtajte dijagram tijeka koji prikazuje korake ML praktičara. Gdje se sada vidite u procesu? Gdje predviđate da ćete naići na poteškoće? Što vam se čini lako?

## [Kviz nakon predavanja](https://ff-quizzes.netlify.app/en/ml/)

## Pregled i samostalno učenje

Potražite na internetu intervjue sa znanstvenicima podataka koji govore o svom svakodnevnom radu. Evo [jednog](https://www.youtube.com/watch?v=Z3IjgbbCEfs).

## Zadatak

[Interview a data scientist](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Odricanje od odgovornosti**:
Ovaj dokument je preveden pomoću AI usluge prijevoda [Co-op Translator](https://github.com/Azure/co-op-translator). Iako težimo točnosti, imajte na umu da automatski prijevodi mogu sadržavati pogreške ili netočnosti. Izvorni dokument na izvornom jeziku treba se smatrati autoritativnim izvorom. Za kritične informacije preporučuje se profesionalni ljudski prijevod. Nismo odgovorni za bilo kakva nesporazume ili pogrešna tumačenja proizašla iz korištenja ovog prijevoda.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->