# Koneoppimisen menetelmät

Koneoppimismallien rakentaminen, käyttäminen ja ylläpito sekä niiden käyttämän datan hallinta ovat hyvin erilaisia prosesseja moniin muihin kehitystyön työnkulkuihin verrattuna. Tässä oppitunnissa avaamme tätä prosessia ja esittelemme tärkeimmät tekniikat, jotka sinun tarvitsee tietää. Sinä:

- Ymmärrät koneoppimisen prosessien perusteet yleisellä tasolla.
- Tutustut peruskäsitteisiin, kuten 'mallit', 'ennusteet' ja 'koulutusdata'.

## [Ennakkotehtäväquiz](https://ff-quizzes.netlify.app/en/ml/)

[![ML for beginners - Techniques of Machine Learning](https://img.youtube.com/vi/4NGM0U2ZSHU/0.jpg)](https://youtu.be/4NGM0U2ZSHU "ML for beginners - Techniques of Machine Learning")

> 🎥 Klikkaa yllä olevaa kuvaa nähdäksesi lyhyen videon, joka käy tämän oppitunnin läpi.

## Johdanto

Yleisellä tasolla koneoppimisprosessin rakentaminen koostuu useista vaiheista:

1. **Päättää kysymys**. Useimmat ML-prosessit alkavat kysymyksellä, johon ei voida vastata yksinkertaisella ehtolausepohjaisella ohjelmalla tai sääntöpohjaisella moottorilla. Nämä kysymykset liittyvät usein ennusteisiin, jotka perustuvat tietoaineistoon.
2. **Kerää ja valmistele data**. Vastataksesi kysymykseen tarvitset dataa. Datan laatu ja joskus myös määrä määrää, kuinka hyvin pystyt vastaamaan alkuperäiseen kysymykseesi. Datan visualisointi on tärkeä osa tätä vaihetta. Tähän vaiheeseen kuuluu myös datan jakaminen koulutus- ja testiryhmiin mallin rakentamista varten.
3. **Valitse koulutusmenetelmä**. Kysymyksesi ja datasi luonteen mukaan sinun täytyy valita, miten koulutat mallin parhaiten heijastamaan dataasi ja tekemään tarkkoja ennusteita. Tämä osa ML-prosessia vaatii erityisosaamista ja usein runsaasti kokeiluja.
4. **Kouluta malli**. Käyttämällä koulutusdataa sovellat erilaisia algoritmeja mallin opettamiseen, jotta malli oppii tunnistamaan datan kaavat. Malli voi hyödyntää sisäisiä painoja, joita säädetään, jotta tietyt datan osat saavat suuremman painoarvon, mikä auttaa rakentamaan parempaa mallia.
5. **Arvioi malli**. Käytät koskaan aiemmin näkemätöntä dataa (testidataa) arvioidaksesi mallisi suorituskykyä.
6. **Parametrien viritys**. Mallin suorituskyvyn perusteella voit toistaa prosessin erilaisilla parametreilla eli muuttujilla, jotka ohjaavat käytettyjen algoritmien toimintaa mallin kouluttamisessa.
7. **Ennusta**. Käytä uusia syötteitä testataksesi mallisi tarkkuutta.

## Mitä kysymystä esittää

Tietokoneet osaavat erityisen hyvin löytää piilotettuja kaavoja datasta. Tämä on erittäin hyödyllistä tutkijoille, joilla on tiettyyn aihealueeseen liittyviä kysymyksiä, joihin ei voida helposti vastata ehdollisesti toimivan sääntöpohjaisen moottorin avulla. Esimerkiksi vakuutustyössä data-analyytikko voisi rakentaa käsin laadittuja sääntöjä tupakoitsijoiden ja ei-tupakoitsijoiden kuolleisuudesta.

Kun monia muuttujia tuodaan mukaan, koneoppimismalli voi osoittautua tehokkaammaksi ennustamaan tulevia kuolleisuuslukuja aiemman terveystiedon perusteella. Iloisempi esimerkki voisi olla säätiedotusten tekeminen huhtikuulle tietyssä paikassa käyttäen dataa, joka sisältää leveys- ja pituusasteet, ilmastonmuutoksen vaikutukset, meren läheisyyden, suihkuvirtausten kuviot ja muuta.

✅ Tämä [esitys](https://www2.cisl.ucar.edu/sites/default/files/2021-10/0900%20June%2024%20Haupt_0.pdf) säämallien käytöstä tarjoaa historiallisen näkökulman koneoppimisen hyödyntämiseen sääanalyysissä.  

## Ennen mallin rakentamista tehtävät tehtävät

Ennen mallin rakentamista on suoritettava useita tehtäviä. Kysymyksen testaamiseksi ja hypoteesin muodostamiseksi mallin ennusteiden perusteella sinun täytyy tunnistaa ja määrittää useita elementtejä.

### Data

Vastataksesi kysymykseesi millään varmuudella tarvitset riittävästi oikeanlaista dataa. Tässä vaiheessa sinun tulee tehdä kaksi asiaa:

- **Kerää dataa**. Muista edellisessä oppitunnissa käsitelty oikeudenmukaisuus datan analyysissa, ja kerää data huolellisesti. Ole tietoinen datan lähteistä, mahdollisista vääristymistä ja dokumentoi sen alkuperä.
- **Valmistele data**. Datan valmistelussa on useita vaiheita. Saatat joutua yhdistämään dataa ja normalisoimaan sitä, jos se tulee eri lähteistä. Voit parantaa datan laatua ja määrää muun muassa muuntamalla merkkijonot numeroiksi (kuten teemme [Klusteroinnissa](../../5-Clustering/1-Visualize/README.md)). Voit myös generoida uutta dataa alkuperäisen pohjalta (kuten teemme [Luokittelussa](../../4-Classification/1-Introduction/README.md)). Voit siivota ja muokata dataa (kuten teemme ennen [Verkkosovellus](../../3-Web-App/README.md) -oppituntia). Lopuksi voit myös joutua satunnaistamaan ja sekoittamaan dataa koulutusmenetelmistä riippuen.

✅ Kun olet kerännyt ja käsitellyt datasi, tarkista, mahdollistaako sen muoto vastaamaan tarkoitettua kysymystä. Käytettävän datan sopimattomuus voi aiheuttaa heikkoa suorituskykyä, kuten huomaamme [Klusterointivaiheessa](../../5-Clustering/1-Visualize/README.md)!

### Ominaisuudet ja tavoite

[Ominaisuus](https://www.datasciencecentral.com/profiles/blogs/an-introduction-to-variable-and-feature-selection) on mitattavissa oleva datan ominaisuus. Usein se esiintyy sarakeotsikkona, kuten 'päivämäärä', 'koko' tai 'väri'. Ominaisuusmuuttuja, yleensä koodissa merkitty `X`:llä, edustaa sisääntulomuuttujaa, jota käytetään mallin kouluttamiseen.

Tavoite on se asia, jota yrität ennustaa. Tavoite, yleensä koodissa `y`:llä, on vastaus kysymykseen, jonka esität datallesi: joulukuussa, minkä **värisiä** kurpitsoja on halvimpia? San Franciscossa, mitkä kaupunginosat tarjoavat parhaat kiinteistöjen **hinnat**? Tavoitetta kutsutaan joskus myös nimilabeliksi (label attribute).

### Ominaisuusmuuttujan valinta

🎓 **Ominaisuuksien valinta ja ominaisuuksien erottelu** Miten tiedät, minkä muuttujan valitset mallia rakentaessasi? Todennäköisesti käyt läpi ominaisuuksien valinta- tai erotteluprosessin valitaksesi sopivimmat muuttujat parhaan mallin rakentamiseksi. Ne eivät kuitenkaan ole sama asia: "Ominaisuuksien erottelu luo uusia ominaisuuksia alkuperäisten ominaisuuksien funktioista, kun taas ominaisuuksien valinta palauttaa osajoukon ominaisuuksista." ([lähde](https://wikipedia.org/wiki/Feature_selection))

### Visualisoi datasi

Tärkeä osa data-analyytikon työkalupakkia on kyky visualisoida dataa erilaisten erinomaisen hyvien kirjastojen, kuten Seabornin tai MatPlotLibin, avulla. Datan visuaalinen esitys voi antaa mahdollisuuden paljastaa piileviä riippuvuuksia, joita voit hyödyntää. Visualisointi voi myös auttaa sinua havaitsemaan harhaa tai epätasapainoista dataa (kuten huomaamme [Luokittelussa](../../4-Classification/2-Classifiers-1/README.md)).

### Jaa datasi osiin

Ennen mallin kouluttamista sinun on jaettava datasi kahteen tai useampaan epätasaiseen osaan, jotka kuitenkin edustavat dataa hyvin.

- **Koulutus**. Tämä osa data-aineistosta sovitetaan malliin sen kouluttamiseksi. Tämä joukko muodostaa suurimman osan alkuperäisestä datasta.
- **Testaus**. Testidatasetti on itsenäinen dataryhmä, usein kerätty alkuperäisestä datasta, jota käytetään mallin suorituskyvyn varmistamiseen.
- **Validointi**. Validointijoukko on pienempi riippumaton esimerkkiryhmä, jota käytetään mallin hyperparametrien eli arkkitehtuurin hienosäätöön parantamaan mallin suorituskykyä. Datasi koosta ja kysymystesi luonteesta riippuen tätä kolmatta joukkoa ei välttämättä tarvita (kuten mainitsemme [Aikasarjojen ennusteissa](../../7-TimeSeries/1-Introduction/README.md)).

## Mallin rakentaminen

Käyttäen koulutusdataa tavoitteesi on rakentaa malli tai tilastollinen esitys datastasi eri algoritmeja hyödyntäen mallin **kouluttamiseksi**. Mallin kouluttaminen altistaa sen datalle ja antaa sille mahdollisuuden tehdä oletuksia havaitsemistaan kuviosta, joita se validoi ja joko hyväksyy tai hylkää.

### Valitse koulutusmenetelmä

Kysymyksesi ja datasi luonteen mukaan valitset koulutusmenetelmän. Käymällä läpi [Scikit-learnin dokumentaatiota](https://scikit-learn.org/stable/user_guide.html) — jota käytämme tässä kurssissa — voit tutkia monia tapoja kouluttaa mallia. Kokemuksestasi riippuen sinun täytyy ehkä kokeilla useita eri menetelmiä parhaan mallin rakentamiseksi. Todennäköisesti käyt läpi prosessin, jossa data-analyytikot arvioivat mallin suorituskykyä syöttämällä sinne näkemätöntä dataa, tarkistamalla tarkkuuden, harhan ja muut suorituskykyä heikentävät tekijät sekä valitsevat tehtävään sopivimman koulutusmenetelmän.

### Kouluta malli

Koulutusdata allasi olet valmis 'sovittamaan' eli fittaamaan sen malliin. Useissa koneoppimiskirjastoissa näet koodin 'model.fit' — tässä vaiheessa syötät ominaisuusmuuttujan arvovektorina (yleensä 'X') ja tavoitemuuttujan (yleensä 'y').

### Arvioi malli

Kun koulutus on valmis (suuren mallin koulutus voi vaatia monta iterointia eli 'aikakautta'), voit arvioida mallin laatua käyttämällä testidataa sen suorituskyvyn mittaamiseen. Tämä data on alkuperäisen joukon osa, jota malli ei ole aiemmin analysoinut. Voit tulostaa taulukon mallin laadun mittareista.

🎓 **Mallin sovitus**

Koneoppimisen yhteydessä mallin sovitus tarkoittaa mallin toiminnan tarkkuutta analysoida dataa, jota se ei ole aiemmin nähnyt.

🎓 **Ali- ja ylisovitus** ovat yleisiä ongelmia, jotka heikentävät mallin laatua, kun malli sovitetaan joko liian löyhästi tai liian tarkasti. Tämä saa mallin tekemään ennusteita, jotka muistuttavat liikaa tai liian vähän koulutusdataa. Ylisovittanut malli ennustaa koulutusdataa liian tarkasti, koska se on oppinut datan yksityiskohdat ja kohinan liiankin hyvin. Alisovittanut malli ei ole tarkka, sillä se ei pysty analysoimaan oikein koulutusdataa eikä myöskään aiemmin näkemätöntä dataa.

![overfitting model](../../../../translated_images/fi/overfitting.1c132d92bfd93cb6.webp)
> Infografiikka: [Jen Looper](https://twitter.com/jenlooper)

## Parametrien viritys

Kun alkuperäinen koulutusprosessi on valmis, tarkkaile mallin laatua ja harkitse sen parantamista säätämällä 'hyperparametreja'. Lue lisää prosessista [dokumentaatiosta](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters?WT.mc_id=academic-77952-leestott).

## Ennustus

Tässä vaiheessa voit käyttää täysin uutta dataa testataksesi mallisi tarkkuutta. Käytännön koneoppimisen ympäristössä, jossa rakennat verkkosovelluksia mallin käyttämiseksi tuotannossa, tähän voi kuulua käyttäjän syötteen (esimerkiksi napin painallus) vastaanottaminen, sen muuttamiseen muuttujaksi ja mallin käyttämiseen päättelyyn eli arviointiin.

Näissä oppitunneissa opit käyttämään näitä vaiheita datan valmisteluun, mallin rakentamiseen, testaamiseen, arviointiin ja ennustamiseen — kaikki data-analyytikon keskeiset toimet, ja vielä enemmän, kun etenet kohti 'full stack' koneoppimisinsinöörin roolia.

---

## 🚀Haaste

Piirrä vuokaavio, jossa esität koneoppimisen toimijan vaiheet. Missä vaiheessa prosessia olet nyt? Missä ennakoit vaikeuksia? Mikä tuntuu sinusta helpolta?

## [Loppukoe](https://ff-quizzes.netlify.app/en/ml/)

## Kertaus & Itsenäinen opiskelu

Etsi verkosta haastatteluja data-analyytikoista, jotka kertovat päivittäisestä työstään. Tässä on [yksi](https://www.youtube.com/watch?v=Z3IjgbbCEfs).

## Tehtävä

[Haastattele data-analyytikkoa](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Vastuuvapauslauseke**:  
Tämä asiakirja on käännetty käyttämällä tekoälypohjaista käännöspalvelua [Co-op Translator](https://github.com/Azure/co-op-translator). Pyrimme tarkkuuteen, mutta huomioithan, että automatisoidut käännökset voivat sisältää virheitä tai epätarkkuuksia. Alkuperäistä asiakirjaa sen alkuperäiskielellä tulisi pitää ensisijaisena lähteenä. Tärkeissä tiedoissa suositellaan ammattimaista ihmiskäännöstä. Emme ota vastuuta tämän käännöksen käytöstä aiheutuvista väärinkäsityksistä tai tulkinnoista.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->