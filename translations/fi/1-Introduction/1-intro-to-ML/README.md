# Johdatus koneoppimiseen

## [Ennakkoluentokysely](https://ff-quizzes.netlify.app/en/ml/)

---

[![ML aloittelijoille - Johdatus koneoppimiseen aloittelijoille](https://img.youtube.com/vi/6mSx_KJxcHI/0.jpg)](https://youtu.be/6mSx_KJxcHI "ML aloittelijoille - Johdatus koneoppimiseen aloittelijoille")

> 🎥 Klikkaa yllä olevaa kuvaa katsomaan lyhyt video, jossa käydään läpi tämä oppitunti.

Tervetuloa tälle kurssille klassisesta koneoppimisesta aloittelijoille! Olitpa täysin uusi aiheessa tai kokenut koneoppimisen harjoittaja, joka haluaa kerrata jotakin aluetta, olemme iloisia, että liityt seuraamme! Haluamme luoda ystävällisen aloituspaikan koneoppimisen opiskelullesi ja otamme mielellämme vastaan, arvioimme ja sisällytämme palautteesi [palautteeseen](https://github.com/microsoft/ML-For-Beginners/discussions).

[![Johdatus ML:ään](https://img.youtube.com/vi/h0e2HAPTGF4/0.jpg)](https://youtu.be/h0e2HAPTGF4 "Johdatus ML:ään")

> 🎥 Klikkaa yllä olevaa kuvaa videoon: MIT:n John Guttag esittelee koneoppimisen

---
## Koneoppimisen aloittaminen

Ennen kuin aloitat tämän opetussuunnitelman, sinun täytyy saada tietokoneesi käyttövalmiiksi ja pystyä suorittamaan muistikirjoja paikallisesti.

- **Konfiguroi koneesi näillä videoilla**. Käytä seuraavia linkkejä oppiaksesi [kuinka Python asennetaan](https://youtu.be/CXZYvNRIAKM) järjestelmääsi ja [tekstieditori kehitykseen](https://youtu.be/EU8eayHWoZg).
- **Opiskele Pythonia**. On myös suositeltavaa saada perustiedot [Pythonista](https://docs.microsoft.com/learn/paths/python-language/?WT.mc_id=academic-77952-leestott), ohjelmointikielestä, joka on hyödyllinen datatieteilijöille ja jota käytämme tässä kurssissa.
- **Opiskele Node.js:ää ja JavaScriptiä**. Käytämme myös JavaScriptiä muutaman kerran tässä kurssissa verkkosovelluksia rakentaessa, joten tarvitset [noden](https://nodejs.org) ja [npm:n](https://www.npmjs.com/) asennettuna sekä [Visual Studio Code](https://code.visualstudio.com/) niin Python- kuin JavaScript-kehitykseen.
- **Luo GitHub-tili**. Koska löysit meidät täältä [GitHubista](https://github.com), sinulla saattaa jo olla tili, mutta jos ei, luo se ja tee kurssin versiosta oma haarasi (fork). (Voit myös antaa meille tähden 😊)
- **Tutustu Scikit-learniin**. Tutustu [Scikit-learniin](https://scikit-learn.org/stable/user_guide.html), joukkoon koneoppimiskirjastoja, joita viittaamme näissä oppitunneissa.

---
## Mikä on koneoppiminen?

Termi 'koneoppiminen' on yksi tämän päivän suosituimmista ja yleisesti käytetyistä termeistä. On hyvin mahdollista, että olet kuullut tämän termin ainakin kerran, jos sinulla on jonkinasteista teknologiaan liittyvää tuttavuutta, riippumatta siitä, millä alalla työskentelet. Koneoppimisen mekanismit ovat kuitenkin useimmille ihmisille mysteeri. Koneoppimisen aloittelijalle aihe voi toisinaan tuntua ylivoimaiselta. Siksi on tärkeää ymmärtää, mitä koneoppiminen todella on, ja oppia siitä askel askeleelta käytännön esimerkkien kautta.

---
## Hype-käyrä

![ml hype curve](../../../../translated_images/fi/hype.07183d711a17aafe.webp)

> Google Trends näyttää termin 'machine learning' viimeaikaisen 'hype-käyrän'

---
## Salaperäinen universumi

Elämme universumissa, joka on täynnä kiehtovia mysteerejä. Suuret tiedemiehet kuten Stephen Hawking, Albert Einstein ja monet muut ovat omistaneet elämänsä merkityksellisen tiedon etsintään, joka paljastaa ympäröivän maailman salaisuudet. Tämä on ihmisen oppimisen ehto: lapsi oppii uusia asioita ja paljastaa maailmansa rakenteita vuosi vuodelta kasvaessaan aikuiseksi.

---
## Lapsen aivot

Lapsen aivot ja aistit havaitsevat ympäristön tosiasiat ja oppivat vähitellen elämän piilotettuja kaavoja, jotka auttavat lasta muodostamaan loogisia sääntöjä oppimiensa kaavojen tunnistamiseksi. Ihmisaivojen oppimisprosessi tekee ihmisistä maailman kehittyneimmät elävät olennot. Jatkuva oppiminen piilotettujen kaavojen löytämisen kautta ja niiden kehittäminen mahdollistaa itsemme parantamisen koko elämämme ajan. Tämä oppimiskyky ja kehittyvä kapasiteetti liittyy käsitteeseen, jota kutsutaan [aivojen plastisuudeksi](https://www.simplypsychology.org/brain-plasticity.html). Pinnallisesti voimme löytää motivaatioita ihmisaivojen oppimisprosessin ja koneoppimisen käsitteiden välillä.

---
## Ihmisen aivot

[Ihminen aivot](https://www.livescience.com/29365-human-brain.html) havaitsevat asioita todellisesta maailmasta, käsittelevät havaittua tietoa, tekevät järkeviä päätöksiä ja suorittavat toimintoja tilanteiden perusteella. Tätä kutsutaan älykkääksi käyttäytymiseksi. Kun ohjelmoimme koneelle älykkään käyttäytymisprosessin kopion, sitä kutsutaan tekoälyksi (AI).

---
## Jotkin termit

Vaikka termejä voi sekoittaa, koneoppiminen (ML) on tärkeä osa tekoälyä. **Koneoppimisessa käytetään erikoistuneita algoritmeja merkityksellisen tiedon löytämiseksi ja piilotettujen kaavojen paljastamiseksi havaituista datoista järkevän päätöksenteon tukemiseksi**.

---
## AI, ML, syväoppiminen

![AI, ML, deep learning, data science](../../../../translated_images/fi/ai-ml-ds.537ea441b124ebf6.webp)

> Kaavio, joka näyttää AI:n, ML:n, syväoppimisen ja datatieteen väliset suhteet. Infografiikka Jen Looperin (@jenlooper) tekemänä, innoittajana [tämä kuva](https://softwareengineering.stackexchange.com/questions/366996/distinction-between-ai-ml-neural-networks-deep-learning-and-data-mining)

---
## Käsitteitä, jotka käsitellään

Tässä opetussuunnitelmassa käsittelemme vain koneoppimisen keskeiset käsitteet, jotka aloittelijan on tunnettava. Käymme läpi niin sanotun 'klassisen koneoppimisen', pääasiassa käyttäen Scikit-learnia, erinomaista kirjastoa, jota monet opiskelijat käyttävät perusteiden oppimiseen. Yleisemmän tekoälyn tai syväoppimisen käsitteiden ymmärtäminen vaatii vahvaa perusosaamista koneoppimisesta, jonka haluamme tarjota täällä.

---
## Tässä kurssissa opit:

- koneoppimisen keskeiset käsitteet
- koneoppimisen historian
- koneoppimisen ja oikeudenmukaisuuden
- regressio-ML-tekniikat
- luokittelu-ML-tekniikat
- klusterointimenetelmät ML:ssä
- luonnollisen kielen käsittelyn ML-tekniikat
- aikasarjaennusteiden ML-tekniikat
- vahvistusoppiminen
- ML:n todellisen maailman sovellukset

---
## Mitä emme käsittele

- syväoppiminen
- neuroverkot
- tekoäly

Parempaa oppimiskokemusta varten vältämme neuroverkkojen, 'syväoppimisen' - monikerroksisten mallien rakentamisen neuroverkkojen avulla - ja tekoälyn monimutkaisuuksia, joita käsittelemme eri opetussuunnitelmassa. Tarjoamme myös tulevaisuudessa datatieteeseen keskittyvän opintopolun tämän laajemman alan osalta.

---
## Miksi opiskella koneoppimista?

Koneoppiminen järjestelmien näkökulmasta määritellään automatisoitujen järjestelmien luomiseksi, jotka voivat oppia piilotettuja kaavoja datasta auttaakseen tekemään älykkäitä päätöksiä.

Tämä motivaatio on löyhästi innoittamana siitä, miten ihmisaivot oppivat tiettyjä asioita niiden havainnoimasta ulkomaailmasta.

✅ Mieti hetki, miksi yrityksen kannattaisi yrittää hyödyntää koneoppimista verrattuna kovakoodattuun sääntöpohjaiseen moottoriin.

---
## Miksi datan laatu on tärkeää

Korkealaatuinen data parantaa mallin suorituskykyä. Huono tai meluisa data voi johtaa epätarkkoihin ennusteisiin, jopa edistyneitä koneoppimisalgoritmeja käytettäessä.

---
## Koneoppimisen sovellukset

Koneoppimisen sovelluksia on nykyään lähes kaikkialla ja ne ovat yhtä yleisiä kuin data, joka virtaa yhteiskunnissamme älypuhelimista, liitetyistä laitteista ja muista järjestelmistä. Huippuluokan koneoppimisalgoritmien valtavan potentiaalin vuoksi tutkijat ovat tutkineet niiden kykyä ratkaista monimuotoisia ja moniammatillisia todellisen elämän ongelmia erinomaisin tuloksin.

---
## Käytännön esimerkkejä ML:stä

**Voit käyttää koneoppimista monin tavoin**:

- Ennustamaan sairauksien todennäköisyyttä potilaan sairaushistorian tai raporttien perusteella.
- Hyödyntämään säädataa sääilmiöiden ennustamiseen.
- Ymmärtämään tekstin tunnetehoa.
- Havaitsemaan valeuutisia propagandan leviämisen pysäyttämiseksi.

Rahoitus, taloustiede, maantiede, avaruustutkimus, biolääketieteellinen tekniikka, kognitiotiede ja jopa humanistiset alat ovat omaksuneet koneoppimisen ratkaisuna kovan datankäsittelyn ongelmiinsa.

---
## Yhteenveto

Koneoppiminen automatisoi kaavojen löytämisen prosessin löytämällä merkityksellisiä havaintoja todellisesta tai tuotetusta datasta. Se on osoittautunut erittäin arvokkaaksi liiketoiminnassa, terveydessä ja rahoituksessa, muiden alojen ohella.

Lähitulevaisuudessa koneoppimisen perusteiden ymmärtäminen on välttämätöntä ihmisille kaikilla aloilla laajan käyttöönoton vuoksi.

---
# 🚀 Haaste

Piirrä paperille tai käytä verkkosovellusta kuten [Excalidrawia](https://excalidraw.com/) kuvaamaan ymmärrystäsi AI:n, ML:n, syväoppimisen ja datatieteen eroista. Lisää ideoita ongelmista, joita kukin näistä tekniikoista osaa ratkaista hyvin.

# [Jälkikokeilu](https://ff-quizzes.netlify.app/en/ml/)

---
# Kertaaminen ja itsenäinen opiskelu

Jos haluat oppia lisää siitä, miten voit työskennellä koneoppimisalgoritmien kanssa pilvessä, seuraa tätä [Oppimispolkua](https://docs.microsoft.com/learn/paths/create-no-code-predictive-models-azure-machine-learning/?WT.mc_id=academic-77952-leestott).

Osallistu [Oppimispolkuun](https://docs.microsoft.com/learn/modules/introduction-to-machine-learning/?WT.mc_id=academic-77952-leestott), jossa käsitellään koneoppimisen perusteet.

---
# Tehtävä

[Ota ensin käyttöön ja käynnisty](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Vastuuvapauslauseke**:
Tämä asiakirja on käännetty käyttämällä tekoälypohjaista käännöspalvelua [Co-op Translator](https://github.com/Azure/co-op-translator). Vaikka pyrimme tarkkuuteen, otathan huomioon, että automaattiset käännökset saattavat sisältää virheitä tai epätarkkuuksia. Alkuperäinen asiakirja sen alkuperäiskielellä on virallinen lähde. Tärkeissä asioissa suositellaan ammattimaista ihmiskäännöstä. Emme ole vastuussa tämän käännöksen käytöstä aiheutuvista väärinymmärryksistä tai tulkinnoista.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->