# Introducere în învățarea automată

## [Chestionar pre-lectură](https://ff-quizzes.netlify.app/en/ml/)

---

[![ML pentru începători - Introducere în Învățarea Automată pentru Începători](https://img.youtube.com/vi/6mSx_KJxcHI/0.jpg)](https://youtu.be/6mSx_KJxcHI "ML pentru începători - Introducere în Învățarea Automată pentru Începători")

> 🎥 Faceți clic pe imaginea de mai sus pentru un scurt videoclip care parcurge această lecție.

Bine ați venit la acest curs despre învățarea automată clasică pentru începători! Fie că sunteți complet nou în acest domeniu, fie că sunteți un practician experimentat în ML care dorește să aprofundeze o arie, suntem bucuroși să vă avem alături! Dorim să creăm un punct de pornire prietenos pentru studiul vostru în ML și am fi bucuroși să evaluăm, să răspundem și să integrăm [feedback-ul](https://github.com/microsoft/ML-For-Beginners/discussions) vostru.

[![Introducere în ML](https://img.youtube.com/vi/h0e2HAPTGF4/0.jpg)](https://youtu.be/h0e2HAPTGF4 "Introducere în ML")

> 🎥 Faceți clic pe imaginea de mai sus pentru un videoclip: John Guttag de la MIT introduce învățarea automată

---
## Începuturi în învățarea automată

Înainte de a începe cu acest curriculum, trebuie să aveți calculatorul configurat și pregătit să ruleze notebook-uri local.

- **Configurați-vă mașina cu aceste videoclipuri**. Folosiți linkurile următoare pentru a învăța [cum să instalați Python](https://youtu.be/CXZYvNRIAKM) pe sistemul vostru și să [configurați un editor de text](https://youtu.be/EU8eayHWoZg) pentru dezvoltare.
- **Învățați Python**. De asemenea, este recomandat să aveți o înțelegere de bază a [Python](https://docs.microsoft.com/learn/paths/python-language/?WT.mc_id=academic-77952-leestott), un limbaj de programare util pentru oamenii de știință în domeniul datelor pe care îl folosim în acest curs.
- **Învățați Node.js și JavaScript**. Folosim și JavaScript de câteva ori în acest curs când construim aplicații web, așa că va trebui să aveți instalate [node](https://nodejs.org) și [npm](https://www.npmjs.com/), precum și [Visual Studio Code](https://code.visualstudio.com/) disponibil atât pentru dezvoltarea Python, cât și JavaScript.
- **Creați un cont GitHub**. Deoarece ne-ați găsit aici pe [GitHub](https://github.com), s-ar putea să aveți deja un cont, dar dacă nu, creați unul și apoi faceți fork la acest curriculum pentru a-l folosi pe cont propriu. (Simțiți-vă liber să ne acordați și o stea 😊)
- **Explorați Scikit-learn**. Familiarizați-vă cu [Scikit-learn](https://scikit-learn.org/stable/user_guide.html), un set de biblioteci ML pe care le referim în aceste lecții.

---
## Ce este învățarea automată?

Termenul „învățare automată” este unul dintre cei mai populari și des folosiți termeni din prezent. Există o probabilitate nontrivială să fi auzit acest termen cel puțin o dată dacă aveți oarecare familiaritate cu tehnologia, indiferent în ce domeniu lucrați. Mecanismele învățării automate, însă, sunt un mister pentru majoritatea oamenilor. Pentru un începător în învățarea automată, subiectul poate părea uneori copleșitor. Prin urmare, este important să înțelegem ce este de fapt învățarea automată și să învățăm despre ea pas cu pas, prin exemple practice.

---
## Curba hype-ului

![ml hype curve](../../../../translated_images/ro/hype.07183d711a17aafe.webp)

> Google Trends arată „curba hype-ului” recentă a termenului „învățare automată”

---
## Un univers misterios

Trăim într-un univers plin de mistere fascinante. Mari oameni de știință precum Stephen Hawking, Albert Einstein și mulți alții și-au dedicat viețile căutării de informații semnificative care descoperă misterele lumii înconjurătoare. Aceasta este condiția umană a învățării: un copil învață lucruri noi și descoperă structura lumii lor an după an pe măsură ce crește spre maturitate.

---
## Creierul copilului

Creierul și simțurile unui copil percep faptele din jur și învață treptat modelele ascunse ale vieții care ajută copilul să elaboreze reguli logice pentru a identifica modelele învățate. Procesul de învățare al creierului uman face din oameni cele mai sofisticate ființe din această lume. Învățarea continuă prin descoperirea modelelor ascunse și apoi inovarea pe baza acelor modele ne permite să ne îmbunătățim pe noi înșine pe tot parcursul vieții. Această capacitate de învățare și abilitate în evoluție este legată de un concept numit [plasticitatea creierului](https://www.simplypsychology.org/brain-plasticity.html). Dintr-o perspectivă superficială, putem trage unele asemănări motivaționale între procesul de învățare al creierului uman și conceptele învățării automate.

---
## Creierul uman

[Creierul uman](https://www.livescience.com/29365-human-brain.html) percepe lucrurile din lumea reală, procesează informațiile percepute, ia decizii raționale și efectuează anumite acțiuni în funcție de circumstanțe. Acesta este ceea ce numim comportament inteligent. Când programăm o facsimilare a procesului de comportament inteligent pe o mașină, aceasta se numește inteligență artificială (IA).

---
## Câteva terminologii

Deși termenii pot fi confuzi, învățarea automată (ML) este un subset important al inteligenței artificiale. **ML se ocupă cu utilizarea algoritmilor specializați pentru a descoperi informații semnificative și a găsi modele ascunse din datele percepute pentru a susține procesul de luare a deciziilor raționale**.

---
## AI, ML, Deep Learning

![AI, ML, deep learning, data science](../../../../translated_images/ro/ai-ml-ds.537ea441b124ebf6.webp)

> O diagramă ce arată relațiile dintre AI, ML, deep learning și știința datelor. Infografic de [Jen Looper](https://twitter.com/jenlooper) inspirat de [această grafică](https://softwareengineering.stackexchange.com/questions/366996/distinction-between-ai-ml-neural-networks-deep-learning-and-data-mining)

---
## Concepte de acoperit

În acest curriculum, vom acoperi doar conceptele de bază ale învățării automate pe care un începător trebuie să le cunoască. Vom acoperi ceea ce numim „învățarea automată clasică” folosind în principal Scikit-learn, o bibliotecă excelentă pe care mulți studenți o folosesc pentru a învăța noțiunile de bază. Pentru a înțelege concepte mai largi de inteligență artificială sau deep learning, o cunoaștere fundamentală puternică a învățării automate este indispensabilă, așa că dorim să o oferim aici.

---
## În acest curs veți învăța:

- conceptele de bază ale învățării automate
- istoria ML
- ML și echitatea
- tehnici de regresie ML
- tehnici de clasificare ML
- tehnici de clusterizare ML
- tehnici de procesare a limbajului natural ML
- tehnici de predicție a seriilor temporale ML
- învățarea prin întărire
- aplicații reale pentru ML

---
## Ce NU vom acoperi

- deep learning
- rețele neuronale
- AI

Pentru o experiență mai bună de învățare, vom evita complexitățile rețelelor neuronale, „deep learning” - modelarea în mai multe straturi folosind rețele neuronale - și AI, pe care le vom discuta într-un alt curriculum. De asemenea, vom oferi un curriculum viitor de știința datelor pentru a ne concentra pe acest aspect al acestui domeniu mai larg.

---
## De ce să studiezi învățarea automată?

Învățarea automată, din perspectiva sistemelor, este definită ca crearea de sisteme automate care pot învăța modele ascunse din date pentru a ajuta la luarea deciziilor inteligente.

Această motivație este inspirată vag de modul în care creierul uman învață anumite lucruri pe baza datelor pe care le percepe din lumea exterioară.

✅ Gândiți-vă pentru un moment de ce o afacere ar dori să încerce să utilizeze strategii de învățare automată în loc să creeze un motor bazat pe reguli hardcodate.

---
## De ce contează calitatea datelor

Datele de înaltă calitate îmbunătățesc performanța modelului. Datele proaste sau zgomotoase pot duce la predicții inexacte, chiar și folosind algoritmi avansați de învățare automată.

---
## Aplicații ale învățării automate

Aplicațiile învățării automate sunt acum aproape peste tot și sunt la fel de omniprezente ca datele care circulă prin societățile noastre, generate de telefoanele noastre inteligente, dispozitivele conectate și alte sisteme. Având în vedere potențialul imens al algoritmilor de învățare automată de ultimă generație, cercetătorii și-au explorat capacitatea de a rezolva probleme multidimensionale și multidisciplinare reale cu rezultate foarte pozitive.

---
## Exemple de ML aplicat

**Puteți folosi învățarea automată în multe moduri**:

- Pentru a prezice probabilitatea unei boli din istoricul medical sau rapoartele unui pacient.
- Pentru a folosi date meteorologice pentru a prezice evenimente meteo.
- Pentru a înțelege sentimentele unui text.
- Pentru a detecta știri false pentru a opri răspândirea propagandei.

Finanțe, economie, știința pământului, explorarea spațială, ingineria biomedicală, știința cognitivă și chiar domenii în umanioare au adaptat învățarea automată pentru a rezolva probleme dificile, intensive în procesarea datelor, din domeniile lor.

---
## Concluzie

Învățarea automată automatizează procesul de descoperire a modelelor prin găsirea de informații semnificative din datele reale sau generate. S-a dovedit a fi extrem de valoroasă în afaceri, sănătate și aplicații financiare, printre altele.

În viitorul apropiat, înțelegerea elementelor de bază ale învățării automate va deveni o necesitate pentru oamenii din orice domeniu, datorită adoptării sale pe scară largă.

---
# 🚀 Provocare

Desenați, pe hârtie sau folosind o aplicație online precum [Excalidraw](https://excalidraw.com/), înțelegerea dvs. despre diferențele dintre AI, ML, deep learning și știința datelor. Adăugați câteva idei despre problemele pe care fiecare dintre aceste tehnici le rezolvă bine.

# [Chestionar post-lectură](https://ff-quizzes.netlify.app/en/ml/)

---
# Recapitulare și Auto-studiu

Pentru a afla mai multe despre cum puteți lucra cu algoritmi ML în cloud, urmați acest [Learning Path](https://docs.microsoft.com/learn/paths/create-no-code-predictive-models-azure-machine-learning/?WT.mc_id=academic-77952-leestott).

Parcurgeți un [Learning Path](https://docs.microsoft.com/learn/modules/introduction-to-machine-learning/?WT.mc_id=academic-77952-leestott) despre elementele de bază ale ML.

---
# Tema

[Porniți și rulați](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Declinare a responsabilității**:
Acest document a fost tradus folosind serviciul de traducere AI [Co-op Translator](https://github.com/Azure/co-op-translator). În timp ce ne străduim pentru acuratețe, vă rugăm să rețineți că traducerile automate pot conține erori sau inexactități. Documentul original în limba sa nativă trebuie considerat sursa autorizată. Pentru informații critice, se recomandă traducerea profesională realizată de un om. Nu ne asumăm responsabilitatea pentru eventualele neînțelegeri sau interpretări greșite care decurg din utilizarea acestei traduceri.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->