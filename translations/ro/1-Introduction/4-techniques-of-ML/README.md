# Tehnici de Învățare Automată

Procesul de construire, utilizare și întreținere a modelelor de învățare automată și a datelor pe care le folosesc este un proces foarte diferit față de multe alte fluxuri de lucru de dezvoltare. În această lecție, vom demistifica procesul și vom contura principalele tehnici pe care trebuie să le cunoști. Vei:

- Înțelege procesele fundamentale ale învățării automate la un nivel înalt.
- Explora concepte de bază precum „modele”, „predicții” și „date de antrenament”.

## [Test pre-lectură](https://ff-quizzes.netlify.app/en/ml/)

[![ML for beginners - Techniques of Machine Learning](https://img.youtube.com/vi/4NGM0U2ZSHU/0.jpg)](https://youtu.be/4NGM0U2ZSHU "ML for beginners - Techniques of Machine Learning")

> 🎥 Click pe imaginea de mai sus pentru un scurt video care parcurge această lecție.

## Introducere

La un nivel înalt, arta creării proceselor de învățare automată (ML) este alcătuită din mai mulți pași:

1. **Decide asupra întrebării**. Majoritatea proceselor ML încep prin a pune o întrebare care nu poate fi răspunsă printr-un program condițional simplu sau printr-un motor bazat pe reguli. Aceste întrebări se referă adesea la predicții bazate pe o colecție de date.
2. **Colectează și pregătește datele**. Pentru a putea răspunde la întrebarea ta, ai nevoie de date. Calitatea și, uneori, cantitatea datelor tale vor determina cât de bine poți răspunde la întrebarea inițială. Vizualizarea datelor este o componentă importantă a acestei faze. Această fază include și împărțirea datelor în grupuri de antrenament și test pentru a construi un model.
3. **Alege o metodă de antrenament**. În funcție de întrebarea ta și de natura datelor, trebuie să alegi cum dorești să antrenezi un model pentru a reflecta cel mai bine datele și pentru a face predicții exacte pe baza acestora. Aceasta este partea procesului tău ML care necesită expertiză specifică și, adesea, o cantitate considerabilă de experimentare.
4. **Antrenează modelul**. Folosind datele de antrenament, vei utiliza diferiți algoritmi pentru a antrena un model să recunoască tipare în date. Modelul poate folosi greutăți interne ajustabile pentru a privilegia anumite părți ale datelor în detrimentul altora pentru a construi un model mai bun.
5. **Evaluează modelul**. Folosești date noi, nevăzute anterior (datele de testare) din setul tău colectat pentru a vedea cum performează modelul.
6. **Ajustarea parametrilor**. Pe baza performanței modelului, poți reface procesul folosind parametri diferiți, sau variabile, care controlează comportamentul algoritmilor folosiți pentru antrenarea modelului.
7. **Fă predicții**. Folosește noi intrări pentru a testa acuratețea modelului tău.

## Ce întrebări să pui

Calculatoarele sunt deosebit de pricepute la descoperirea tiparelor ascunse în date. Această utilitate este foarte folositoare pentru cercetătorii care au întrebări legate de un anumit domeniu și care nu pot fi ușor răspunse prin crearea unui motor bazat pe reguli condiționale. De exemplu, pentru o sarcină actuarială, un data scientist ar putea construi reguli manuale privind mortalitatea fumătorilor față de nefumători.

Totuși, când multe alte variabile sunt incluse în ecuație, un model ML poate fi mai eficient pentru a prezice ratele viitoare de mortalitate bazate pe istoricul medical anterior. Un exemplu mai optimist ar fi realizarea predicțiilor meteo pentru luna aprilie într-o anumită locație în baza datelor care includ latitudinea, longitudinea, schimbările climatice, proximitatea față de ocean, modelele fluxului jet și altele.

✅ Acest [set de slide-uri](https://www2.cisl.ucar.edu/sites/default/files/2021-10/0900%20June%2024%20Haupt_0.pdf) despre modele meteorologice oferă o perspectivă istorică asupra folosirii ML în analiza meteo.

## Sarcini pre-construire

Înainte de a începe să construiești modelul, există mai multe sarcini pe care trebuie să le finalizezi. Pentru a testa întrebarea și a formula o ipoteză bazată pe predicțiile unui model, trebuie să identifici și să configurezi mai multe elemente.

### Date

Pentru a răspunde întrebării tale cu orice fel de certitudine, ai nevoie de o cantitate bună de date potrivite. Sunt două lucruri de făcut în acest moment:

- **Colectează date**. Ținând cont de lecția anterioară despre corectitudinea în analiza datelor, colectează datele cu grijă. Fii conștient de sursele acestor date, de orice părtinire inerentă și documentează originea acestora.
- **Pregătește datele**. Există mai mulți pași în procesul de pregătire a datelor. Este posibil să fie nevoie să le unești și să le normalizezi dacă provin din surse diverse. Poți îmbunătăți calitatea și cantitatea datelor prin diferite metode, cum ar fi convertirea șirurilor de caractere în numere (așa cum facem în [Clustering](../../5-Clustering/1-Visualize/README.md)). Poți genera date noi, bazate pe cele originale (așa cum facem în [Classification](../../4-Classification/1-Introduction/README.md)). Poți curăța și edita datele (așa cum vom face înainte de lecția [Web App](../../3-Web-App/README.md)). În final, poate fi necesar să le randomizezi și să le amesteci, în funcție de tehnicile de antrenament folosite.

✅ După colectarea și procesarea datelor, ia un moment să vezi dacă forma acestora îți va permite să adresezi întrebarea dorită. Este posibil ca datele să nu performeze bine în sarcina ta, așa cum descoperim în lecțiile de [Clustering](../../5-Clustering/1-Visualize/README.md)!

### Caracteristici și țintă

O [caracteristică](https://www.datasciencecentral.com/profiles/blogs/an-introduction-to-variable-and-feature-selection) este o proprietate măsurabilă a datelor tale. În multe seturi de date, este exprimată prin intermediul unui titlu de coloană precum „data”, „dimensiune” sau „culoare”. Variabila ta caracteristică, de obicei reprezentată ca `X` în cod, reprezintă variabila de intrare ce va fi folosită pentru antrenarea modelului.

O țintă este ceva ce încerci să prezici. Ținta, de obicei reprezentată ca `y` în cod, reprezintă răspunsul la întrebarea pe care încerci să o adresezi datelor tale: în decembrie, ce culoare vor avea dovlecii cei mai ieftini? în San Francisco, care cartiere vor avea cel mai bun preț imobiliar? Ocazional ținta este denumită și atribut etichetă.

### Selectarea variabilei caracteristice

🎓 **Selecția și extracția caracteristicilor** Cum știi ce variabilă să alegi când construiești un model? Probabil vei trece printr-un proces de selecție sau extracție a caracteristicilor pentru a alege variabilele potrivite pentru cel mai performant model. Totuși, acestea nu sunt același lucru: „Extracția caracteristicilor creează caracteristici noi bazate pe funcții ale caracteristicilor originale, în timp ce selecția caracteristicilor returnează un subset al caracteristicilor.” ([sursă](https://wikipedia.org/wiki/Feature_selection))

### Vizualizează-ți datele

O componentă importantă în trusa unui data scientist este puterea de a vizualiza datele folosind mai multe biblioteci excelente precum Seaborn sau MatPlotLib. Reprezentarea datelor vizual te poate ajuta să descoperi corelații ascunse pe care le poți folosi. Vizualizările ar putea de asemenea să te ajute să descoperi părtinire sau date dezechilibrate (așa cum descoperim în [Classification](../../4-Classification/2-Classifiers-1/README.md)).

### Împarte setul tău de date

Înainte de antrenare, trebuie să împarți setul de date în două sau mai multe părți de mărimi inegale care să reprezinte în continuare bine datele.

- **Antrenament**. Această parte a setului de date este folosită pentru a antrena modelul. Acest set constituie majoritatea setului de date original.
- **Testare**. Un set de date de test este un grup independent de date, adesea extras din datele inițiale, pe care îl folosești pentru a confirma performanța modelului construit.
- **Validare**. Un set de validare este un grup mai mic independent de exemple folosit pentru a ajusta hiperparametrii sau arhitectura modelului pentru a-l îmbunătăți. În funcție de mărimea datelor tale și întrebarea pe care o adresezi, s-ar putea să nu fie necesar să construiești acest al treilea set (așa cum notăm în [Time Series Forecasting](../../7-TimeSeries/1-Introduction/README.md)).

## Construirea unui model

Folosindu-te de datele de antrenament, scopul tău este să construiești un model, sau o reprezentare statistică a datelor, folosind diferiți algoritmi pentru a-l **antrena**. Antrenarea unui model îl expune la date și îi permite să facă presupuneri despre tiparele percepute pe care le descoperă, validează și acceptă sau respinge.

### Alege o metodă de antrenament

În funcție de întrebarea ta și de natura datelor, vei alege o metodă de antrenament. Parcurgând [documentația Scikit-learn](https://scikit-learn.org/stable/user_guide.html) — pe care o folosim în acest curs — poți explora multe metode de a antrena un model. În funcție de experiența ta, este posibil să trebuiască să încerci mai multe metode diferite pentru a construi cel mai bun model. Este probabil să treci printr-un proces în care data scientist-ii evaluează performanța unui model prin alimentarea sa cu date noi, verificând acuratețea, părtinirea și alte probleme care pot degrada calitatea, și aleg metoda de antrenament cea mai potrivită pentru sarcina respectivă.

### Antrenează un model

Înarmat cu datele tale de antrenament, ești gata să-l „potrivești” pentru a crea modelul. Vei observa că în multe biblioteci ML găsești codul „model.fit” – este în acest moment când trimiți variabila ta caracteristică ca un șir de valori (de obicei „X”) și o variabilă țintă (de obicei „y”).

### Evaluează modelul

Odată ce procesul de antrenament s-a încheiat (poate dura multe iterații, sau „epoci”, pentru a antrena un model mare), vei putea evalua calitatea modelului folosind datele de test pentru a-i măsura performanța. Aceste date sunt un subset al datelor originale pe care modelul nu le-a analizat anterior. Poți afișa un tabel de metrici despre calitatea modelului tău.

🎓 **Potrivirea modelului**

În contextul învățării automate, potrivirea modelului se referă la acuratețea funcției interne a modelului în timp ce încearcă să analizeze date cu care nu este familiar.

🎓 **Suprapotrivirea** și **subpotrivirea** sunt probleme comune care degradează calitatea modelului, deoarece modelul se potrivește fie insuficient, fie prea mult. Acest lucru determină modelul să facă predicții prea rigide sau prea slabe față de datele de antrenament. Un model suprapotrivit prezice datele de antrenament prea bine pentru că a învățat foarte bine detaliile și zgomotul din date. Un model subpotrivit nu este precis, neputând analiza corect nici datele de antrenament și nici datele nevăzute anterior.

![overfitting model](../../../../translated_images/ro/overfitting.1c132d92bfd93cb6.webp)
> Infografic de [Jen Looper](https://twitter.com/jenlooper)

## Ajustarea parametrilor

Odată ce antrenamentul inițial s-a încheiat, observă calitatea modelului și ia în considerare îmbunătățirea sa prin ajustarea „hiperparametrilor”. Citește mai multe despre proces [în documentație](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters?WT.mc_id=academic-77952-leestott).

## Predicție

Acesta este momentul în care poți folosi date complet noi pentru a testa acuratețea modelului. Într-un context ML „aplicat”, unde construiești active web pentru a folosi modelul în producție, acest proces poate implica colectarea inputului utilizatorului (de exemplu, apăsarea unui buton) pentru a seta o variabilă și a o trimite modelului pentru inferență sau evaluare.

În aceste lecții, vei descoperi cum să folosești acești pași pentru a pregăti, construi, testa, evalua și prezice — toate gesturile unui data scientist și mai mult, pe măsură ce avansezi în călătoria ta de a deveni un inginer ML „full stack”.

---

## 🚀Provocare

Desenează o diagramă de flux care reflectă pașii unui practician ML. Unde te vezi acum în proces? Unde crezi că vei întâmpina dificultăți? Ce ți se pare ușor?

## [Test post-lectură](https://ff-quizzes.netlify.app/en/ml/)

## Recapitulare & Auto-studiu

Caută online interviuri cu data scientist-i care discută despre munca lor zilnică. Iată [unul](https://www.youtube.com/watch?v=Z3IjgbbCEfs).

## Temă

[Intervievează un data scientist](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Declinare de responsabilitate**:  
Acest document a fost tradus folosind serviciul de traducere AI [Co-op Translator](https://github.com/Azure/co-op-translator). Deși depunem eforturi pentru acuratețe, vă rugăm să rețineți că traducerile automate pot conține erori sau inexactități. Documentul original în limba sa nativă trebuie considerat sursa autorizată. Pentru informații critice, se recomandă traducerea profesională realizată de un traducător uman. Nu ne asumăm răspunderea pentru eventualele neînțelegeri sau interpretări greșite rezultate din utilizarea acestei traduceri.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->