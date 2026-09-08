# Cuisine classifiers 2

Tässä toisessa luokittelutunnissa tutustut lisää tapoihin luokitella numeerista dataa. Opit myös, mitä seurauksia yhden luokittimen valinnalla on toisen sijaan.

## [Esiluentokoe](https://ff-quizzes.netlify.app/en/ml/)

### Ennakkoedellytys

Oletetaan, että olet suorittanut edelliset oppitunnit ja sinulla on siivottu aineisto kansiossasi nimeltä _cleaned_cuisines.csv_ tämän neljän oppitunnin kansion juurikansiossa.

### Valmistelut

Olemme ladanneet _notebook.ipynb_-tiedostosi siivotun aineiston kanssa ja jakaneet sen X- ja y-datafreimeiksi, valmiina mallin rakentamisprosessiin.

## Luokittelukartta

Aiemmin opit Microsoftin kikan avulla eri vaihtoehdoista datan luokittelussa. Scikit-learn tarjoaa samanlaisen mutta tarkemman kikan, joka voi auttaa kaventamaan arvioijiasi (toinen nimi luokittimille):

![ML Map from Scikit-learn](../../../../translated_images/fi/map.e963a6a51349425a.webp)
> Vinkki: [vieraile tällä kartalla verkossa](https://scikit-learn.org/stable/tutorial/machine_learning_map/) ja klikkaa polkua lukeaksesi dokumentaatiota.

### Suunnitelma

Tämä kartta on hyvin hyödyllinen, kun ymmärrät datasi selkeästi, sillä voit ’kävellä’ sen polkuja päätökseen:

- Meillä on yli 50 näytettä
- Haluamme ennustaa kategorian
- Meillä on merkittyä dataa
- Näytteitä on alle 100 000
- ✨ Voimme valita Linear SVC:n
- Jos se ei toimi, koska meillä on numeerista dataa
    - Voimme kokeilla ✨ KNeighbors-luokitinta
      - Jos sekään ei toimi, kokeile ✨ SVC:tä ja ✨ yhdistelmäluokittimia

Tämä on hyvin hyödyllinen polku seurata.

## Harjoitus - jaa data

Seuraamalla tätä polkua, aloitamme tuomalla käyttöön tarvittavat kirjastot.

1. Tuo tarvittavat kirjastot:

    ```python
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    import numpy as np
    ```

1. Jaa harjoitus- ja testidata:

    ```python
    X_train, X_test, y_train, y_test = train_test_split(cuisines_features_df, cuisines_label_df, test_size=0.3)
    ```

## Linear SVC luokitin

Support-Vector clustering (SVC) kuuluu Support-Vector machines -menetelmien perheeseen (lue lisää niistä alla). Tässä menetelmässä voit valita ’ytimen’, joka päättää, miten tunnisteet ryhmitellään. ’C’-parametri viittaa ’regulointiin’, joka säätelee parametrien vaikutusta. Ydin voi olla yksi [useista](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC); tässä asetamme sen ’lineaariseksi’ varmistaaksemme lineaarisen SVC:n käytön. Todennäköisyysasetuksena on oletuksena ’false’; tässä asetamme sen ’true’ saadaksemme todennäköisyyksiä. Asetamme satunnaistilan ’0’:ksi, jotta data sekoittuu todennäköisyyksien saadessa.

### Harjoitus - käytä lineaarista SVC:tä

Aloita luomalla taulukko luokittimista. Lisäät taulukkoon vähitellen testatessasi.

1. Aloita Linear SVC:llä:

    ```python
    C = 10
    # Luo erilaisia luokittelijoita.
    classifiers = {
        'Linear SVC': SVC(kernel='linear', C=C, probability=True,random_state=0)
    }
    ```

2. Kouluta mallisi käyttäen Linear SVC:tä ja tulosta raportti:

    ```python
    n_classifiers = len(classifiers)
    
    for index, (name, classifier) in enumerate(classifiers.items()):
        classifier.fit(X_train, np.ravel(y_train))
    
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy (train) for %s: %0.1f%% " % (name, accuracy * 100))
        print(classification_report(y_test,y_pred))
    ```

    Tulokset ovat varsin hyvät:

    ```output
    Accuracy (train) for Linear SVC: 78.6% 
                  precision    recall  f1-score   support
    
         chinese       0.71      0.67      0.69       242
          indian       0.88      0.86      0.87       234
        japanese       0.79      0.74      0.76       254
          korean       0.85      0.81      0.83       242
            thai       0.71      0.86      0.78       227
    
        accuracy                           0.79      1199
       macro avg       0.79      0.79      0.79      1199
    weighted avg       0.79      0.79      0.79      1199
    ```

## K-Neighbors luokitin

K-Neighbors kuuluu ”naapuri”-menetelmien koneoppimisperheeseen, jota voidaan käyttää valvottuun ja valvomattomaan oppimiseen. Tässä menetelmässä luodaan ennalta määritelty määrä pisteitä, joihin data kerätään niin, että datan yleistäminen ja tunnisteiden ennustaminen onnistuu.

### Harjoitus - käytä K-Neighbors luokitinta

Edellinen luokitin toimi hyvin aineistolla, mutta ehkä voimme saada paremman tarkkuuden. Kokeile K-Neighbors-luokitinta.

1. Lisää rivi luokittimien taulukkoon (lisää pilkku Linear SVC:n jälkeen):

    ```python
    'KNN classifier': KNeighborsClassifier(C),
    ```

    Tulokset ovat hieman heikommat:

    ```output
    Accuracy (train) for KNN classifier: 73.8% 
                  precision    recall  f1-score   support
    
         chinese       0.64      0.67      0.66       242
          indian       0.86      0.78      0.82       234
        japanese       0.66      0.83      0.74       254
          korean       0.94      0.58      0.72       242
            thai       0.71      0.82      0.76       227
    
        accuracy                           0.74      1199
       macro avg       0.76      0.74      0.74      1199
    weighted avg       0.76      0.74      0.74      1199
    ```

    ✅ Lue lisää [K-Neighborsista](https://scikit-learn.org/stable/modules/neighbors.html#neighbors)

## Support Vector Classifier

Support-Vector -luokittimet kuuluvat [Support-Vector Machine](https://wikipedia.org/wiki/Support-vector_machine) -menetelmien koneoppimisperheeseen, joita käytetään luokittelu- ja regressiotehtävissä. SVM:t ”karttavat harjoitusesimerkit pisteiksi avaruuteen” maksimoidakseen kahden kategorian eron. Seuraavat datat kartoitetaan tähän avaruuteen, jotta niiden kategoria voidaan ennustaa.

### Harjoitus - käytä Support Vector Classifieria

Yritetään hieman parempaa tarkkuutta Support Vector Classifierilla.

1. Lisää pilkku K-Neighborsin jälkeen ja lisää sitten tämä rivi:

    ```python
    'SVC': SVC(),
    ```

    Tulokset ovat varsin hyvät!

    ```output
    Accuracy (train) for SVC: 83.2% 
                  precision    recall  f1-score   support
    
         chinese       0.79      0.74      0.76       242
          indian       0.88      0.90      0.89       234
        japanese       0.87      0.81      0.84       254
          korean       0.91      0.82      0.86       242
            thai       0.74      0.90      0.81       227
    
        accuracy                           0.83      1199
       macro avg       0.84      0.83      0.83      1199
    weighted avg       0.84      0.83      0.83      1199
    ```

    ✅ Lue lisää [Support-Vektoreista](https://scikit-learn.org/stable/modules/svm.html#svm)

## Yhdistelmäluokittimet

Seurataan polkua aivan loppuun asti, vaikka edellinen testi oli varsin hyvä. Kokeillaan ’Yhdistelmäluokittimia’, erityisesti Random Forestia ja AdaBoostia:

```python
  'RFST': RandomForestClassifier(n_estimators=100),
  'ADA': AdaBoostClassifier(n_estimators=100)
```

Tulokset ovat erittäin hyvät, erityisesti Random Forestilla:

```output
Accuracy (train) for RFST: 84.5% 
              precision    recall  f1-score   support

     chinese       0.80      0.77      0.78       242
      indian       0.89      0.92      0.90       234
    japanese       0.86      0.84      0.85       254
      korean       0.88      0.83      0.85       242
        thai       0.80      0.87      0.83       227

    accuracy                           0.84      1199
   macro avg       0.85      0.85      0.84      1199
weighted avg       0.85      0.84      0.84      1199

Accuracy (train) for ADA: 72.4% 
              precision    recall  f1-score   support

     chinese       0.64      0.49      0.56       242
      indian       0.91      0.83      0.87       234
    japanese       0.68      0.69      0.69       254
      korean       0.73      0.79      0.76       242
        thai       0.67      0.83      0.74       227

    accuracy                           0.72      1199
   macro avg       0.73      0.73      0.72      1199
weighted avg       0.73      0.72      0.72      1199
```

✅ Lue lisää [Yhdistelmäluokittimista](https://scikit-learn.org/stable/modules/ensemble.html)

Tämä koneoppimismenetelmä ”yhdistää usean perusarvioijan ennusteet” parantaakseen mallin laatua. Esimerkissämme käytimme satunnaisia puita ja AdaBoostia. 

- [Random Forest](https://scikit-learn.org/stable/modules/ensemble.html#forest), keskiarvomenetelmä, rakentaa ’metsän’ ’päätöspuita’, joita satunnaistetaan ylisovittamisen välttämiseksi. n_estimators-parametri on puiden lukumäärä.

- [AdaBoost](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html) sovittaa luokittimen aineistoon ja sovittaa sitten kopioita tästä luokittimesta samaan aineistoon. Se keskittyy virheellisesti luokiteltujen kohteiden painoihin ja säätää seuraavan luokittimen sovitusta korjatakseen ne.

---

## 🚀Haaste

Jokaisella näistä menetelmistä on suuri määrä parametreja, joita voit säätää. Tutki kunkin oletusparametreja ja mieti, mitä niiden muuttaminen merkitsisi mallin laadulle.

## [Jälkiluennon koe](https://ff-quizzes.netlify.app/en/ml/)

## Kertaus & Itsenäinen opiskelu

Näissä oppitunneissa on paljon ammattisanastoa, joten ota hetki ja kertaile [tätä listaa](https://docs.microsoft.com/dotnet/machine-learning/resources/glossary?WT.mc_id=academic-77952-leestott) hyödyllisistä termeistä!

## Tehtävä 

[Parametrileikki](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Vastuuvapauslauseke**:
Tämä asiakirja on käännetty käyttämällä tekoälypohjaista käännöspalvelua [Co-op Translator](https://github.com/Azure/co-op-translator). Vaikka pyrimme tarkkuuteen, automaattiset käännökset saattavat sisältää virheitä tai epätarkkuuksia. Alkuperäinen asiakirja sen alkuperäiskielellä tulee pitää virallisena lähteenä. Tärkeissä asioissa suositellaan ammattimaista ihmiskäännöstä. Emme ole vastuussa tämän käännöksen käytöstä johtuvista väärinymmärryksistä tai virhetulkinnoista.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->