# Vainisha za vyakula 2

Katika somo hili la pili la uainishaji, utachunguza njia zaidi za kuainisha data ya nambari. Pia utajifunza kuhusu athari za kuchagua vainisha mmoja badala ya mwingine.

## [Jaribio kabla ya mihadhara](https://ff-quizzes.netlify.app/en/ml/)

### Sharti

Tunadhani umefanya masomo ya awali na una seti safi ya data katika folda yako ya `data` inayoitwa _cleaned_cuisines.csv_ kwa mzizi wa folda hii ya masomo 4.

### Maandalizi

Tumepakia faili lako la _notebook.ipynb_ lenye seti safi ya data na tumeigawanya kuwa dataframes za X na y, tayari kwa mchakato wa ujenzi wa modeli.

## Ramani ya uainishaji

Hapo awali, ulijifunza kuhusu chaguzi mbalimbali ulizo nazo wakati wa kuainisha data kwa kutumia karatasi ya hila ya Microsoft. Scikit-learn inatoa karatasi ya hila inayofanana, lakini yenye maelezo zaidi ambayo inaweza kusaidia zaidi kupunguza vainisha wako (neno jingine la vainisha ni makadirio):

![ML Map from Scikit-learn](../../../../translated_images/sw/map.e963a6a51349425a.webp)
> Vidokezo: [tembelea ramani hii mtandaoni](https://scikit-learn.org/stable/tutorial/machine_learning_map/) na bonyeza mfululizo wa njia kusoma nyaraka.

### Mpango

Ramani hii ni msaada mkubwa mara unapokuwa na uelewa wazi wa data yako, kwani unaweza 'kutembea' kwenye njia zake kuelekea uamuzi:

- Tuna sampuli >50
- Tunataka kutabiri kategoria
- Tuna data zilizo na lebo
- Tuna sampuli chini ya 100K
- ✨ Tunaweza kuchagua Linear SVC
- Ikiwa hiyo haitumiki, kwa kuwa tuna data za nambari
    - Tunaweza kujaribu ✨ KNeighbors Classifier 
      - Ikiwa hiyo haitumiki, jaribu ✨ SVC na ✨ Ensemble Classifiers

Hii ni njia yenye msaada mkubwa kufuata.

## Zoha - gawanya data

Kwa kufuata njia hii, tunapaswa kuanza kwa kuingiza maktaba kadhaa kutumia.

1. Ingiza maktaba zinazohitajika:

    ```python
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    import numpy as np
    ```

1. Gawanya data yako ya mafunzo na mtihani:

    ```python
    X_train, X_test, y_train, y_test = train_test_split(cuisines_features_df, cuisines_label_df, test_size=0.3)
    ```

## Vainisha la Linear SVC

Support-Vector clustering (SVC) ni sehemu ya familia ya Mashine za Support-Vector za mbinu za ML (jifunza zaidi kuhusu hizi hapa chini). Katika njia hii, unaweza kuchagua 'kernel' kuamua jinsi ya kuunganisha lebo. Parameter ya 'C' inahusu 'regularization' ambayo hudhibiti ushawishi wa vigezo. Kernel inaweza kuwa moja ya [nguvu kadhaa](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC); hapa tumeiseti kuwa 'linear' kuhakikisha tunatumia linear SVC. Probability kwa default ni 'false'; hapa tumeiseti kuwa 'true' kukusanya makadirio ya uwezekano. Tumeweka hali ya bahati nasibu kuwa '0' kuchanganya data kupata uwezekano.

### Zoha - tumia linear SVC

Anza kwa kuunda safu ya vainisha. Utaiongeza polepole kwenye safu hii tunapojaribu.

1. Anza na Linear SVC:

    ```python
    C = 10
    # Unda waainishaji tofauti.
    classifiers = {
        'Linear SVC': SVC(kernel='linear', C=C, probability=True,random_state=0)
    }
    ```

2. Funza modeli yako kwa kutumia Linear SVC na chapisha ripoti:

    ```python
    n_classifiers = len(classifiers)
    
    for index, (name, classifier) in enumerate(classifiers.items()):
        classifier.fit(X_train, np.ravel(y_train))
    
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy (train) for %s: %0.1f%% " % (name, accuracy * 100))
        print(classification_report(y_test,y_pred))
    ```

    Matokeo ni mazuri kweli:

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

## Vainisha wa K-Neighbors

K-Neighbors ni sehemu ya familia ya mbinu za ML za "neighbors", ambazo zinaweza kutumika kwa kujifunza kwa uangalizi au bila uangalizi. Katika njia hii, idadi iliyowekwa ya pointi huundwa na data hukusanywa karibu na pointi hizi ili lebo jumla zitabiriwe kwa data.

### Zoha - tumia vainisha wa K-Neighbors

Vainisha wa awali ulikuwa mzuri, na ulikuwa na utendaji mzuri kwenye data, lakini labda tunaweza kupata usahihi bora zaidi. Jaribu vainisha wa K-Neighbors.

1. Ongeza mstari kwenye safu yako ya vainisha (ongeza koma baada ya kipengele cha Linear SVC):

    ```python
    'KNN classifier': KNeighborsClassifier(C),
    ```

    Matokeo ni mabaya kidogo:

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

    ✅ Jifunze kuhusu [K-Neighbors](https://scikit-learn.org/stable/modules/neighbors.html#neighbors)

## Vainisha wa Support Vector

Support-Vector vainisha ni sehemu ya familia ya mbinu za ML za [Support-Vector Machine](https://wikipedia.org/wiki/Support-vector_machine) zinazotumika kwa kazi za uainishaji na regression. SVMs "huweka mifano ya mafunzo kwenye pointi katika nafasi" ili kuongeza umbali kati ya vikundi viwili. Data inayofuata huwekwa kwenye nafasi hii ili kategoria yake itabiriwe.

### Zoha - tumia Vainisha wa Support Vector

Tujaribu usahihi kidogo bora na Vainisha wa Support Vector.

1. Ongeza koma baada ya kipengele cha K-Neighbors, kisha ongeza mstari huu:

    ```python
    'SVC': SVC(),
    ```

    Matokeo ni mazuri sana!

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

    ✅ Jifunze kuhusu [Support-Vectors](https://scikit-learn.org/stable/modules/svm.html#svm)

## Vainisha za Ensemble

Tufuate njia hadi mwisho kabisa, ingawa jaribio la awali lilikuwa zuri sana. Tujaribu 'Vainisha za Ensemble', hasa Random Forest na AdaBoost:

```python
  'RFST': RandomForestClassifier(n_estimators=100),
  'ADA': AdaBoostClassifier(n_estimators=100)
```

Matokeo ni mazuri sana, hasa kwa Random Forest:

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

✅ Jifunze kuhusu [Vainisha za Ensemble](https://scikit-learn.org/stable/modules/ensemble.html)

Njia hii ya Kujifunza Mashine "inaunganisha utabiri wa makadirio mengi ya msingi" ili kuboresha ubora wa modeli. Katika mfano wetu, tulitumia Miti ya Bahati nasibu na AdaBoost.

- [Random Forest](https://scikit-learn.org/stable/modules/ensemble.html#forest), njia ya kusawazisha, hujenga 'msitu' wa 'miti ya uamuzi' umetawanywa kwa bahati nasibu ili kuepuka kufikia kiwango cha juu sana cha kufaa data. Parameter ya n_estimators imewekwa kwa idadi ya miti.

- [AdaBoost](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html) huendana na vainisha kwa seti ya data kisha huiga vainisha huo kwa seti ile ile ya data. Inazingatia uzito wa vitu vilivyokosewa na kurekebisha utendakazi kwa vainisha inayofuata kurekebisha kosa.

---

## 🚀Changamoto

Kila moja ya mbinu hizi ina vigezo vingi unaweza kubadilisha. Fanya utafiti wa vigezo vyao vya default na fikiria maana ya kubadilisha vigezo hivi kwa ubora wa modeli.

## [Jaribio baada ya mihadhara](https://ff-quizzes.netlify.app/en/ml/)

## Mapitio & Kusoma Kibinafsi

Kuna maneno mengi magumu katika masomo haya, hivyo chukua dakika moja kupitia [orodha hii](https://docs.microsoft.com/dotnet/machine-learning/resources/glossary?WT.mc_id=academic-77952-leestott) ya istilahi muhimu!

## Kazi ya nyumbani

[Cheza na vigezo](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Kiarifa cha Hapana Dhima**:
Nyaraka hii imetafsiriwa kwa kutumia huduma ya tafsiri ya AI [Co-op Translator](https://github.com/Azure/co-op-translator). Ingawa tunajitahidi kuwa sahihi, tafadhali fahamu kuwa tafsiri za moja kwa moja zinaweza kuwa na makosa au kasoro. Nyaraka asilia katika lugha yake ya asili inapaswa kuzingatiwa kama chanzo cha uhakika. Kwa taarifa muhimu, tafsiri ya kitaalamu ya mwanadamu inashauriwa. Hatuwajibiki kwa kutoelewana au tafsiri za makosa zinazotokana na matumizi ya tafsiri hii.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->