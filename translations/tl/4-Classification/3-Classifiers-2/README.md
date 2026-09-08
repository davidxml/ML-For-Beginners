# Mga classifier ng lutuin 2

Sa ikalawang aralin sa klasipikasyon na ito, susuriin mo ang higit pang mga paraan upang iklasipika ang numerikong datos. Malalaman mo rin ang mga epekto ng pagpili ng isang classifier kaysa sa iba.

## [Pre-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

### Paunang Kaalaman

Ipinagpapalagay namin na nakumpleto mo na ang mga nakaraang aralin at mayroon kang malinis na dataset sa iyong `data` folder na tinatawag na _cleaned_cuisines.csv_ sa root ng 4-lesson folder na ito.

### Paghahanda

Na-load namin sa iyong _notebook.ipynb_ file ang malinis na dataset at hinati ito sa mga dataframe na X at y, na handa na para sa proseso ng paggawa ng modelo.

## Isang mapa ng klasipikasyon

Dati, natutunan mo ang iba't ibang mga opsyon na mayroon ka sa pagpapangkat ng data gamit ang cheat sheet ng Microsoft. Nag-aalok ang Scikit-learn ng katulad, ngunit mas detalyadong cheat sheet na makakatulong pang paliitin ang iyong mga estimator (ibang tawag sa mga classifier):

![ML Map from Scikit-learn](../../../../translated_images/tl/map.e963a6a51349425a.webp)
> Tip: [bisitahin ang mapang ito online](https://scikit-learn.org/stable/tutorial/machine_learning_map/) at i-click ang mga landas upang basahin ang dokumentasyon.

### Ang plano

Napakabisa ng mapang ito kapag malinaw na ang iyong pagkakaintindi sa data, dahil maaari kang ‘maglakad’ sa mga landas nito upang makagawa ng desisyon:

- Mayroon tayong >50 samples
- Gusto nating hulaan ang isang kategorya
- Mayroon tayong tinag na data
- Mas kaunti tayo sa 100K samples
- ✨ Pwede tayong pumili ng Linear SVC
- Kung hindi ito gumana, dahil numeriko ang data natin
    - Pwede nating subukan ang ✨ KNeighbors Classifier 
      - Kung hindi pa rin ito gumana, subukan ang ✨ SVC at ✨ Ensemble Classifiers

Napakagandang landas ito na sundan.

## Ehersisyo - hatiin ang data

Sa pagsunod sa landas na ito, magsisimula tayong mag-import ng ilang mga library na gagamitin.

1. I-import ang mga kinakailangang library:

    ```python
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    import numpy as np
    ```

1. Hatiin ang iyong training at test data:

    ```python
    X_train, X_test, y_train, y_test = train_test_split(cuisines_features_df, cuisines_label_df, test_size=0.3)
    ```

## Linear SVC classifier

Ang Support-Vector clustering (SVC) ay bahagi ng pamilya ng Support-Vector machines na mga teknik sa ML (matuto pa tungkol dito sa ibaba). Sa pamamaraang ito, maaari kang pumili ng 'kernel' para tukuyin kung paano i-cluster ang mga label. Ang parameter na 'C' ay tumutukoy sa 'regularization' na nagreregula sa impluwensya ng mga parameter. Ang kernel ay isa sa [marami](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC); dito inilagay natin ito sa 'linear' upang matiyak na magagamit natin ang linear SVC. Ang default ng probability ay 'false'; dito itinakda natin ito sa 'true' upang makuha ang mga pagtatantiya ng posibilidad. Itinakda namin ang random state sa '0' upang i-shuffle ang data para makuha ang mga posibilidad.

### Ehersisyo - ipatupad ang linear SVC

Magsimula sa paggawa ng array ng mga classifier. Unang idadagdag mo rito habang nagt-test tayo.

1. Magsimula sa isang Linear SVC:

    ```python
    C = 10
    # Gumawa ng iba't ibang mga classifier.
    classifiers = {
        'Linear SVC': SVC(kernel='linear', C=C, probability=True,random_state=0)
    }
    ```

2. I-train ang iyong modelo gamit ang Linear SVC at i-print ang umiulat na ulat:

    ```python
    n_classifiers = len(classifiers)
    
    for index, (name, classifier) in enumerate(classifiers.items()):
        classifier.fit(X_train, np.ravel(y_train))
    
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy (train) for %s: %0.1f%% " % (name, accuracy * 100))
        print(classification_report(y_test,y_pred))
    ```

    Maganda ang resulta:

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

## K-Neighbors classifier

Ang K-Neighbors ay bahagi ng "neighbors" na pamilya ng mga ML na pamamaraan, na maaaring gamitin sa parehong supervised at unsupervised na pagkatuto. Sa pamamaraang ito, isang paunang tinukoy na bilang ng mga punto ang nilikha at nakaipon ang data sa paligid ng mga puntong ito upang mahulaan ang mga generalized na label para sa data.

### Ehersisyo - ipatupad ang K-Neighbors classifier

Maganda ang naunang classifier, at mabisa sa data, pero baka pwede pa nating mapahusay ang katumpakan. Subukan ang K-Neighbors classifier.

1. Magdagdag ng linya sa iyong classifier array (maglagay ng kuwit pagkatapos ng Linear SVC item):

    ```python
    'KNN classifier': KNeighborsClassifier(C),
    ```

    Medyo mas mababa ang resulta:

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

    ✅ Matuto tungkol sa [K-Neighbors](https://scikit-learn.org/stable/modules/neighbors.html#neighbors)

## Support Vector Classifier

Ang Support-Vector classifiers ay bahagi ng pamilya ng [Support-Vector Machine](https://wikipedia.org/wiki/Support-vector_machine) na mga teknik sa ML na ginagamit sa klasipikasyon at regression na mga gawain. Ang SVM ay "nagma-map ng mga halimbawa ng pagsasanay sa mga puntos sa espasyo" upang i-maximize ang distansya sa pagitan ng dalawang kategorya. Ang mga kasunod na data ay pinapasok sa puwang na ito upang mahulaan ang kanilang kategorya.

### Ehersisyo - ipatupad ang Support Vector Classifier

Subukan natin ang medyo mas mataas na katumpakan gamit ang Support Vector Classifier.

1. Magdagdag ng kuwit pagkatapos ng K-Neighbors item, at idagdag ang linyang ito:

    ```python
    'SVC': SVC(),
    ```

    Medyo maganda ang resulta!

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

    ✅ Matuto tungkol sa [Support-Vectors](https://scikit-learn.org/stable/modules/svm.html#svm)

## Ensemble Classifiers

Sundan natin ang landas hanggang sa dulo, kahit maganda na ang nauna nating test. Subukan natin ang ilang 'Ensemble Classifiers', partikular ang Random Forest at AdaBoost:

```python
  'RFST': RandomForestClassifier(n_estimators=100),
  'ADA': AdaBoostClassifier(n_estimators=100)
```

Napakabuti ang resulta, lalo na sa Random Forest:

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

✅ Matuto tungkol sa [Ensemble Classifiers](https://scikit-learn.org/stable/modules/ensemble.html)

Ang pamamaraang ito sa Machine Learning ay "pinaghalong prediksyon ng ilang base estimator" para mapabuti ang kalidad ng modelo. Sa aming halimbawa, gumamit kami ng Random Trees at AdaBoost. 

- [Random Forest](https://scikit-learn.org/stable/modules/ensemble.html#forest), isang paraan ng pag-average, ay bumubuo ng 'gubat' ng mga 'puno ng desisyon' na may kasamang randomness upang maiwasan ang overfitting. Ang parameter na n_estimators ay itinakda sa bilang ng mga puno.

- [AdaBoost](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html) ay nagta-takda ng classifier sa isang dataset at pagkatapos ay nagta-takda ng mga kopya ng classifier na iyon sa parehong dataset. Nakatuon ito sa mga timbang ng mga maling naklasipikang item at inaayos ang fit para sa susunod na classifier upang itama ito.

---

## 🚀Pagsubok

Bawat isa sa mga teknik na ito ay may maraming parameter na pwede mong i-tweak. Siyasatin ang mga default na parametro ng bawat isa at pag-isipang ano ang ibig sabihin ng pag-aayos ng mga parameter na ito para sa kalidad ng modelo.

## [Post-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## Review at Pansariling Pag-aaral

Maraming jargon sa mga araling ito, kaya maglaan ng sandali para suriin ang [listahang ito](https://docs.microsoft.com/dotnet/machine-learning/resources/glossary?WT.mc_id=academic-77952-leestott) ng mga kapaki-pakinabang na termino!

## Takdang-aralin 

[Parameter play](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Paunawa**:
Ang dokumentong ito ay isinalin gamit ang serbisyo ng AI na pagsasalin na [Co-op Translator](https://github.com/Azure/co-op-translator). Bagamat nagsusumikap kami para sa katumpakan, mangyaring tandaan na ang mga awtomatikong pagsasalin ay maaaring maglaman ng mga pagkakamali o di-tumpak na impormasyon. Ang orihinal na dokumento sa orihinal nitong wika ang dapat ituring na pangunahing sanggunian. Para sa mahahalagang impormasyon, inirerekomenda ang propesyonal na pagsasaling-tao. Hindi kami mananagot sa anumang hindi pagkakaunawaan o maling interpretasyon na maaaring magmula sa paggamit ng pagsasaling ito.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->