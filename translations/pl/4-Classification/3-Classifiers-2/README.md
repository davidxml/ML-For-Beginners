# Klasyfikatory kuchni 2

W tej drugiej lekcji klasyfikacji poznasz więcej sposobów klasyfikacji danych numerycznych. Dowiesz się również o konsekwencjach wyboru jednego klasyfikatora zamiast drugiego.

## [Quiz przed wykładem](https://ff-quizzes.netlify.app/en/ml/)

### Wymagania wstępne

Zakładamy, że ukończyłeś poprzednie lekcje i masz wyczyszczony zestaw danych w folderze `data` o nazwie _cleaned_cuisines.csv_ w katalogu głównym tego folderu z 4 lekcjami.

### Przygotowanie

Wczytaliśmy do twojego pliku _notebook.ipynb_ wyczyszczony zestaw danych i podzieliliśmy go na ramki danych X i y, gotowe do procesu budowania modelu.

## Mapa klasyfikacji

Wcześniej poznałeś różne opcje klasyfikacji danych za pomocą ściągawki Microsoftu. Scikit-learn oferuje podobną, ale bardziej szczegółową ściągawkę, która może jeszcze bardziej zawęzić wybór twoich estymatorów (inna nazwa dla klasyfikatorów):

![ML Map from Scikit-learn](../../../../translated_images/pl/map.e963a6a51349425a.webp)
> Wskazówka: [odwiedź tę mapę online](https://scikit-learn.org/stable/tutorial/machine_learning_map/) i klikaj po ścieżce, aby czytać dokumentację.

### Plan

Ta mapa jest bardzo pomocna, gdy masz jasne zrozumienie swoich danych, ponieważ możesz 'wędrować' jej ścieżkami do decyzji:

- Mamy >50 próbek
- Chcemy przewidzieć kategorię
- Mamy dane oznaczone
- Mamy mniej niż 100 tysięcy próbek
- ✨ Możemy wybrać Linear SVC
- Jeśli to nie zadziała, ponieważ mamy dane numeryczne
    - Możemy spróbować ✨ KNeighbors Classifier
      - Jeśli to nie zadziała, spróbuj ✨ SVC i ✨ Ensemble Classifiers

To bardzo pomocna ścieżka do podążania.

## Ćwiczenie - podział danych

Podążając tą ścieżką, powinniśmy zacząć od zaimportowania kilku potrzebnych bibliotek.

1. Zaimportuj potrzebne biblioteki:

    ```python
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    import numpy as np
    ```

1. Podziel dane na treningowe i testowe:

    ```python
    X_train, X_test, y_train, y_test = train_test_split(cuisines_features_df, cuisines_label_df, test_size=0.3)
    ```

## Klasyfikator Linear SVC

Support-Vector clustering (SVC) jest techniką z rodziny maszyn wektorów nośnych (ang. Support-Vector machines), o których dowiesz się więcej poniżej. W tej metodzie możesz wybrać 'jądro' decydujące, jak grupować etykiety. Parametr 'C' odnosi się do 'regularizacji', która reguluje wpływ parametrów. Jądro może być jednym z [wielu](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC); tutaj ustawiamy je na 'linear', aby wykorzystać liniowy SVC. Parametr probability domyślnie jest 'false'; tutaj ustawiamy na 'true', aby zebrać oszacowania prawdopodobieństwa. Ustawiamy stan losowy na '0', aby przetasować dane, co pozwala otrzymać prawdopodobieństwa.

### Ćwiczenie - zastosuj Linear SVC

Zacznij od stworzenia tablicy klasyfikatorów. Będziesz ją stopniowo uzupełniać podczas testów.

1. Zacznij od Linear SVC:

    ```python
    C = 10
    # Utwórz różne klasyfikatory.
    classifiers = {
        'Linear SVC': SVC(kernel='linear', C=C, probability=True,random_state=0)
    }
    ```

2. Naucz model przy użyciu Linear SVC i wydrukuj raport:

    ```python
    n_classifiers = len(classifiers)
    
    for index, (name, classifier) in enumerate(classifiers.items()):
        classifier.fit(X_train, np.ravel(y_train))
    
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy (train) for %s: %0.1f%% " % (name, accuracy * 100))
        print(classification_report(y_test,y_pred))
    ```

    Wynik jest całkiem dobry:

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

## Klasyfikator K-Neighbors

K-Neighbors należy do rodziny metod "sąsiadów", które można stosować zarówno w uczeniu nadzorowanym, jak i nienadzorowanym. W tej metodzie tworzona jest zdefiniowana liczba punktów, a dane są grupowane wokół tych punktów, tak aby można było przewidzieć uogólnione etykiety dla danych.

### Ćwiczenie - zastosuj klasyfikator K-Neighbors

Poprzedni klasyfikator działał dobrze z danymi, ale może uzyskamy lepszą dokładność. Spróbuj klasyfikatora K-Neighbors.

1. Dodaj linię do tablicy klasyfikatorów (dodaj przecinek po elemencie Linear SVC):

    ```python
    'KNN classifier': KNeighborsClassifier(C),
    ```

    Wynik jest nieco gorszy:

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

    ✅ Dowiedz się więcej o [K-Neighbors](https://scikit-learn.org/stable/modules/neighbors.html#neighbors)

## Klasyfikator Support Vector

Klasyfikatory support-vector są częścią rodziny metod [Support-Vector Machine](https://wikipedia.org/wiki/Support-vector_machine) stosowanych do zadań klasyfikacji i regresji. SVM „mapuje przykłady treningowe na punkty w przestrzeni”, aby zmaksymalizować odległość między dwiema kategoriami. Kolejne dane są mapowane do tej przestrzeni, aby można było przewidzieć ich kategorię.

### Ćwiczenie - zastosuj Support Vector Classifier

Spróbujmy uzyskać nieco lepszą dokładność za pomocą Support Vector Classifier.

1. Dodaj przecinek po elemencie K-Neighbors, a następnie dodaj tę linię:

    ```python
    'SVC': SVC(),
    ```

    Wynik jest całkiem dobry!

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

    ✅ Dowiedz się więcej o [Support-Vectors](https://scikit-learn.org/stable/modules/svm.html#svm)

## Klasyfikatory zespołowe (Ensemble)

Podążmy ścieżką aż do końca, chociaż poprzedni test był całkiem dobry. Spróbujmy kilku 'klasyfikatorów zespołowych', w szczególności Random Forest i AdaBoost:

```python
  'RFST': RandomForestClassifier(n_estimators=100),
  'ADA': AdaBoostClassifier(n_estimators=100)
```

Wynik jest bardzo dobry, szczególnie dla Random Forest:

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

✅ Dowiedz się o [Klasyfikatorach zespołowych](https://scikit-learn.org/stable/modules/ensemble.html)

Ta metoda uczenia maszynowego „łączy predykcje kilku bazowych estymatorów”, aby poprawić jakość modelu. W naszym przykładzie użyliśmy drzew losowych i AdaBoost.

- [Random Forest](https://scikit-learn.org/stable/modules/ensemble.html#forest), metoda uśredniająca, buduje 'las' 'drzew decyzyjnych' z elementem losowości, aby uniknąć przeuczenia. Parametr n_estimators określa liczbę drzew.

- [AdaBoost](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html) dopasowuje klasyfikator do zestawu danych, a następnie dopasowuje kopie tego klasyfikatora do tego samego zestawu. Skupia się na wagach błędnie sklasyfikowanych elementów i dostosowuje dopasowanie dla kolejnego klasyfikatora, aby to poprawić.

---

## 🚀Wyzwanie

Każda z tych technik ma dużą liczbę parametrów, które możesz dostosować. Zbadaj domyślne parametry każdego z nich i zastanów się, co zmiana tych parametrów mogłaby oznaczać dla jakości modelu.

## [Quiz po wykładzie](https://ff-quizzes.netlify.app/en/ml/)

## Powtórka i samodzielna nauka

W tych lekcjach jest dużo żargonu, więc poświęć chwilę na przegląd [tej listy](https://docs.microsoft.com/dotnet/machine-learning/resources/glossary?WT.mc_id=academic-77952-leestott) przydatnej terminologii!

## Zadanie

[Zabawa z parametrami](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Zastrzeżenie**:  
Ten dokument został przetłumaczony przy użyciu automatycznej usługi tłumaczeniowej [Co-op Translator](https://github.com/Azure/co-op-translator). Mimo że dążymy do dokładności, prosimy mieć na uwadze, że tłumaczenia automatyczne mogą zawierać błędy lub nieścisłości. Oryginalny dokument w języku macierzystym należy uznawać za źródło autorytatywne. W przypadku krytycznych informacji zalecane jest skorzystanie z profesjonalnego tłumaczenia wykonanego przez człowieka. Nie ponosimy odpowiedzialności za jakiekolwiek nieporozumienia lub błędne interpretacje wynikające z korzystania z tego tłumaczenia.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->