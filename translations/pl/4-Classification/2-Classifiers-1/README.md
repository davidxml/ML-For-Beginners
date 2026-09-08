# Klasyfikatory kuchni 1

W tej lekcji użyjesz zestawu danych, który zapisałeś z poprzedniej lekcji, pełnego zrównoważonych, czystych danych dotyczących kuchni.

Użyjesz tego zestawu danych z różnymi klasyfikatorami, aby _przewidzieć daną kuchnię narodową na podstawie grupy składników_. Przy okazji dowiesz się więcej o różnych sposobach wykorzystywania algorytmów do zadań klasyfikacji.

## [Quiz przed wykładem](https://ff-quizzes.netlify.app/en/ml/)
# Przygotowanie

Zakładając, że ukończyłeś [Lekcję 1](../1-Introduction/README.md), upewnij się, że w głównym folderze `/data` na potrzeby tych czterech lekcji istnieje plik _cleaned_cuisines.csv_.

## Ćwiczenie - przewidywanie kuchni narodowej

1. Pracując w folderze _notebook.ipynb_ z tej lekcji, zaimportuj ten plik wraz z biblioteką Pandas:

    ```python
    import pandas as pd
    cuisines_df = pd.read_csv("../data/cleaned_cuisines.csv")
    cuisines_df.head()
    ```

    Dane wyglądają tak:

|     | Unnamed: 0 | cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
| --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
| 0   | 0          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 1   | 1          | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 2   | 2          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 3   | 3          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 4   | 4          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |
  

1. Teraz zaimportuj kilka innych bibliotek:

    ```python
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    from sklearn.svm import SVC
    import numpy as np
    ```

1. Podziel współrzędne X i y na dwa dataframe’y do trenowania. `cuisine` może być dataframe’m etykiet:

    ```python
    cuisines_label_df = cuisines_df['cuisine']
    cuisines_label_df.head()
    ```

    Będzie wyglądać tak:

    ```output
    0    indian
    1    indian
    2    indian
    3    indian
    4    indian
    Name: cuisine, dtype: object
    ```

1. Usuń kolumnę `Unnamed: 0` oraz kolumnę `cuisine`, wywołując `drop()`. Resztę danych zapisz jako cechy do trenowania:

    ```python
    cuisines_feature_df = cuisines_df.drop(['Unnamed: 0', 'cuisine'], axis=1)
    cuisines_feature_df.head()
    ```

    Twoje cechy wyglądają tak:

|      | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | artemisia | artichoke |  ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood |  yam | yeast | yogurt | zucchini |
| ---: | -----: | -------: | ----: | ---------: | ----: | -----------: | ------: | -------: | --------: | --------: | ---: | ------: | ----------: | ---------: | ----------------------: | ---: | ---: | ---: | ----: | -----: | -------: |
|    0 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    1 |      1 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    2 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    3 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    4 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      1 |        0 | 0 |

Teraz jesteś gotów do trenowania modelu!

## Wybór klasyfikatora

Po oczyszczeniu danych i przygotowaniu ich do trenowania musisz zdecydować, którego algorytmu użyć.

Scikit-learn grupuje klasyfikację w ramach Uczenia Nadzorowanego (Supervised Learning), a w tej kategorii znajdziesz wiele sposobów klasyfikacji. [Różnorodność](https://scikit-learn.org/stable/supervised_learning.html) może na początku przytłaczać. Poniższe metody obejmują techniki klasyfikacyjne:

- Model liniowy
- Maszyny wektorów nośnych (Support Vector Machines)
- Stoachastyczny spadek gradientu (Stochastic Gradient Descent)
- Najbliżsi sąsiedzi (Nearest Neighbors)
- Procesy Gaussowskie
- Drzewa decyzyjne
- Metody zespołowe (voting Classifier)
- Algorytmy wieloklasowe i wielowyjściowe (wieloklasowa i wieloetykietowa klasyfikacja, wieloklasowa klasyfikacja wielowyjściowa)

> Możesz również użyć [sieci neuronowych do klasyfikacji danych](https://scikit-learn.org/stable/modules/neural_networks_supervised.html#classification), ale to wykracza poza zakres tej lekcji.

### Który klasyfikator wybrać?

Zatem, który klasyfikator wybrać? Często sposób polega na przetestowaniu kilku i poszukaniu dobrego wyniku. Scikit-learn oferuje [porównanie obok siebie](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html) na utworzonym zbiorze danych, porównując KNeighbors, SVC dwoma metodami, GaussianProcessClassifier, DecisionTreeClassifier, RandomForestClassifier, MLPClassifier, AdaBoostClassifier, GaussianNB i QuadraticDiscrinationAnalysis, pokazując wyniki wizualizowane:

![porównanie klasyfikatorów](../../../../translated_images/pl/comparison.edfab56193a85e7f.webp)
> Wykresy wygenerowane w dokumentacji Scikit-learn

> AutoML rozwiązuje ten problem sprawnie, uruchamiając te porównania w chmurze i pozwalając na wybór najlepszego algorytmu dla twoich danych. Spróbuj [tutaj](https://docs.microsoft.com/learn/modules/automate-model-selection-with-azure-automl/?WT.mc_id=academic-77952-leestott)

### Lepsze podejście

Lepszym, niż przypadkowe próby, jest zapoznanie się z ideami zawartymi w do pobrania [ściągawce ML](https://docs.microsoft.com/azure/machine-learning/algorithm-cheat-sheet?WT.mc_id=academic-77952-leestott). Tutaj odkrywamy, że dla naszego problemu wieloklasowego mamy pewne opcje:

![ściągawka dla problemów wieloklasowych](../../../../translated_images/pl/cheatsheet.07a475ea444d2223.webp)
> Fragment ściągawki Microsoftu dotyczącej wyboru algorytmu, opisujący możliwości klasyfikacji wieloklasowej

✅ Pobierz tę ściągawkę, wydrukuj i powieś na ścianie!

### Rozumowanie

Zobaczmy, czy możemy rozumowo przeanalizować różne podejścia biorąc pod uwagę dostępne ograniczenia:

- **Sieci neuronowe są zbyt ciężkie**. Mając czysty, ale minimalny zestaw danych oraz fakt, że trening odbywa się lokalnie w notebookach, sieci neuronowe są zbyt zasobożerne do tego zadania.
- **Brak klasyfikatora dwu-klasowego**. Nie korzystamy z klasyfikatora dwu-klasowego, więc wykluczamy one-vs-all.
- **Drzewo decyzyjne lub regresja logistyczna mogą zadziałać**. Drzewo decyzyjne może się sprawdzić, lub regresja logistyczna dla danych wieloklasowych.
- **Wieloklasowe Boosted Decision Trees rozwiązują inny problem**. Wieloklasowe zwiększane drzewa decyzyjne nadają się do zadań nieparametrycznych, np. budowania rankingów, więc nie są dla nas przydatne.

### Używanie Scikit-learn

Będziemy używać Scikit-learn do analizy naszych danych. Jednak istnieje wiele sposobów użycia regresji logistycznej w Scikit-learn. Spójrz na [parametry, które można przekazać](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).

Istotne są dwa parametry - `multi_class` i `solver` - które musimy określić, gdy prosimy Scikit-learn o regresję logistyczną. Wartość `multi_class` określa określone zachowanie. `solver` to wybrany algorytm. Nie wszystkie solvery mogą być łączone z wszystkimi wartościami `multi_class`.

Z dokumentacji wynika, że w przypadku wieloklasowym algorytm treningu:

- **używa schematu one-vs-rest (OvR)**, jeśli opcja `multi_class` jest ustawiona na `ovr`
- **używa funkcji straty entropii krzyżowej**, jeśli opcja `multi_class` jest ustawiona na `multinomial`. (Obecnie opcja `multinomial` jest wspierana tylko przez solvery ‘lbfgs’, ‘sag’, ‘saga’ oraz ‘newton-cg’)."

> 🎓 'Schemat' może być „ovr” (one-vs-rest) lub „multinomial”. Regresja logistyczna jest projektowana głównie do klasyfikacji binarnej, te schematy umożliwiają lepszą obsługę zadań klasyfikacji wieloklasowej. [źródło](https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/)

> 🎓 'Solver' to „algorytm używany w problemie optymalizacji”. [źródło](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).

Scikit-learn oferuje tę tabelę wyjaśniającą, jak solvery radzą sobie z różnymi wyzwaniami wynikającymi z różnego rodzaju struktur danych:

![solvery](../../../../translated_images/pl/solvers.5fc648618529e627.webp)

## Ćwiczenie - podziel dane

Możemy skupić się na regresji logistycznej jako naszym pierwszym próbnym treningu, ponieważ niedawno uczyłeś się o niej na poprzedniej lekcji.  
Podziel dane na grupy treningowe i testowe, wywołując `train_test_split()`:

```python
X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
```

## Ćwiczenie - zastosuj regresję logistyczną

Ponieważ używasz przypadku wieloklasowego, musisz zdecydować, jaki _schemat_ zastosować i jaki _solver_ ustawić. Użyj LogisticRegression z ustawieniem wieloklasowym i solverem **liblinear** do treningu.

1. Utwórz regresję logistyczną z multi_class ustawionym na `ovr` i solverem ustawionym na `liblinear`:

    ```python
    lr = LogisticRegression(multi_class='ovr',solver='liblinear')
    model = lr.fit(X_train, np.ravel(y_train))
    
    accuracy = model.score(X_test, y_test)
    print ("Accuracy is {}".format(accuracy))
    ```

    ✅ Spróbuj innego solvera, np. `lbfgs`, który często jest ustawiony jako domyślny

    > Uwaga, użyj funkcji Pandas [`ravel`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.ravel.html) do spłaszczania danych, gdy jest to potrzebne.

    Dokładność jest dobra, ponad **80%**!

1. Możesz zobaczyć działanie modelu, testując jeden wiersz danych (#50):

    ```python
    print(f'ingredients: {X_test.iloc[50][X_test.iloc[50]!=0].keys()}')
    print(f'cuisine: {y_test.iloc[50]}')
    ```

    Wynik jest wyświetlany:

   ```output
   ingredients: Index(['cilantro', 'onion', 'pea', 'potato', 'tomato', 'vegetable_oil'], dtype='object')
   cuisine: indian
   ```

   ✅ Spróbuj innego numeru wiersza i sprawdź wyniki
1. Zagłębiając się bardziej, możesz sprawdzić dokładność tej prognozy:

    ```python
    test= X_test.iloc[50].values.reshape(-1, 1).T
    proba = model.predict_proba(test)
    classes = model.classes_
    resultdf = pd.DataFrame(data=proba, columns=classes)
    
    topPrediction = resultdf.T.sort_values(by=[0], ascending = [False])
    topPrediction.head()
    ```

    Wynik jest wydrukowany - kuchnia indyjska to najlepsze przypuszczenie, z dużym prawdopodobieństwem:

    |          |        0 |
    | -------: | -------: |
    |   indian | 0.715851 |
    |  chinese | 0.229475 |
    | japanese | 0.029763 |
    |   korean | 0.017277 |
    |     thai | 0.007634 |

    ✅ Czy możesz wyjaśnić, dlaczego model jest dość pewny, że to kuchnia indyjska?

1. Uzyskaj więcej szczegółów, drukując raport klasyfikacji, tak jak robiłeś to na lekcjach regresji:

    ```python
    y_pred = model.predict(X_test)
    print(classification_report(y_test,y_pred))
    ```

    |              | precision | recall | f1-score | support |
    | ------------ | --------- | ------ | -------- | ------- |
    | chinese      | 0.73      | 0.71   | 0.72     | 229     |
    | indian       | 0.91      | 0.93   | 0.92     | 254     |
    | japanese     | 0.70      | 0.75   | 0.72     | 220     |
    | korean       | 0.86      | 0.76   | 0.81     | 242     |
    | thai         | 0.79      | 0.85   | 0.82     | 254     |
    | accuracy     |           |        | 0.80     | 1199    |
    | macro avg    | 0.80      | 0.80   | 0.80     | 1199    |
    | weighted avg | 0.80      | 0.80   | 0.80     | 1199    |

## 🚀Wyzwanie

Na tej lekcji użyłeś swoich oczyszczonych danych do zbudowania modelu uczenia maszynowego, który potrafi przewidzieć narodową kuchnię na podstawie zestawu składników. Poświęć trochę czasu, aby zapoznać się z wieloma opcjami oferowanymi przez Scikit-learn do klasyfikacji danych. Zagłęb się w pojęcie 'solver', aby zrozumieć, co dzieje się za kulisami.

## [Quiz po wykładzie](https://ff-quizzes.netlify.app/en/ml/)

## Przegląd i samodzielna nauka

Zagłęb się nieco bardziej w matematykę stojącą za regresją logistyczną w [tej lekcji](https://people.eecs.berkeley.edu/~russell/classes/cs194/f11/lectures/CS194%20Fall%202011%20Lecture%2006.pdf)
## Zadanie 

[Przestudiuj solvery](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Zastrzeżenie**:  
Niniejszy dokument został przetłumaczony za pomocą usługi tłumaczenia AI [Co-op Translator](https://github.com/Azure/co-op-translator). Chociaż dążymy do dokładności, prosimy pamiętać, że tłumaczenia automatyczne mogą zawierać błędy lub niedokładności. Oryginalny dokument w języku źródłowym powinien być uważany za autorytatywne źródło. W przypadku informacji krytycznych zalecane jest skorzystanie z profesjonalnego tłumaczenia wykonanego przez człowieka. Nie ponosimy odpowiedzialności za jakiekolwiek nieporozumienia lub błędne interpretacje wynikające z korzystania z tego tłumaczenia.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->