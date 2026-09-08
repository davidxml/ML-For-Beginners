# Techniki uczenia maszynowego

Proces budowania, używania i utrzymywania modeli uczenia maszynowego oraz danych, których używają, różni się znacznie od wielu innych przepływów pracy programistycznej. W tej lekcji rozwiejemy tajemnice tego procesu i przedstawimy główne techniki, które musisz znać. Nauczysz się:

- Rozumieć procesy leżące u podstaw uczenia maszynowego na wysokim poziomie.
- Poznawać podstawowe pojęcia, takie jak „modele”, „prognozy” i „dane treningowe”.

## [Quiz przed wykładem](https://ff-quizzes.netlify.app/en/ml/)

[![ML for beginners - Techniques of Machine Learning](https://img.youtube.com/vi/4NGM0U2ZSHU/0.jpg)](https://youtu.be/4NGM0U2ZSHU "ML for beginners - Techniques of Machine Learning")

> 🎥 Kliknij powyższy obraz, aby obejrzeć krótki film omawiający tę lekcję.

## Wprowadzenie

Na wysokim poziomie, rzemiosło tworzenia procesów uczenia maszynowego (ML) składa się z kilku etapów:

1. **Zdecyduj o pytaniu**. Większość procesów ML zaczyna się od postawienia pytania, na które nie można odpowiedzieć za pomocą prostego programu warunkowego lub silnika opartego na regułach. Pytania te często dotyczą prognoz opartych na zbiorze danych.
2. **Zbierz i przygotuj dane**. Aby odpowiedzieć na swoje pytanie, potrzebujesz danych. Jakość, a czasami ilość danych, określi, jak dobrze możesz odpowiedzieć na pytanie. Wizualizacja danych jest ważnym aspektem tej fazy. Ta faza obejmuje także podział danych na grupę treningową i testową do budowy modelu.
3. **Wybierz metodę treningu**. W zależności od pytania i charakteru danych musisz wybrać, jak chcesz trenować model, aby najlepiej odzwierciedlał dane i wykonywał dokładne prognozy. To część procesu ML, która wymaga specjalistycznej wiedzy i często sporej ilości eksperymentów.
4. **Wytrenuj model**. Korzystając z danych treningowych, zastosujesz różne algorytmy do wytrenowania modelu, aby rozpoznawał wzorce w danych. Model może wykorzystywać wewnętrzne wagi, które można dostosowywać, aby faworyzować pewne części danych i zbudować lepszy model.
5. **Oceń model**. Używasz dotychczas niewidzianych danych (danych testowych) z twojego zbioru, aby sprawdzić, jak model działa.
6. **Dostrajanie parametrów**. Na podstawie wydajności modelu możesz powtórzyć proces, stosując różne parametry lub zmienne, które kontrolują zachowanie algorytmów używanych do trenowania modelu.
7. **Prognozuj**. Użyj nowych danych wejściowych do przetestowania dokładności modelu.

## Jakie pytanie zadać

Komputery są szczególnie dobre w odkrywaniu ukrytych wzorców w danych. Ta użyteczność jest niezwykle pomocna dla badaczy, którzy mają pytania dotyczące danej dziedziny, na które trudno odpowiedzieć, tworząc warunkowy silnik reguł. Na przykład, przy zadaniu aktuarialnym, data scientist może zbudować ręcznie opracowane reguły dotyczące śmiertelności palaczy w porównaniu z niepalącymi.

Gdy do równania wprowadzonych zostanie wiele innych zmiennych, model ML może okazać się bardziej efektywny w przewidywaniu przyszłych wskaźników śmiertelności na podstawie historii zdrowotnej. Bardziej optymistycznym przykładem może być prognozowanie pogody na kwiecień dla danego miejsca na podstawie danych, które obejmują szerokość geograficzną, długość geograficzną, zmiany klimatu, bliskość oceanu, wzorce prądów strumieniowych i więcej.

✅ Ten [pakiet slajdów](https://www2.cisl.ucar.edu/sites/default/files/2021-10/0900%20June%2024%20Haupt_0.pdf) dotyczący modeli pogodowych oferuje historyczną perspektywę wykorzystania ML w analizie pogodowej.

## Zadania przedbudowy

Zanim zaczniesz budować model, musisz wykonać kilka zadań. Aby przetestować swoje pytanie i sformułować hipotezę na podstawie prognoz modelu, musisz zidentyfikować i skonfigurować kilka elementów.

### Dane

Aby z dużą pewnością odpowiedzieć na swoje pytanie, potrzebujesz odpowiedniej ilości danych odpowiedniego typu. W tym momencie musisz wykonać dwie rzeczy:

- **Zbierz dane**. Mając na uwadze poprzednią lekcję na temat równości w analizie danych, zbieraj dane ostrożnie. Bądź świadomy źródeł tych danych, ewentualnych wbudowanych uprzedzeń oraz dokumentuj ich pochodzenie.
- **Przygotuj dane**. W procesie przygotowania danych jest kilka etapów. Może być konieczne zebranie danych i ich normalizacja, jeśli pochodzą z różnych źródeł. Możesz poprawić jakość i ilość danych poprzez różne metody, takie jak konwersja łańcuchów znaków na liczby (jak robimy w [Klasteryzacji](../../5-Clustering/1-Visualize/README.md)). Możesz także wygenerować nowe dane na bazie oryginalnych (jak robimy w [Klasyfikacji](../../4-Classification/1-Introduction/README.md)). Możesz oczyścić i edytować dane (jak zrobimy przed lekcją [Aplikacja Webowa](../../3-Web-App/README.md)). Na koniec może być także potrzebne losowe uporządkowanie i przetasowanie danych, zależnie od wybranych technik treningowych.

✅ Po zebraniu i przetworzeniu danych, poświęć chwilę, by sprawdzić, czy ich forma pozwoli na odpowiedź na planowane pytanie. Może się okazać, że dane nie sprawdzą się dobrze przy danym zadaniu, co odkrywamy w lekcjach [Klasteryzacji](../../5-Clustering/1-Visualize/README.md)!

### Cechy i cel

[Cechą](https://www.datasciencecentral.com/profiles/blogs/an-introduction-to-variable-and-feature-selection) jest mierzalna właściwość danych. W wielu zbiorach danych jest ona wyrażana jako nagłówek kolumny, taki jak 'data', 'rozmiar' lub 'kolor'. Zmienna cechy, zwykle oznaczana w kodzie jako `X`, reprezentuje zmienną wejściową, która będzie używana do trenowania modelu.

Cel to rzecz, którą próbujesz przewidzieć. Cel, zwykle oznaczany w kodzie jako `y`, reprezentuje odpowiedź na pytanie, które zadajesz danym: w grudniu, jakie kolory dyni będą najtańsze? w San Francisco, które dzielnice będą miały najlepsze ceny nieruchomości? Czasem cel nazywany jest też etykietą (label).

### Wybór zmiennej cechy

🎓 **Wybór cech i ekstrakcja cech** Skąd wiesz, którą zmienną wybrać przy budowie modelu? Prawdopodobnie przejdziesz przez proces wyboru cech lub ekstrakcji cech, aby wybrać odpowiednie zmienne do najlepszego modelu. Nie są to jednak tożsame procesy: „Ekstrakcja cech tworzy nowe cechy na podstawie funkcji oryginalnych cech, natomiast wybór cech zwraca podzbiór cech.” ([źródło](https://wikipedia.org/wiki/Feature_selection))

### Wizualizuj swoje dane

Ważnym elementem zestawu narzędzi data scientistów jest możliwość wizualizacji danych przy użyciu znakomitych bibliotek, takich jak Seaborn czy MatPlotLib. Graficzne przedstawienie danych może pozwolić na odkrycie ukrytych korelacji, które można wykorzystać. Wizualizacje mogą także pomóc w wykryciu uprzedzeń lub niezrównoważonych danych (co odkrywamy w [Klasyfikacji](../../4-Classification/2-Classifiers-1/README.md)).

### Podział zbioru danych

Przed treningiem musisz podzielić zbiór danych na dwie lub więcej części o nierównej wielkości, które nadal dobrze reprezentują dane.

- **Treningowy**. Ta część zbioru danych służy do dopasowania modelu i jego treningu. Ta grupa stanowi większość oryginalnego zbioru danych.
- **Testowy**. Zbiór testowy to niezależna grupa danych, często wyodrębniona z oryginalnego zbioru, którą używasz do potwierdzenia jakości wytrenowanego modelu.
- **Walidacyjny**. Zbiór walidacyjny to mniejsza, niezależna grupa przykładów, którą używasz do dostrojenia hiperparametrów lub architektury, aby poprawić model. W zależności od wielkości twoich danych i pytania, które zadajesz, możesz nie potrzebować tworzyć tego trzeciego zbioru (jak zaznaczamy w [Prognozowaniu szeregów czasowych](../../7-TimeSeries/1-Introduction/README.md)).

## Budowa modelu

Korzystając z danych treningowych, twoim celem jest zbudowanie modelu, czyli statystycznej reprezentacji danych, używając różnych algorytmów do jego **wytrenowania**. Trening modelu polega na „ekspozycji” modelu na dane, co pozwala mu dokonywać założeń na temat rozpoznawanych wzorców, które następnie weryfikuje, akceptuje lub odrzuca.

### Wybierz metodę treningu

W zależności od pytania i charakteru danych wybierzesz metodę treningu. Przeglądając [dokumentację Scikit-learn](https://scikit-learn.org/stable/user_guide.html) – której używamy w tym kursie – możesz poznać wiele sposobów trenowania modelu. W zależności od doświadczenia, możesz spróbować kilku różnych metod, aby zbudować najlepszy model. Najczęściej proces ten obejmuje ocenę wydajności modelu przez data scientistów - podają im dane niewidziane wcześniej, sprawdzają dokładność, uprzedzenia i inne problemy obniżające jakość, a następnie wybierają najbardziej odpowiednią metodę treningu.

### Wytrenuj model

Mając dane treningowe, jesteś gotów „dopasować” je, aby stworzyć model. W wielu bibliotekach ML znajdziesz polecenie 'model.fit' - właśnie wtedy przesyłasz swoją zmienną cechy jako tablicę wartości (zwykle 'X') oraz zmienną celu (zwykle 'y').

### Oceń model

Po zakończeniu procesu treningowego (który może wymagać wielu iteracji, zwanych 'epokami', by wytrenować duży model), możesz ocenić jakość modelu, używając danych testowych do pomiaru jego wydajności. Dane te to podzbiór oryginalnych danych, których model wcześniej nie analizował. Możesz wyświetlić tabelę z metrykami jakości swojego modelu.

🎓 **Dopasowanie modelu**

W kontekście uczenia maszynowego dopasowanie modelu odnosi się do dokładności funkcji leżącej u podstaw modelu podczas próby analizy nieznanych mu danych.

🎓 **Niedopasowanie** i **przeuczenie** to częste problemy obniżające jakość modelu, gdy model dopasowuje się albo zbyt słabo, albo zbyt mocno. Powoduje to, że prognozy modelu są zbyt dokładnie (overfit) lub zbyt niedokładnie (underfit) dopasowane do danych treningowych. Model overfit przewiduje dane treningowe zbyt dokładnie, ponieważ zbyt dobrze poznał szczegóły i szumy danych. Model underfit jest niedokładny, ponieważ nie potrafi ani poprawnie analizować danych treningowych, ani danych, których wcześniej nie „widział”.

![overfitting model](../../../../translated_images/pl/overfitting.1c132d92bfd93cb6.webp)
> Infografika autorstwa [Jen Looper](https://twitter.com/jenlooper)

## Dostrajanie parametrów

Gdy wstępny trening jest ukończony, obserwuj jakość modelu i rozważ jego ulepszenie poprzez dostrajanie 'hiperparametrów'. Więcej informacji o tym procesie znajdziesz [w dokumentacji](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters?WT.mc_id=academic-77952-leestott).

## Prognozowanie

To moment, w którym możesz użyć całkowicie nowych danych, aby przetestować dokładność modelu. W praktycznym zastosowaniu ML, gdzie budujesz zasoby webowe do użycia modelu w produkcji, proces ten może wiązać się z pozyskaniem danych od użytkownika (np. naciśnięcie przycisku), ustawieniem zmiennej i przesłaniem jej do modelu do inferencji, czyli oceny.

W tych lekcjach poznasz, jak korzystać z tych kroków, aby przygotować, zbudować, przetestować, ocenić i prognozować – wszystkie gesty data scientista i jeszcze więcej, na drodze do zostania inżynierem ML 'full stack'.

---

## 🚀Wyzwanie

Narysuj schemat blokowy odzwierciedlający kroki praktyka ML. Gdzie widzisz siebie obecnie w tym procesie? Gdzie przewidujesz trudności? Co wydaje ci się łatwe?

## [Quiz po wykładzie](https://ff-quizzes.netlify.app/en/ml/)

## Powtórka i samodzielna nauka

Wyszukaj w internecie wywiady z data scientistami, którzy opowiadają o swojej codziennej pracy. Oto [jeden](https://www.youtube.com/watch?v=Z3IjgbbCEfs).

## Zadanie

[Przeprowadź wywiad z data scientistą](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Zastrzeżenie**:  
Dokument ten został przetłumaczony za pomocą usługi tłumaczeń AI [Co-op Translator](https://github.com/Azure/co-op-translator). Choć dokładamy starań, aby tłumaczenie było precyzyjne, prosimy pamiętać, że automatyczne tłumaczenia mogą zawierać błędy lub niedokładności. Oryginalny dokument w języku źródłowym powinien być uznawany za wersję autorytatywną. W przypadku informacji o kluczowym znaczeniu zalecane jest skorzystanie z profesjonalnego tłumaczenia wykonanego przez człowieka. Nie ponosimy odpowiedzialności za jakiekolwiek nieporozumienia lub błędne interpretacje wynikające z użycia tego tłumaczenia.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->