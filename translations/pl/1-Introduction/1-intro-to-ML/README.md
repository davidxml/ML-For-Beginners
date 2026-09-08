# Wprowadzenie do uczenia maszynowego

## [Quiz przed wykładem](https://ff-quizzes.netlify.app/en/ml/)

---

[![ML for beginners - Introduction to Machine Learning for Beginners](https://img.youtube.com/vi/6mSx_KJxcHI/0.jpg)](https://youtu.be/6mSx_KJxcHI "ML for beginners - Introduction to Machine Learning for Beginners")

> 🎥 Kliknij powyższy obraz, aby obejrzeć krótki film omawiający tę lekcję.

Witamy na kursie klasycznego uczenia maszynowego dla początkujących! Niezależnie od tego, czy jesteś zupełnie nowy w tym temacie, czy doświadczonym praktykiem ML, który chce odświeżyć swoją wiedzę, cieszymy się, że do nas dołączasz! Chcemy stworzyć przyjazne miejsce startowe dla Twojej nauki ML i chętnie ocenimy, odpowiemy na, oraz włączymy Twoje [opinie](https://github.com/microsoft/ML-For-Beginners/discussions).

[![Introduction to ML](https://img.youtube.com/vi/h0e2HAPTGF4/0.jpg)](https://youtu.be/h0e2HAPTGF4 "Introduction to ML")

> 🎥 Kliknij powyższy obraz, aby obejrzeć film: John Guttag z MIT wprowadza w uczenie maszynowe

---
## Zacznij swoją przygodę z uczeniem maszynowym

Przed rozpoczęciem pracy z tym materiałem, musisz mieć komputer skonfigurowany i gotowy do uruchamiania notatników lokalnie.

- **Skonfiguruj swój komputer korzystając z tych filmów**. Użyj poniższych linków, aby nauczyć się [jak zainstalować Pythona](https://youtu.be/CXZYvNRIAKM) na swoim systemie oraz [jak ustawić edytor tekstu](https://youtu.be/EU8eayHWoZg) do pracy.
- **Naucz się Pythona**. Zalecane jest również posiadanie podstawowej wiedzy o [Pythonie](https://docs.microsoft.com/learn/paths/python-language/?WT.mc_id=academic-77952-leestott), języku programowania przydatnym dla data scientistów, którego używamy w tym kursie.
- **Naucz się Node.js i JavaScriptu**. W tym kursie korzystamy też kilkakrotnie z JavaScriptu przy budowie aplikacji webowych, dlatego potrzebujesz mieć zainstalowane [node](https://nodejs.org) oraz [npm](https://www.npmjs.com/), a także [Visual Studio Code](https://code.visualstudio.com/) do rozwoju zarówno w Pythonie, jak i JavaScriptcie.
- **Załóż konto na GitHubie**. Jeśli trafiłeś tutaj przez [GitHub](https://github.com), możesz mieć już konto, ale jeśli nie, załóż je, a następnie sforkuj ten materiał, aby korzystać z niego na własne potrzeby. (Możesz też zostawić nam gwiazdkę 😊)
- **Poznaj Scikit-learn**. Zapoznaj się z [Scikit-learn](https://scikit-learn.org/stable/user_guide.html), zestawem bibliotek ML, na które często się powołujemy w tych lekcjach.

---
## Czym jest uczenie maszynowe?

Termin „uczenie maszynowe” jest jednym z najpopularniejszych i najczęściej używanych obecnie. Istnieje spore prawdopodobieństwo, że słyszałeś ten termin przynajmniej raz, jeśli masz jakiekolwiek pojęcie o technologii, bez względu na dziedzinę, w której pracujesz. Mechanika uczenia maszynowego jest jednak dla większości osób tajemnicą. Dla początkującego w ML temat może być nieco przytłaczający. Dlatego ważne jest, aby zrozumieć, czym naprawdę jest uczenie maszynowe, i uczyć się o nim krok po kroku, przez praktyczne przykłady.

---
## Krzywa hype'u

![ml hype curve](../../../../translated_images/pl/hype.07183d711a17aafe.webp)

> Google Trends pokazuje ostatnią „krzywą hype’u” terminu „uczenie maszynowe”

---
## Tajemniczy wszechświat

Żyjemy w wszechświecie pełnym fascynujących tajemnic. Wielcy naukowcy, tacy jak Stephen Hawking, Albert Einstein i wielu innych, poświęcili swoje życie poszukiwaniu znaczących informacji, które ujawniają tajemnice świata wokół nas. To kondycja ludzka uczenia się: dziecko poznaje nowe rzeczy i odkrywa strukturę swojego świata z roku na rok, dorastając do dorosłości.

---
## Mózg dziecka

Mózg i zmysły dziecka postrzegają fakty otoczenia i stopniowo uczą się ukrytych wzorców życia, które pomagają dziecku stworzyć logiczne reguły identyfikujące nauczone wzorce. Proces uczenia się ludzkiego mózgu czyni ludzi najbardziej wyrafinowanym żywym stworzeniem na świecie. Ciągłe uczenie się przez odkrywanie ukrytych wzorców, a następnie innowacje na ich podstawie pozwala nam się stale ulepszać przez całe życie. Ta zdolność uczenia się i rozwijające się możliwości wiążą się z pojęciem zwanym [plastycznością mózgu](https://www.simplypsychology.org/brain-plasticity.html). Na powierzchni możemy dostrzec pewne motywacyjne podobieństwa między procesem uczenia się ludzkiego mózgu a koncepcjami uczenia maszynowego.

---
## Ludzki mózg

[Ludzki mózg](https://www.livescience.com/29365-human-brain.html) postrzega rzeczy ze świata realnego, przetwarza odebrane informacje, podejmuje racjonalne decyzje i wykonuje określone działania w zależności od okoliczności. To nazywamy inteligentnym zachowaniem. Kiedy programujemy na maszynie naśladując ten inteligentny proces zachowań, nazywamy to sztuczną inteligencją (AI).

---
## Kilka terminów

Chociaż terminy mogą się mylić, uczenie maszynowe (ML) jest ważnym podzbiorem sztucznej inteligencji. **ML zajmuje się stosowaniem wyspecjalizowanych algorytmów do odkrywania znaczących informacji oraz znajdowania ukrytych wzorców z odebranych danych, aby potwierdzić racjonalny proces podejmowania decyzji**.

---
## AI, ML, uczenie głębokie

![AI, ML, deep learning, data science](../../../../translated_images/pl/ai-ml-ds.537ea441b124ebf6.webp)

> Diagram pokazujący relacje między sztuczną inteligencją, uczeniem maszynowym, uczeniem głębokim i nauką o danych. Infografika autorstwa [Jen Looper](https://twitter.com/jenlooper) inspirowana [tym obrazem](https://softwareengineering.stackexchange.com/questions/366996/distinction-between-ai-ml-neural-networks-deep-learning-and-data-mining)

---
## Koncepcje do omówienia

W tym programie omówimy tylko podstawowe koncepcje uczenia maszynowego, które początkujący musi znać. Skupimy się na tzw. „klasycznym uczeniu maszynowym”, głównie korzystając ze Scikit-learn, znakomitej biblioteki, z której wielu studentów korzysta, aby poznać podstawy. Aby zrozumieć szersze pojęcia sztucznej inteligencji lub uczenia głębokiego, solidna wiedza podstawowa z uczenia maszynowego jest niezbędna, dlatego chcemy ją tutaj zapewnić.

---
## Na tym kursie nauczysz się:

- podstawowych koncepcji uczenia maszynowego
- historii ML
- ML a sprawiedliwości
- technik regresji ML
- technik klasyfikacji ML
- technik klastrowania ML
- technik przetwarzania języka naturalnego ML
- technik prognozowania szeregów czasowych ML
- uczenia ze wzmocnieniem
- zastosowań ML w praktyce

---
## Czego nie omówimy

- uczenia głębokiego
- sieci neuronowych
- sztucznej inteligencji

Aby zapewnić lepsze doświadczenie edukacyjne, unikniemy złożoności sieci neuronowych, „uczenia głębokiego” – wielowarstwowego tworzenia modeli za pomocą sieci neuronowych – oraz AI, które omówimy w osobnym kursie. Wkrótce zaproponujemy również kurs nauki o danych, skupiający się na tym obszarze.

---
## Dlaczego warto uczyć się uczenia maszynowego?

Uczenie maszynowe, z perspektywy systemów, definiuje się jako tworzenie zautomatyzowanych systemów, które potrafią uczyć się ukrytych wzorców z danych, aby wspomagać podejmowanie inteligentnych decyzji.

Ta motywacja jest luźno inspirowana tym, jak ludzki mózg uczy się pewnych rzeczy na podstawie danych, które odbiera z otaczającego świata.

✅ Pomyśl przez chwilę, dlaczego firma chciałaby wykorzystać strategie uczenia maszynowego zamiast tworzyć silnik oparty na kodowanych na twardo regułach.

---
## Dlaczego jakość danych ma znaczenie

Dane wysokiej jakości poprawiają wydajność modelu. Słabe lub zaszumione dane mogą prowadzić do niedokładnych predykcji, nawet przy stosowaniu zaawansowanych algorytmów uczenia maszynowego.

---
## Zastosowania uczenia maszynowego

Zastosowania uczenia maszynowego są dziś niemal wszędzie i są tak powszechne, jak dane które przepływają w naszych społeczeństwach, generowane przez smartfony, urządzenia połączone i inne systemy. Biorąc pod uwagę ogromny potencjał najnowocześniejszych algorytmów ML, badacze eksplorują ich zdolności do rozwiązywania wielowymiarowych i interdyscyplinarnych problemów rzeczywistych z bardzo pozytywnymi wynikami.

---
## Przykłady zastosowanego ML

**Uczenie maszynowe można wykorzystywać na wiele sposobów**:

- Aby przewidywać prawdopodobieństwo choroby na podstawie historii medycznej lub raportów pacjenta.
- Aby wykorzystać dane pogodowe do prognozowania zjawisk atmosferycznych.
- Aby zrozumieć sentyment tekstu.
- Aby wykrywać fałszywe wiadomości i powstrzymać rozprzestrzenianie się propagandy.

Finanse, ekonomia, nauki o ziemi, badania kosmiczne, inżynieria biomedyczna, nauki kognitywne, a nawet dziedziny humanistyczne dostosowały uczenie maszynowe do rozwiązywania trudnych, opartych na danych problemów swoich obszarów.

---
## Podsumowanie

Uczenie maszynowe automatyzuje proces odkrywania wzorców przez znajdowanie znaczących informacji z danych rzeczywistych lub generowanych. Udowodniło swoją dużą wartość w biznesie, zdrowiu i finansach, między innymi.

W niedalekiej przyszłości znajomość podstaw uczenia maszynowego stanie się obowiązkowa dla ludzi z każdej dziedziny ze względu na jego powszechne zastosowanie.

---
# 🚀 Wyzwanie

Naszkicuj na papierze lub używając aplikacji online, takiej jak [Excalidraw](https://excalidraw.com/), swoje rozumienie różnic między AI, ML, uczeniem głębokim i nauką o danych. Dodaj kilka pomysłów na problemy, które każda z tych technik dobrze rozwiązuje.

# [Quiz po wykładzie](https://ff-quizzes.netlify.app/en/ml/)

---
# Przegląd i samodzielna nauka

Aby dowiedzieć się więcej o tym, jak pracować z algorytmami ML w chmurze, skorzystaj z tego [Learning Path](https://docs.microsoft.com/learn/paths/create-no-code-predictive-models-azure-machine-learning/?WT.mc_id=academic-77952-leestott).

Ukończ [Learning Path](https://docs.microsoft.com/learn/modules/introduction-to-machine-learning/?WT.mc_id=academic-77952-leestott) o podstawach ML.

---
# Zadanie

[Zacznij pracę](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Zastrzeżenie**:
Niniejszy dokument został przetłumaczony za pomocą usługi tłumaczenia AI [Co-op Translator](https://github.com/Azure/co-op-translator). Choć dążymy do dokładności, prosimy pamiętać, że automatyczne tłumaczenia mogą zawierać błędy lub niedokładności. Oryginalny dokument w jego języku źródłowym należy uznawać za autorytatywne źródło. W przypadku informacji krytycznych zalecane jest skorzystanie z profesjonalnego tłumaczenia wykonanego przez człowieka. Nie ponosimy odpowiedzialności za jakiekolwiek nieporozumienia lub błędne interpretacje wynikające z użycia tego tłumaczenia.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->