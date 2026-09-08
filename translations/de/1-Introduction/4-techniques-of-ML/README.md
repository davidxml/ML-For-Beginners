# Techniken des maschinellen Lernens

Der Prozess des Erstellens, Verwendens und Wartens von Modellen des maschinellen Lernens und deren genutzten Daten ist ein ganz anderer Prozess als viele andere Entwicklungsworkflows. In dieser Lektion werden wir den Prozess entmystifizieren und die wichtigsten Techniken skizzieren, die Sie kennen müssen. Sie werden:

- Die Prozesse, die dem maschinellen Lernen zugrunde liegen, auf hohem Niveau verstehen.
- Grundkonzepte wie „Modelle“, „Vorhersagen“ und „Trainingsdaten“ erkunden.

## [Vorlesungsquiz](https://ff-quizzes.netlify.app/en/ml/)

[![ML for beginners - Techniques of Machine Learning](https://img.youtube.com/vi/4NGM0U2ZSHU/0.jpg)](https://youtu.be/4NGM0U2ZSHU "ML for beginners - Techniques of Machine Learning")

> 🎥 Klicken Sie auf das Bild oben für ein kurzes Video, das durch diese Lektion führt.

## Einführung

Auf hohem Niveau besteht das Handwerk der Erstellung von Prozessen des maschinellen Lernens (ML) aus mehreren Schritten:

1. **Die Frage festlegen.** Die meisten ML-Prozesse beginnen mit einer Frage, die nicht durch ein einfaches bedingtes Programm oder regelbasiertes System beantwortet werden kann. Diese Fragen drehen sich oft um Vorhersagen basierend auf einer Datensammlung.
2. **Daten sammeln und vorbereiten.** Um Ihre Frage beantworten zu können, benötigen Sie Daten. Die Qualität und manchmal auch die Menge Ihrer Daten bestimmt, wie gut Sie Ihre ursprüngliche Frage beantworten können. Die Visualisierung der Daten ist ein wichtiger Aspekt dieser Phase. Diese Phase umfasst auch die Aufteilung der Daten in Trainings- und Testgruppen zum Aufbau eines Modells.
3. **Eine Trainingsmethode wählen.** Abhängig von Ihrer Frage und der Natur Ihrer Daten müssen Sie entscheiden, wie Sie ein Modell trainieren möchten, um Ihre Daten am besten widerzuspiegeln und genaue Vorhersagen zu treffen. Dies ist der Teil Ihres ML-Prozesses, der spezifische Expertise und oft eine beträchtliche Menge an Experimenten erfordert.
4. **Das Modell trainieren.** Mithilfe Ihrer Trainingsdaten verwenden Sie verschiedene Algorithmen, um ein Modell zu trainieren, das Muster in den Daten erkennt. Das Modell kann interne Gewichte verwenden, die angepasst werden können, um bestimmten Teilen der Daten Vorrang zu geben und so ein besseres Modell zu erstellen.
5. **Das Modell bewerten.** Sie verwenden zuvor nie gesehene Daten (Ihre Testdaten) aus Ihrem gesammelten Datensatz, um die Leistung des Modells zu überprüfen.
6. **Parameteranpassung.** Basierend auf der Leistung Ihres Modells können Sie den Prozess wiederholen und verschiedene Parameter oder Variablen verwenden, die das Verhalten der zum Training verwendeten Algorithmen steuern.
7. **Vorhersagen treffen.** Verwenden Sie neue Eingaben, um die Genauigkeit Ihres Modells zu testen.

## Welche Frage sollte man stellen

Computer sind besonders gut darin, verborgene Muster in Daten zu entdecken. Diese Nützlichkeit ist sehr hilfreich für Forscher, die Fragen zu einem bestimmten Gebiet haben, die nicht leicht durch ein regelbasiertes System beantwortet werden können. Bei einer versicherungsmathematischen Aufgabe könnte ein Datenwissenschaftler zum Beispiel handgefertigte Regeln zur Mortalität von Rauchern versus Nichtrauchern erstellen.

Wenn jedoch viele andere Variablen mit einbezogen werden, könnte sich ein ML-Modell als effizienter erweisen, um zukünftige Sterblichkeitsraten basierend auf der bisherigen Gesundheitsgeschichte vorherzusagen. Ein fröhlicheres Beispiel könnte das Erstellen von Wettervorhersagen für den Monat April an einem bestimmten Ort sein, basierend auf Daten, die Breite, Länge, Klimawandel, Nähe zum Ozean, Muster des Jetstreams und mehr enthalten.

✅ Diese [Präsentation](https://www2.cisl.ucar.edu/sites/default/files/2021-10/0900%20June%2024%20Haupt_0.pdf) über Wettermodelle bietet eine historische Perspektive zur Verwendung von ML in der Wetteranalyse.

## Aufgaben vor dem Aufbau

Bevor Sie mit dem Aufbau Ihres Modells beginnen, müssen Sie mehrere Aufgaben erledigen. Um Ihre Frage zu testen und eine Hypothese auf Basis der Vorhersagen eines Modells zu bilden, müssen Sie mehrere Elemente identifizieren und konfigurieren.

### Daten

Um Ihre Frage mit einer gewissen Sicherheit beantworten zu können, benötigen Sie eine gute Menge an Daten vom richtigen Typ. An diesem Punkt müssen Sie zwei Dinge tun:

- **Daten sammeln.** Unter Berücksichtigung der vorherigen Lektion zur Fairness in der Datenanalyse sammeln Sie Ihre Daten sorgfältig. Seien Sie sich der Herkunft dieser Daten bewusst, etwaiger inhärenter Verzerrungen und dokumentieren Sie deren Ursprung.
- **Daten vorbereiten.** Im Datenvorbereitungsprozess sind mehrere Schritte erforderlich. Möglicherweise müssen Sie Daten aus verschiedenen Quellen sammeln und normalisieren. Sie können die Qualität und Menge der Daten durch verschiedene Methoden verbessern, z. B. durch Umwandlung von Zeichenketten in Zahlen (wie wir es in [Clustering](../../5-Clustering/1-Visualize/README.md) tun). Sie können auch neue Daten basierend auf den Originaldaten erzeugen (wie wir es in [Classification](../../4-Classification/1-Introduction/README.md) tun). Sie können die Daten säubern und bearbeiten (wie wir es vor der [Web-App](../../3-Web-App/README.md) Lektion tun). Schließlich müssen Sie sie je nach Trainingsmethode möglicherweise auch zufällig mischen und durchmischen.

✅ Nachdem Sie Ihre Daten gesammelt und verarbeitet haben, nehmen Sie sich einen Moment Zeit, um zu sehen, ob deren Form es Ihnen erlaubt, Ihre beabsichtigte Frage zu bearbeiten. Es kann sein, dass die Daten für Ihre Aufgabe nicht gut geeignet sind, wie wir in unseren [Clustering](../../5-Clustering/1-Visualize/README.md) Lektionen entdecken!

### Merkmale und Ziel

Ein [Merkmal](https://www.datasciencecentral.com/profiles/blogs/an-introduction-to-variable-and-feature-selection) ist eine messbare Eigenschaft Ihrer Daten. In vielen Datensätzen wird es als Spaltenüberschrift wie „Datum“, „Größe“ oder „Farbe“ dargestellt. Ihre Merkmalsvariable, meist als `X` im Code dargestellt, repräsentiert die Eingangsvariable, die zum Trainieren eines Modells verwendet wird.

Ein Ziel ist das, was Sie vorhersagen möchten. Das Ziel, meist als `y` im Code dargestellt, repräsentiert die Antwort auf die Frage, die Sie Ihren Daten stellen möchten: Im Dezember, welche **Farbe** werden Kürbisse am günstigsten haben? In San Francisco, welche Viertel werden den besten Immobilien-**Preis** haben? Manchmal wird das Ziel auch als Label-Attribut bezeichnet.

### Auswahl Ihrer Merkmalsvariablen

🎓 **Merkmalsauswahl und Merkmalsextraktion** Wie wissen Sie, welche Variable Sie beim Aufbau eines Modells wählen sollen? Wahrscheinlich durchlaufen Sie einen Prozess der Merkmalsauswahl oder Merkmalsextraktion, um die richtigen Variablen für das leistungsfähigste Modell auszuwählen. Sie sind jedoch nicht dasselbe: „Merkmalsextraktion erstellt neue Merkmale aus Funktionen der ursprünglichen Merkmale, während Merkmalsauswahl eine Teilmenge der Merkmale zurückgibt.“ ([Quelle](https://wikipedia.org/wiki/Feature_selection))

### Visualisieren Sie Ihre Daten

Ein wichtiger Bestandteil des Werkzeugsatzes eines Datenwissenschaftlers ist die Fähigkeit, Daten mit mehreren exzellenten Bibliotheken wie Seaborn oder MatPlotLib zu visualisieren. Die visuelle Darstellung Ihrer Daten kann Ihnen helfen, verborgene Korrelationen zu entdecken, die Sie nutzen können. Ihre Visualisierungen können Ihnen auch helfen, Verzerrungen oder unausgewogene Daten zu erkennen (wie wir in [Classification](../../4-Classification/2-Classifiers-1/README.md) entdecken).

### Teilen Sie Ihren Datensatz auf

Vor dem Training müssen Sie Ihren Datensatz in zwei oder mehr unterschiedlich große Teile aufteilen, die dennoch die Daten gut repräsentieren.

- **Training.** Dieser Teil des Datensatzes wird verwendet, um Ihr Modell zu trainieren. Dieser Satz macht den Großteil des ursprünglichen Datensatzes aus.
- **Testen.** Ein Testdatensatz ist eine unabhängige Datenmenge, oft aus den Originaldaten gewonnen, die Sie verwenden, um die Leistung des gebauten Modells zu bestätigen.
- **Validierung.** Ein Validierungssatz ist eine kleinere unabhängige Gruppe von Beispielen, die Sie zur Feinabstimmung der Hyperparameter oder Architektur des Modells nutzen, um das Modell zu verbessern. Je nach Größe Ihrer Daten und der gestellten Frage benötigen Sie diesen dritten Satz möglicherweise nicht (wie wir in [Time Series Forecasting](../../7-TimeSeries/1-Introduction/README.md) anmerken).

## Aufbau eines Modells

Mit Ihren Trainingsdaten ist es Ihr Ziel, ein Modell oder eine statistische Darstellung Ihrer Daten durch verschiedene Algorithmen zu **trainieren**. Das Trainieren eines Modells setzt es den Daten aus und ermöglicht ihm, Annahmen über wahrgenommene Muster zu treffen, diese zu validieren und anzunehmen oder abzulehnen.

### Wählen Sie eine Trainingsmethode

Je nach Ihrer Frage und der Natur Ihrer Daten wählen Sie eine Methode, um das Modell zu trainieren. Wenn Sie durch die [Dokumentation von Scikit-learn](https://scikit-learn.org/stable/user_guide.html) - die wir in diesem Kurs verwenden - gehen, finden Sie viele Möglichkeiten, ein Modell zu trainieren. Je nach Ihrer Erfahrung müssen Sie möglicherweise mehrere verschiedene Methoden ausprobieren, um das beste Modell zu erstellen. Häufig durchlaufen Datenwissenschaftler einen Prozess, bei dem sie die Leistung eines Modells durch die Überprüfung von zuvor unbekannten Daten, die auf Genauigkeit, Verzerrung und andere Qualitätsprobleme geprüft werden, evaluieren und die passendste Trainingsmethode für die jeweilige Aufgabe auswählen.

### Trainieren Sie ein Modell

Mit Ihren Trainingsdaten sind Sie bereit, es zu „fitten“, um ein Modell zu erstellen. Sie werden feststellen, dass Sie in vielen ML-Bibliotheken den Code `model.fit` sehen – zu diesem Zeitpunkt übergeben Sie Ihre Merkmalsvariable als Array von Werten (normalerweise 'X') und eine Zielvariable (normalerweise 'y').

### Bewerten Sie das Modell

Sobald der Trainingsprozess abgeschlossen ist (es können viele Iterationen oder „Epochen“ nötig sein, um ein großes Modell zu trainieren), können Sie die Qualität des Modells anhand von Testdaten einschätzen. Diese Daten sind ein Teil des ursprünglichen Datensatzes, den das Modell noch nicht analysiert hat. Sie können eine Tabelle mit Metriken zur Qualität Ihres Modells ausgeben.

🎓 **Modellanpassung**

Im Kontext des maschinellen Lernens bezieht sich Modellanpassung auf die Genauigkeit der zugrundeliegenden Funktion des Modells, während es versucht, Daten zu analysieren, mit denen es nicht vertraut ist.

🎓 **Underfitting** und **Overfitting** sind häufige Probleme, die die Qualität des Modells beeinträchtigen, wenn das Modell entweder nicht gut genug oder zu gut anpasst. Dies führt dazu, dass das Modell entweder zu genau oder zu lose mit seinen Trainingsdaten übereinstimmt. Ein überangepasstes Modell sagt die Trainingsdaten zu gut vorher, da es die Details und das Rauschen der Daten zu genau gelernt hat. Ein unterangepasstes Modell ist ungenau, da es weder seine Trainingsdaten noch neue Daten, die es noch nicht „gesehen“ hat, genau analysieren kann.

![overfitting model](../../../../translated_images/de/overfitting.1c132d92bfd93cb6.webp)
> Infografik von [Jen Looper](https://twitter.com/jenlooper)

## Parameteranpassung

Sobald Ihr erstes Training abgeschlossen ist, beobachten Sie die Qualität des Modells und überlegen, es zu verbessern, indem Sie dessen „Hyperparameter“ anpassen. Lesen Sie mehr über den Prozess [in der Dokumentation](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters?WT.mc_id=academic-77952-leestott).

## Vorhersage

Dies ist der Moment, in dem Sie völlig neue Daten verwenden können, um die Genauigkeit Ihres Modells zu testen. In einem „angewandten“ ML-Setting, bei dem Sie Web-Anwendungen erstellen, um das Modell in der Produktion zu nutzen, könnte dieser Prozess die Erfassung von Benutzereingaben (beispielsweise ein Button-Klick) umfassen, um eine Variable zu setzen und an das Modell zur Inferenz oder Bewertung zu senden.

In diesen Lektionen entdecken Sie, wie Sie diese Schritte nutzen, um Daten vorzubereiten, Modelle zu erstellen, zu testen, zu bewerten und Vorhersagen zu treffen – all die Gesten eines Datenwissenschaftlers und mehr, während Sie auf dem Weg sind, ein „Full Stack“ ML Engineer zu werden.

---

## 🚀Herausforderung

Zeichnen Sie ein Flussdiagramm, das die Schritte eines ML-Praktikers widerspiegelt. Wo sehen Sie sich gerade im Prozess? Wo prognostizieren Sie Schwierigkeiten? Was erscheint Ihnen leicht?

## [Nachvorlesungs-Quiz](https://ff-quizzes.netlify.app/en/ml/)

## Rückblick & Selbststudium

Suchen Sie online nach Interviews mit Datenwissenschaftlern, die ihren Arbeitsalltag beschreiben. Hier ist [eines](https://www.youtube.com/watch?v=Z3IjgbbCEfs).

## Aufgabe

[Interviewen Sie einen Datenwissenschaftler](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Haftungsausschluss**:  
Dieses Dokument wurde mithilfe des KI-Übersetzungsdienstes [Co-op Translator](https://github.com/Azure/co-op-translator) übersetzt. Obwohl wir auf Genauigkeit achten, beachten Sie bitte, dass automatisierte Übersetzungen Fehler oder Ungenauigkeiten enthalten können. Das Originaldokument in seiner ursprünglichen Sprache gilt als maßgebliche Quelle. Für kritische Informationen wird eine professionelle menschliche Übersetzung empfohlen. Wir übernehmen keine Haftung für Missverständnisse oder Fehlinterpretationen, die aus der Verwendung dieser Übersetzung entstehen.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->