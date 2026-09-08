# Einführung in maschinelles Lernen

## [Pre-Vorlesungsquiz](https://ff-quizzes.netlify.app/en/ml/)

---

[![ML für Anfänger - Einführung in maschinelles Lernen für Anfänger](https://img.youtube.com/vi/6mSx_KJxcHI/0.jpg)](https://youtu.be/6mSx_KJxcHI "ML für Anfänger - Einführung in maschinelles Lernen für Anfänger")

> 🎥 Klicke auf das Bild oben für ein kurzes Video, das diese Lektion durchgeht.

Willkommen zu diesem Kurs über klassisches maschinelles Lernen für Anfänger! Egal, ob du völlig neu auf diesem Gebiet bist oder ein erfahrener ML-Praktiker, der sein Wissen in einem Bereich auffrischen möchte – wir freuen uns, dass du dabei bist! Wir möchten einen freundlichen Ausgangspunkt für dein ML-Studium schaffen und sind gerne bereit, dein [Feedback](https://github.com/microsoft/ML-For-Beginners/discussions) zu bewerten, zu beantworten und einzubeziehen.

[![Einführung in ML](https://img.youtube.com/vi/h0e2HAPTGF4/0.jpg)](https://youtu.be/h0e2HAPTGF4 "Einführung in ML")

> 🎥 Klicke auf das Bild oben für ein Video: John Guttag vom MIT stellt maschinelles Lernen vor

---
## Einstieg in maschinelles Lernen

Bevor du mit diesem Lehrplan beginnst, musst du deinen Computer einrichten und bereit machen, um Notebooks lokal auszuführen.

- **Konfiguriere deine Maschine mit diesen Videos**. Nutze die folgenden Links, um zu lernen, [wie man Python installiert](https://youtu.be/CXZYvNRIAKM) und einen [Texteditor einrichtet](https://youtu.be/EU8eayHWoZg) für die Entwicklung.
- **Lerne Python**. Es wird außerdem empfohlen, grundlegende Kenntnisse in [Python](https://docs.microsoft.com/learn/paths/python-language/?WT.mc_id=academic-77952-leestott) zu haben, einer Programmiersprache, die für Datenwissenschaftler nützlich ist und die wir in diesem Kurs verwenden.
- **Lerne Node.js und JavaScript**. Wir verwenden JavaScript auch einige Male in diesem Kurs beim Erstellen von Web-Apps, daher benötigst du [node](https://nodejs.org) und [npm](https://www.npmjs.com/) installiert sowie [Visual Studio Code](https://code.visualstudio.com/) für die Python- und JavaScript-Entwicklung.
- **Erstelle ein GitHub-Konto**. Da du uns hier auf [GitHub](https://github.com) gefunden hast, hast du vielleicht schon eins, aber falls nicht, erstelle eins und forke diesen Lehrplan, um ihn selbst zu verwenden. (Gib uns gerne auch einen Stern 😊)
- **Erkunde Scikit-learn**. Mache dich mit [Scikit-learn](https://scikit-learn.org/stable/user_guide.html) vertraut, einem Satz von ML-Bibliotheken, die wir in diesen Lektionen referenzieren.

---
## Was ist maschinelles Lernen?

Der Begriff „maschinelles Lernen“ ist einer der populärsten und am häufigsten verwendeten Begriffe heute. Es ist sehr wahrscheinlich, dass du diesen Begriff zumindest einmal gehört hast, wenn du irgendeine Vertrautheit mit Technologie hast, unabhängig davon, in welchem Bereich du arbeitest. Die Mechanik des maschinellen Lernens ist jedoch für die meisten Menschen ein Rätsel. Für einen ML-Anfänger kann das Thema manchmal überwältigend wirken. Daher ist es wichtig, zu verstehen, was maschinelles Lernen tatsächlich ist, und es Schritt für Schritt durch praktische Beispiele kennenzulernen.

---
## Die Hype-Kurve

![ml hype curve](../../../../translated_images/de/hype.07183d711a17aafe.webp)

> Google Trends zeigt die aktuelle „Hype-Kurve“ des Begriffs „machine learning“

---
## Ein geheimnisvolles Universum

Wir leben in einem Universum voller faszinierender Geheimnisse. Große Wissenschaftler wie Stephen Hawking, Albert Einstein und viele weitere haben ihr Leben der Suche nach bedeutungsvollen Informationen gewidmet, die die Geheimnisse der Welt um uns herum enthüllen. Das ist der menschliche Zustand des Lernens: Ein Kind lernt Jahr für Jahr neue Dinge und entdeckt die Struktur seiner Welt, während es zum Erwachsenen heranwächst.

---
## Das Gehirn des Kindes

Das Gehirn und die Sinne eines Kindes nehmen die Fakten seiner Umgebung wahr und lernen allmählich die verborgenen Muster des Lebens, die dem Kind helfen, logische Regeln zu erstellen, um gelernte Muster zu erkennen. Der Lernprozess des menschlichen Gehirns macht den Menschen zum anspruchsvollsten Lebewesen auf dieser Welt. Indem wir kontinuierlich lernen, versteckte Muster entdecken und diese dann innovativ weiterentwickeln, können wir uns im Laufe unseres Lebens immer weiter verbessern. Diese Lernfähigkeit und sich entwickelnde Kapazität steht im Zusammenhang mit einem Konzept namens [Gehirnplastizität](https://www.simplypsychology.org/brain-plasticity.html). Oberflächlich betrachtet kann man einige motivierende Ähnlichkeiten zwischen dem Lernprozess des menschlichen Gehirns und den Konzepten des maschinellen Lernens ziehen.

---
## Das menschliche Gehirn

Das [menschliche Gehirn](https://www.livescience.com/29365-human-brain.html) nimmt Dinge aus der realen Welt wahr, verarbeitet die wahrgenommenen Informationen, trifft rationale Entscheidungen und führt je nach Situation bestimmte Handlungen aus. Dies nennen wir intelligentes Verhalten. Wenn wir einen Nachbau dieses intelligenten Verhaltensprozesses in eine Maschine programmieren, nennt man das künstliche Intelligenz (KI).

---
## Einige Begriffserklärungen

Obwohl die Begriffe verwechselt werden können, ist maschinelles Lernen (ML) ein wichtiger Teilbereich der künstlichen Intelligenz. **ML beschäftigt sich damit, spezialisierte Algorithmen zu verwenden, um bedeutungsvolle Informationen zu entdecken und verborgene Muster aus wahrgenommenen Daten zu finden, um den rationalen Entscheidungsprozess zu unterstützen**.

---
## KI, ML, Deep Learning

![AI, ML, deep learning, data science](../../../../translated_images/de/ai-ml-ds.537ea441b124ebf6.webp)

> Ein Diagramm, das die Beziehungen zwischen KI, ML, Deep Learning und Data Science zeigt. Infografik von [Jen Looper](https://twitter.com/jenlooper) inspiriert von [dieser Grafik](https://softwareengineering.stackexchange.com/questions/366996/distinction-between-ai-ml-neural-networks-deep-learning-and-data-mining)

---
## Zu behandelnde Konzepte

In diesem Lehrplan behandeln wir nur die Kernkonzepte des maschinellen Lernens, die ein Anfänger kennen muss. Wir decken das an, was wir als „klassisches maschinelles Lernen“ bezeichnen, hauptsächlich mit Scikit-learn, einer ausgezeichneten Bibliothek, die viele Studenten verwenden, um die Grundlagen zu lernen. Um breitere Konzepte der künstlichen Intelligenz oder des Deep Learnings zu verstehen, ist ein starkes Grundwissen im maschinellen Lernen unverzichtbar, und daher möchten wir es hier anbieten.

---
## In diesem Kurs lernst du:

- Kernkonzepte des maschinellen Lernens
- Die Geschichte des ML
- ML und Fairness
- Regressions-ML-Techniken
- Klassifikations-ML-Techniken
- Clustering-ML-Techniken
- Natürliche Sprachverarbeitung-ML-Techniken
- Zeitreihen-Prognose-ML-Techniken
- Verstärkendes Lernen
- Anwendungen von ML in der Praxis

---
## Was wir nicht behandeln

- Deep Learning
- Neuronale Netzwerke
- KI

Um ein besseres Lernerlebnis zu ermöglichen, vermeiden wir die Komplexität von neuronalen Netzwerken, „Deep Learning“ – vielschichtige Modellbildung mittels neuronalen Netzwerken – und KI, die wir in einem anderen Lehrplan besprechen werden. Wir werden außerdem einen zukünftigen Lehrplan zu Data Science anbieten, der sich auf diesen Aspekt dieses größeren Fachgebiets konzentriert.

---
## Warum maschinelles Lernen studieren?

Maschinelles Lernen wird aus Systemsicht als die Erstellung automatisierter Systeme definiert, die verborgene Muster aus Daten lernen können, um intelligente Entscheidungen zu unterstützen.

Diese Motivation ist lose inspiriert davon, wie das menschliche Gehirn bestimmte Dinge basierend auf Daten lernt, die es aus der Außenwelt wahrnimmt.

✅ Überlege einen Moment, warum ein Unternehmen versuchen würde, maschinelle Lernstrategien zu verwenden, anstatt eine regelbasierte Hard-Coded-Engine zu erstellen.

---
## Warum Datenqualität wichtig ist

Hochwertige Daten verbessern die Modellleistung. Schlechte oder verrauschte Daten können zu ungenauen Vorhersagen führen, selbst bei der Verwendung fortschrittlicher maschineller Lernalgorithmen.

---
## Anwendungen des maschinellen Lernens

Anwendungen des maschinellen Lernens sind inzwischen fast überall zu finden und ebenso allgegenwärtig wie die Daten, die unsere Gesellschaften durchströmen, erzeugt von unseren Smartphones, vernetzten Geräten und anderen Systemen. Angesichts des enormen Potenzials moderner maschineller Lernalgorithmen erforschen Forscher deren Fähigkeit, multidimensionale und multidisziplinäre reale Probleme mit großartigen positiven Ergebnissen zu lösen.

---
## Beispiele angewandten ML

**Maschinelles Lernen kann auf viele Arten genutzt werden**:

- Um die Wahrscheinlichkeit einer Krankheit aus der medizinischen Vorgeschichte oder Berichten eines Patienten vorherzusagen.
- Um Wetterdaten zu nutzen, um Wetterereignisse vorherzusagen.
- Um die Stimmung eines Texts zu verstehen.
- Um Fake-News zu erkennen, um die Verbreitung von Propaganda zu stoppen.

Finanzen, Wirtschaft, Geowissenschaften, Raumfahrt, Biomedizintechnik, Kognitionswissenschaft und sogar Fachgebiete der Geisteswissenschaften haben maschinelles Lernen adaptiert, um die schwierigen, datenverarbeitungsintensiven Probleme ihres Bereichs zu lösen.

---
## Fazit

Maschinelles Lernen automatisiert den Prozess der Mustererkennung, indem es bedeutungsvolle Erkenntnisse aus realen oder generierten Daten findet. Es hat sich im Geschäfts-, Gesundheits- und Finanzwesen als äußerst wertvoll erwiesen, unter anderem.

In naher Zukunft wird das Verstehen der Grundlagen des maschinellen Lernens für Menschen aus allen Bereichen ein Muss sein, wegen seiner weit verbreiteten Anwendung.

---
# 🚀 Herausforderung

Skizziere auf Papier oder mit einer Online-App wie [Excalidraw](https://excalidraw.com/) dein Verständnis der Unterschiede zwischen KI, ML, Deep Learning und Data Science. Füge einige Ideen über Probleme hinzu, die jede dieser Techniken gut lösen kann.

# [Post-Vorlesungsquiz](https://ff-quizzes.netlify.app/en/ml/)

---
# Rückblick & Selbststudium

Um mehr darüber zu lernen, wie du mit ML-Algorithmen in der Cloud arbeiten kannst, folge diesem [Learning Path](https://docs.microsoft.com/learn/paths/create-no-code-predictive-models-azure-machine-learning/?WT.mc_id=academic-77952-leestott).

Mache einen [Learning Path](https://docs.microsoft.com/learn/modules/introduction-to-machine-learning/?WT.mc_id=academic-77952-leestott) zu den Grundlagen des ML.

---
# Aufgabe

[Starte und loslegen](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Haftungsausschluss**:
Dieses Dokument wurde mit dem KI-Übersetzungsdienst [Co-op Translator](https://github.com/Azure/co-op-translator) übersetzt. Obwohl wir uns um Genauigkeit bemühen, beachten Sie bitte, dass automatisierte Übersetzungen Fehler oder Ungenauigkeiten enthalten können. Das Originaldokument in seiner Ursprungssprache gilt als maßgebliche Quelle. Bei kritischen Informationen wird eine professionelle menschliche Übersetzung empfohlen. Wir übernehmen keine Haftung für Missverständnisse oder Fehlinterpretationen, die aus der Verwendung dieser Übersetzung entstehen.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->