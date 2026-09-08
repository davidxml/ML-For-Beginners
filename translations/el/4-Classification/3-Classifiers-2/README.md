# Ταξινομητές κουζίνας 2

Σε αυτό το δεύτερο μάθημα ταξινόμησης, θα εξερευνήσετε περισσότερους τρόπους για να ταξινομήσετε αριθμητικά δεδομένα. Θα μάθετε επίσης για τις επιπτώσεις της επιλογής ενός ταξινομητή έναντι του άλλου.

## [Quiz πριν το μάθημα](https://ff-quizzes.netlify.app/en/ml/)

### Προαπαιτούμενο

Υποθέτουμε ότι έχετε ολοκληρώσει τα προηγούμενα μαθήματα και έχετε ένα καθαρισμένο σύνολο δεδομένων στον φάκελο `data` που ονομάζεται _cleaned_cuisines.csv_ στη ρίζα αυτού του φακέλου με τα 4 μαθήματα.

### Προετοιμασία

Έχουμε φορτώσει το αρχείο σας _notebook.ipynb_ με το καθαρισμένο σύνολο δεδομένων και το έχουμε χωρίσει σε πίνακες δεδομένων X και y, έτοιμους για τη διαδικασία δημιουργίας μοντέλου.

## Ένας χάρτης ταξινόμησης

Προηγουμένως, μάθατε για τις διάφορες επιλογές που έχετε κατά την ταξινόμηση δεδομένων χρησιμοποιώντας το cheat sheet της Microsoft. Το Scikit-learn προσφέρει ένα παρόμοιο, αλλά πιο λεπτομερές cheat sheet που μπορεί να βοηθήσει περαιτέρω στο να περιορίσετε τους εκτιμητές σας (άλλος όρος για τους ταξινομητές):

![ML Map from Scikit-learn](../../../../translated_images/el/map.e963a6a51349425a.webp)
> Συμβουλή: [επισκεφτείτε αυτόν τον χάρτη διαδικτυακά](https://scikit-learn.org/stable/tutorial/machine_learning_map/) και κάντε κλικ στο μονοπάτι για να διαβάσετε την τεκμηρίωση.

### Το σχέδιο

Αυτός ο χάρτης είναι πολύ χρήσιμος μόλις κατανοήσετε καλά τα δεδομένα σας, καθώς μπορείτε να 'περπατήσετε' κατά μήκος των διαδρομών του για μια απόφαση:

- Έχουμε >50 δείγματα
- Θέλουμε να προβλέψουμε μια κατηγορία
- Έχουμε ετικετοποιημένα δεδομένα
- Έχουμε λιγότερα από 100K δείγματα
- ✨ Μπορούμε να επιλέξουμε ένα Linear SVC
- Αν αυτό δεν δουλέψει, αφού έχουμε αριθμητικά δεδομένα
    - Μπορούμε να δοκιμάσουμε έναν ✨ KNeighbors Classifier
      - Αν αυτό δεν δουλέψει, δοκιμάστε ✨ SVC και ✨ Ensemble Classifiers

Αυτό είναι ένα πολύ χρήσιμο μονοπάτι για να ακολουθήσετε.

## Άσκηση - διαχωρίστε τα δεδομένα

Ακολουθώντας αυτό το μονοπάτι, πρέπει να ξεκινήσουμε εισάγοντας μερικές βιβλιοθήκες προς χρήση.

1. Εισάγετε τις απαραίτητες βιβλιοθήκες:

    ```python
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    import numpy as np
    ```

1. Διαχωρίστε τα δεδομένα εκπαίδευσης και ελέγχου σας:

    ```python
    X_train, X_test, y_train, y_test = train_test_split(cuisines_features_df, cuisines_label_df, test_size=0.3)
    ```

## Ταξινομητής Linear SVC

Η ομαδοποίηση Support-Vector (SVC) είναι ένας απόγονος της οικογένειας μηχανών Support-Vector για τεχνικές ML (μάθετε περισσότερα γι' αυτές παρακάτω). Σε αυτή τη μέθοδο, μπορείτε να επιλέξετε ένα 'kernel' για να αποφασίσετε πώς θα ομαδοποιηθούν οι ετικέτες. Η παράμετρος 'C' αναφέρεται στη 'κανονικοποίηση' που ρυθμίζει την επιρροή των παραμέτρων. Το kernel μπορεί να είναι ένα από [πολλά](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC· εδώ το ορίσαμε σε 'linear' για να διασφαλίσουμε ότι εκμεταλλευόμαστε το γραμμικό SVC. Η πιθανότητα έχει προεπιλογή το 'false'· εδώ την ορίσαμε σε 'true' για να συλλέξουμε εκτιμήσεις πιθανοτήτων. Ορίσαμε το random state σε '0' για να ανακατέψουμε τα δεδομένα ώστε να λάβουμε πιθανότητες.

### Άσκηση - εφαρμόστε ένα γραμμικό SVC

Ξεκινήστε δημιουργώντας έναν πίνακα ταξινομητών. Θα προσθέσετε σταδιακά σε αυτόν τον πίνακα καθώς δοκιμάζουμε.

1. Ξεκινήστε με ένα Linear SVC:

    ```python
    C = 10
    # Δημιουργήστε διάφορους ταξινομητές.
    classifiers = {
        'Linear SVC': SVC(kernel='linear', C=C, probability=True,random_state=0)
    }
    ```

2. Εκπαιδεύστε το μοντέλο σας χρησιμοποιώντας το Linear SVC και εκτυπώστε μια αναφορά:

    ```python
    n_classifiers = len(classifiers)
    
    for index, (name, classifier) in enumerate(classifiers.items()):
        classifier.fit(X_train, np.ravel(y_train))
    
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy (train) for %s: %0.1f%% " % (name, accuracy * 100))
        print(classification_report(y_test,y_pred))
    ```

    Το αποτέλεσμα είναι αρκετά καλό:

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

## Ταξινομητής K-Neighbors

Ο K-Neighbors είναι μέρος της οικογένειας μεθόδων "neighbors" για ML, που μπορούν να χρησιμοποιηθούν για επιβλεπόμενη και μη επιβλεπόμενη μάθηση. Σε αυτή τη μέθοδο δημιουργείται ένας προκαθορισμένος αριθμός σημείων και τα δεδομένα συγκεντρώνονται γύρω από αυτά τα σημεία έτσι ώστε να μπορούν να προβλεφθούν γενικευμένες ετικέτες για τα δεδομένα.

### Άσκηση - εφαρμόστε τον ταξινομητή K-Neighbors

Ο προηγούμενος ταξινομητής ήταν καλός και δούλεψε καλά με τα δεδομένα, αλλά ίσως μπορούμε να πετύχουμε καλύτερη ακρίβεια. Δοκιμάστε έναν ταξινομητή K-Neighbors.

1. Προσθέστε μια γραμμή στον πίνακα ταξινομητών σας (προσθέστε κόμμα μετά από το στοιχείο Linear SVC):

    ```python
    'KNN classifier': KNeighborsClassifier(C),
    ```

    Το αποτέλεσμα είναι λίγο χειρότερο:

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

    ✅ Μάθετε για τους [K-Neighbors](https://scikit-learn.org/stable/modules/neighbors.html#neighbors)

## Ταξινομητής Support Vector

Οι ταξινομητές Support-Vector είναι μέρος της οικογένειας [Support-Vector Machine](https://wikipedia.org/wiki/Support-vector_machine) τεχνικών ML που χρησιμοποιούνται για ταξινόμηση και εργασίες παλινδρόμησης. Τα SVM "χαρτογραφούν παραδείγματα εκπαίδευσης σε σημεία στο χώρο" για να μεγιστοποιήσουν την απόσταση μεταξύ δύο κατηγοριών. Μετέπειτα δεδομένα χαρτογραφούνται σε αυτόν τον χώρο ώστε να μπορεί να προβλεφθεί η κατηγορία τους.

### Άσκηση - εφαρμόστε έναν ταξινομητή Support Vector

Ας δοκιμάσουμε για λίγο καλύτερη ακρίβεια με έναν ταξινομητή Support Vector.

1. Προσθέστε κόμμα μετά το στοιχείο K-Neighbors και κατόπιν προσθέστε αυτή τη γραμμή:

    ```python
    'SVC': SVC(),
    ```

    Το αποτέλεσμα είναι αρκετά καλό!

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

    ✅ Μάθετε για τους [Support-Vectors](https://scikit-learn.org/stable/modules/svm.html#svm)

## Σύνολα ταξινομητών (Ensemble Classifiers)

Ας ακολουθήσουμε τη διαδρομή μέχρι το τέλος, παρόλο που το προηγούμενο τεστ ήταν αρκετά καλό. Ας δοκιμάσουμε κάποιους 'Ensemble Classifiers', συγκεκριμένα Random Forest και AdaBoost:

```python
  'RFST': RandomForestClassifier(n_estimators=100),
  'ADA': AdaBoostClassifier(n_estimators=100)
```

Το αποτέλεσμα είναι πολύ καλό, ειδικά για το Random Forest:

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

✅ Μάθετε για τους [Ensemble Classifiers](https://scikit-learn.org/stable/modules/ensemble.html)

Αυτή η μέθοδος Μηχανικής Μάθησης "συνδυάζει τις προβλέψεις πολλών βασικών εκτιμητών" για να βελτιώσει την ποιότητα του μοντέλου. Στο παράδειγμά μας, χρησιμοποιήσαμε τυχαία δέντρα (Random Trees) και AdaBoost.

- [Random Forest](https://scikit-learn.org/stable/modules/ensemble.html#forest), μια μέθοδος μέσης τιμής, κατασκευάζει ένα 'δάσος' από 'δέντρα αποφάσεων' με τυχαίο χαρακτήρα ώστε να αποφευχθεί η υπερεκπαίδευση. Η παράμετρος n_estimators ορίζεται στον αριθμό δέντρων.

- [AdaBoost](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html) προσαρμόζει έναν ταξινομητή σε ένα σύνολο δεδομένων και κατόπιν προσαρμόζει αντίγραφα αυτού του ταξινομητή στο ίδιο σύνολο δεδομένων. Επικεντρώνεται στα βάρη των εσφαλμένα ταξινομημένων στοιχείων και προσαρμόζει την εφαρμογή για τον επόμενο ταξινομητή για διόρθωση.

---

## 🚀Πρόκληση

Καθένα από αυτά τα τεχνικές έχει μεγάλο αριθμό παραμέτρων που μπορείτε να προσαρμόσετε. Ερευνήστε τις προεπιλεγμένες παραμέτρους της καθεμίας και σκεφτείτε τι σημαίνει για την ποιότητα του μοντέλου η τροποποίηση αυτών των παραμέτρων.

## [Quiz μετά το μάθημα](https://ff-quizzes.netlify.app/en/ml/)

## Ανασκόπηση & Αυτοδιδασκαλία

Υπάρχει πολύς όρος σε αυτά τα μαθήματα, οπότε αφιερώστε ένα λεπτό για να αναθεωρήσετε [αυτή τη λίστα](https://docs.microsoft.com/dotnet/machine-learning/resources/glossary?WT.mc_id=academic-77952-leestott) με χρήσιμες ορολογίες!

## Ανάθεση

[Παίξτε με τις παραμέτρους](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Αποποίηση ευθυνών**:  
Αυτό το έγγραφο έχει μεταφραστεί χρησιμοποιώντας την υπηρεσία αυτόματης μετάφρασης AI [Co-op Translator](https://github.com/Azure/co-op-translator). Παρότι επιδιώκουμε την ακρίβεια, παρακαλούμε να λάβετε υπόψη ότι οι αυτόματες μεταφράσεις μπορεί να περιέχουν λάθη ή ανακρίβειες. Το πρωτότυπο έγγραφο στη γλώσσα του θεωρείται η αυθεντική πηγή. Για κρίσιμες πληροφορίες, συνιστάται επαγγελματική μετάφραση από ανθρώπινο μεταφραστή. Δεν φέρουμε ευθύνη για οποιεσδήποτε παρεξηγήσεις ή λανθασμένες ερμηνείες που προκύπτουν από τη χρήση αυτής της μετάφρασης.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->