# Uvod u strojno učenje

## [Pre-predavanje kviz](https://ff-quizzes.netlify.app/en/ml/)

---

[![ML za početnike - Uvod u strojno učenje za početnike](https://img.youtube.com/vi/6mSx_KJxcHI/0.jpg)](https://youtu.be/6mSx_KJxcHI "ML za početnike - Uvod u strojno učenje za početnike")

> 🎥 Kliknite na gornju sliku za kratki video kroz ovaj lekciju.

Dobrodošli na ovaj tečaj klasičnog strojnog učenja za početnike! Bilo da ste potpuno novi u ovoj temi ili iskusni praktičar ML-a koji želi obnoviti znanje u nekom području, drago nam je što ste nam se pridružili! Želimo stvoriti prijateljsko polazište za vaše proučavanje ML-a i rado ćemo procijeniti, odgovoriti i uključiti vaše [povratne informacije](https://github.com/microsoft/ML-For-Beginners/discussions).

[![Uvod u ML](https://img.youtube.com/vi/h0e2HAPTGF4/0.jpg)](https://youtu.be/h0e2HAPTGF4 "Uvod u ML")

> 🎥 Kliknite na gornju sliku za video: John Guttag s MIT-a uvodi u strojno učenje

---
## Početak sa strojnim učenjem

Prije nego započnete s ovim kurikulumom, trebate imati svoj računar postavljen i spreman za pokretanje bilješki lokalno.

- **Konfigurirajte svoje računalo uz ove videozapise**. Koristite sljedeće poveznice kako biste naučili [kako instalirati Python](https://youtu.be/CXZYvNRIAKM) na svoj sustav i [postaviti uređivač teksta](https://youtu.be/EU8eayHWoZg) za razvoj.
- **Naučite Python**. Također je preporučljivo imati osnovno razumijevanje [Pythona](https://docs.microsoft.com/learn/paths/python-language/?WT.mc_id=academic-77952-leestott), programski jezik koristan za znanstvenike podataka koji koristimo u ovom tečaju.
- **Naučite Node.js i JavaScript**. Također koristimo JavaScript nekoliko puta u ovom tečaju prilikom izrade web aplikacija, pa ćete trebati imati instalirane [node](https://nodejs.org) i [npm](https://www.npmjs.com/), kao i [Visual Studio Code](https://code.visualstudio.com/) dostupan za razvoj u Pythonu i JavaScriptu.
- **Kreirajte GitHub račun**. Kako ste nas pronašli ovdje na [GitHubu](https://github.com), možda već imate račun, ali ako nemate, napravite ga i zatim forkajte ovaj kurikulum za vlastitu upotrebu. (Slobodno nam i dodajte zvjezdicu 😊)
- **Istražite Scikit-learn**. Upoznajte se s [Scikit-learn](https://scikit-learn.org/stable/user_guide.html), skupom ML biblioteka na koje se pozivamo u ovim lekcijama.

---
## Što je strojno učenje?

Pojam 'strojno učenje' jedan je od najpopularnijih i najčešće korištenih pojmova danas. Postoji značajna vjerojatnost da ste ovaj pojam barem jednom čuli ako imate ikakvu povezanost s tehnologijom, bez obzira u kojem području radili. Međutim, mehanika strojnog učenja većini ljudi ostaje tajna. Za početnika u strojnome učenju, predmet ponekad može djelovati zastrašujuće. Stoga je važno razumjeti što strojno učenje zapravo jest te ga učiti korak po korak, kroz praktične primjere.

---
## Hype krivulja

![ml hype curve](../../../../translated_images/hr/hype.07183d711a17aafe.webp)

> Google Trends prikazuje nedavnu 'hype krivulju' termina 'strojno učenje'

---
## Tajanstveni svemir

Živimo u svemiru punom fascinantnih misterija. Veliki znanstvenici poput Stephena Hawkinga, Alberta Einsteina i mnogih drugih posvetili su svoje živote traženju značajnih informacija koje otkrivaju misterije svijeta oko nas. Ovo je ljudsko stanje učenja: ljudsko dijete uči nove stvari i otkriva strukturu svog svijeta iz godine u godinu dok odrasta.

---
## Mozak djeteta

Mozak i osjetila djeteta percipiraju činjenice svog okruženja i postupno uče skrivene obrasce života koji pomažu djetetu da oblikuje logička pravila za prepoznavanje naučenih obrazaca. Proces učenja ljudskog mozga čini ljude najsloženijim živim bićem na ovom svijetu. Kontinuirano učenje kroz otkrivanje skrivenih obrazaca i zatim inoviranje na tim obrascima omogućuje nam da se tijekom života neprestano usavršavamo. Ova sposobnost učenja i evoluirajuća mogućnost povezani su s pojmom zvanim [plastičnost mozga](https://www.simplypsychology.org/brain-plasticity.html). Površno gledano, možemo izvući motivacijske sličnosti između procesa učenja ljudskog mozga i koncepta strojnog učenja.

---
## Ljudski mozak

[Čovjekov mozak](https://www.livescience.com/29365-human-brain.html) percipira stvari iz stvarnog svijeta, obrađuje primljene informacije, donosi racionalne odluke i izvodi određene radnje temeljem okolnosti. To nazivamo inteligentnim ponašanjem. Kada programiramo sličan proces inteligentnog ponašanja na stroju, to se naziva umjetna inteligencija (AI).

---
## Neka terminologija

Iako se pojmovi mogu brkati, strojno učenje (ML) važan je podskup umjetne inteligencije. **ML se bavi korištenjem specijaliziranih algoritama za otkrivanje značajnih informacija i pronalaženje skrivenih obrazaca iz primljenih podataka kako bi se potkrijepio racionalni proces donošenja odluka**.

---
## AI, ML, duboko učenje

![AI, ML, deep learning, data science](../../../../translated_images/hr/ai-ml-ds.537ea441b124ebf6.webp)

> Dijagram koji prikazuje odnose između AI, ML, dubokog učenja i znanosti o podacima. Infografika autorice [Jen Looper](https://twitter.com/jenlooper) inspirirana [ovim grafičkim prikazom](https://softwareengineering.stackexchange.com/questions/366996/distinction-between-ai-ml-neural-networks-deep-learning-and-data-mining)

---
## Koncepti koje ćemo obraditi

U ovom kurikulumu pokrit ćemo samo osnovne koncepte strojnog učenja koje svaki početnik mora znati. Pokrivamo ono što nazivamo 'klasičnim strojnim učenjem' prvenstveno koristeći Scikit-learn, izvrsnu biblioteku koju mnogi studenti koriste za učenje osnova. Da bismo razumjeli šire koncepte umjetne inteligencije ili dubokog učenja, snažno temeljno znanje strojnog učenja je neizostavno, pa ga ovdje želimo ponuditi.

---
## U ovom tečaju ćete naučiti:

- osnovne koncepte strojnog učenja
- povijest ML-a
- ML i pravednost
- tehnike regresije u ML-u
- tehnike klasifikacije u ML-u
- tehnike grupiranja u ML-u
- tehnike obrade prirodnog jezika u ML-u
- tehnike vremenskog predviđanja u ML-u
- učenje s pojačanjem
- primjenu ML-a u stvarnom svijetu

---
## Što nećemo obraditi

- duboko učenje
- neuronske mreže
- AI

Da bismo omogućili bolje iskustvo učenja, izbjegavat ćemo složenosti neuronskih mreža, 'dubokog učenja' – višeslojno modeliranje korištenjem neuronskih mreža – i AI, o kojima ćemo govoriti u drugom kurikulumu. Također ćemo uskoro ponuditi kurikulum iz znanosti o podacima koji se fokusira na taj aspekt ovog šireg područja.

---
## Zašto učiti strojno učenje?

Strojno učenje, iz perspektive sustava, definira se kao stvaranje automatiziranih sustava koji mogu učiti skrivene obrasce iz podataka kako bi pomogli u donošenju inteligentnih odluka.

Ova motivacija je labavo inspirirana načinom na koji ljudski mozak uči određene stvari na temelju podataka koje prima iz vanjskog svijeta.

✅ Razmislite na trenutak zašto bi poslovanje željelo koristiti strategije strojnog učenja umjesto stvaranja strogo kodiranog sustava pravila.

---
## Zašto je kvaliteta podataka važna

Visokokvalitetni podaci poboljšavaju performanse modela. Loši ili šumni podaci mogu dovesti do netočnih predviđanja, čak i kada se koriste napredni algoritmi strojnog učenja.

---
## Primjene strojnog učenja

Primjene strojnog učenja danas su gotovo svugdje i jednako su raširene kao i podaci koji kruže našim društvima, generirani naših pametnim telefonima, povezanim uređajima i ostalim sustavima. Imajući u vidu golem potencijal najmodernijih algoritama strojnog učenja, istraživači su istraživali njihove mogućnosti rješavanja višedimenzionalnih i multidisciplinarnih problema iz stvarnog života s velikim pozitivnim rezultatima.

---
## Primjeri primijenjenog ML-a

**Možete koristiti strojno učenje na mnoge načine**:

- Za predviđanje vjerojatnosti bolesti iz povijesti bolesti ili izvještaja pacijenta.
- Za korištenje vremenskih podataka za predviđanje vremenskih događaja.
- Za razumijevanje sentimenta teksta.
- Za otkrivanje lažnih vijesti kako bi se zaustavilo širenje propagande.

Financije, ekonomija, znanost o Zemlji, istraživanje svemira, biomedicinsko inženjerstvo, kognitivne znanosti, pa čak i humanistička područja prilagodila su strojno učenje kako bi rješavala zahtjevne probleme obrade podataka svojeg područja.

---
## Zaključak

Strojno učenje automatizira proces otkrivanja obrazaca pronalaskom značajnih uvida iz stvarnih ili generiranih podataka. Pokazalo se kao vrlo vrijedno u poslovanju, zdravstvu i financijama, među ostalim područjima.

U bliskoj budućnosti, razumijevanje osnova strojnog učenja postat će obavezno za ljude iz bilo kojeg područja zbog njegove široke primjene.

---
# 🚀 Izazov

Nacrtajte, na papiru ili koristeći online aplikaciju poput [Excalidraw](https://excalidraw.com/), svoje razumijevanje razlika između AI, ML, dubokog učenja i znanosti o podacima. Dodajte neke ideje o problemima koje je svaka od ovih tehnika dobra u rješavanju.

# [Posljednji kviz](https://ff-quizzes.netlify.app/en/ml/)

---
# Pregled i samostalno učenje

Da biste saznali više o tome kako raditi s ML algoritmima u oblaku, slijedite ovaj [put učenja](https://docs.microsoft.com/learn/paths/create-no-code-predictive-models-azure-machine-learning/?WT.mc_id=academic-77952-leestott).

Pohađajte [put učenja](https://docs.microsoft.com/learn/modules/introduction-to-machine-learning/?WT.mc_id=academic-77952-leestott) o osnovama ML-a.

---
# Zadatak

[Pokrenite se](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Napomena**:
Ovaj dokument je preveden korištenjem AI prevoditeljskog servisa [Co-op Translator](https://github.com/Azure/co-op-translator). Iako težimo točnosti, imajte na umu da automatski prijevodi mogu sadržavati greške ili netočnosti. Izvorni dokument na izvornom jeziku treba smatrati autoritativnim izvorom. Za važne informacije preporuča se profesionalni ljudski prijevod. Nismo odgovorni za bilo kakva nesporazumevanja ili pogrešne interpretacije koje proizlaze iz korištenja ovog prijevoda.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->