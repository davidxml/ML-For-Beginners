# Uvod v strojno učenje

## [Pred predavanjem kviz](https://ff-quizzes.netlify.app/en/ml/)

---

[![ML za začetnike - Uvod v strojno učenje za začetnike](https://img.youtube.com/vi/6mSx_KJxcHI/0.jpg)](https://youtu.be/6mSx_KJxcHI "ML za začetnike - Uvod v strojno učenje za začetnike")

> 🎥 Kliknite na zgornjo sliko za kratek video, ki obravnava to lekcijo.

Dobrodošli v tem tečaju o klasičnem strojnem učenju za začetnike! Ne glede na to, ali ste popolnoma novi na to temo ali izkušen izvajalec ML, ki želi osvežiti znanje v določenem področju, smo veseli, da ste se nam pridružili! Želimo ustvariti prijazno izhodišče za vaše študij strojnjega učenja in z veseljem bomo ocenili, odgovorili na in vključili vaše [povratne informacije](https://github.com/microsoft/ML-For-Beginners/discussions).

[![Uvod v ML](https://img.youtube.com/vi/h0e2HAPTGF4/0.jpg)](https://youtu.be/h0e2HAPTGF4 "Uvod v ML")

> 🎥 Kliknite na zgornjo sliko za video: John Guttag z MIT predstavlja strojno učenje

---
## Začetek s strojnim učenjem

Pred začetkom s tem učnim načrtom morate imeti svoj računalnik pripravljen in konfiguriran za lokalno izvajanje zvezkov.

- **Konfigurirajte svoj računalnik z temi videi**. Uporabite naslednje povezave, da se naučite [kako namestiti Python](https://youtu.be/CXZYvNRIAKM) v svoj sistem in [nastaviti urejevalnik besedil](https://youtu.be/EU8eayHWoZg) za razvoj.
- **Naučite se Pythona**. Priporočljivo je tudi imeti osnovno razumevanje [Pythona](https://docs.microsoft.com/learn/paths/python-language/?WT.mc_id=academic-77952-leestott), programskega jezika, ki je uporaben za podatkovne znanstvenike in ga uporabljamo v tem tečaju.
- **Naučite se Node.js in JavaScript**. V tem tečaju JavaScript uporabimo tudi nekajkrat pri gradnji spletnih aplikacij, zato boste morali imeti nameščen [node](https://nodejs.org) in [npm](https://www.npmjs.com/), prav tako pa je za razvoj v Pythonu in JavaScriptu priporočljiv [Visual Studio Code](https://code.visualstudio.com/).
- **Ustvarite GitHub račun**. Ker ste nas našli tukaj na [GitHubu](https://github.com), imate morda že račun, če ne, pa si ga ustvarite in nato odvežite ta učni načrt za svojo uporabo. (Lahko nam tudi daste zvezdico 😊)
- **Raziskujte Scikit-learn**. Spoznajte [Scikit-learn](https://scikit-learn.org/stable/user_guide.html), nabor knjižnic ML, na katere se sklicujemo v teh lekcijah.

---
## Kaj je strojno učenje?

Izraz 'strojno učenje' je eden najbolj priljubljenih in pogosto uporabljenih izrazov danes. Obstaja zelo velika verjetnost, da ste ta izraz zasledili vsaj enkrat, če imate kakršnokoli poznanstvo s tehnologijo, ne glede na to, v katerem področju delate. Mehanika strojnjega učenja pa je za večino ljudi skrivnost. Za začetnika strojnjega učenja je lahko ta tema včasih preobširna. Zato je pomembno razumeti, kaj strojno učenje dejansko je, in se z njim učiti korak za korakom, skozi praktične primere.

---
## Hipe krivulja

![ml hype curve](../../../../translated_images/sl/hype.07183d711a17aafe.webp)

> Google Trends prikazuje nedavno 'hype krivuljo' pojma 'strojno učenje'

---
## skrivnostno vesolje

Živimo v vesolju, polnem fascinantnih skrivnosti. Veliki znanstveniki, kot so Stephen Hawking, Albert Einstein in mnogi drugi, so svoj življenjski čas posvetili iskanju pomembnih informacij, ki razkrivajo skrivnosti sveta okoli nas. To je človeško stanje učenja: otrok se uči novih stvari in odkriva strukturo svojega sveta iz leta v leto, ko odrašča do odraslosti.

---
## Otrokov možgan

Otrokov možgan in čutila zaznavajo dejstva svojega okolja in postopoma spoznavajo skrite vzorce življenja, ki otroku pomagajo oblikovati logična pravila za prepoznavanje naučenih vzorcev. Proces učenja človeških možganov naredi ljudi najbolj dovršena živa bitja na tem svetu. Nenehno učenje s odkrivanjem skritih vzorcev in nato inoviranje na teh vzorcih nam omogoča, da se čez celo življenje izboljšujemo. Ta sposobnost učenja in razvijajoča se zmogljivost sta povezana s konceptom, imenovanim [plastičnost možganov](https://www.simplypsychology.org/brain-plasticity.html). Površinsko lahko najdemo nekaj motivacijskih podobnosti med procesom učenja človeških možganov in koncepti strojnjega učenja.

---
## Človeški možgani

[Človeški možgani](https://www.livescience.com/29365-human-brain.html) zaznavajo stvari iz resničnega sveta, obdelujejo zaznane informacije, sprejemajo racionalne odločitve in izvajajo določena dejanja glede na okoliščine. To imenujemo inteligentno vedenje. Ko stroj programiramo s simulacijo inteligentnega vedenjskega procesa, temu pravimo umetna inteligenca (UI).

---
## Nekateri izrazi

Čeprav se izrazi lahko zamenjujejo, je strojno učenje (ML) pomemben podskup umetne inteligence. **ML se ukvarja z uporabo specializiranih algoritmov za odkrivanje pomembnih informacij in iskanje skritih vzorcev iz zaznanih podatkov za podporo racionalnemu procesu odločanja**.

---
## UI, ML, globoko učenje

![AI, ML, deep learning, data science](../../../../translated_images/sl/ai-ml-ds.537ea441b124ebf6.webp)

> Diagram, ki prikazuje odnose med UI, ML, globokim učenjem in podatkovno znanostjo. Infografika avtorice [Jen Looper](https://twitter.com/jenlooper) po vzoru [te grafike](https://softwareengineering.stackexchange.com/questions/366996/distinction-between-ai-ml-neural-networks-deep-learning-and-data-mining)

---
## Koncepti, ki jih bomo obravnavali

V tem učnem načrtu bomo pokrili le osnovne koncepte strojnjega učenja, ki jih mora začetnik poznati. Obravnavamo tisto, kar imenujemo 'klasično strojno učenje', predvsem s pomočjo Scikit-learn, odlične knjižnice, ki jo mnogi študentje uporabljajo za učenje osnov. Za razumevanje širših konceptov umetne inteligence ali globokega učenja je trdno temeljno znanje strojnjega učenja nepogrešljivo, zato vam ga želimo tukaj ponuditi.

---
## V tem tečaju boste spoznali:

- osnovne koncepte strojnjega učenja
- zgodovino ML
- ML in pravičnost
- regresijske tehnike ML
- klasifikacijske tehnike ML
- tehnike združevanja (clustering) ML
- tehnike obdelave naravnega jezika ML
- tehnike napovedovanja časovnih vrst ML
- krepitveno učenje
- praktične uporabe ML

---
## Česa ne bomo obravnavali

- globoko učenje
- nevronske mreže
- UI

Za boljšo izkušnjo učenja se bomo izognili zapletenostim nevronskih mrež, 'globokemu učenju' – večplastnemu modeliranju z nevronskimi mrežami – in UI, o katerih bomo govorili v drugem učnem načrtu. Prav tako bomo ponudili prihajajoči učni načrt podatkovne znanosti, ki bo osredotočen na ta del širšega področja.

---
## Zakaj študirati strojno učenje?

Strojno učenje je z vidika sistemov definirano kot ustvarjanje avtomatiziranih sistemov, ki se lahko učijo skrite vzorce iz podatkov za pomoč pri sprejemanju inteligentnih odločitev.

Ta motivacija je v grobem navdihnjena s tem, kako se človeški možgani učijo določenih stvari na podlagi podatkov, ki jih zaznajo iz zunanjega sveta.

✅ Za minuto premislite, zakaj bi podjetje želelo uporabiti strategije strojnjega učenja namesto ustvarjanja sistema z vnaprej določenimi pravili.

---
## Zakaj je pomembna kakovost podatkov

Podatki visoke kakovosti izboljšajo delovanje modela. Slabi ali šumni podatki lahko vodijo do netočnih napovedi, tudi pri uporabi naprednih algoritmov strojnjega učenja.

---
## Uporabe strojnjega učenja

Uporabe strojnjega učenja so danes skoraj povsod in so tako razširjene kot tudi podatki, ki se pretakajo skozi naše družbe, ustvarjeni z našimi pametnimi telefoni, povezanimi napravami in drugimi sistemi. Glede na izjemen potencial najnovejših algoritmov strojnjega učenja raziskovalci preučujejo njihovo zmožnost reševanja večdimenzionalnih in večdisciplinarnih resničnih problemskih izzivov z odličnimi pozitivnimi rezultati.

---
## Primeri uporabljenega ML

**Strojno učenje lahko uporabite na več načinov**:

- Za napovedovanje verjetnosti bolezni na podlagi zgodovine ali poročil pacienta.
- Za uporabo vremenskih podatkov za napovedovanje vremenskih pojavov.
- Za razumevanje sentimenta v besedilu.
- Za odkrivanje lažnih novic in preprečevanje širjenja propagande.

Finance, ekonomija, znanost o Zemlji, raziskovanje vesolja, biomedicinsko inženirstvo, kognitivna znanost in celo področja humanistike so sprejela strojno učenje za reševanje zahtevnih, podatkovno intenzivnih problemov na svojem področju.

---
## Zaključek

Strojno učenje avtomatizira proces odkrivanja vzorcev z iskanjem pomembnih vpogledov iz resničnih ali generiranih podatkov. Dokazalo se je kot zelo dragoceno v poslovnih, zdravstvenih in finančnih aplikacijah med drugimi.

V bližnji prihodnosti bo razumevanje osnov strojnjega učenja nujno za ljudi iz vseh področij zaradi njegove široke uporabe.

---
# 🚀 Izziv

Na papir narišite ali s pomočjo spletne aplikacije, kot je [Excalidraw](https://excalidraw.com/), predstavite svoje razumevanje razlik med UI, ML, globokim učenjem in podatkovno znanostjo. Dodajte nekaj idej o problemih, ki jih vsaka od teh tehnik dobro rešuje.

# [Kviz po predavanju](https://ff-quizzes.netlify.app/en/ml/)

---
# Pregled & Samostojno učenje

Če želite izvedeti več o tem, kako lahko delate z ML algoritmi v oblaku, sledite temu [učnemu poti](https://docs.microsoft.com/learn/paths/create-no-code-predictive-models-azure-machine-learning/?WT.mc_id=academic-77952-leestott).

Opravite [učni pot](https://docs.microsoft.com/learn/modules/introduction-to-machine-learning/?WT.mc_id=academic-77952-leestott) o osnovah ML.

---
# Naloga

[Zaženite se](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Omejitev odgovornosti**:
Ta dokument je bil preveden z uporabo AI prevajalske storitve [Co-op Translator](https://github.com/Azure/co-op-translator). Čeprav si prizadevamo za natančnost, vas prosimo, da upoštevate, da avtomatizirani prevodi lahko vsebujejo napake ali netočnosti. Izvirni dokument v njegovem izvirnem jeziku je treba obravnavati kot avtoritativni vir. Za kritične informacije je priporočljiv strokovni človeški prevod. Ne odgovarjamo za morebitna nesporazume ali napačne interpretacije, ki izhajajo iz uporabe tega prevoda.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->