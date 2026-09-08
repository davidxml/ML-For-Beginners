# Úvod do strojového učenia

## [Prednáškový kvíz](https://ff-quizzes.netlify.app/en/ml/)

---

[![ML pre začiatočníkov - Úvod do strojového učenia pre začiatočníkov](https://img.youtube.com/vi/6mSx_KJxcHI/0.jpg)](https://youtu.be/6mSx_KJxcHI "ML pre začiatočníkov - Úvod do strojového učenia pre začiatočníkov")

> 🎥 Kliknite na obrázok vyššie pre krátke video, ktoré prechádza touto lekciou.

Vitajte v tomto kurze klasického strojového učenia pre začiatočníkov! Či už ste úplne noví v tejto oblasti, alebo skúsený odborník na ML, ktorý chce zdokonaliť svoje znalosti, radi vás tu máme! Chceme vytvoriť priateľské štartovacie miesto pre vaše štúdium ML a budeme radi, keď nám poskytnete, zareagujeme na a začleníme vaše [spätné väzby](https://github.com/microsoft/ML-For-Beginners/discussions).

[![Úvod do ML](https://img.youtube.com/vi/h0e2HAPTGF4/0.jpg)](https://youtu.be/h0e2HAPTGF4 "Úvod do ML")

> 🎥 Kliknite na obrázok vyššie pre video: John Guttag z MIT predstavuje strojové učenie

---
## Začíname so strojovým učením

Pred začatím tohto učebného plánu je potrebné mať počítač nastavený a pripravený na spúšťanie notebookov lokálne.

- **Nakonfigurujte svoj počítač pomocou týchto videí**. Použite nasledujúce odkazy na naučenie sa [ako nainštalovať Python](https://youtu.be/CXZYvNRIAKM) vo vašom systéme a [nastaviť textový editor](https://youtu.be/EU8eayHWoZg) pre vývoj.
- **Naučte sa Python**. Tiež sa odporúča mať základné znalosti [Pythonu](https://docs.microsoft.com/learn/paths/python-language/?WT.mc_id=academic-77952-leestott), programovacieho jazyka užitočného pre dátových vedcov, ktorý v tomto kurze používame.
- **Naučte sa Node.js a JavaScript**. JavaScript používame niekoľkokrát v tomto kurze pri tvorbe webových aplikácií, takže budete potrebovať mať nainštalované [node](https://nodejs.org) a [npm](https://www.npmjs.com/), ako aj [Visual Studio Code](https://code.visualstudio.com/) dostupné pre vývoj v Pythone aj JavaScripte.
- **Vytvorte si účet na GitHub-e**. Keďže nás našli tu na [GitHub-e](https://github.com), možno už účet máte, ak nie, vytvorte si ho a potom si tento učebný plán forknete, aby ste ho mohli použiť sami. (Neváhajte nám tiež dať hviezdičku 😊)
- **Preskúmajte Scikit-learn**. Oboznámte sa so [Scikit-learn](https://scikit-learn.org/stable/user_guide.html), súborom ML knižníc, na ktoré sa v týchto lekciách odkazujeme.

---
## Čo je to strojové učenie?

Termín „strojové učenie“ je dnes jedným z najpopulárnejších a najčastejšie používaných výrazov. Existuje značná pravdepodobnosť, že ste tento termín počuli aspoň raz, ak máte akúkoľvek znalosť o technológiách, nezávisle od toho, v akej oblasti pracujete. Mechanizmy strojového učenia však väčšine ľudí zostávajú záhadou. Pre začiatočníka v strojovom učení môže byť táto téma niekedy ohromujúca. Preto je dôležité pochopiť, čo vlastne strojové učenie je, a učiť sa o ňom krok za krokom, prostredníctvom praktických príkladov.

---
## Krivka hypu

![ml hype curve](../../../../translated_images/sk/hype.07183d711a17aafe.webp)

> Google Trends ukazuje nedávnu „krivku hypu“ termínu „strojové učenie“

---
## Záhadný vesmír

Žijeme vo vesmíre plnom fascinujúcich záhad. Veľkí vedci ako Stephen Hawking, Albert Einstein a mnohí ďalší zasvätili svoj život hľadaniu významných informácií, ktoré odhaľujú záhady sveta okolo nás. Toto je ľudský stav učenia: ľudské dieťa sa učí nové veci a postupne objavuje štruktúru svojho sveta rok za rokom, keď rastie do dospelosti.

---
## Detský mozog

Detský mozog a zmysly vnímajú fakty o svojom okolí a postupne sa učia skryté vzory života, ktoré dieťaťu pomáhajú vytvárať logické pravidlá na rozpoznávanie naučených vzorov. Proces učenia ľudského mozgu robí z ľudí najsofistikovanejší živý tvor na tejto planéte. Učenie sa neustále objavovaním skrytých vzorov a ich následným zdokonaľovaním nám umožňuje robiť sa lepšími počas celého života. Táto schopnosť učenia a neustále sa vyvíjajúca kapacita súvisí s konceptom nazývaným [plasticita mozgu](https://www.simplypsychology.org/brain-plasticity.html). Povrchovo môžeme nájsť niektoré motivačné podobnosti medzi procesom učenia sa ľudského mozgu a konceptmi strojového učenia.

---
## Ľudský mozog

[Ľudský mozog](https://www.livescience.com/29365-human-brain.html) vníma veci z reálneho sveta, spracováva vnímané informácie, robí racionálne rozhodnutia a vykonáva určité akcie na základe okolností. Toto nazývame inteligentným správaním. Keď programujeme podobu tohto inteligentného správania na stroj, nazývame to umelou inteligenciou (AI).

---
## Niektorá terminológia

Hoci sa termíny môžu zamieňať, strojové učenie (ML) je dôležitou podmnožinou umelej inteligencie. **ML sa zaoberá používaním špecializovaných algoritmov na odhaľovanie významných informácií a hľadanie skrytých vzorov z vnímaných dát, aby podporilo racionálny rozhodovací proces**.

---
## AI, ML, hlboké učenie

![AI, ML, deep learning, data science](../../../../translated_images/sk/ai-ml-ds.537ea441b124ebf6.webp)

> Diagram zobrazujúci vzťahy medzi AI, ML, hlbokým učením a dátovou vedou. Infografika od [Jen Looper](https://twitter.com/jenlooper) inšpirovaná [týmto grafom](https://softwareengineering.stackexchange.com/questions/366996/distinction-between-ai-ml-neural-networks-deep-learning-and-data-mining)

---
## Koncepty, ktoré pokryjeme

V tomto učebnom pláne pokryjeme len základné koncepty strojového učenia, ktoré musí začiatočník poznať. Pokrývame to, čo nazývame 'klasické strojové učenie' primárne pomocou Scikit-learn, vynikajúcej knižnice, ktorú používa mnoho študentov na osvojenie si základov. Na pochopenie širších konceptov umelej inteligencie alebo hlbokého učenia je nevyhnutné mať silné základy v strojovom učení, a preto vám ich tu chceme poskytnúť.

---
## V tomto kurze sa naučíte:

- základné koncepty strojového učenia
- históriu ML
- ML a spravodlivosť
- techniky regresie v ML
- techniky klasifikácie v ML
- techniky zhlukovania v ML
- techniky spracovania prirodzeného jazyka v ML
- techniky predikcie časových radov v ML
- posilňovacie učenie
- reálne aplikácie ML

---
## Čo nebudeme pokrývať

- hlboké učenie
- neurónové siete
- AI

Pre lepší zážitok z učenia sa vyhneme zložitostiam neurónových sietí, „hlbokého učenia“ – viacvrstvového budovania modelov pomocou neurónových sietí – a AI, o ktorej budeme diskutovať v inom kurze. Tiež pripravujeme nadchádzajúci kurz dátovej vedy, ktorý sa zameria na túto oblasť väčšieho odboru.

---
## Prečo študovať strojové učenie?

Strojové učenie je z pohľadu systémov definované ako tvorba automatizovaných systémov, ktoré sa dokážu učiť skryté vzory z dát, aby pomohli robiť inteligentné rozhodnutia.

Táto motivácia je voľne inšpirovaná tým, ako sa ľudský mozog učí určité veci na základe dát, ktoré vníma z vonkajšieho sveta.

✅ Premyslite si na chvíľu, prečo by podnik chcel použiť stratégie strojového učenia namiesto vytvárania systému založeného na pevne zakódovaných pravidlách.

---
## Prečo je kvalita dát dôležitá

Vysokokvalitné dáta zlepšujú výkon modelu. Slabé alebo šumové dáta môžu viesť k nepresným predikciám, aj keď sa používajú pokročilé algoritmy strojového učenia.

---
## Aplikácie strojového učenia

Aplikácie strojového učenia sú dnes takmer všade, rovnako bežné ako dáta plynúce v našich spoločnostiach, generované našimi smartfónmi, pripojenými zariadeniami a ďalšími systémami. S ohľadom na obrovský potenciál najmodernejších algoritmov strojového učenia výskumníci skúmajú ich schopnosť riešiť viacrozmerné a multidisciplinárne reálne problémy s veľmi pozitívnymi výsledkami.

---
## Príklady aplikovaného ML

**Strojové učenie môžete využívať mnohými spôsobmi**:

- Predpovedať pravdepodobnosť ochorenia na základe lekárskej histórie alebo správ pacienta.
- Využiť meteorologické dáta na predikciu poveternostných udalostí.
- Pochopiť sentiment textu.
- Detegovať falošné správy, aby sa zastavilo šírenie propagandy.

Financie, ekonómia, veda o Zemi, vesmírny výskum, biomedicínske inžinierstvo, kognitívna veda a dokonca aj humanitné odbory prijali strojové učenie na riešenie zložitých, dátovo náročných problémov svojho odboru.

---
## Záver

Strojové učenie automatizuje proces objavovania vzorov nájdením významných poznatkov z reálnych alebo generovaných dát. Preukázalo svoju vysokú hodnotu v podnikaní, zdravotníctve a finančných aplikáciách, medzi inými.

V blízkej budúcnosti bude znalosť základov strojového učenia nevyhnutná pre ľudí z akéhokoľvek odboru vzhľadom na jeho široké prijatie.

---
# 🚀 Výzva

Nakreslite na papieri alebo pomocou online aplikácie ako [Excalidraw](https://excalidraw.com/) svoje pochopenie rozdielov medzi AI, ML, hlbokým učením a dátovou vedou. Pridajte niekoľko nápadov na problémy, ktoré je každá z týchto techník schopná dobre riešiť.

# [Poprednáškový kvíz](https://ff-quizzes.netlify.app/en/ml/)

---
# Prehľad & samostatné štúdium

Ak sa chcete dozvedieť viac o tom, ako môžete pracovať s ML algoritmami v cloude, sledujte tento [Learning Path](https://docs.microsoft.com/learn/paths/create-no-code-predictive-models-azure-machine-learning/?WT.mc_id=academic-77952-leestott).

Absolvujte [Learning Path](https://docs.microsoft.com/learn/modules/introduction-to-machine-learning/?WT.mc_id=academic-77952-leestott) o základoch ML.

---
# Zadanie

[Začnite a rozbiehajte sa](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Vyhlásenie o zodpovednosti**:
Tento dokument bol preložený pomocou AI prekladateľskej služby [Co-op Translator](https://github.com/Azure/co-op-translator). Hoci sa snažíme o presnosť, vezmite prosím na vedomie, že automatické preklady môžu obsahovať chyby alebo nepresnosti. Pôvodný dokument v jeho natívnom jazyku by mal byť považovaný za autoritatívny zdroj. Pre kritické informácie sa odporúča profesionálny ľudský preklad. Nie sme zodpovední za žiadne nedorozumenia alebo nesprávne interpretácie vyplývajúce z použitia tohto prekladu.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->