# Úvod do strojového učení

## [Přednáškový kvíz](https://ff-quizzes.netlify.app/en/ml/)

---

[![ML for beginners - Introduction to Machine Learning for Beginners](https://img.youtube.com/vi/6mSx_KJxcHI/0.jpg)](https://youtu.be/6mSx_KJxcHI "ML for beginners - Introduction to Machine Learning for Beginners")

> 🎥 Klikněte na obrázek výše pro krátké video, které prochází touto lekcí.

Vítejte v tomto kurzu klasického strojového učení pro začátečníky! Ať už jste na toto téma zcela noví, nebo jste zkušený praktik ML, který si chce zopakovat určitou oblast, jsme rádi, že jste s námi! Chceme vytvořit přátelské místo pro vaše studium ML a rádi vyhodnotíme, odpovíme a zapracujeme vaše [zpětné vazby](https://github.com/microsoft/ML-For-Beginners/discussions).

[![Introduction to ML](https://img.youtube.com/vi/h0e2HAPTGF4/0.jpg)](https://youtu.be/h0e2HAPTGF4 "Introduction to ML")

> 🎥 Klikněte na obrázek výše pro video: John Guttag z MIT představuje strojové učení

---
## Začínáme se strojovým učením

Než začnete s tímto kurikulem, musíte mít připravený počítač a mít možnost spouštět poznámkové bloky lokálně.

- **Nakonfigurujte svůj počítač podle těchto videí**. Použijte následující odkazy k naučení [jak nainstalovat Python](https://youtu.be/CXZYvNRIAKM) do vašeho systému a [nastavit textový editor](https://youtu.be/EU8eayHWoZg) pro vývoj.
- **Naučte se Python**. Doporučujeme také mít základní znalosti [Pythonu](https://docs.microsoft.com/learn/paths/python-language/?WT.mc_id=academic-77952-leestott), programovacího jazyka užitečného pro datové vědce, který používáme v tomto kurzu.
- **Naučte se Node.js a JavaScript**. V kurzu používáme JavaScript několikrát při tvorbě webových aplikací, takže budete potřebovat mít nainstalované [node](https://nodejs.org) a [npm](https://www.npmjs.com/), stejně jako [Visual Studio Code](https://code.visualstudio.com/) pro vývoj v Pythonu i JavaScriptu.
- **Vytvořte si účet na GitHubu**. Pokud nás našli zde na [GitHubu](https://github.com), možná již účet máte, ale pokud ne, založte si ho a potom si tento kurz naklonujte (forkněte) pro vlastní použití. (Klidně nám také dejte hvězdičku 😊)
- **Prozkoumejte Scikit-learn**. Seznamte se s [Scikit-learn](https://scikit-learn.org/stable/user_guide.html), sadou knihoven pro ML, na které se v lekcích odkazujeme.

---
## Co je strojové učení?

Termín „strojové učení“ je jedním z nejpopulárnějších a nejčastěji používaných termínů dneška. Existuje velká pravděpodobnost, že jste tento termín slyšeli alespoň jednou, pokud máte nějaké povědomí o technologii, ať už pracujete v jakékoli oblasti. Mechanika strojového učení je však pro většinu lidí záhadou. Pro začátečníka ve strojovém učení může být toto téma někdy ohromující. Proto je důležité pochopit, co vlastně strojové učení je, a učit se o něm krok za krokem prostřednictvím praktických příkladů.

---
## Křivka hype

![ml hype curve](../../../../translated_images/cs/hype.07183d711a17aafe.webp)

> Google Trends ukazuje aktuální 'hype křivku' termínu „strojové učení“

---
## Tajemný vesmír

Žijeme ve vesmíru plném fascinujících tajemství. Velcí vědci jako Stephen Hawking, Albert Einstein a mnozí další věnovali svůj život hledání smysluplných informací, které odkrývají záhady světa kolem nás. To je lidský stav učení: lidské dítě se učí nové věci a odkrývá strukturu svého světa rok za rokem, jak roste do dospělosti.

---
## Mozek dítěte

Mozek dítěte a jeho smysly vnímají fakta svého okolí a postupně se učí skryté vzory života, které dítěti pomáhají vytvořit logická pravidla k rozpoznávání naučených vzorů. Proces učení lidského mozku dělá z lidí nejsložitější živý tvor na tomto světě. Neustálým učením se pomocí odhalování skrytých vzorů a jejich následnou inovací si můžeme v průběhu života stále zlepšovat sebe sama. Tato schopnost učení a vyvíjející se kapacita souvisí s konceptem nazývaným [plasticita mozku](https://www.simplypsychology.org/brain-plasticity.html). Povrchně lze najít některé motivační podobnosti mezi procesem učení lidského mozku a koncepty strojového učení.

---
## Lidský mozek

[Lidský mozek](https://www.livescience.com/29365-human-brain.html) vnímá věci ze skutečného světa, zpracovává získané informace, učiní racionální rozhodnutí a na základě okolností vykonává určité akce. Tomu říkáme inteligentní chování. Když naprogramujeme stroj, který napodobuje tento inteligentní proces chování, nazývá se to umělá inteligence (AI).

---
## Některá terminologie

Ačkoli se termíny mohou zaměňovat, strojové učení (ML) je důležitou podmnožinou umělé inteligence. **ML se zabývá používáním specializovaných algoritmů k odhalování smysluplných informací a nalezení skrytých vzorů v získaných datech na podporu racionálního rozhodovacího procesu**.

---
## AI, ML, hluboké učení

![AI, ML, deep learning, data science](../../../../translated_images/cs/ai-ml-ds.537ea441b124ebf6.webp)

> Diagram ukazující vztahy mezi AI, ML, hlubokým učením a datovou vědou. Infografika od [Jen Looper](https://twitter.com/jenlooper) inspirovaná [tímto grafem](https://softwareengineering.stackexchange.com/questions/366996/distinction-between-ai-ml-neural-networks-deep-learning-and-data-mining)

---
## Koncepty k pokrytí

V tomto kurikulu budeme pokrývat pouze základní koncepty strojového učení, které by měl začátečník znát. Pokryjeme to, co nazýváme „klasické strojové učení“, primárně pomocí Scikit-learn, vynikající knihovny, kterou mnoho studentů využívá k učení základů. Pro pochopení širších konceptů umělé inteligence nebo hlubokého učení je nezbytné mít pevné základy ve strojovém učení, a proto vám je chceme zde nabídnout.

---
## V tomto kurzu se naučíte:

- základní koncepty strojového učení
- historii ML
- ML a spravedlnost
- regresní techniky ML
- klasifikační techniky ML
- shlukovací techniky ML
- techniky zpracování přirozeného jazyka ML
- techniky předpovědi časových řad ML
- posilované učení
- reálné aplikace ML

---
## Co nepokryjeme

- hluboké učení
- neuronové sítě
- AI

Pro lepší vzdělávací zážitek se vyhneme složitostem neuronových sítí, „hlubokému učení“ – vícevrsťovému modelování pomocí neuronových sítí – a AI, o kterých si povíme v jiném kurikulu. Také nabídneme budoucí kurikulum datové vědy, které se bude zaměřovat na tento aspekt širší oblasti.

---
## Proč studovat strojové učení?

Strojové učení je z pohledu systémů definováno jako tvorba automatizovaných systémů, které se dokážou naučit skryté vzory z dat, aby pomohly při inteligentním rozhodování.

Tato motivace je volně inspirována tím, jak se lidský mozek učí určité věci na základě dat, která vnímá z vnějšího světa.

✅ Zamyslete se na chvíli, proč by firma chtěla používat strategie strojového učení místo vytváření pevně zakódovaného pravidlového systému.

---
## Proč záleží na kvalitě dat

Kvalitní data zlepšují výkonnost modelu. Špatná nebo hlučná data mohou vést k nepřesným předpovědím, i když používáte pokročilé algoritmy strojového učení.

---
## Aplikace strojového učení

Aplikace strojového učení jsou dnes téměř všude a jsou tak běžné jako data, která proudí naší společností, generovaná našimi chytrými telefony, připojenými zařízeními a dalšími systémy. S ohledem na obrovský potenciál špičkových algoritmů strojového učení zkoumají výzkumníci jejich schopnost řešit mnohorozměrné a multidisciplinární reálné problémy s pozitivními výsledky.

---
## Příklady aplikovaného ML

**Strojové učení lze využít mnoha způsoby**:

- Predikovat pravděpodobnost nemoci z lékařské historie nebo zpráv pacienta.
- Využít data o počasí k předpovědi meteorologických jevů.
- Porozumět sentimentu textu.
- Detekovat falešné zprávy k zastavení šíření propagandy.

Finance, ekonomie, zemědělství, průzkum vesmíru, biomedicínské inženýrství, kognitivní věda a dokonce i humanitní obory si přizpůsobily strojové učení k řešení náročných, daty náročných problémů své oblasti.

---
## Závěr

Strojové učení automatizuje proces objevování vzorů tím, že nalézá smysluplné poznatky z reálných nebo generovaných dat. Ukázalo se, že je velmi cenné v obchodní, zdravotní a finanční oblasti, mimo jiné.

V nejbližší budoucnosti bude porozumění základům strojového učení nezbytností pro lidi z jakéhokoli oboru kvůli jeho širokému přijetí.

---
# 🚀 Výzva

Náčrt, na papíře nebo pomocí online aplikace jako [Excalidraw](https://excalidraw.com/), vlastním porozuměním rozdílů mezi AI, ML, hlubokým učením a datovou vědou. Přidejte i nějaké nápady na problémy, které jsou tyto techniky dobré řešit.

# [Po přednáškový kvíz](https://ff-quizzes.netlify.app/en/ml/)

---
# Recenze a samostudium

Chcete-li se naučit více o tom, jak pracovat s ML algoritmy v cloudu, sledujte tuto [Learning Path](https://docs.microsoft.com/learn/paths/create-no-code-predictive-models-azure-machine-learning/?WT.mc_id=academic-77952-leestott).

Abyste se naučili základy ML, sledujte tento [Learning Path](https://docs.microsoft.com/learn/modules/introduction-to-machine-learning/?WT.mc_id=academic-77952-leestott).

---
# Zadání

[Začněte a spusťte se](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Prohlášení o omezení odpovědnosti**:
Tento dokument byl přeložen pomocí AI překladatelské služby [Co-op Translator](https://github.com/Azure/co-op-translator). Přestože usilujeme o co největší přesnost, mějte prosím na paměti, že automatizované překlady mohou obsahovat chyby nebo nepřesnosti. Originální dokument v jeho mateřském jazyce by měl být považován za autoritativní zdroj. Pro kritické informace se doporučuje profesionální lidský překlad. Nejsme odpovědní za jakékoli nedorozumění nebo nesprávné interpretace vzniklé použitím tohoto překladu.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->