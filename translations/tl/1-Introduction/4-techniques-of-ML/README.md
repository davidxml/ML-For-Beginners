# Mga Teknik ng Machine Learning

Ang proseso ng paggawa, paggamit, at pagpapanatili ng mga modelo ng machine learning at ng data na ginagamit nila ay isang napakaibang proseso kumpara sa maraming iba pang mga daloy ng pag-unlad. Sa leksyong ito, bibigyang-katuwiran natin ang proseso, at ilalahad ang mga pangunahing teknik na kailangan mong malaman. Iyong gagawin:

- Maunawaan ang mga prosesong nagpapailalim sa machine learning sa mataas na antas.
- Tuklasin ang mga pangunahing konsepto tulad ng 'mga modelo', 'mga hula', at 'pagsasanay ng data'.

## [Pre-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

[![ML for beginners - Techniques of Machine Learning](https://img.youtube.com/vi/4NGM0U2ZSHU/0.jpg)](https://youtu.be/4NGM0U2ZSHU "ML for beginners - Techniques of Machine Learning")

> 🎥 I-click ang larawan sa itaas para sa isang maikling video na ginagabayan ang lektyong ito.

## Panimula

Sa mataas na antas, ang sining ng paglikha ng mga proseso ng machine learning (ML) ay binubuo ng ilang mga hakbang:

1. **Magpasya sa tanong**. Karamihan sa mga proseso ng ML ay nagsisimula sa pagtatanong ng isang tanong na hindi masasagot ng simpleng programang de-kondisyunal o ng mga patakarang batay-sa-patakarang makina. Madalas na ang mga tanong na ito ay nakatuon sa mga hula batay sa isang koleksyon ng data.
2. **Kolektahin at ihanda ang data**. Para masagot mo ang iyong tanong, kailangan mo ng data. Ang kalidad at, minsan, dami ng iyong data ang magtatakda kung gaano kaayos mong masasagot ang iyong paunang tanong. Ang pag-visualize ng data ay isang mahalagang bahagi ng yugtong ito. Kabilang din sa yugtong ito ang paghati ng data sa isang pangkat ng pagsasanay at pagsusulit upang makabuo ng modelo.
3. **Pumili ng paraan ng pagsasanay**. Depende sa iyong tanong at sa katangian ng iyong data, kailangan mong piliin kung paano mo nais sanayin ang modelo upang pinakamahusay na maipakita ang iyong data at makagawa ng tumpak na mga hula laban dito. Ito ang bahagi ng iyong proseso ng ML na nangangailangan ng espesipikong kadalubhasaan at, madalas, ng maraming eksperimento.
4. **Sanayin ang modelo**. Gamit ang iyong data sa pagsasanay, gagamitin mo ang iba't ibang mga algorithm upang sanayin ang modelo na kilalanin ang mga pattern sa data. Maaaring gamitin ng modelo ang mga panloob na bigat na maaaring ayusin upang bigyang-pansin ang ilang bahagi ng data kaysa sa iba para makabuo ng mas mahusay na modelo.
5. **Suriin ang modelo**. Gagamitin mo ang data na hindi pa nasilipan dati (ang iyong testing data) mula sa iyong koleksyon upang makita kung paano gumagana ang modelo.
6. **Parameter tuning**. Batay sa pagganap ng modelo mo, maaari mong ulitin ang proseso gamit ang ibang mga parameter, o mga baryabol, na kumokontrol sa kilos ng mga algorithm na ginamit sa pagsasanay ng modelo.
7. **Mangguhula**. Gamitin ang mga bagong input para subukan ang katumpakan ng iyong modelo.

## Anong tanong ang dapat itanong

Sanay ang mga computer sa pagtuklas ng mga nakatagong pattern sa data. Ang kapaki-pakinabang na ito ay napaka-kapaki-pakinabang para sa mga mananaliksik na may mga tanong tungkol sa isang tiyak na larangan na hindi madaling masagot sa pamamagitan ng paggawa ng kondisyonal na rules engine. Halimbawa, sa isang actuarial na gawain, maaaring makagawa ang isang data scientist ng mga gawang-kamay na mga patakaran tungkol sa mortalidad ng mga naninigarilyo kumpara sa hindi naninigarilyo.

Gayunpaman, kapag maraming iba pang mga baryabol ang isinama sa ekwasyon, maaaring mas maging epektibo ang isang ML model upang mahulaan ang mga hinaharap na rate ng mortalidad batay sa nakaraang kasaysayan ng kalusugan. Isang mas masiglang halimbawa ay ang paggawa ng mga prediksyon sa panahon para sa buwan ng Abril sa isang tiyak na lokasyon, batay sa data na kinabibilangan ng latitude, longitude, pagbabago ng klima, kalapitan sa dagat, mga pattern ng jet stream, at iba pa.

✅ Ang [slide deck](https://www2.cisl.ucar.edu/sites/default/files/2021-10/0900%20June%2024%20Haupt_0.pdf) tungkol sa mga modelo ng panahon ay nag-aalok ng pangkasaysayang pananaw para sa paggamit ng ML sa pagsusuri ng panahon.

## Mga gawain bago magtayo

Bago magsimula na bumuo ng iyong modelo, may ilang mga gawain na kailangan mong tapusin. Para subukan ang iyong tanong at bumuo ng hipotesis batay sa mga hulang ginawa ng modelo, kailangan mong tukuyin at i-configure ang ilang mga elemento.

### Data

Para masagot ang iyong tanong nang may katiyakan, kailangan mo ng sapat na dami ng tamang uri ng data. May dalawang bagay na kailangan mong gawin sa puntong ito:

- **Kolektahin ang data**. Isinasaalang-alang ang nakaraang leksyon tungkol sa pagiging patas sa pagsusuri ng data, kolektahin ang iyong data nang maingat. Maging maingat sa mga pinagkuhanan ng data, anumang likas na pagkiling na maaaring mayroon ito, at idokumento ang pinagmulan nito.
- **Ihanda ang data**. Mayroong ilang mga hakbang sa proseso ng paghahanda ng data. Maaaring kailanganin mong pag-isa-isaing muli ang data at i-normalize ito kung galing ito sa iba't ibang pinagmulan. Maaari mong pagandahin ang kalidad at dami ng data sa pamamagitan ng iba't ibang paraan tulad ng pag-convert ng mga string sa numero (tulad ng ginagawa natin sa [Clustering](../../5-Clustering/1-Visualize/README.md)). Maaari ka ring lumikha ng bagong data, batay sa orihinal (tulad ng ginagawa natin sa [Classification](../../4-Classification/1-Introduction/README.md)). Maaari mo ring linisin at i-edit ang data (tulad ng gagawin natin bago ang leksyon sa [Web App](../../3-Web-App/README.md)). Panghuli, maaaring kailanganin mo ring i-randomize at i-shuffle ito, depende sa iyong mga teknik sa pagsasanay.

✅ Matapos kolektahin at iproseso ang iyong data, maglaan ng sandali upang tingnan kung ang hugis nito ay magbibigay-daan upang masagot ang iyong nais na tanong. Maaaring mangyari na hindi gagana nang maayos ang data sa iyong ibinigay na gawain, tulad ng nadiskubre natin sa mga aralin sa [Clustering](../../5-Clustering/1-Visualize/README.md)!

### Mga Katangian at Target

Ang isang [feature](https://www.datasciencecentral.com/profiles/blogs/an-introduction-to-variable-and-feature-selection) ay isang nasusukat na katangian ng iyong data. Sa maraming dataset, ito ay ipinapahayag bilang isang ulo ng kolum tulad ng 'date', 'size', o 'color'. Ang iyong feature variable, karaniwang kinakatawan bilang `X` sa code, ay kumakatawan sa input variable na gagamitin upang sanayin ang modelo.

Ang target ay isang bagay na sinusubukan mong hulaan. Ang target, karaniwang kinakatawan bilang `y` sa code, ay kumakatawan sa sagot sa tanong na sinusubukan mong itanong sa iyong data: sa Disyembre, anong **kulay** ng mga kalabasa ang magiging pinakamura? Sa San Francisco, anong mga kapitbahayan ang magkakaroon ng pinakamahusay na presyo sa real estate? Minsan tinutukoy din ang target bilang isang label attribute.

### Pagpili ng iyong feature variable

🎓 **Feature Selection and Feature Extraction** Paano mo malalaman kung aling variable ang pipiliin kapag bumubuo ng modelo? Marahil ay daraan ka sa proseso ng feature selection o feature extraction upang piliin ang tamang mga variable para sa pinakaepektibong modelo. Hindi sila pareho: "Ang Feature extraction ay lumilikha ng mga bagong feature mula sa mga function ng orihinal na feature, samantalang ang feature selection ay nagbabalik ng isang subset ng mga feature." ([pinagmulan](https://wikipedia.org/wiki/Feature_selection))

### I-visualize ang iyong data

Isang mahalagang aspeto ng toolkit ng data scientist ay ang kapangyarihan na i-visualize ang data gamit ang ilang mga mahusay na library tulad ng Seaborn o MatPlotLib. Ang paglalarawan ng iyong data sa pamamagitan ng visual ay maaaring magbigay-daan sa iyong matuklasan ang mga nakatagong kaugnayan na maaari mong gamitin. Maaaring makatulong din ang iyong mga visualization upang matuklasan ang pagkiling o hindi balanseng data (tulad ng nadiskubre natin sa [Classification](../../4-Classification/2-Classifiers-1/README.md)).

### Hatiin ang iyong dataset

Bago ang pagsasanay, kailangan mong hatiin ang iyong dataset sa dalawa o higit pang mga bahagi na hindi pantay ang laki ngunit mahusay na kumakatawan sa data.

- **Training**. Ang bahaging ito ng dataset ang itinutugma sa iyong modelo upang sanayin ito. Ang set na ito ang bumubuo ng karamihan ng orihinal na dataset.
- **Testing**. Ang test dataset ay isang independiyenteng pangkat ng data, madalas na kinuha mula sa orihinal na data, na ginagamit mo upang kumpirmahin ang pagganap ng nabuo na modelo.
- **Validating**. Ang validation set ay isang mas maliit na independiyenteng pangkat ng mga halimbawa na ginagamit mo upang i-tune ang mga hyperparameter o arkitektura ng modelo upang mapabuti ito. Depende sa laki ng iyong data at ang tanong na iyong tinatanong, maaaring hindi mo kailangang bumuo ng ikatlong set na ito (tulad ng napansin natin sa [Time Series Forecasting](../../7-TimeSeries/1-Introduction/README.md)).

## Pagtatayo ng modelo

Gamit ang iyong training data, ang layunin mo ay bumuo ng modelo, o estadistikang representasyon ng iyong data, gamit ang iba't ibang mga algorithm upang **sanayin** ito. Ang pagsasanay ng isang modelo ay nagpapakita nito sa data at pinapayagan itong gumawa ng mga palagay tungkol sa mga natukoy nitong pattern, sinusuri, at tinatanggap o tinatanggihan.

### Magpasya sa paraan ng pagsasanay

Depende sa iyong tanong at sa katangian ng iyong data, pipili ka ng paraan upang sanayin ito. Sa pagsunod sa [Scikit-learn's documentation](https://scikit-learn.org/stable/user_guide.html) - na ginagamit natin sa kursong ito - maaari mong tuklasin ang maraming paraan sa pagsasanay ng modelo. Depende sa iyong karanasan, maaaring kailanganin mong subukan ang iba't ibang mga paraan upang mabuo ang pinakamahusay na modelo. Malamang na dadaan ka sa isang proseso kung saan sinusuri ng mga data scientist ang pagganap ng modelo sa pamamagitan ng pagsuplay dito ng mga hindi pa nakikitang data, sinusuri ang katumpakan, pagkiling, at iba pang mga isyu na nagpapababa ng kalidad, at pinipili ang pinakaangkop na paraan ng pagsasanay para sa gawain.

### Sanayin ang modelo

Hawakan ang iyong training data, handa ka nang 'i-fit' ito upang lumikha ng modelo. Mapapansin mo na sa maraming ML libraries ay makikita mo ang code na 'model.fit' - dito mo ipapasa ang iyong feature variable bilang array ng mga halaga (karaniwang 'X') at ang target variable (karaniwang 'y').

### Suriin ang modelo

Kapag natapos na ang proseso ng pagsasanay (maaaring tumagal ito ng maraming pag-ulit, o 'epochs', upang masanay ang isang malaking modelo), magagawa mong suriin ang kalidad ng modelo gamit ang test data upang sukatin ang pagganap nito. Ang data na ito ay isang bahagi ng orihinal na data na hindi pa nasuri ng modelo. Maaari kang mag-print ng talaan ng mga sukatan tungkol sa kalidad ng iyong modelo.

🎓 **Model fitting**

Sa konteksto ng machine learning, ang model fitting ay tumutukoy sa katumpakan ng pundasyong function ng modelo habang sinusubukan nitong suriin ang data na hindi nito alam.

🎓 Karaniwan ang mga problemang **underfitting** at **overfitting** na nagpapababa sa kalidad ng modelo, sapagkat ang modelo ay hindi sapat ang fit o sobra-sobra ang fit. Nagiging sanhi ito upang gumawa ang modelo ng mga hula nang masyadong malapit o masyadong malaya sa data ng pagsasanay. Ang overfit na modelo ay naghuhula nang napakahusay sa data ng pagsasanay dahil natutunan nito nang mabuti ang mga detalye at ingay ng data. Ang underfit na modelo ay hindi tumpak dahil hindi nito kayang suriin nang tumpak ang data ng pagsasanay o ang data na hindi pa nito 'nakikita'.

![overfitting model](../../../../translated_images/tl/overfitting.1c132d92bfd93cb6.webp)
> Infographic ni [Jen Looper](https://twitter.com/jenlooper)

## Parameter tuning

Kapag natapos na ang iyong paunang pagsasanay, obserbahan ang kalidad ng modelo at isaalang-alang na pagandahin ito sa pamamagitan ng pag-aayos ng mga 'hyperparameter'. Basahin pa tungkol sa proseso [sa dokumentasyon](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters?WT.mc_id=academic-77952-leestott).

## Prediction

Ito na ang sandali kung saan maaari mong gamitin ang ganap na bagong data upang subukan ang katumpakan ng iyong modelo. Sa isang 'applied' na setting ng ML, kung saan bumubuo ka ng mga web asset upang gamitin ang modelo sa produksyon, maaaring kasama sa prosesong ito ang pangangalap ng input mula sa user (isang pagpindot ng button, halimbawa) upang itakda ang variable at ipadala ito sa modelo para sa inference o pagsusuri.

Sa mga araling ito, matutuklasan mo kung paano gamitin ang mga hakbang na ito upang ihanda, bumuo, subukan, suriin, at manghula — lahat ng mga galaw ng isang data scientist at higit pa, habang nagpapatuloy ka sa iyong paglalakbay upang maging isang 'full stack' ML engineer.

---

## 🚀Hamunin

Gumuhit ng flow chart na nagpapakita ng mga hakbang ng isang ML practitioner. Saan mo nakikita ang iyong sarili ngayon sa proseso? Saan mo inaasahan na mahihirapan? Ano ang tila madali para sa iyo?

## [Post-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## Repaso at Sariling Pag-aaral

Maghanap online ng mga panayam sa mga data scientist na nagsasalaysay ng kanilang araw-araw na gawain. Narito ang [isa](https://www.youtube.com/watch?v=Z3IjgbbCEfs).

## Takdang Aralin

[Interview a data scientist](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Pagtanggi**:  
Ang dokumentong ito ay isinalin gamit ang serbisyo ng AI translation na [Co-op Translator](https://github.com/Azure/co-op-translator). Bagama't nagsusumikap kami para sa katumpakan, pakatandaan na ang awtomatikong pagsasalin ay maaaring maglaman ng mga pagkakamali o di-katumpakan. Ang orihinal na dokumento sa orihinal nitong wika ang dapat ituring na pangunahing sanggunian. Para sa mahahalagang impormasyon, inirerekomenda ang propesyonal na pagsasalin ng tao. Hindi kami mananagot sa anumang hindi pagkakaunawaan o maling interpretasyon na nagmumula sa paggamit ng pagsasaling ito.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->