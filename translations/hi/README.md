[![GitHub license](https://img.shields.io/github/license/microsoft/ML-For-Beginners.svg)](https://github.com/microsoft/ML-For-Beginners/blob/master/LICENSE)
[![GitHub contributors](https://img.shields.io/github/contributors/microsoft/ML-For-Beginners.svg)](https://GitHub.com/microsoft/ML-For-Beginners/graphs/contributors/)
[![GitHub issues](https://img.shields.io/github/issues/microsoft/ML-For-Beginners.svg)](https://GitHub.com/microsoft/ML-For-Beginners/issues/)
[![GitHub pull-requests](https://img.shields.io/github/issues-pr/microsoft/ML-For-Beginners.svg)](https://GitHub.com/microsoft/ML-For-Beginners/pulls/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)

[![GitHub watchers](https://img.shields.io/github/watchers/microsoft/ML-For-Beginners.svg?style=social&label=Watch)](https://GitHub.com/microsoft/ML-For-Beginners/watchers/)
[![GitHub forks](https://img.shields.io/github/forks/microsoft/ML-For-Beginners.svg?style=social&label=Fork)](https://GitHub.com/microsoft/ML-For-Beginners/network/)
[![GitHub stars](https://img.shields.io/github/stars/microsoft/ML-For-Beginners.svg?style=social&label=Star)](https://GitHub.com/microsoft/ML-For-Beginners/stargazers/)

### 🌐 बहुभाषी समर्थन

#### GitHub Action के माध्यम से समर्थित (स्वचालित और हमेशा अद्यतित)

<!-- CO-OP TRANSLATOR LANGUAGES TABLE START -->
[Arabic](../ar/README.md) | [Bengali](../bn/README.md) | [Bulgarian](../bg/README.md) | [Burmese (Myanmar)](../my/README.md) | [Chinese (Simplified)](../zh-CN/README.md) | [Chinese (Traditional, Hong Kong)](../zh-HK/README.md) | [Chinese (Traditional, Macau)](../zh-MO/README.md) | [Chinese (Traditional, Taiwan)](../zh-TW/README.md) | [Croatian](../hr/README.md) | [Czech](../cs/README.md) | [Danish](../da/README.md) | [Dutch](../nl/README.md) | [Estonian](../et/README.md) | [Finnish](../fi/README.md) | [French](../fr/README.md) | [German](../de/README.md) | [Greek](../el/README.md) | [Hebrew](../he/README.md) | [Hindi](./README.md) | [Hungarian](../hu/README.md) | [Indonesian](../id/README.md) | [Italian](../it/README.md) | [Japanese](../ja/README.md) | [Kannada](../kn/README.md) | [Khmer](../km/README.md) | [Korean](../ko/README.md) | [Lithuanian](../lt/README.md) | [Malay](../ms/README.md) | [Malayalam](../ml/README.md) | [Marathi](../mr/README.md) | [Nepali](../ne/README.md) | [Nigerian Pidgin](../pcm/README.md) | [Norwegian](../no/README.md) | [Persian (Farsi)](../fa/README.md) | [Polish](../pl/README.md) | [Portuguese (Brazil)](../pt-BR/README.md) | [Portuguese (Portugal)](../pt-PT/README.md) | [Punjabi (Gurmukhi)](../pa/README.md) | [Romanian](../ro/README.md) | [Russian](../ru/README.md) | [Serbian (Cyrillic)](../sr/README.md) | [Slovak](../sk/README.md) | [Slovenian](../sl/README.md) | [Spanish](../es/README.md) | [Swahili](../sw/README.md) | [Swedish](../sv/README.md) | [Tagalog (Filipino)](../tl/README.md) | [Tamil](../ta/README.md) | [Telugu](../te/README.md) | [Thai](../th/README.md) | [Turkish](../tr/README.md) | [Ukrainian](../uk/README.md) | [Urdu](../ur/README.md) | [Vietnamese](../vi/README.md)

> **स्थानीय रूप से क्लोन करना पसंद करते हैं?**
>
> यह रिपॉजिटरी 50+ भाषा अनुवादों को शामिल करती है जो डाउनलोड आकार को काफी बढ़ाती है। अनुवाद के बिना क्लोन करने के लिए, sparse checkout का उपयोग करें:
>
> **Bash / macOS / Linux:**
> ```bash
> git clone --filter=blob:none --sparse https://github.com/microsoft/ML-For-Beginners.git
> cd ML-For-Beginners
> git sparse-checkout set --no-cone '/*' '!translations' '!translated_images'
> ```
>
> **CMD (Windows):**
> ```cmd
> git clone --filter=blob:none --sparse https://github.com/microsoft/ML-For-Beginners.git
> cd ML-For-Beginners
> git sparse-checkout set --no-cone "/*" "!translations" "!translated_images"
> ```
>
> इससे आपको कोर्स पूरा करने के लिए आवश्यक सब कुछ तेज़ी से डाउनलोड होगा।
<!-- CO-OP TRANSLATOR LANGUAGES TABLE END -->

#### हमारे समुदाय में शामिल हों

[![Microsoft Foundry Discord](https://dcbadge.limes.pink/api/server/nTYy5BXMWG)](https://discord.gg/nTYy5BXMWG)

हमारे पास एक Discord पर AI के साथ सीखने की श्रृंखला चल रही है, अधिक जानने और जुड़ने के लिए जाएं [Learn with AI Series](https://aka.ms/learnwithai/discord) 18 - 30 सितंबर, 2025 को। आपको डेटा साइंस के लिए GitHub Copilot का उपयोग करने के टिप्स और ट्रिक्स मिलेंगे।

![Learn with AI series](../../translated_images/hi/3.9b58fd8d6c373c20.webp)

# शुरुआती लोगों के लिए मशीन लर्निंग - एक पाठ्यक्रम

> 🌍 विश्व की संस्कृतियों के माध्यम से मशीन लर्निंग का अन्वेषण करते हुए विश्वभर की यात्रा करें 🌍

Microsoft के क्लाउड एडवोकेट्स आपको **मशीन लर्निंग** के बारे में 12 सप्ताह, 26-पाठ के पाठ्यक्रम की पेशकश करते हैं। इस पाठ्यक्रम में, आप कभी-कभी 'क्लासिक मशीन लर्निंग' कहा जाने वाला विषय सीखेंगे, जो मुख्य रूप से Scikit-learn लाइब्रेरी का उपयोग करता है और डीप लर्निंग से बचता है, जिसका कवरेज हमारी [AI for Beginners' curriculum](https://aka.ms/ai4beginners) में है। इन पाठों को हमारे ['डेटा साइंस फॉर बिगिनर्स' पाठ्यक्रम](https://aka.ms/ds4beginners) के साथ संयोजित करें।

हमारे साथ विश्वभर की यात्रा करें क्योंकि हम विभिन्न क्षेत्रों के डेटा पर क्लासिक तकनीकों को लागू करते हैं। प्रत्येक पाठ में प्री और पोस्ट-लेसन क्विज, लिखित निर्देश, समाधान, असाइनमेंट आदि शामिल हैं। हमारी परियोजना आधारित शिक्षण पद्धति आपको निर्माण करते हुए सीखने का मौका देती है, जो नई कौशलों को 'अटके' रहने का सिद्ध तरीका है।

**✍️ हमारे लेखकों को हार्दिक धन्यवाद** जे़न लूपर, स्टीफन हाउल, फ्रांसेस्का लज़्ज़ेरी, टोमोमी इमुरा, कैसी ब्रेवियू, दिमित्री सोशनिकोव, क्रिस नोरिंग, अनिर्बान मुखर्जी, ऑर्नेला अल्टुन्यान, रुथ याकुबू और ऐमी बॉयड

**🎨 हमारे चित्रकारों को धन्यवाद** टोमोमी इमुरा, दिसानी मैडीपाली, और जे़न लूपर

**🙏 Microsoft Student Ambassador लेखकों, समीक्षकों और सामग्री योगदानकर्ताओं को विशेष धन्यवाद**, विशेष रूप से ऋषित दागली, मुहम्मद साकिब खान इनान, रोहन राज, अलेक्जेंड्रू पेट्रेस्कु, अभिषेक जायसवाल, नवरिन ताबस्सुम, इवान सामुइला, और स्निग्धा अग्रवाल

**🤩 Microsoft Student Ambassadors एरिक वंजाऊ, जसलिन सोंधी, और विदुषी गुप्ता को R पाठों के लिए अतिरिक्त आभार!**

# शुरुआत

इन चरणों का पालन करें:
1. **रिपॉजिटरी को फोर्क करें**: इस पेज के ऊपर-दाएँ कोने में "Fork" बटन पर क्लिक करें।
2. **रिपॉजिटरी को क्लोन करें**: `git clone https://github.com/microsoft/ML-For-Beginners.git`

> [इस कोर्स के सभी अतिरिक्त संसाधन Microsoft Learn संग्रह में पाएँ](https://learn.microsoft.com/en-us/collections/qrqzamz1nn2wx3?WT.mc_id=academic-77952-bethanycheum)

> 🔧 **मदद चाहिए?** आम समस्याओं के समाधान के लिए हमारा [Troubleshooting Guide](TROUBLESHOOTING.md) देखें।

**[छात्रों](https://aka.ms/student-page)**, इस पाठ्यक्रम का उपयोग करने के लिए, पूरे रिपॉजिटरी को अपने GitHub खाते पर फोर्क करें और स्वयं या समूह के साथ अभ्यास पूरा करें:

- प्री-लेक्चर क्विज से शुरुआत करें।
- व्याख्यान पढ़ें और गतिविधियाँ पूरी करें, प्रत्येक ज्ञान जांच पर रुकें और सोचें।
- समाधान कोड चलाने के बजाय पाठ को समझकर परियोजनाएं बनाने का प्रयास करें; हालांकि कोड प्रत्येक परियोजना आधारित पाठ के `/solution` फ़ोल्डर में उपलब्ध है।
- पोस्ट-लेक्चर क्विज लें।
- चैलेंज पूरा करें।
- असाइनमेंट पूरा करें।
- किसी भी लेसन समूह को पूरा करने के बाद, [Discussion Board](https://github.com/microsoft/ML-For-Beginners/discussions) पर जाएं और सही PAT रूपांकन भरकर "उच्चारित करें"। PAT एक प्रगति मूल्यांकन उपकरण है जो आपकी सीख को आगे बढ़ाता है। आप अन्य PAT पर प्रतिक्रियाएँ भी दे सकते हैं ताकि हम एक साथ सीखें।

> आगे पढ़ाई के लिए, हम इन [Microsoft Learn](https://docs.microsoft.com/en-us/users/jenlooper-2911/collections/k7o7tg1gp306q4?WT.mc_id=academic-77952-leestott) मॉड्यूल और सीखने के रास्तों का पालन करने की सलाह देते हैं।

**शिक्षकों**, हमने इस पाठ्यक्रम का उपयोग कैसे करें, इस पर कुछ [सुझाव शामिल किए हैं](for-teachers.md)।

---

## वीडियो वॉकथ्रू

कुछ पाठ छोटे वीडियो के रूप में उपलब्ध हैं। इन्हें आप पाठों में इन-लाइन देख सकते हैं, या Microsoft Developer YouTube चैनल पर [ML for Beginners प्लेलिस्ट](https://aka.ms/ml-beginners-videos) में नीचे दिए चित्र पर क्लिक करके देख सकते हैं।

[![ML for beginners banner](../../translated_images/hi/ml-for-beginners-video-banner.63f694a100034bc6.webp)](https://aka.ms/ml-beginners-videos)

---

## टीम से मिलें

[![Promo video](../../images/ml.gif)](https://youtu.be/Tj1XWrDSYJU)

**गिफ़ के निर्माता** [Mohit Jaisal](https://linkedin.com/in/mohitjaisal)

> 🎥 परियोजना और इसे बनाने वालों के बारे में वीडियो देखने के लिए ऊपर के चित्र पर क्लिक करें!

---

## शिक्षण पद्धति

इस पाठ्यक्रम को बनाते समय हमने दो शैक्षणिक सिद्धांत चुनें हैं: यह सुनिश्चित करना कि यह **प्रोजेक्ट-आधारित** और हैंड्स-ऑन हो तथा इसमें **बार-बार क्विज़** शामिल हों। इसके अलावा, इस पाठ्यक्रम का एक साझा **थीम** है जो इसे एकजुट बनाता है।

सामग्री का प्रोजेक्ट्स के साथ मिलान करने से छात्रों के लिए प्रक्रिया अधिक रोचक हो जाती है और अवधारणाओं की स्थिरता बढ़ती है। इसके अलावा, कक्षा से पहले का एक कम दबाव वाला क्विज छात्र की विषय सीखने की इच्छा को स्थापित करता है, जबकि कक्षा के बाद दूसरा क्विज चारों ओर प्रतिधारण सुनिश्चित करता है। यह पाठ्यक्रम लचीला और मनोरंजक बनाया गया है और इसे पूरा या अंशतः लिया जा सकता है। प्रोजेक्ट्स छोटे शुरू होते हैं और 12-सप्ताह के अंत तक धीरे-धीरे जटिल हो जाते हैं। इस पाठ्यक्रम में वास्तविक दुनिया के ML अनुप्रयोगों पर पोस्टस्क्रिप्ट भी शामिल है, जिसे अतिरिक्त क्रेडिट या चर्चा के आधार के रूप में इस्तेमाल किया जा सकता है।

> हमारे [व्यवहार संहिता](CODE_OF_CONDUCT.md), [योगदान](CONTRIBUTING.md), [अनुवाद](..), और [समस्याओं का समाधान](TROUBLESHOOTING.md) दिशानिर्देश देखें। हम आपकी रचनात्मक प्रतिक्रिया का स्वागत करते हैं!

## प्रत्येक पाठ में शामिल हैं

- वैकल्पिक स्केचनोट
- वैकल्पिक पूरक वीडियो
- वीडियो वॉकथ्रू (कुछ पाठों में)
- [प्री-लेक्चर वार्मअप क्विज](https://ff-quizzes.netlify.app/en/ml/)
- लिखित पाठ
- परियोजना आधारित पाठों के लिए, परियोजना को कैसे बनाएं पर चरण-दर-चरण गाइड
- ज्ञान जांच
- एक चुनौती
- पूरक पठन सामग्री
- असाइनमेंट
- [पोस्ट-लेक्चर क्विज](https://ff-quizzes.netlify.app/en/ml/)
> **भाषाओं के बारे में एक नोट**: ये पाठ मुख्य रूप से Python में लिखे गए हैं, लेकिन कई R में भी उपलब्ध हैं। R पाठ पूरा करने के लिए, `/solution` फ़ोल्डर पर जाएं और R पाठ देखें। इनमें .rmd एक्सटेंशन होता है जो एक **R Markdown** फ़ाइल को दर्शाता है जिसे सरलता से `कोड खंड` (R या अन्य भाषाओं के) और एक `YAML हेडर` (जो PDF जैसी आउटपुट स्वरूपों को प्रारूपित करने का मार्गदर्शन करता है) को `Markdown दस्तावेज़` में एम्बेड करने के रूप में परिभाषित किया जा सकता है। इसलिए, यह डेटा विज्ञान के लिए एक आदर्श लेखन ढांचा के रूप में कार्य करता है क्योंकि इससे आप अपना कोड, उसका आउटपुट, और अपने विचार एक साथ Markdown में लिख सकते हैं। इसके अलावा, R Markdown दस्तावेज़ PDF, HTML, या Word जैसे आउटपुट स्वरूपों में प्रस्तुत किए जा सकते हैं।

> **क्विज़ के बारे में एक नोट**: सभी क्विज़ [Quiz App फ़ोल्डर](../../quiz-app) में पाए जाते हैं, कुल 52 क्विज़ हैं जिनमें से प्रत्येक में तीन प्रश्न होते हैं। वे पाठों के भीतर से जुड़े होते हैं लेकिन क्विज़ ऐप स्थानीय रूप से चलाया जा सकता है; `quiz-app` फ़ोल्डर में निर्देशों का पालन करें ताकि आप इसे स्थानीय रूप से होस्ट या Azure पर तैनात कर सकें।

| पाठ संख्या |                             विषय                              |                   पाठ समूह                   | सीखने के उद्देश्य                                                                                                             |                                                              लिंक्ड पाठ                                                               |                        लेखक                        |
| :--------: | :------------------------------------------------------------: | :------------------------------------------: | ----------------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------: |
|      01    |                मशीन लर्निंग का परिचय                |      [परिचय](1-Introduction/README.md)       | मशीन लर्निंग के मूलभूत सिद्धांत सीखें                                                                                |                                             [पाठ](1-Introduction/1-intro-to-ML/README.md)                                             |                       Muhammad                       |
|      02    |                मशीन लर्निंग का इतिहास                 |      [परिचय](1-Introduction/README.md)       | इस क्षेत्र के इतिहास को जानें                                                                                         |                                            [पाठ](1-Introduction/2-history-of-ML/README.md)                                            |                     Jen and Amy                      |
|      03    |                 निष्पक्षता और मशीन लर्निंग                  |      [परिचय](1-Introduction/README.md)       | मशीन लर्निंग मॉडल बनाते और लागू करते समय छात्रों को विचार करने वाले निष्पक्षता के महत्वपूर्ण दार्शनिक मुद्दे क्या हैं? |                                              [पाठ](1-Introduction/3-fairness/README.md)                                               |                        Tomomi                        |
|      04    |                मशीन लर्निंग के तकनीक                  |      [परिचय](1-Introduction/README.md)       | मशीन लर्निंग अनुसंधानकर्ता मशीन लर्निंग मॉडल बनाने के लिए कौन-कौन सी तकनीकें इस्तेमाल करते हैं?                                                                       |                                          [पाठ](1-Introduction/4-techniques-of-ML/README.md)                                           |                    Chris and Jen                     |
|      05    |                   प्रतिगमन का परिचय                   |        [Regression](2-Regression/README.md)         | प्रतिगमन मॉडलों के लिए Python और Scikit-learn के साथ शुरूआत करें                                                                  |         [Python](2-Regression/1-Tools/README.md) • [R](../../2-Regression/1-Tools/solution/R/lesson_1.html)         |      Jen • Eric Wanjau       |
|      06    |                उत्तरी अमेरिका कद्दू के दाम 🎃                |        [Regression](2-Regression/README.md)         | ML के लिए डेटा को विज़ुअलाइज़ और साफ करें                                                                                  |          [Python](2-Regression/2-Data/README.md) • [R](../../2-Regression/2-Data/solution/R/lesson_2.html)          |      Jen • Eric Wanjau       |
|      07    |                उत्तरी अमेरिका कद्दू के दाम 🎃                |        [Regression](2-Regression/README.md)         | रैखिक और बहुपदीय प्रतिगमन मॉडल बनाएं                                                                                   |        [Python](2-Regression/3-Linear/README.md) • [R](../../2-Regression/3-Linear/solution/R/lesson_3.html)        |      Jen and Dmitry • Eric Wanjau       |
|      08    |                उत्तरी अमेरिका कद्दू के दाम 🎃                |        [Regression](2-Regression/README.md)         | एक लॉजिस्टिक प्रतिगमन मॉडल बनाएं                                                                                               |     [Python](2-Regression/4-Logistic/README.md) • [R](../../2-Regression/4-Logistic/solution/R/lesson_4.html)      |      Jen • Eric Wanjau       |
|      09    |                          एक वेब ऐप 🔌                          |           [Web App](3-Web-App/README.md)            | अपने प्रशिक्षित मॉडल का उपयोग करने के लिए एक वेब ऐप बनाएं                                                                                       |                                                 [Python](3-Web-App/1-Web-App/README.md)                                                  |                         Jen                          |
|      10    |                 वर्गीकरण का परिचय                 |    [Classification](4-Classification/README.md)     | अपने डेटा को साफ, तैयार, और विज़ुअलाइज़ करें; वर्गीकरण का परिचय                                                            | [Python](4-Classification/1-Introduction/README.md) • [R](../../4-Classification/1-Introduction/solution/R/lesson_10.html)  | Jen and Cassie • Eric Wanjau |
|      11    |             स्वादिष्ट एशियाई और भारतीय व्यंजन 🍜             |    [Classification](4-Classification/README.md)     | वर्गीकरणकर्ताओं का परिचय                                                                                                     | [Python](4-Classification/2-Classifiers-1/README.md) • [R](../../4-Classification/2-Classifiers-1/solution/R/lesson_11.html) | Jen and Cassie • Eric Wanjau |
|      12    |             स्वादिष्ट एशियाई और भारतीय व्यंजन 🍜             |    [Classification](4-Classification/README.md)     | और अधिक वर्गीकरणकर्ता                                                                                                                | [Python](4-Classification/3-Classifiers-2/README.md) • [R](../../4-Classification/3-Classifiers-2/solution/R/lesson_12.html) | Jen and Cassie • Eric Wanjau |
|      13    |             स्वादिष्ट एशियाई और भारतीय व्यंजन 🍜             |    [Classification](4-Classification/README.md)     | अपने मॉडल का उपयोग करके एक अनुशंषा वेब ऐप बनाएं                                                                                    |                                              [Python](4-Classification/4-Applied/README.md)                                              |                         Jen                          |
|      14    |                   क्लस्टरिंग का परिचय                   |        [Clustering](5-Clustering/README.md)         | अपने डेटा को साफ, तैयार, और विज़ुअलाइज़ करें; क्लस्टरिंग का परिचय                                                                |         [Python](5-Clustering/1-Visualize/README.md) • [R](../../5-Clustering/1-Visualize/solution/R/lesson_14.html)         |      Jen • Eric Wanjau       |
|      15    |              नाइजीरियाई संगीत स्वादों का अन्वेषण 🎧              |        [Clustering](5-Clustering/README.md)         | K-Means क्लस्टरिंग विधि का अन्वेषण                                                                                           |           [Python](5-Clustering/2-K-Means/README.md) • [R](../../5-Clustering/2-K-Means/solution/R/lesson_15.html)           |      Jen • Eric Wanjau       |
|      16    |        प्राकृतिक भाषा प्रसंस्करण का परिचय ☕️         |   [Natural language processing](6-NLP/README.md)    | एक सरल बॉट बनाकर NLP के मूलभूत बातें सीखें                                                                             |                                             [Python](6-NLP/1-Introduction-to-NLP/README.md)                                              |                       Stephen                        |
|      17    |                      सामान्य NLP कार्य ☕️                      |   [Natural language processing](6-NLP/README.md)    | भाषा संरचनाओं के साथ काम करने में आवश्यक सामान्य कार्यों को समझकर अपने NLP ज्ञान को गहरा करें                          |                                                    [Python](6-NLP/2-Tasks/README.md)                                                     |                       Stephen                        |
|      18    |             अनुवाद और भावना विश्लेषण ♥️              |   [Natural language processing](6-NLP/README.md)    | जेन ऑस्टेन के साथ अनुवाद और भावना विश्लेषण                                                                             |                                            [Python](6-NLP/3-Translation-Sentiment/README.md)                                             |                       Stephen                        |
|      19    |                  यूरोप के रोमांटिक होटल ♥️                  |   [Natural language processing](6-NLP/README.md)    | होटल समीक्षाओं के साथ भावना विश्लेषण 1                                                                                         |                                               [Python](6-NLP/4-Hotel-Reviews-1/README.md)                                                |                       Stephen                        |
|      20    |                  यूरोप के रोमांटिक होटल ♥️                  |   [Natural language processing](6-NLP/README.md)    | होटल समीक्षाओं के साथ भावना विश्लेषण 2                                                                                         |                                               [Python](6-NLP/5-Hotel-Reviews-2/README.md)                                                |                       Stephen                        |
|      21    |            टाइम सीरीज पूर्वानुमान का परिचय             |        [Time series](7-TimeSeries/README.md)        | टाइम सीरीज पूर्वानुमान का परिचय                                                                                         |                                             [Python](7-TimeSeries/1-Introduction/README.md)                                              |                      Francesca                       |
|      22    | ⚡️ विश्व ऊर्जा उपयोग ⚡️ - ARIMA के साथ टाइम सीरीज पूर्वानुमान |        [Time series](7-TimeSeries/README.md)        | ARIMA के साथ टाइम सीरीज पूर्वानुमान                                                                                              |                                                 [Python](7-TimeSeries/2-ARIMA/README.md)                                                 |                      Francesca                       |
|      23    |  ⚡️ विश्व ऊर्जा उपयोग ⚡️ - SVR के साथ टाइम सीरीज पूर्वानुमान  |        [Time series](7-TimeSeries/README.md)        | Support Vector Regressor के साथ टाइम सीरीज पूर्वानुमान                                                                           |                                                  [Python](7-TimeSeries/3-SVR/README.md)                                                  |                       Anirban                        |
|      24    |             सुदृढीकरण शिक्षण का परिचय             | [Reinforcement learning](8-Reinforcement/README.md) | Q-लर्निंग के साथ सुदृढीकरण शिक्षण का परिचय                                                                          |                                             [Python](8-Reinforcement/1-QLearning/README.md)                                              |                        Dmitry                        |
|      25    |                 पीटर को भेड़िये से बचाएँ! 🐺                  | [Reinforcement learning](8-Reinforcement/README.md) | सुदृढीकरण शिक्षण जिम                                                                                                      |                                                [Python](8-Reinforcement/2-Gym/README.md)                                                 |                        Dmitry                        |
|  पोस्टस्क्रिप्ट   |            वास्तविक दुनिया के ML परिदृश्य और अनुप्रयोग            |      [ML in the Wild](9-Real-World/README.md)       | क्लासिकल ML के दिलचस्प और प्रकट करते हुए वास्तविक दुनिया के अनुप्रयोग                                                               |                                             [पाठ](9-Real-World/1-Applications/README.md)                                              |                         टीम                         |
|  पोस्टस्क्रिप्ट   |            RAI डैशबोर्ड का उपयोग करके ML में मॉडल डिबगिंग          |      [ML in the Wild](9-Real-World/README.md)       | जिम्मेदार AI डैशबोर्ड कॉम्पोनेंट्स का उपयोग कर मशीन लर्निंग में मॉडल डिबगिंग                                                              |                                             [पाठ](9-Real-World/2-Debugging-ML-Models/README.md)                                              |                         Ruth Yakubu                       |

> [इस पाठ्यक्रम के लिए हमारे Microsoft Learn संग्रह में सभी अतिरिक्त संसाधन देखें](https://learn.microsoft.com/en-us/collections/qrqzamz1nn2wx3?WT.mc_id=academic-77952-bethanycheum)

## ऑफलाइन पहुंच

आप इस दस्तावेज़ को ऑफलाइन [Docsify](https://docsify.js.org/#/) का उपयोग करके चला सकते हैं। इस रिपॉजिटरी को फोर्क करें, अपने स्थानीय मशीन पर [Docsify स्थापित करें](https://docsify.js.org/#/quickstart), और फिर इस रिपॉजिटरी के रूट फ़ोल्डर में टाइप करें `docsify serve`। वेबसाइट आपके लोकलहोस्ट पर पोर्ट 3000 पर चलती रहेगी: `localhost:3000`।

## पीडीएफ़

पाठ्यक्रम की पीडीएफ़ फाइल लिंक के साथ [यहाँ](https://microsoft.github.io/ML-For-Beginners/pdf/readme.pdf) देखें।


## 🎒 अन्य पाठ्यक्रम 

हमारी टीम अन्य पाठ्यक्रम बनाती है! देखें:

<!-- CO-OP TRANSLATOR OTHER COURSES START -->
### LangChain
[![LangChain4j for Beginners](https://img.shields.io/badge/LangChain4j%20for%20Beginners-22C55E?style=for-the-badge&&labelColor=E5E7EB&color=0553D6)](https://aka.ms/langchain4j-for-beginners)
[![LangChain.js for Beginners](https://img.shields.io/badge/LangChain.js%20for%20Beginners-22C55E?style=for-the-badge&labelColor=E5E7EB&color=0553D6)](https://aka.ms/langchainjs-for-beginners?WT.mc_id=m365-94501-dwahlin)
[![LangChain for Beginners](https://img.shields.io/badge/LangChain%20for%20Beginners-22C55E?style=for-the-badge&labelColor=E5E7EB&color=0553D6)](https://github.com/microsoft/langchain-for-beginners?WT.mc_id=m365-94501-dwahlin)
---

### Azure / Edge / MCP / Agents
[![AZD for Beginners](https://img.shields.io/badge/AZD%20for%20Beginners-0078D4?style=for-the-badge&labelColor=E5E7EB&color=0078D4)](https://github.com/microsoft/AZD-for-beginners?WT.mc_id=academic-105485-koreyst)
[![Edge AI for Beginners](https://img.shields.io/badge/Edge%20AI%20for%20Beginners-00B8E4?style=for-the-badge&labelColor=E5E7EB&color=00B8E4)](https://github.com/microsoft/edgeai-for-beginners?WT.mc_id=academic-105485-koreyst)
[![MCP for Beginners](https://img.shields.io/badge/MCP%20for%20Beginners-009688?style=for-the-badge&labelColor=E5E7EB&color=009688)](https://github.com/microsoft/mcp-for-beginners?WT.mc_id=academic-105485-koreyst)
[![AI Agents for Beginners](https://img.shields.io/badge/AI%20Agents%20for%20Beginners-00C49A?style=for-the-badge&labelColor=E5E7EB&color=00C49A)](https://github.com/microsoft/ai-agents-for-beginners?WT.mc_id=academic-105485-koreyst)

---
 
### जेनरेटिव एआई श्रृंखला
[![Generative AI for Beginners](https://img.shields.io/badge/Generative%20AI%20for%20Beginners-8B5CF6?style=for-the-badge&labelColor=E5E7EB&color=8B5CF6)](https://github.com/microsoft/generative-ai-for-beginners?WT.mc_id=academic-105485-koreyst)
[![Generative AI (.NET)](https://img.shields.io/badge/Generative%20AI%20(.NET)-9333EA?style=for-the-badge&labelColor=E5E7EB&color=9333EA)](https://github.com/microsoft/Generative-AI-for-beginners-dotnet?WT.mc_id=academic-105485-koreyst)
[![Generative AI (Java)](https://img.shields.io/badge/Generative%20AI%20(Java)-C084FC?style=for-the-badge&labelColor=E5E7EB&color=C084FC)](https://github.com/microsoft/generative-ai-for-beginners-java?WT.mc_id=academic-105485-koreyst)
[![Generative AI (JavaScript)](https://img.shields.io/badge/Generative%20AI%20(JavaScript)-E879F9?style=for-the-badge&labelColor=E5E7EB&color=E879F9)](https://github.com/microsoft/generative-ai-with-javascript?WT.mc_id=academic-105485-koreyst)

---
 
### कोर लर्निंग
[![ML for Beginners](https://img.shields.io/badge/ML%20for%20Beginners-22C55E?style=for-the-badge&labelColor=E5E7EB&color=22C55E)](https://aka.ms/ml-beginners?WT.mc_id=academic-105485-koreyst)
[![Data Science for Beginners](https://img.shields.io/badge/Data%20Science%20for%20Beginners-84CC16?style=for-the-badge&labelColor=E5E7EB&color=84CC16)](https://aka.ms/datascience-beginners?WT.mc_id=academic-105485-koreyst)
[![AI for Beginners](https://img.shields.io/badge/AI%20for%20Beginners-A3E635?style=for-the-badge&labelColor=E5E7EB&color=A3E635)](https://aka.ms/ai-beginners?WT.mc_id=academic-105485-koreyst)
[![Cybersecurity for Beginners](https://img.shields.io/badge/Cybersecurity%20for%20Beginners-F97316?style=for-the-badge&labelColor=E5E7EB&color=F97316)](https://github.com/microsoft/Security-101?WT.mc_id=academic-96948-sayoung)
[![Web Dev for Beginners](https://img.shields.io/badge/Web%20Dev%20for%20Beginners-EC4899?style=for-the-badge&labelColor=E5E7EB&color=EC4899)](https://aka.ms/webdev-beginners?WT.mc_id=academic-105485-koreyst)
[![IoT for Beginners](https://img.shields.io/badge/IoT%20for%20Beginners-14B8A6?style=for-the-badge&labelColor=E5E7EB&color=14B8A6)](https://aka.ms/iot-beginners?WT.mc_id=academic-105485-koreyst)
[![XR Development for Beginners](https://img.shields.io/badge/XR%20Development%20for%20Beginners-38BDF8?style=for-the-badge&labelColor=E5E7EB&color=38BDF8)](https://github.com/microsoft/xr-development-for-beginners?WT.mc_id=academic-105485-koreyst)

---
 
### कॉपिलट श्रृंखला
[![Copilot for AI Paired Programming](https://img.shields.io/badge/Copilot%20for%20AI%20Paired%20Programming-FACC15?style=for-the-badge&labelColor=E5E7EB&color=FACC15)](https://aka.ms/GitHubCopilotAI?WT.mc_id=academic-105485-koreyst)
[![Copilot for C#/.NET](https://img.shields.io/badge/Copilot%20for%20C%23/.NET-FBBF24?style=for-the-badge&labelColor=E5E7EB&color=FBBF24)](https://github.com/microsoft/mastering-github-copilot-for-dotnet-csharp-developers?WT.mc_id=academic-105485-koreyst)
[![Copilot Adventure](https://img.shields.io/badge/Copilot%20Adventure-FDE68A?style=for-the-badge&labelColor=E5E7EB&color=FDE68A)](https://github.com/microsoft/CopilotAdventures?WT.mc_id=academic-105485-koreyst)
<!-- CO-OP TRANSLATOR OTHER COURSES END -->

## सहायता प्राप्त करना

यदि आप मशीन लर्निंग सीखते समय या एआई अनुप्रयोग बनाने में फंस जाते हैं या आपके पास प्रश्न हैं, तो चिंता न करें — सहायता उपलब्ध है।

आप अन्य सीखने वालों और डेवलपर्स के साथ चर्चाओं में शामिल हो सकते हैं, प्रश्न पूछ सकते हैं, और अपनी विचारधाराएं समुदाय के साथ साझा कर सकते हैं।

- प्रश्न पूछने और दूसरों के साथ सीखने के लिए समुदाय में शामिल हों
- मशीन लर्निंग अवधारणाओं और परियोजना विचारों पर चर्चा करें
- अनुभवी डेवलपर्स से मार्गदर्शन प्राप्त करें

एक सहायक समुदाय आपके कौशल को विकसित करने और समस्याओं के समाधान को तेज़ करने का एक शानदार तरीका है।

[Microsoft Foundry Discord Community](https://discord.gg/nTYy5BXMWG)

यदि आपको बग, त्रुटियाँ मिलती हैं, या बेहतर बनाने के लिए सुझाव हैं, तो आप इस रिपॉजिटरी में एक **Issue** भी खोल सकते हैं ताकि समस्या रिपोर्ट की जा सके।

उत्पाद फीडबैक के लिए या मौजूदा समुदाय पोस्ट खोजने के लिए, डेवलपर फोरम देखें:

[![Microsoft Foundry Developer Forum](https://img.shields.io/badge/GitHub-Microsoft_Foundry_Developer_Forum-blue?style=for-the-badge&logo=github&color=000000&logoColor=fff)](https://aka.ms/foundry/forum)

## अतिरिक्त सीखने के सुझाव

- बेहतर समझ के लिए प्रत्येक पाठ के बाद नोटबुक की समीक्षा करें।
- अपने आप एल्गोरिदम को लागू करने का अभ्यास करें।
- सीखे गए सिद्धांतों का उपयोग करके वास्तविक दुनिया के डेटा सेट खोजें।

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**अस्वीकरण**:  
यह दस्तावेज़ AI अनुवाद सेवा [Co-op Translator](https://github.com/Azure/co-op-translator) का उपयोग करके अनूदित किया गया है। जबकि हम सटीकता के लिए प्रयासरत हैं, कृपया ध्यान दें कि स्वचालित अनुवाद में त्रुटियां या असंगतियाँ हो सकती हैं। मूल दस्तावेज़ अपनी स्थानीय भाषा में अधिकारिक स्रोत माना जाना चाहिए। महत्वपूर्ण जानकारी के लिए, पेशेवर मानव अनुवाद की सिफारिश की जाती है। इस अनुवाद के उपयोग से उत्पन्न किसी भी गलतफहमी या गलत व्याख्या के लिए हम जिम्मेदार नहीं हैं।
<!-- CO-OP TRANSLATOR DISCLAIMER END -->