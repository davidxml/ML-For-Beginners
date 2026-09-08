# Introduction to machine learning

## [Pre-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

---

[![ML for beginners - Introduction to Machine Learning for Beginners](https://img.youtube.com/vi/6mSx_KJxcHI/0.jpg)](https://youtu.be/6mSx_KJxcHI "ML for beginners - Introduction to Machine Learning for Beginners")

> 🎥 點擊上方圖片觀看本課程的短片教學。

歡迎參加這個為初學者而設的經典機器學習課程！無論你是對這個主題毫無認識，還是有經驗的機器學習從業者想要溫習某個範疇，我們都非常歡迎你加入！我們希望為你的機器學習學習之路建立一個友善的起點，並很樂意評估、回應並納入你的[反饋](https://github.com/microsoft/ML-For-Beginners/discussions)。

[![Introduction to ML](https://img.youtube.com/vi/h0e2HAPTGF4/0.jpg)](https://youtu.be/h0e2HAPTGF4 "Introduction to ML")

> 🎥 點擊上方圖片觀看視頻：MIT 的 John Guttag 介紹機器學習

---
## Getting started with machine learning

在開始本課程之前，你需要先設定好你的電腦，準備好本地執行 notebook。

- <strong>跟著這些影片設定你的機器</strong>。利用以下連結學習[如何安裝 Python](https://youtu.be/CXZYvNRIAKM)於你的系統，以及[設定文字編輯器](https://youtu.be/EU8eayHWoZg)作開發環境。
- **學習 Python**。同時建議具備 [Python](https://docs.microsoft.com/learn/paths/python-language/?WT.mc_id=academic-77952-leestott) 的基本認識，這是數據科學家常用的程式語言，本課程亦會使用。
- **學習 Node.js 與 JavaScript**。本課程中建構網頁應用時會用到 JavaScript，因此你需要安裝 [node](https://nodejs.org) 及 [npm](https://www.npmjs.com/)，並安裝 [Visual Studio Code](https://code.visualstudio.com/) 作 Python 和 JavaScript 開發之用。
- **建立 GitHub 帳號**。既然你已在 [GitHub](https://github.com) 上找到我們，你可能已有帳號，如果沒有，請建立一個，然後 fork 本課程以便個人使用。（也歡迎給我們點個 star 😊）
- **探索 Scikit-learn**。熟悉一下我們課程中會提及的 [Scikit-learn](https://scikit-learn.org/stable/user_guide.html)，這是一套非常好用的機器學習函式庫，許多學生用它來學習基礎。

---
## What is machine learning?

「機器學習」是現今最流行且經常被提及的詞彙之一。如果你對科技有一點認識，無論你工作在哪個領域，很大機會你都至少聽過這個詞。可是大多數人對機器學習的運作原理仍是一團迷霧。對於初學者來說，這個主題有時候會顯得非常難以掌握。因此，理解機器學習究竟是甚麼、一步步透過實際例子來學習它是非常重要的。

---
## The hype curve

![ml hype curve](../../../../translated_images/zh-HK/hype.07183d711a17aafe.webp)

> Google 趨勢展示了「機器學習」一詞近期的「熱潮曲線」

---
## A mysterious universe

我們生活在一個充滿迷人謎團的宇宙中。偉大的科學家如 Stephen Hawking、Albert Einstein 等，獻出生命追尋有意義的資訊，以揭示我們周遭世界的奧秘。這是人類學習的本質：一個小孩隨著成長年年學習新事物，揭開其世界的結構。

---
## The child's brain

孩童的大腦與感官感知周遭事實，逐漸學習生命中隱藏的模式，這幫助孩子制定邏輯規則來識別學到的模式。人類大腦的學習過程造就了人類作為世界上最複雜的生物。透過持續發現隱藏模式並加以創新，我們能終生改進自己。這種學習能力及不斷演進的能力與一種稱為[大腦可塑性](https://www.simplypsychology.org/brain-plasticity.html)的概念相關。表面上，我們可從人腦學習過程與機器學習的概念中找到一些激勵性的相似之處。

---
## The human brain

[人類大腦](https://www.livescience.com/29365-human-brain.html)感知世界，處理所得資訊，依據情況做出理性決策，並執行某些行動。這就是我們所謂的智能行為。當我們將智能行為過程模擬編程到機器上，即稱為人工智能 (AI)。

---
## Some terminology

雖然這些名詞容易混淆，機器學習 (ML) 是人工智能的重要子集。**機器學習專注於使用專門的演算法，從感知到的數據中挖掘有意義的資訊與隱藏模式，輔助理性決策過程。**

---
## AI, ML, Deep Learning

![AI, ML, deep learning, data science](../../../../translated_images/zh-HK/ai-ml-ds.537ea441b124ebf6.webp)

> 顯示人工智能、機器學習、深度學習及數據科學之間關係的圖表。資訊圖由 [Jen Looper](https://twitter.com/jenlooper) 製作，靈感來自於[此圖](https://softwareengineering.stackexchange.com/questions/366996/distinction-between-ai-ml-neural-networks-deep-learning-and-data-mining)

---
## Concepts to cover

本課程將只涵蓋初學者必須知道的機器學習核心概念。我們主要使用 Scikit-learn 探討所謂的「經典機器學習」，這是許多學生用來學習基礎的優秀函式庫。想要理解更廣泛的人工智能或深度學習概念，紮實的機器學習基礎知識必不可少，因此我們希望在這裡提供它。

---
## In this course you will learn:

- 機器學習核心概念
- 機器學習歷史
- 機器學習與公平性
- 回歸機器學習技術
- 分類機器學習技術
- 叢集機器學習技術
- 自然語言處理機器學習技術
- 時間序列預測機器學習技術
- 強化學習
- 機器學習的真實世界應用

---
## What we will not cover

- 深度學習
- 神經網絡
- AI

為了提供更好的學習體驗，我們將避開神經網絡、「深度學習」(使用多層神經網絡的模型建構)以及人工智能相關複雜性，這些會於另一個課程詳細探討。我們亦將提供即將推出的數據科學課程以聚焦這個更大的領域。

---
## Why study machine learning?

從系統角度來看，機器學習是創造能自動學習數據中隱藏模式，幫助做出智能決策的系統。

這種動機大致來自於人腦如何根據外界感知到的數據學習某些東西。

✅ 想一想，為何企業會用機器學習策略，而非只用硬編碼規則引擎。

---
## Why data quality matters

高質量數據能提升模型表現。即使使用先進的機器學習演算法，劣質或雜訊數據仍會導致預測不準。

---
## Applications of machine learning

機器學習應用已存在於各處，如同不斷流動於我們社會的數據，這些數據由智能手機、連接設備及其他系統產生。考慮到尖端機器學習演算法的巨大潛力，研究人員一直探索其解決多維且跨學科現實問題的能力，並取得良好成果。

---
## Examples of applied ML

<strong>機器學習的用法多種多樣</strong>：

- 從患者的醫療歷史或報告預測疾病可能性。
- 利用天氣數據預測氣象事件。
- 理解文本的情感傾向。
- 偵測假新聞以停止宣傳散播。

金融、經濟、地球科學、太空探索、生物醫學工程、認知科學甚至人文領域都已採用機器學習來解決其艱難且數據處理密集的問題。

---
## Conclusion

機器學習通過從現實或生成的數據中挖掘有意義的洞見，自動化了模式發現過程。它已證明在商業、健康及金融等應用中極具價值。

不久的將來，由於其廣泛應用，各行各業人員學習機器學習基礎將成為必須。

---
# 🚀 Challenge

用紙張或線上軟件如[Excalidraw](https://excalidraw.com/)繪製你對人工智能、機器學習、深度學習及數據科學的差異的理解。加上這些技術適合解決問題的一些想法。

# [Post-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

---
# Review & Self Study

欲了解如何在雲端操作機器學習演算法，請跟隨此[學習路徑](https://docs.microsoft.com/learn/paths/create-no-code-predictive-models-azure-machine-learning/?WT.mc_id=academic-77952-leestott)。

上這個[學習路徑](https://docs.microsoft.com/learn/modules/introduction-to-machine-learning/?WT.mc_id=academic-77952-leestott)了解機器學習基礎。

---
# Assignment

[Get up and running](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**免責聲明**：
本文件由 AI 翻譯服務 [Co-op Translator](https://github.com/Azure/co-op-translator) 翻譯而成。雖然我們致力於確保準確性，但請注意，機器自動翻譯可能包含錯誤或不準確之處。原始文件的母語版本應被視為權威來源。對於重要資訊，建議進行專業人工翻譯。我們不對因使用本翻譯而產生的任何誤解或誤釋承擔責任。
<!-- CO-OP TRANSLATOR DISCLAIMER END -->