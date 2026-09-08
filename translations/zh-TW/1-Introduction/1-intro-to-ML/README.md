# 機器學習簡介

## [課前測驗](https://ff-quizzes.netlify.app/en/ml/)

---

[![針對初學者的機器學習 - 機器學習入門](https://img.youtube.com/vi/6mSx_KJxcHI/0.jpg)](https://youtu.be/6mSx_KJxcHI "針對初學者的機器學習 - 機器學習入門")

> 🎥 點擊上方圖片觀看本課程的短片教學。

歡迎來到這門初學者的經典機器學習課程！無論你是完全不熟悉這個領域的初學者，或是希望加強某部分知識的有經驗 ML 從業者，我們都很高興你加入我們！我們希望為你的機器學習學習打造一個友善的起點，也歡迎你提出[反饋](https://github.com/microsoft/ML-For-Beginners/discussions)，我們會評估、回應並考慮納入改進。

[![機器學習簡介](https://img.youtube.com/vi/h0e2HAPTGF4/0.jpg)](https://youtu.be/h0e2HAPTGF4 "機器學習簡介")

> 🎥 點擊上方圖片觀看影片：MIT 的 John Guttag 介紹機器學習

---
## 開始使用機器學習

在開始本課程內容之前，你需要先設定好電腦，以本地運行筆記本。

- <strong>透過這些影片設定你的電腦</strong>。使用以下連結學習[如何安裝 Python](https://youtu.be/CXZYvNRIAKM)及[設定文字編輯器](https://youtu.be/EU8eayHWoZg)以便開發。
- **學習 Python**。建議具備 [Python](https://docs.microsoft.com/learn/paths/python-language/?WT.mc_id=academic-77952-leestott) 的基本認識，這是數據科學家常用的程式語言，也是本課程中的主要語言。
- **學習 Node.js 和 JavaScript**。本課程部分時候會使用 JavaScript 來建立網頁應用，所以請安裝 [node](https://nodejs.org) 與 [npm](https://www.npmjs.com/)，並準備好 [Visual Studio Code](https://code.visualstudio.com/) 來開發 Python 和 JavaScript。
- **建立 GitHub 帳號**。既然你已找到本課程於 [GitHub](https://github.com)，可能已經有帳號，如無，請註冊一個並分叉此課程內容以作自用。（也歡迎給我們點個⭐️）
- **熟悉 Scikit-learn**。了解 [Scikit-learn](https://scikit-learn.org/stable/user_guide.html)，這是一組在課程中常用到的機器學習函式庫。

---
## 什麼是機器學習？

「機器學習」這個詞是當今最流行且經常使用的術語之一。假如你對科技有一定程度的熟悉，不論你屬於哪個領域，你很可能至少聽過這個詞一次。不過，機器學習的實際運作對大多數人來說還是一個謎。對於機器學習初學者來說，這個主題有時會感到壓力很大。因此，了解機器學習到底是什麼，並藉由實際案例循序漸進學習它，是非常重要的。

---
## 熱潮曲線

![ml hype curve](../../../../translated_images/zh-TW/hype.07183d711a17aafe.webp)

> Google 趨勢顯示「機器學習」一詞近期的熱潮曲線

---
## 神秘的宇宙

我們生活在充滿神秘的宇宙中。像史蒂芬·霍金、艾爾伯特·愛因斯坦等偉大科學家，都終身致力於尋找揭開周遭世界奧秘的有意義資訊。這是人類學習的本質：一個孩子隨著成長，逐年學習新事物並揭露他們世界的結構。

---
## 兒童的大腦

兒童的大腦和感覺器官感知周圍事實，逐漸學習生命中隱藏的模式，這幫助兒童建構邏輯規則以識別所學的模式。人類大腦的學習過程造就了人類成為世界上最精密的生物。透過持續學習隱藏模式並在這些模式基礎上創新，使我們能夠在一生中逐步提升自己。這種學習能力和不斷演變的能力，與稱為[大腦可塑性](https://www.simplypsychology.org/brain-plasticity.html)的概念有關。粗略地說，我們可以在大腦的學習過程與機器學習概念之間找到一些啟發性的相似之處。

---
## 人體大腦

[人體大腦](https://www.livescience.com/29365-human-brain.html)感知真實世界的事物，處理所感知的信息，基於情境做出理性決策並執行行動。這就是我們所說的智能行為。當我們為機器編寫模擬智能行為過程的程式時，這稱為人工智慧（AI）。

---
## 一些術語

雖然這些術語易混淆，但機器學習（ML）是人工智慧的重要子集。**機器學習關注於使用專用算法，從感知到的數據中揭露有意義的信息及隱藏的模式，以佐證理性決策過程**。

---
## AI、ML、深度學習

![AI, ML, deep learning, data science](../../../../translated_images/zh-TW/ai-ml-ds.537ea441b124ebf6.webp)

> 一張說明 AI、ML、深度學習與資料科學間關係的圖示。資訊圖表由 [Jen Looper](https://twitter.com/jenlooper) 製作，靈感來源於[此圖](https://softwareengineering.stackexchange.com/questions/366996/distinction-between-ai-ml-neural-networks-deep-learning-and-data-mining)

---
## 將涵蓋的概念

本課程將只介紹初學者必須了解的機器學習核心概念。我們主要使用 Scikit-learn，這是許多學生學習基礎時優良的函式庫。我們希望提供良好的基礎知識讓你未來學習更廣泛的人工智慧或深度學習概念時更為扎實。

---
## 本課程你將學習：

- 機器學習核心概念
- 機器學習歷史
- 機器學習與公平性
- 回歸機器學習技術
- 分類機器學習技術
- 分群機器學習技術
- 自然語言處理相關機器學習技術
- 時序預測相關機器學習技術
- 強化學習
- 機器學習的實際應用

---
## 我們不會涵蓋的內容

- 深度學習
- 神經網路
- 人工智慧

為了讓學習體驗更佳，我們將避免涉及神經網路的複雜性、使用多層神經網路建立模型的「深度學習」以及人工智慧相關主題，那些將在其他課程中詳細討論。我們也會提供即將推出的資料科學課程，專注於這個更廣泛領域的部分。

---
## 為何要學機器學習？

從系統觀點看，機器學習是創建自動化系統，以從數據中學習隱藏模式，協助做出智能決策。

這個動機大致來自人腦如何依據感知到的外部數據學習特定事物。

✅ 思考一下：為什麼企業會想用機器學習策略，而不是建立硬編碼的規則引擎呢？

---
## 為何資料品質很重要

高品質資料可改善模型性能。即使使用先進的機器學習算法，品質差或雜訊多的資料也會導致不準確的預測結果。

---
## 機器學習的應用

機器學習應用幾乎無處不在，與我們社會中由智慧型手機、連網裝置及其他系統產生的海量數據同樣普遍。鑑於尖端機器學習算法的強大潛力，研究人員一直在探索它們解決多維度、多學科現實問題的能力，且成效良好。

---
## 已應用機器學習的範例

<strong>你可以在許多場景中使用機器學習</strong>：

- 透過病患的醫療史或報告預測疾病風險。
- 利用氣象資料預測天氣事件。
- 解析文本的情感傾向。
- 偵測假新聞以阻止宣傳擴散。

金融、經濟、地球科學、太空探索、生醫工程、認知科學，甚至人文領域都已採用機器學習，以解決該領域中龐大且繁重的數據處理問題。

---
## 結論

機器學習透過從真實世界或生成數據中發掘有意義的洞見，自動化模式發現過程。它在商業、健康與金融應用等領域已被證明極具價值。

未來，理解機器學習基礎將成為所有領域人士的必備能力，因為其廣泛採用將持續擴大。

---
# 🚀 挑戰

用紙筆或線上軟體如 [Excalidraw](https://excalidraw.com/) 繪製你對 AI、ML、深度學習與資料科學間差異的理解。並列出每種技術適合解決的問題。

# [課後測驗](https://ff-quizzes.netlify.app/en/ml/)

---
# 複習與自學

欲了解更多在雲端如何使用機器學習算法，請參考此[學習路徑](https://docs.microsoft.com/learn/paths/create-no-code-predictive-models-azure-machine-learning/?WT.mc_id=academic-77952-leestott)。

搭配此[學習路徑](https://docs.microsoft.com/learn/modules/introduction-to-machine-learning/?WT.mc_id=academic-77952-leestott)學習機器學習基礎。

---
# 作業

[開始上手](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**免責聲明**：
此文件已使用 AI 翻譯服務 [Co-op Translator](https://github.com/Azure/co-op-translator) 進行翻譯。雖然我們努力追求準確性，但請注意自動翻譯可能包含錯誤或不準確之處。原始文件的母語版本應視為權威來源。對於關鍵資訊，建議採用專業人工翻譯。我們不對因使用此翻譯所產生的任何誤解或誤譯承擔責任。
<!-- CO-OP TRANSLATOR DISCLAIMER END -->