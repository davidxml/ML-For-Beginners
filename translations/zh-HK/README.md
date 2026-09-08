[![GitHub license](https://img.shields.io/github/license/microsoft/ML-For-Beginners.svg)](https://github.com/microsoft/ML-For-Beginners/blob/master/LICENSE)
[![GitHub contributors](https://img.shields.io/github/contributors/microsoft/ML-For-Beginners.svg)](https://GitHub.com/microsoft/ML-For-Beginners/graphs/contributors/)
[![GitHub issues](https://img.shields.io/github/issues/microsoft/ML-For-Beginners.svg)](https://GitHub.com/microsoft/ML-For-Beginners/issues/)
[![GitHub pull-requests](https://img.shields.io/github/issues-pr/microsoft/ML-For-Beginners.svg)](https://GitHub.com/microsoft/ML-For-Beginners/pulls/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)

[![GitHub watchers](https://img.shields.io/github/watchers/microsoft/ML-For-Beginners.svg?style=social&label=Watch)](https://GitHub.com/microsoft/ML-For-Beginners/watchers/)
[![GitHub forks](https://img.shields.io/github/forks/microsoft/ML-For-Beginners.svg?style=social&label=Fork)](https://GitHub.com/microsoft/ML-For-Beginners/network/)
[![GitHub stars](https://img.shields.io/github/stars/microsoft/ML-For-Beginners.svg?style=social&label=Star)](https://GitHub.com/microsoft/ML-For-Beginners/stargazers/)

### 🌐 多語言支援

#### 透過 GitHub Action 支援（自動且永遠保持最新）

<!-- CO-OP TRANSLATOR LANGUAGES TABLE START -->
[阿拉伯文](../ar/README.md) | [孟加拉文](../bn/README.md) | [保加利亞文](../bg/README.md) | [緬甸語](../my/README.md) | [中文（簡體）](../zh-CN/README.md) | [中文（繁體，香港）](./README.md) | [中文（繁體，澳門）](../zh-MO/README.md) | [中文（繁體，台灣）](../zh-TW/README.md) | [克羅地亞文](../hr/README.md) | [捷克文](../cs/README.md) | [丹麥文](../da/README.md) | [荷蘭文](../nl/README.md) | [愛沙尼亞文](../et/README.md) | [芬蘭文](../fi/README.md) | [法文](../fr/README.md) | [德文](../de/README.md) | [希臘文](../el/README.md) | [希伯來文](../he/README.md) | [印地文](../hi/README.md) | [匈牙利文](../hu/README.md) | [印尼文](../id/README.md) | [義大利文](../it/README.md) | [日文](../ja/README.md) | [卡納達文](../kn/README.md) | [高棉文](../km/README.md) | [韓文](../ko/README.md) | [立陶宛文](../lt/README.md) | [馬來文](../ms/README.md) | [馬拉雅拉姆文](../ml/README.md) | [馬拉地文](../mr/README.md) | [尼泊爾文](../ne/README.md) | [奈及利亞俚語](../pcm/README.md) | [挪威文](../no/README.md) | [波斯文（法爾西語）](../fa/README.md) | [波蘭文](../pl/README.md) | [葡萄牙文（巴西）](../pt-BR/README.md) | [葡萄牙文（葡萄牙）](../pt-PT/README.md) | [旁遮普文（古魯穆奇體）](../pa/README.md) | [羅馬尼亞文](../ro/README.md) | [俄文](../ru/README.md) | [塞爾維亞文（西里爾字母）](../sr/README.md) | [斯洛伐克文](../sk/README.md) | [斯洛文尼亞文](../sl/README.md) | [西班牙文](../es/README.md) | [斯瓦希里文](../sw/README.md) | [瑞典文](../sv/README.md) | [他加祿文（菲律賓語）](../tl/README.md) | [泰米爾文](../ta/README.md) | [泰盧固文](../te/README.md) | [泰文](../th/README.md) | [土耳其文](../tr/README.md) | [烏克蘭文](../uk/README.md) | [烏爾都文](../ur/README.md) | [越南文](../vi/README.md)

> **想本地克隆？**
>
> 此儲存庫包含超過 50 種語言翻譯，會大幅增加下載大小。若想不含翻譯克隆，請使用 sparse checkout：
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
> 這樣你能以更快速度完成下載，並擁有完成課程所需的所有內容。
<!-- CO-OP TRANSLATOR LANGUAGES TABLE END -->

#### 加入我們的社群

[![Microsoft Foundry Discord](https://dcbadge.limes.pink/api/server/nTYy5BXMWG)](https://discord.gg/nTYy5BXMWG)

我們推出了 Discord 上的「與 AI 一起學習」系列，了解詳情並於 2025 年 9 月 18 至 30 日加入我們，網址：[Learn with AI Series](https://aka.ms/learnwithai/discord)。你將學習如何用 GitHub Copilot 於資料科學中獲得小技巧和秘訣。

![Learn with AI series](../../translated_images/zh-HK/3.9b58fd8d6c373c20.webp)

# 初學者機器學習課程大綱

> 🌍 環遊世界，透過世界文化探索機器學習 🌍

微軟的 Cloud Advocates 很高興提供一套 12 週，共 26 節課的機器學習課程。在本課程中，你會學習所謂的 <strong>經典機器學習</strong>，主要使用 Scikit-learn 函式庫，避開深度學習，後者可在我們的 [AI初學者課程](https://aka.ms/ai4beginners)中學習。這些課程也可搭配我們的 ['初學者資料科學'課程](https://aka.ms/ds4beginners)使用！

跟著我們環遊世界，將這些經典技術應用於各地的數據。每節課都包含課前及課後小測驗、書面指引、解答、作業等。我們的專案導向教學法讓你在動手實做中學習，是幫助新技能內化的有效方式。

**✍️ 衷心感謝作者團隊：** Jen Looper, Stephen Howell, Francesca Lazzeri, Tomomi Imura, Cassie Breviu, Dmitry Soshnikov, Chris Noring, Anirban Mukherjee, Ornella Altunyan, Ruth Yakubu 和 Amy Boyd

**🎨 亦感謝插畫師：** Tomomi Imura, Dasani Madipalli, 和 Jen Looper

**🙏 特別感謝微軟學生大使作者、審核與內容貢獻者，尤其是 Rishit Dagli、Muhammad Sakib Khan Inan、Rohan Raj、Alexandru Petrescu、Abhishek Jaiswal、Nawrin Tabassum、Ioan Samuila 和 Snigdha Agarwal**

**🤩 額外感謝微軟學生大使 Eric Wanjau、Jasleen Sondhi 和 Vidushi Gupta 協助 R 課程！**

# 開始使用

請依照以下步驟：
1. **Fork 儲存庫**：點擊頁面右上角的「Fork」按鈕。
2. **Clone 儲存庫**： `git clone https://github.com/microsoft/ML-For-Beginners.git`

> [在我們的 Microsoft Learn 集合中尋找本課程所有附加資源](https://learn.microsoft.com/en-us/collections/qrqzamz1nn2wx3?WT.mc_id=academic-77952-bethanycheum)

> 🔧 **需要幫助？** 請查看我們的 [故障排除指南](TROUBLESHOOTING.md)，協助解決安裝、設定和執行課程常見問題。

**[學生](https://aka.ms/student-page)**，使用本課程時，請 fork 整個儲存庫到你的 GitHub 帳號，自行或與組員一起完成練習：

- 先從課前小測開始。
- 閱讀課程內容並完成任務，每個知識檢測時刻停思考。
- 嘗試理解課程內容後自己動手創作專案，解答程式碼位於每個專案課程中的 `/solution` 資料夾。
- 完成課後小測。
- 完成挑戰題。
- 完成作業。
- 完成一組課程後，前往 [討論區](https://github.com/microsoft/ML-For-Beginners/discussions)，透過填寫合適的 PAT 評分表「大聲學習」。'PAT' 是進度評估工具，透過表格幫助你學習。你也可以回應他人 PAT，一起進步。

> 進一步學習，我們推薦參考以下 [Microsoft Learn](https://docs.microsoft.com/en-us/users/jenlooper-2911/collections/k7o7tg1gp306q4?WT.mc_id=academic-77952-leestott) 模組及學習路徑。

<strong>教師們</strong>，我們有在 [使用建議](for-teachers.md) 提供一些參考。

---

## 影片教學

部分課程提供短影片說明。你可於課程中直接觀看，或至 [Microsoft Developer YouTube 頻道的 ML for Beginners 播放列表](https://aka.ms/ml-beginners-videos)點擊下方圖片觀看。

[![ML for beginners banner](../../translated_images/zh-HK/ml-for-beginners-video-banner.63f694a100034bc6.webp)](https://aka.ms/ml-beginners-videos)

---

## 認識團隊

[![Promo video](../../images/ml.gif)](https://youtu.be/Tj1XWrDSYJU)

**GIF 製作由** [Mohit Jaisal](https://linkedin.com/in/mohitjaisal)

> 🎥 點擊上方圖片觀看本專案及團隊影片介紹！

---

## 教學法

本課程設計遵循兩大教學原則：確保內容是實作 <strong>專案導向</strong>，並包含 <strong>頻繁的小測驗</strong>。此外，本課程有統一的 <strong>主題</strong> ，增強一致性。

透過專案對應課程內容，讓學習更投入並提升觀念記憶。課前低壓力小測目的是啟發學習意圖，課後測驗則有助鞏固記憶。本課程設計靈活有趣，可全程參與或挑選部分學習。專案由淺入深，12 週結束時達較高難度。本課程另附機器學習真實應用的補充章節，可用做額外加分或討論話題。

> 查看我們的 [行為守則](CODE_OF_CONDUCT.md)、[貢獻指南](CONTRIBUTING.md)、[翻譯](..) 及 [故障排除](TROUBLESHOOTING.md) 文件，我們歡迎你的建設性反饋！

## 每節課程包含

- 可選的手繪筆記
- 可選的附加影片
- 影片講解（部分課程）
- [課前暖身小測驗](https://ff-quizzes.netlify.app/en/ml/)
- 書面課程內容
- 專案課程時，逐步專案建立指南
- 知識檢測
- 挑戰題
- 補充閱讀
- 作業
- [課後小測驗](https://ff-quizzes.netlify.app/en/ml/)
> <strong>關於語言的說明</strong>：這些課程主要以 Python 編寫，但許多也有提供 R 語言版本。要完成 R 課程，請前往 `/solution` 資料夾尋找 R 課程。它們包含一個 .rmd 副檔名，代表 **R Markdown** 檔案，這可簡單定義為在一個 `Markdown 文件` 中嵌入 `程式碼區塊`（R 或其他語言）和一個 `YAML 標頭`（指導如何格式化輸出，如 PDF）。因此，它是數據科學的典範創作框架，因為它允許你將程式碼、輸出結果和思考結合起來，並允許你用 Markdown 書寫。此外，R Markdown 文件可以渲染成 PDF、HTML 或 Word 等輸出格式。

> <strong>關於測驗的說明</strong>：所有測驗都包含在 [Quiz App folder](../../quiz-app) 中，共 52 組，每組有三個問題。它們在課程中有連結，但測驗應用程式可以本地執行；請遵循 `quiz-app` 資料夾中的指示進行本地架設或部署到 Azure。

| 課程編號 |                             主題                              |                   課程分組                   | 學習目標                                                                                                                     |                                                              合作課程                                                               |                        作者                        |
| :------: | :------------------------------------------------------------: | :------------------------------------------: | ---------------------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------: |
|    01    |                      機器學習簡介                      |      [Introduction](1-Introduction/README.md)       | 學習機器學習的基本概念                                                                                                        |                                             [課程](1-Introduction/1-intro-to-ML/README.md)                                             |                       Muhammad                       |
|    02    |                       機器學習的歷史                       |      [Introduction](1-Introduction/README.md)       | 學習此領域的歷史背景                                                                                                          |                                            [課程](1-Introduction/2-history-of-ML/README.md)                                            |                     Jen and Amy                      |
|    03    |                       公平性與機器學習                       |      [Introduction](1-Introduction/README.md)       | 建構及應用機器學習模型時，學生應考慮哪些重要的公平性哲學議題？                                                                |                                              [課程](1-Introduction/3-fairness/README.md)                                               |                        Tomomi                        |
|    04    |                       機器學習技術                       |      [Introduction](1-Introduction/README.md)       | 機器學習研究者用哪些技術來建構機器學習模型？                                                                                  |                                          [課程](1-Introduction/4-techniques-of-ML/README.md)                                           |                    Chris and Jen                     |
|    05    |                    回歸模型介紹                    |        [Regression](2-Regression/README.md)         | 使用 Python 和 Scikit-learn 入門回歸模型                                                                                      |         [Python](2-Regression/1-Tools/README.md) • [R](../../2-Regression/1-Tools/solution/R/lesson_1.html)         |      Jen • Eric Wanjau       |
|    06    |                北美南瓜價格 🎃                |        [Regression](2-Regression/README.md)         | 進行視覺化以及資料清理以準備機器學習                                                                                          |          [Python](2-Regression/2-Data/README.md) • [R](../../2-Regression/2-Data/solution/R/lesson_2.html)          |      Jen • Eric Wanjau       |
|    07    |                北美南瓜價格 🎃                |        [Regression](2-Regression/README.md)         | 建構線性與多項式回歸模型                                                                                                      |        [Python](2-Regression/3-Linear/README.md) • [R](../../2-Regression/3-Linear/solution/R/lesson_3.html)        |      Jen and Dmitry • Eric Wanjau       |
|    08    |                北美南瓜價格 🎃                |        [Regression](2-Regression/README.md)         | 建構邏輯斯迴歸模型                                                                                                            |     [Python](2-Regression/4-Logistic/README.md) • [R](../../2-Regression/4-Logistic/solution/R/lesson_4.html)      |      Jen • Eric Wanjau       |
|    09    |                          網頁應用程式 🔌                          |           [Web App](3-Web-App/README.md)            | 建構網頁應用程式以使用你的訓練模型                                                                                            |                                                 [Python](3-Web-App/1-Web-App/README.md)                                                  |                         Jen                          |
|    10    |                 分類介紹                 |    [Classification](4-Classification/README.md)     | 清理、準備並視覺化你的資料；分類概論                                                                                          | [Python](4-Classification/1-Introduction/README.md) • [R](../../4-Classification/1-Introduction/solution/R/lesson_10.html)  | Jen and Cassie • Eric Wanjau |
|    11    |             美味的亞洲和印度料理 🍜             |    [Classification](4-Classification/README.md)     | 分類器介紹                                                                                                                    | [Python](4-Classification/2-Classifiers-1/README.md) • [R](../../4-Classification/2-Classifiers-1/solution/R/lesson_11.html) | Jen and Cassie • Eric Wanjau |
|    12    |             美味的亞洲和印度料理 🍜             |    [Classification](4-Classification/README.md)     | 更多分類器                                                                                                                   | [Python](4-Classification/3-Classifiers-2/README.md) • [R](../../4-Classification/3-Classifiers-2/solution/R/lesson_12.html) | Jen and Cassie • Eric Wanjau |
|    13    |             美味的亞洲和印度料理 🍜             |    [Classification](4-Classification/README.md)     | 使用你的模型建構推薦網頁應用程式                                                                                            |                                              [Python](4-Classification/4-Applied/README.md)                                              |                         Jen                          |
|    14    |                   分群介紹                   |        [Clustering](5-Clustering/README.md)         | 清理、準備並視覺化你的資料；分群介紹                                                                                          |         [Python](5-Clustering/1-Visualize/README.md) • [R](../../5-Clustering/1-Visualize/solution/R/lesson_14.html)         |      Jen • Eric Wanjau       |
|    15    |              探索尼日利亞音樂喜好 🎧              |        [Clustering](5-Clustering/README.md)         | 探索 K 均值分群法                                                                                                            |           [Python](5-Clustering/2-K-Means/README.md) • [R](../../5-Clustering/2-K-Means/solution/R/lesson_15.html)           |      Jen • Eric Wanjau       |
|    16    |        自然語言處理介紹 ☕️         |   [Natural language processing](6-NLP/README.md)    | 透過建立簡單機器人學習 NLP 基礎知識                                                                                          |                                             [Python](6-NLP/1-Introduction-to-NLP/README.md)                                              |                       Stephen                        |
|    17    |                      常見 NLP 任務 ☕️                      |   [Natural language processing](6-NLP/README.md)    | 增進 NLP 知識，了解處理語言結構時所需的常見任務                                                                               |                                                    [Python](6-NLP/2-Tasks/README.md)                                                     |                       Stephen                        |
|    18    |             翻譯與情感分析 ♥️              |   [Natural language processing](6-NLP/README.md)    | 使用簡·奧斯汀進行翻譯與情感分析                                                                                              |                                            [Python](6-NLP/3-Translation-Sentiment/README.md)                                             |                       Stephen                        |
|    19    |                  歐洲浪漫旅館 ♥️                  |   [Natural language processing](6-NLP/README.md)    | 透過旅館評論進行情感分析 1                                                                                                  |                                               [Python](6-NLP/4-Hotel-Reviews-1/README.md)                                                |                       Stephen                        |
|    20    |                  歐洲浪漫旅館 ♥️                  |   [Natural language processing](6-NLP/README.md)    | 透過旅館評論進行情感分析 2                                                                                                  |                                               [Python](6-NLP/5-Hotel-Reviews-2/README.md)                                                |                       Stephen                        |
|    21    |            時間序列預測介紹             |        [Time series](7-TimeSeries/README.md)        | 時間序列預測介紹                                                                                                            |                                             [Python](7-TimeSeries/1-Introduction/README.md)                                              |                      Francesca                       |
|    22    | ⚡️ 世界用電量 ⚡️ - 使用 ARIMA 的時間序列預測 |        [Time series](7-TimeSeries/README.md)        | 使用 ARIMA 進行時間序列預測                                                                                                |                                                 [Python](7-TimeSeries/2-ARIMA/README.md)                                                 |                      Francesca                       |
|    23    |  ⚡️ 世界用電量 ⚡️ - 使用 SVR 的時間序列預測  |        [Time series](7-TimeSeries/README.md)        | 使用支援向量回歸器進行時間序列預測                                                                                          |                                                  [Python](7-TimeSeries/3-SVR/README.md)                                                  |                       Anirban                        |
|    24    |             強化學習介紹             | [Reinforcement learning](8-Reinforcement/README.md) | 使用 Q-Learning 介紹強化學習                                                                                                |                                             [Python](8-Reinforcement/1-QLearning/README.md)                                              |                        Dmitry                        |
|    25    |                 幫助彼得避開狼！🐺                  | [Reinforcement learning](8-Reinforcement/README.md) | 強化學習 Gym                                                                                                               |                                                [Python](8-Reinforcement/2-Gym/README.md)                                                 |                        Dmitry                        |
|  附錄   |            現實世界中的機器學習場景與應用            |      [ML in the Wild](9-Real-World/README.md)       | 有趣且啟發性的經典機器學習真實應用                                                                                          |                                             [課程](9-Real-World/1-Applications/README.md)                                              |                         Team                         |
|  附錄   |            使用 RAI 控制台進行機器學習模型除錯          |      [ML in the Wild](9-Real-World/README.md)       | 使用 Responsible AI 控制台元件進行機器學習模型除錯                                                                                 |                                             [課程](9-Real-World/2-Debugging-ML-Models/README.md)                                              |                         Ruth Yakubu                       |

> [在我們的 Microsoft Learn 系列中找到本課程的所有額外資源](https://learn.microsoft.com/en-us/collections/qrqzamz1nn2wx3?WT.mc_id=academic-77952-bethanycheum)

## 離線存取

你可以使用 [Docsify](https://docsify.js.org/#/) 離線瀏覽此文件。Fork 此倉庫，在本機安裝 [Docsify](https://docsify.js.org/#/quickstart)，然後在此倉庫根目錄下執行 `docsify serve`。網站會在本地主機（localhost） 3000 埠上提供：`localhost:3000`。

## PDF 檔案

在此處找到帶有連結的課程大綱 pdf [here](https://microsoft.github.io/ML-For-Beginners/pdf/readme.pdf)。

## 🎒 其他課程

我們團隊製作了其他課程！請參考：

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
 
### 生成式 AI 系列
[![Generative AI for Beginners](https://img.shields.io/badge/Generative%20AI%20for%20Beginners-8B5CF6?style=for-the-badge&labelColor=E5E7EB&color=8B5CF6)](https://github.com/microsoft/generative-ai-for-beginners?WT.mc_id=academic-105485-koreyst)
[![Generative AI (.NET)](https://img.shields.io/badge/Generative%20AI%20(.NET)-9333EA?style=for-the-badge&labelColor=E5E7EB&color=9333EA)](https://github.com/microsoft/Generative-AI-for-beginners-dotnet?WT.mc_id=academic-105485-koreyst)
[![Generative AI (Java)](https://img.shields.io/badge/Generative%20AI%20(Java)-C084FC?style=for-the-badge&labelColor=E5E7EB&color=C084FC)](https://github.com/microsoft/generative-ai-for-beginners-java?WT.mc_id=academic-105485-koreyst)
[![Generative AI (JavaScript)](https://img.shields.io/badge/Generative%20AI%20(JavaScript)-E879F9?style=for-the-badge&labelColor=E5E7EB&color=E879F9)](https://github.com/microsoft/generative-ai-with-javascript?WT.mc_id=academic-105485-koreyst)

---
 
### 核心學習
[![ML for Beginners](https://img.shields.io/badge/ML%20for%20Beginners-22C55E?style=for-the-badge&labelColor=E5E7EB&color=22C55E)](https://aka.ms/ml-beginners?WT.mc_id=academic-105485-koreyst)
[![Data Science for Beginners](https://img.shields.io/badge/Data%20Science%20for%20Beginners-84CC16?style=for-the-badge&labelColor=E5E7EB&color=84CC16)](https://aka.ms/datascience-beginners?WT.mc_id=academic-105485-koreyst)
[![AI for Beginners](https://img.shields.io/badge/AI%20for%20Beginners-A3E635?style=for-the-badge&labelColor=E5E7EB&color=A3E635)](https://aka.ms/ai-beginners?WT.mc_id=academic-105485-koreyst)
[![Cybersecurity for Beginners](https://img.shields.io/badge/Cybersecurity%20for%20Beginners-F97316?style=for-the-badge&labelColor=E5E7EB&color=F97316)](https://github.com/microsoft/Security-101?WT.mc_id=academic-96948-sayoung)
[![Web Dev for Beginners](https://img.shields.io/badge/Web%20Dev%20for%20Beginners-EC4899?style=for-the-badge&labelColor=E5E7EB&color=EC4899)](https://aka.ms/webdev-beginners?WT.mc_id=academic-105485-koreyst)
[![IoT for Beginners](https://img.shields.io/badge/IoT%20for%20Beginners-14B8A6?style=for-the-badge&labelColor=E5E7EB&color=14B8A6)](https://aka.ms/iot-beginners?WT.mc_id=academic-105485-koreyst)
[![XR Development for Beginners](https://img.shields.io/badge/XR%20Development%20for%20Beginners-38BDF8?style=for-the-badge&labelColor=E5E7EB&color=38BDF8)](https://github.com/microsoft/xr-development-for-beginners?WT.mc_id=academic-105485-koreyst)

---
 
### Copilot 系列
[![Copilot for AI Paired Programming](https://img.shields.io/badge/Copilot%20for%20AI%20Paired%20Programming-FACC15?style=for-the-badge&labelColor=E5E7EB&color=FACC15)](https://aka.ms/GitHubCopilotAI?WT.mc_id=academic-105485-koreyst)
[![Copilot for C#/.NET](https://img.shields.io/badge/Copilot%20for%20C%23/.NET-FBBF24?style=for-the-badge&labelColor=E5E7EB&color=FBBF24)](https://github.com/microsoft/mastering-github-copilot-for-dotnet-csharp-developers?WT.mc_id=academic-105485-koreyst)
[![Copilot Adventure](https://img.shields.io/badge/Copilot%20Adventure-FDE68A?style=for-the-badge&labelColor=E5E7EB&color=FDE68A)](https://github.com/microsoft/CopilotAdventures?WT.mc_id=academic-105485-koreyst)
<!-- CO-OP TRANSLATOR OTHER COURSES END -->

## 尋求協助

如果您在學習機器學習或構建 AI 應用時遇到困難或有問題，別擔心 — 有幫助可以取得。

您可以與其他學習者和開發者一起參與討論、提問，以及分享您的想法。

- 加入社群，向他人提問並一起學習
- 討論機器學習概念和專案想法
- 獲取有經驗開發者的指導

有支援性的社群是成長技能和更快解決問題的好方法。

[Microsoft Foundry Discord 社群](https://discord.gg/nTYy5BXMWG)

如果您遇到錯誤、異常或有改進建議，也可以在此庫中開啟 **Issue** 來回報問題。

如需產品反饋或搜尋現有社群帖子，請訪問開發者論壇：

[![Microsoft Foundry Developer Forum](https://img.shields.io/badge/GitHub-Microsoft_Foundry_Developer_Forum-blue?style=for-the-badge&logo=github&color=000000&logoColor=fff)](https://aka.ms/foundry/forum)

## 其他學習建議

- 每課後複習筆記本以加深理解。
- 練習自己實作演算法。
- 運用所學概念探索真實世界資料集。

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**免責聲明**：  
本文件使用 AI 翻譯服務 [Co-op Translator](https://github.com/Azure/co-op-translator) 進行翻譯。雖然我們致力於確保準確性，但請注意自動翻譯可能包含錯誤或不準確之處。原始文件的母語版本應被視為權威來源。對於重要資訊，建議採用專業人工翻譯。我們對因使用此翻譯而引起的任何誤解或錯誤解釋概不負責。
<!-- CO-OP TRANSLATOR DISCLAIMER END -->