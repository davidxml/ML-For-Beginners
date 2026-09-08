# 机器学习简介

## [课前测验](https://ff-quizzes.netlify.app/en/ml/)

---

[![初学者机器学习 - 面向初学者的机器学习介绍](https://img.youtube.com/vi/6mSx_KJxcHI/0.jpg)](https://youtu.be/6mSx_KJxcHI "初学者机器学习 - 面向初学者的机器学习介绍")

> 🎥 点击上方图片观看本课的简短讲解视频。

欢迎加入本课程，面向初学者的经典机器学习课程！无论你是对这一主题完全陌生，还是有经验的机器学习从业者希望复习某个领域，我们都很高兴你加入我们！我们希望为你的机器学习学习打造一个友好的起点，并乐意评估、回应以及采纳你的[反馈](https://github.com/microsoft/ML-For-Beginners/discussions)。

[![机器学习介绍](https://img.youtube.com/vi/h0e2HAPTGF4/0.jpg)](https://youtu.be/h0e2HAPTGF4 "机器学习介绍")

> 🎥 点击上方图片观看视频：MIT的John Guttag介绍机器学习

---
## 开始机器学习之旅

在开始本课程之前，你需要准备好电脑并能在本地运行笔记本文件。

- <strong>通过这些视频配置你的机器</strong>。使用以下链接学习[如何安装Python](https://youtu.be/CXZYvNRIAKM)以及[设置文本编辑器](https://youtu.be/EU8eayHWoZg)进行开发。
- **学习Python**。建议具备[Python](https://docs.microsoft.com/learn/paths/python-language/?WT.mc_id=academic-77952-leestott)的基础知识，这是一种对数据科学家非常有用的编程语言，我们在本课程中会使用到。
- **学习Node.js和JavaScript**。本课程中构建网页应用时会用到JavaScript，因此你需要安装[node](https://nodejs.org)和[npm](https://www.npmjs.com/)，以及[Visual Studio Code](https://code.visualstudio.com/)用于Python和JavaScript开发。
- **创建一个GitHub账户**。既然你已经在[GitHub](https://github.com)找到这里，可能已有账户，若没有，请创建一个并Fork本课程，以方便你自己的使用。（也欢迎给我们点个星哦😊）
- **了解Scikit-learn**。熟悉[Scikit-learn](https://scikit-learn.org/stable/user_guide.html)，这是我们课程中常用的一套机器学习库。

---
## 什么是机器学习？

“机器学习”这个术语是当今最流行、最常用的词汇之一。如果你对科技有一定了解，无论你从事哪个领域，几乎可以肯定你至少听过这词一次。然而，机器学习的运作机制对多数人来说仍是个谜。对于机器学习初学者来说，这个主题有时会令人觉得难以应付。因此，理解机器学习到底是什么，并一步步通过实践案例去学习它就显得尤为重要。

---
## 热潮曲线

![ml hype curve](../../../../translated_images/zh-MO/hype.07183d711a17aafe.webp)

> Google Trends 显示了“机器学习”一词近期的“热潮曲线”

---
## 一个神秘的宇宙

我们生活在一个充满着迷人谜团的宇宙中。著名科学家如Stephen Hawking、Albert Einstein等人投入毕生心血寻找揭示世界奥秘的重要信息。这就是人类学习的本质：一个人类儿童逐年成长为成年人，认识新的事物，发掘他所处世界的结构和规律。

---
## 儿童的大脑

儿童的大脑和感官感知周围事实，并逐步学习到生活中的隐藏模式，帮助孩子形成逻辑规则来识别学到的模式。人脑的学习过程使人类成为这个世界上最复杂的生物。通过不断地发现隐藏模式并加以创新，我们得以不断提升自我，这种学习能力和不断进化的能力与[脑可塑性](https://www.simplypsychology.org/brain-plasticity.html)的概念相关联。从表面上看，我们可以发现人脑的学习过程和机器学习理念之间存在激励性的相似点。

---
## 人脑

[人脑](https://www.livescience.com/29365-human-brain.html)感知来自现实世界的信息，处理所感知的信息，做出理性决策，并基于情境执行特定动作。这就是我们所说的智能行为。当我们向机器编写这种智能行为过程的仿真时，这就是人工智能（AI）。

---
## 一些术语

虽然概念之间可能混淆，机器学习（ML）是人工智能的重要子集。**机器学习关注的是利用专门算法从感知到的数据中发现有意义的信息和隐藏模式，从而支持理性决策的过程**。

---
## AI、ML、深度学习

![AI, ML, deep learning, data science](../../../../translated_images/zh-MO/ai-ml-ds.537ea441b124ebf6.webp)

> 展示AI、ML、深度学习和数据科学间关系的示意图。信息图由[Jen Looper](https://twitter.com/jenlooper)制作，灵感来自[this graphic](https://softwareengineering.stackexchange.com/questions/366996/distinction-between-ai-ml-neural-networks-deep-learning-and-data-mining)

---
## 要覆盖的概念

本课程将只覆盖初学者必须了解的机器学习核心概念。我们主要使用Scikit-learn来讲解所谓的“经典机器学习”，这是许多学生用来学习基础知识的优秀库。要理解更广泛的人工智能或深度学习概念，坚实的机器学习基础知识是不可或缺的，因此我们想在这里提供它。

---
## 本课程你将学习：

- 机器学习的核心概念
- 机器学习的历史
- 机器学习与公平性
- 回归机器学习技术
- 分类机器学习技术
- 聚类机器学习技术
- 自然语言处理机器学习技术
- 时间序列预测机器学习技术
- 强化学习
- 机器学习的现实应用

---
## 我们不会覆盖的内容

- 深度学习
- 神经网络
- 人工智能

为了更好的学习体验，我们将避免神经网络、“深度学习”（即利用多层神经网络进行模型构建）及人工智能的复杂内容，这些内容将在其他课程中讨论。我们还将推出一个数据科学的相关课程，专注于该领域。

---
## 为什么学习机器学习？

从系统角度来看，机器学习定义为创建能够自动学习数据中隐藏模式以辅助做出智能决策的自动化系统。

这一动机大致来源于人脑如何基于从外部世界感知到的数据学习某些事物。

✅ 花一分钟思考为什么企业会倾向于使用机器学习策略，而不是创建硬编码的基于规则的引擎。

---
## 数据质量为何重要

高质量数据提升模型表现。即使使用先进的机器学习算法，低质量或含噪声的数据也会导致不准确的预测。

---
## 机器学习的应用

机器学习的应用几乎无处不在，它们与流动于我们社会的数据同样普遍，这些数据由智能手机、联网设备及其他系统生成。考虑到最先进的机器学习算法强大潜力，研究人员探索了利用它们解决多维多学科现实问题的能力，并取得了显著成效。

---
## 机器学习应用示例

<strong>你可以以多种方式使用机器学习</strong>：

- 根据病人的病历或报告预测疾病可能性。
- 利用气象数据预测气象事件。
- 识别文本情绪。
- 侦测假新闻以阻止宣传传播。

金融、经济、地球科学、太空探索、生物医学工程、认知科学甚至人文学科领域都采用机器学习来解决其领域内繁重的数据处理问题。

---
## 总结

机器学习通过从真实或生成数据中发现有意义的见解，实现了模式发现过程的自动化。它在商业、健康和金融等领域证明了其高度价值。

在不远的未来，任何领域的人都必须掌握机器学习基础知识，因为它被广泛采用。

---
# 🚀 挑战

用纸笔或在线应用如[Excalidraw](https://excalidraw.com/)绘制出你对AI、ML、深度学习和数据科学之间区别的理解。添加一些这些技术擅长解决的问题的想法。

# [课后测验](https://ff-quizzes.netlify.app/en/ml/)

---
# 复习与自学

想了解如何在云端使用机器学习算法，请访问此[学习路线](https://docs.microsoft.com/learn/paths/create-no-code-predictive-models-azure-machine-learning/?WT.mc_id=academic-77952-leestott)。

参加有关机器学习基础的[学习路径](https://docs.microsoft.com/learn/modules/introduction-to-machine-learning/?WT.mc_id=academic-77952-leestott)。

---
# 作业

[开始动手](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**免責聲明**：
本文件使用 AI 翻譯服務 [Co-op Translator](https://github.com/Azure/co-op-translator) 進行翻譯。雖然我們力求準確，但請注意，自動翻譯可能包含錯誤或不準確之處。原始文件的母語版本應被視為權威來源。對於重要資訊，建議尋求專業人工翻譯。我們不對因使用本翻譯而引起的任何誤解或曲解承擔責任。
<!-- CO-OP TRANSLATOR DISCLAIMER END -->