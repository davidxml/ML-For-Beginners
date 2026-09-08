# 机器学习简介

## [课前测验](https://ff-quizzes.netlify.app/en/ml/)

---

[![ML for beginners - Introduction to Machine Learning for Beginners](https://img.youtube.com/vi/6mSx_KJxcHI/0.jpg)](https://youtu.be/6mSx_KJxcHI "ML for beginners - Introduction to Machine Learning for Beginners")

> 🎥 点击上方图片观看本课程的简短视频讲解。

欢迎来到面向初学者的经典机器学习课程！无论你是完全不了解这个主题的新手，还是希望巩固某个领域知识的经验丰富的机器学习从业者，我们都很高兴你能加入！我们希望为你的机器学习学习创建一个友好的起点，并乐于评估、回应和采纳你的[反馈](https://github.com/microsoft/ML-For-Beginners/discussions)。

[![Introduction to ML](https://img.youtube.com/vi/h0e2HAPTGF4/0.jpg)](https://youtu.be/h0e2HAPTGF4 "Introduction to ML")

> 🎥 点击上方图片观看视频：MIT的John Guttag介绍机器学习

---
## 机器学习入门

在开始本课程内容之前，你需要将你的计算机配置好并准备好在本地运行笔记本。

- <strong>通过这些视频配置你的机器</strong>。使用以下链接学习如何在你的系统中[安装Python](https://youtu.be/CXZYvNRIAKM)以及[设置文本编辑器](https://youtu.be/EU8eayHWoZg)进行开发。
- **学习Python**。建议对[Python](https://docs.microsoft.com/learn/paths/python-language/?WT.mc_id=academic-77952-leestott)有基本了解，这是一门对数据科学家有用的编程语言，我们将在本课程中使用。
- **学习Node.js和JavaScript**。本课程在构建Web应用时会多次用到JavaScript，因此你需要安装[node](https://nodejs.org)和[npm](https://www.npmjs.com/)，以及安装[Visual Studio Code](https://code.visualstudio.com/)以支持Python和JavaScript开发。
- **创建GitHub账号**。既然你在这通过[GitHub](https://github.com)找到我们，你可能已经有账号了。如果没有，请创建一个账号，然后fork本课程来供你自己使用。（顺便给我们点个星也欢迎 😊）
- **了解Scikit-learn**。熟悉[Scikit-learn](https://scikit-learn.org/stable/user_guide.html)，这是我们在课程中参考的一套机器学习库，许多学生都用它来学习基础知识。

---
## 什么是机器学习？

“机器学习”是当今最流行、使用频率最高的术语之一。如果你对科技有一定了解，无论你从事哪个领域，都很可能至少听过这个词。然而大多数人对机器学习的机制其实了解甚少。对于机器学习初学者来说，这个主题有时会让人感觉难以应对。因此，理解什么是真正的机器学习，并通过实际示例逐步学习它，显得尤为重要。

---
## 热潮曲线

![ml hype curve](../../../../translated_images/zh-CN/hype.07183d711a17aafe.webp)

> Google Trends 显示了“机器学习”这一术语的近期热潮曲线

---
## 一个神秘的宇宙

我们生活在一个充满迷人奥秘的宇宙中。伟大的科学家如Stephen Hawking、Albert Einstein和许多人奉献毕生精力，寻找揭开我们周围世界奥秘的有意义信息。这正是人类学习的本质：一个孩子随着成长，年复一年地学习新事物，揭示世界的结构，最终成为成人。

---
## 儿童的大脑

孩子的大脑及感官感知周围环境的事实，渐渐学习生活中的隐藏模式，这帮助孩子创建逻辑规则以识别所学的模式。人脑的学习过程使人类成为世界上最复杂的生物。通过不断发现隐藏模式并在这些模式上创新，我们得以在一生中不断进步。这种学习能力和进化能力与被称为[脑可塑性](https://www.simplypsychology.org/brain-plasticity.html)的概念相关。从表面上看，我们可以在大脑的学习过程和机器学习的概念之间找到一些激励性的相似之处。

---
## 人脑

[人脑](https://www.livescience.com/29365-human-brain.html)感知真实世界的事物，处理所感知的信息，做出理性决策，并基于环境执行特定行动。这就是我们所说的智能行为。当我们将这种智能行为过程的模拟编写到机器中时，这称为人工智能（AI）。

---
## 一些术语

虽然这些术语容易混淆，但机器学习（ML）是人工智能的一个重要子集。**ML关注于使用专门的算法从感知到的数据中发掘有意义的信息，找到隐藏的模式，从而支持理性决策过程。**

---
## AI、ML、深度学习

![AI, ML, deep learning, data science](../../../../translated_images/zh-CN/ai-ml-ds.537ea441b124ebf6.webp)

> 显示AI、ML、深度学习与数据科学之间关系的图示。信息图由[Jen Looper](https://twitter.com/jenlooper)制作，灵感来自[这张图](https://softwareengineering.stackexchange.com/questions/366996/distinction-between-ai-ml-neural-networks-deep-learning-and-data-mining)

---
## 要覆盖的概念

在本课程中，我们将只涵盖初学者必须了解的机器学习核心概念。我们主要使用Scikit-learn进行所谓的“经典机器学习”教学，这是一款许多学生学习基础知识时使用的优秀库。要理解更广泛的人工智能或深度学习概念，扎实的机器学习基础知识必不可少，因此我们在这里提供这一部分内容。

---
## 在本课程你将学习：

- 机器学习核心概念
- 机器学习历史
- 机器学习与公平性
- 回归机器学习技术
- 分类机器学习技术
- 聚类机器学习技术
- 自然语言处理机器学习技术
- 时间序列预测机器学习技术
- 强化学习
- 机器学习的实际应用

---
## 我们不会涵盖的内容

- 深度学习
- 神经网络
- 人工智能

为了提供更好的学习体验，我们将避免神经网络、“深度学习”（使用多层神经网络的模型构建）及人工智能的复杂性，这些将在另一个课程中讨论。我们还会提供即将推出的数据科学课程，专注于该领域的内容。

---
## 为什么要学习机器学习？

从系统角度看，机器学习被定义为创建能够从数据中学习隐藏模式以辅助智能决策的自动化系统。

这一动机大致受到人脑如何基于从外部世界感知的数据学习某些事物的启发。

✅ 想一想，为什么企业会想使用机器学习策略，而不是创建硬编码的规则引擎。

---
## 为什么数据质量很重要

高质量数据提升模型表现。即使使用先进的机器学习算法，数据质量差或噪声大也会导致预测不准确。

---
## 机器学习的应用

机器学习的应用几乎无处不在，与我们社会中不断流动的数据一样普遍，这些数据由智能手机、联网设备及其他系统生成。考虑到最先进的机器学习算法的巨大潜力，研究人员一直在探索它们解决多维、多学科现实问题的能力，并取得了显著积极成果。

---
## 机器学习应用实例

<strong>你可以通过多种方式使用机器学习</strong>：

- 预测患者病史或报告中疾病的可能性。
- 利用气象数据预测天气事件。
- 理解文本情感。
- 识别假新闻以阻止宣传传播。

金融、经济、地球科学、太空探索、生物医学工程、认知科学甚至人文学科领域都采用机器学习解决他们领域中庞大且繁重的数据处理问题。

---
## 结论

机器学习通过从真实世界或生成数据中发现有意义的洞见，自动化了模式发现的过程。它已被证明在商业、健康及金融等应用领域极具价值。

在不久的将来，鉴于其广泛应用，理解机器学习基础将成为任何领域人员的必备技能。

---
# 🚀 挑战

用纸笔或像[Excalidraw](https://excalidraw.com/)这样的在线工具，草绘你对AI、ML、深度学习和数据科学差异的理解。列举这些技术各自适合解决的问题。

# [课后测验](https://ff-quizzes.netlify.app/en/ml/)

---
# 复习与自学

想了解如何在云中使用机器学习算法，请访问此[学习路径](https://docs.microsoft.com/learn/paths/create-no-code-predictive-models-azure-machine-learning/?WT.mc_id=academic-77952-leestott)。

参加关于机器学习基础的[学习路径](https://docs.microsoft.com/learn/modules/introduction-to-machine-learning/?WT.mc_id=academic-77952-leestott)。

---
# 作业

[开始运行](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**免责声明**：
本文件由 AI 翻译服务 [Co-op Translator](https://github.com/Azure/co-op-translator) 翻译完成。尽管我们力求准确，但请注意，自动翻译可能包含错误或不准确之处。原始语言版文件应视为权威来源。对于重要信息，建议使用专业人工翻译。我们对因使用本翻译而产生的任何误解或误释不承担责任。
<!-- CO-OP TRANSLATOR DISCLAIMER END -->