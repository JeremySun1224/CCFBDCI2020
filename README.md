## CCFBDCI2020-Multi-Classification-from-Labeled-and-Unlabeled-Context

CCF大数据与计算智能大赛（CCF BDCI）中的一个文本分类的比赛——[面向数据安全治理的数据内容智能发现与分级分类](https://www.datafountain.cn/competitions/471)。本仓库拟提供一种通用的应对中文无监督文本分类的解题思路分享。

文本分类任务相对而言是较为容易的NLP任务。但是，这个比赛的重点是测试集需要对20000个文本归为10个类别，可只提供了7个类别共计7000个训练样本，还有3个类别（分别是```游戏，娱乐和体育```）的文章需要在主办方提供的33000的无监督语料中自行利用算法提取。

利用[LOTClass](https://github.com/yumeng5/LOTClass)抽取无监督文本后线上成绩单模可以逼近```88.6%```。以下是3个无监督文本的提取结果（部分展示）。其中，抽取出来的```游戏```类文本共计2431、```娱乐```类文本共计3582、```体育```类文本共计1999。

```游戏```
<img src='https://github.com/JeremySun1224/CCFBDCI2020-Multi-Classification-from-Labeled-and-Unlabeled-Context/blob/main/img/%E6%B8%B8%E6%88%8F.png'>
```娱乐```
<img src='https://github.com/JeremySun1224/CCFBDCI2020-Multi-Classification-from-Labeled-and-Unlabeled-Context/blob/main/img/%E5%A8%B1%E4%B9%90.png'>
```体育```
<img src='https://github.com/JeremySun1224/CCFBDCI2020-Multi-Classification-from-Labeled-and-Unlabeled-Context/blob/main/img/%E4%BD%93%E8%82%B2.png'>

### 1.预训练
首先，在哈工大开源的中文```RoBerta-base```的基础上，对下游任务语料继续进行领域内预训练，得到模型```RoBerta-base-self```。实验表明，线上宏F1可以提升0.7个点。
### 2.无监督训练
然后，基于根据下游任务，通过定制化Tokenizer方法进行了词级别的预训练工作，得到适合本赛题使用的```WoBERT-base```模型。然后，利用```WoBERT-base```找到和标签名称语义相关性较高的词汇，查找类别指示性单词并基于这些单词训练单词分类模型，再进行自训练提升模型，进而从33000的无监督语料中提取出了任务需要的3个领域文本。
### 3.微调
最后，基于```RoBerta-base-self```进行微调，做好分类分级的对应工作，即可输出```pred.csv```和```result.csv```文件。

# 将在近期详细整理开源。谢谢。
