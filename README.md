# MonkeyFace-Verification
利用对齐的猴脸分为gallery和probe数据集，做验证问题

总体思路为：猴脸数据集极小，使用在ImageNet预训练的模型迁移学习，进而提取猴脸特征；参考行人重识别的方法，使用 Part-based Convolutional Baseline 提高准确率

猴脸相关文章：

- [AG 2016 - Chimpanzee Faces in the Wild: Log-Euclidean CNNs for Predicting Identities and Attributes of Primates](https://www.researchgate.net/publication/307090482_Chimpanzee_Faces_in_the_Wild_Log-Euclidean_CNNs_for_Predicting_Identities_and_Attributes_of_Primates) - [code in github](https://github.com/cvjena/analyzing-chimpanzees)

- [ICCV 2017 - Towards Automated Visual Monitoring of Individual Gorillas in the Wild](http://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w41/Brust_Towards_Automated_Visual_ICCV_2017_paper.pdf)

重识别相关文章：

- [2017-Deep Representation Learning with Part Loss for Person Re-Identification](https://arxiv.org/abs/1707.00798)
- [2017-Beyond Part Models-Person Retrieval with Refined Part Pooling](https://arxiv.org/abs/1711.09349)

## 迁移学习
在迁移学习中，我们首先在一个基础数据集和基础任务上训练一个基础网络，然后我们再微调一下学到的特征，或者说将它们迁移到第二个目标网络中，用目标数据集和目标任务训练网络。如果特征是泛化的，那么这个过程会奏效，也就是说，这些特征对基础任务和目标任务都是适用的，而不是特定的适用于某个基础任务。

全面迁移资料：[transferlearning](https://github.com/jindongwang/transferlearning)

### 训练技巧
把预训练的CNN模型当做特征提取器；finetune model 包括全局微调和局部微调，根据问题本身数据库的特点有几种情况可参考：
1. 新的数据库较小，并且和pre-trained model所使用的训练数据库相似度较高： 
由于数据库较小，在进行finetune存在overfit的风险，又由于数据库和原始数据库相似度较高，因此二者不论是local feature还是global feature都比较相近，所以此时最佳的方法是把CNN网络当做特征提取器然后训练一个分类器进行分类 
2. 新的数据库较大，并且和pre-trained model所使用的训练数据库相似度较高： 
很明显，此时我们不用担心overfit，因此对全部网络结构进行finetune是较好的。 
3. 新的数据库较小，并且和pre-trained model所使用的训练数据库差异很大： 
由于数据库较小，不适合进行finetune，由于数据库差异大，应该在单独训练网络结构中较高的层，前面几层local的就不用训练了，直接固定权值。在实际中，这种问题下较好的解决方案一般是从网络的某层开始取出特征，然后训练网络顶层的分类器。 
4. 新的数据库较大，并且和pre-trained model所使用的训练数据库差异很大： 
本来由于数据库较大，可以从头开始训练的，但是在实际中更偏向于训练整个pre-trained model的网络。

在不移除原始结构中的层或者更改其参数的条件下，可以对结构做出修改，比如复制几份顶层的网络；学习率不应该设置的太大，一般在finetune部分的学习率一般设置在 1e-5，而全新部分的学习率设置的大一点为 1e-3。

更详细的信息参考 [迁移学习技巧以及如何更好的finetune 模型](http://blog.csdn.net/u014381600/article/details/71511794)

### 实践代码
使用过 pytorch 和 caffe 微调网络。

使用caffe时，先对prototxt文件进行修改再利用pycaffe接口微调，参考文章
- [caffe fine-tuning - 利用已有模型训练其他数据集](https://zhuanlan.zhihu.com/p/22624331)

- [caffe 中 fine-tuning 模型三重天](http://blog.csdn.net/sinat_26917383/article/details/54999868)

使用pytorch调参及修改网络更加方便，参考文章
- [pytorch 迁移学习](https://zhuanlan.zhihu.com/p/30119664)
- [pytorch 中的 pre-train 函数模型引用及修改](http://blog.csdn.net/whut_ldz/article/details/78845947)

### 预训练模型

* [牛津 VGG 模型](http://www.robots.ox.ac.uk/~vgg/research/very_deep/)
* [谷歌 inception 模型](https://github.com/tensorflow/models/tree/master/inception)
* [微软 ResNet 模型](https://github.com/KaimingHe/deep-residual-networks)
* [谷歌 Word2vec 模型](https://code.google.com/archive/p/word2vec/)
* [斯坦福 GloVe 模型](https://nlp.stanford.edu/projects/glove/)
* [Caffe 模型库 Model Zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo)
* [Pytorch 除官网外的模型库](https://github.com/Cadene/pretrained-models.pytorch)
* [Tensorflow 除官网外的模型库](https://github.com/tensorflow/models/tree/master/research/slim)

## 面部验证
面部验证问题可以参考人脸识别、行人重识别，提取gallery与probe中图片特征，属于基于表征学习的ReID方法，本项目加入基于局部特征的ReID方法。更多的信息参考[基于深度学习的行人重识别研究综述](https://zhuanlan.zhihu.com/p/31921944)、[基于深度学习的人脸识别技术综述](https://zhuanlan.zhihu.com/p/24816781)

行人重识别有些简单的baseline：[简单行人重识别代码到88%准确率](https://zhuanlan.zhihu.com/p/32585203)

## 实验
猴脸图片是比较标准的图片，而且猴脸特征明显，得到的准确率能轻易地高于无限制人脸识别和行人重识别。

网络结构：

IDE：直接微调 resnet18

PCB：将 global feature 划分为 part feature ，对每份 part feature 作某程度的卷积+池化，分别以 classification loss 训练

PGB：将 global feature 划分为 part feature 的同时，global feature 与 part feature 继续通过某程度卷积+池化，分别以 classification loss 训练

验证方式：

concat：all parts are concatenated and vote one label.

score: every part votes for a label. 

结果：

单论网络结构之间差异，使用 part loss 提升不高，但加上 score 验证方法时，效果尤其好。原因可能在于，猴脸各部分差异较大，各个part的度量空间有互补性，提升整体准确率；单纯使用 concat 方法无法利用度量空间的互补性。