# Model

OpenNMT-tf can be used to train several types of models, here is a non exhaustive overview of supported models:

**Maching translation**

* [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215) (Sutskever et al. 2014)
* [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) (Bahdanau et al. 2014)
* [Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025) (Luong et al. 2015)
* [Guided Alignment Training for Topic-Aware Neural Machine Translation](https://arxiv.org/abs/1607.01628) (Chen et al. 2016)
* [Linguistic Input Features Improve Neural Machine Translation](https://arxiv.org/abs/1606.02892) (Sennrich et al. 2016)
* [Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation](https://arxiv.org/abs/1609.08144) (Wu et al. 2016)
* [A Convolutional Encoder Model for Neural Machine Translation](https://arxiv.org/abs/1611.02344) (Gehring et al. 2016)
* [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al. 2017)
* [MS-UEdin Submission to the WMT2018 APE Shared Task: Dual-Source Transformer for Automatic Post-Editing](https://arxiv.org/abs/1809.00188) (Junczys-Dowmunt et al. 2018)
* [Scaling Neural Machine Translation](https://arxiv.org/abs/1806.00187) (Ott et al. 2018)
* [The Best of Both Worlds: Combining Recent Advances in Neural Machine Translation](https://arxiv.org/abs/1804.09849) (Chen et al. 2018)

**Speech recognition**

* [Listen, Attend and Spell](https://arxiv.org/abs/1508.01211) (Chan et al. 2015)

**Language modeling**

* [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf) (Radford et al. 2019)

**Sequence tagging**

* [End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF](https://arxiv.org/abs/1603.01354) (Ma et al. 2016)

and most ideas and modules coming from these papers can be reused for other models or tasks.

## Catalog

OpenNMT-tf comes with a set of standard models that are defined in the [catalog](package/opennmt.models.catalog.html). These models can be directly selected with the `--model_type` command line option, e.g.:

```bash
onmt-main train --model_type Transformer [...]
```

## Custom models

Models are defined from Python code to allow a high level of modeling freedom. They are `opennmt.models.Model` instances that use [available](package/opennmt.html) or user-defined modules. Some of these modules are defined to contain other modules and can be used to design complex architectures:

* `opennmt.encoders.ParallelEncoder`
* `opennmt.encoders.SequentialEncoder`
* `opennmt.inputters.MixedInputter`
* `opennmt.inputters.ParallelInputter`

For example, these container modules can be used to implement multi source inputs, multi modal training, mixed word/character embeddings, and arbitrarily complex encoder architectures (e.g. mixing convolution, RNN, self-attention, etc.).

Some examples are available in the directory [`config/models`](https://github.com/OpenNMT/OpenNMT-tf/tree/master/config/models) in the Git repository. The custom model file should be selected with the `--model` command line option, e.g.:

```bash
onmt-main train --model config/models/custom_model.py [...]
```
