
This code trains ALBERT-large-v2 model on Bengali Wikipedia and OSCAR.

# What is this experiment about?
This experiment is a project between Yandex and HF in which we aim to train an ALBERT model for Bengali collaboratively. That’s where you come in! Each participant can contribute by sharing their compute power, so the more we are, the faster the training is. This is achieved using the [Hivemind library](https://github.com/learning-at-home/hivemind/). You can read more about it in [this paper](https://arxiv.org/abs/2103.03239).

# How to participate?

**Kaggle**. This option provides you a P100 and lasts longer than Colab. This requires a Kaggle account. **You must enable Internet access and switch kernel to GPU mode explicitly.**
* If it is stuck at "installing dependencies" for over 5 minutes, it means you changed session type too late. Simply restart with GPU / internet enabled and it should work just fine.

**[Colab](https://colab.research.google.com/github/mryab/collaborative-training/blob/main/colab_starter.ipynb)**.

**Please do not run multiple GPU instances on the same service!** You can use Kaggle in one tab and Colab in another, but avoid having two Colab GPU instances at the same time.

**Local run** if you have a GPU. Another option to join the training will be made available later in the week. We will keep you informed when this option is available.