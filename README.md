
# What is this experiment about?
This experiment is a project between Yandex and HF in which we aim to train an ALBERT model for Bengali collaboratively. That’s where you come in! Each participant can contribute by sharing their compute power, so the more we are, the faster the training is. This is achieved using the [Hivemind library](https://github.com/learning-at-home/hivemind/). You can read more about it in [this paper](https://arxiv.org/abs/2103.03239).


# How to participate?

1. As a reminder, you need to provide your Hugging Face username to be able to participate. 
    * Current participants are already in the allowlist.
    * New participants need to join the #albert-allowlist channel and add their username. Someone from the team will add you to the allowlist. If you see a :hourglass:  reaction, we’re on it! If you see a :white_check_mark:, you should be added by then. Feel free to reach out to us if you don’t have access.

2. You can join the training in few ways:

* **[Kaggle](https://www.kaggle.com/yhn112/collaborative-training-d87a28)**. This option provides you a P100 and lasts longer than Colab. This requires a Kaggle account. **You must enable Internet access and switch kernel to GPU mode explicitly.**
    * If it is stuck at "installing dependencies" for over 5 minutes, it means you changed session type too late. Simply restart with GPU / internet enabled and it should work just fine.
    * If you ran Kaggle during last week's training, please create a new copy to update the code. On the 3 dots on the screenshot and than create a new copy (don’t click on Edit My Copy).

* **[Google Colab](https://colab.research.google.com/github/mryab/collaborative-training/blob/main/colab_starter.ipynb)**.

**Please do not run multiple GPU instances on the same service!** You can use Kaggle in one tab and Colab in another, but avoid having two Colab GPU instances at the same time.

* **Local run** if you have a GPU. Another option to join the training will be made available later in the week. We will keep you informed when this option is available.

3. You can track the status of the training in [our dashboard](https://wandb.ai/learning-at-home/Main_metrics).

4. We’re inviting all of you to choose the name of the resulting model. You can propose and vote as many names as you want. [Poll link](https://poll.ly/#/LzpmDbRx).

Thank you all for participating!
