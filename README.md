
This code trains ALBERT-large-v2 model on BookCorpus on decentraized participants.

__Requirements (for all participants):__
* `git clone https://github.com/learning-at-home/hivemind -b albert_exp2`
* `cd hivemind && pip install -e .`
* `pip install --upgrade transformers datasets sentencepiece`
* Build and install apex for GPU (see [instructions](https://github.com/NVIDIA/apex#linux))

__AWS quickstart:__ see the [__deployment and training notebook__](./run_training_aws.ipynb) for AWS cloud instances using boto3.

__Run manually:__
1. Download the preprocessed BookCorpus from [this url](https://www.dropbox.com/s/k77ihomdjmwofo5/archive.tar.gz?dl=0)
   * Alternatively, use huggingface.datasets to preprocess OpenBookCorpus using parameters from [ALBERT](https://arxiv.org/abs/1909.11942)
2. Create some means for trainers to load the dataset: upload to S3 storage or an FTP server
   * Under no circumstances should you use the dropbox URL above for actual training.
     Dropbox is not meant for distributing datasets to many peers and will ban you for doing so.
3. Run the first DHT peer (aka "coordinator") on a node that is accessible to all trainers:
``` python run_first_peer.py --listen_on [::]:1337 ```  (see details below)
4. For all GPU trainers, run

```
python run_trainer.py \
  --output_dir ./outputs --overwrite_output_dir \
  --logging_dir ./logs --logging_first_step --logging_steps 100 \
  --initial_peers COORDINATOR_IP:COORDINATOR_PORT --seed 0
```



__The coordinator__ node exists solely to welcome other peers onto the DHT. It requires neither GPU nor high bandwidth, 
the only prerequisite is that coordinator should have high uptime. If no high uptime server is available, one can
also run multiple coordinators on different servers and list all of them as `--initial_peers`. The system will work as long as at least one coordinator is
available.

__The trainer node__ can be launched on any computer with a GPU, such as AWS VM or vast.ai instance.
Trainer nodes can be added to the system at any time.

__Evaluation__ should be performed on a separate instance that periodically runs `averager.load_state_from_peers()`
