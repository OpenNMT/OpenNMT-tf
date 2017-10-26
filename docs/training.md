# Training

## Monitoring

OpenNMT-tf uses [TensorBoard](https://github.com/tensorflow/tensorboard) to log information during the training. Simply start `tensorboard` by setting the active log directory, e.g.:

```bash
tensorboard --logdir="."
```

then open the URL displayed in the shell to monitor and visualize several data, including:

* training and evaluation loss
* training speed
* learning rate
* gradients norm
* computation graphs
* word embeddings
* decoder sampling probability

## Distributed

OpenNMT-tf supports asynchronous distributed training. The user should set on the command line:

* a **chief worker** host that runs a training loop and manages checkpoints, summaries, etc.
* a list of **worker** hosts that run a training loop
* a list of **parameter server** hosts that synchronize the parameters

Then a training instance should be started on each host with a selected task, e.g.:

```bash
CUDA_VISIBLE_DEVICES=0 python -m bin.main train [...] --ps_hosts localhost:2222 --chief_host localhost:2223 --worker_hosts localhost:2224,localhost:2225 --task_type worker --task_index 1
```

will start the worker 1 on the current machine and first GPU.

For more details, see the documentation of [`tf.estimator.train_and_evaluate`](https://www.tensorflow.org/versions/r1.4/api_docs/python/tf/estimator/train_and_evaluate). Also see [tensorflow/ecosystem](https://github.com/tensorflow/ecosystem) to integrate distributed training with open-source frameworks like Docker or Kubernetes.
