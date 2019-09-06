# Inference with Python

The script `ende_client.py` is an example of running interactive translations with a `SavedModel`:

**1\. Go into this directory, as assumed by the rest of the commands:**

```bash
cd examples/serving/python
```

**2\. Download the English-German pretrained model:**

```bash
wget https://s3.amazonaws.com/opennmt-models/averaged-ende-export500k-v2.tar.gz
tar xf averaged-ende-export500k-v2.tar.gz
```

**3\. Run the interactive client:**

```bash
python ende_client.py averaged-ende-export500k-v2
```
