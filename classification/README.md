## Code for Training an MLP on top of PDR

To train, check out the run\_trainer.py file and modify args as desired. Then, to train, run:

```
python run_trainer.py
```

The last epoch of training will also run the model on your test set and report test accuracy. For interpretability/attribution analysis, see test\_trainer.py. To run:

```
python test_trainer.py
```

The GRAD-Cam library will need to be pip installed for GRAD-Cam and Captum will need to be installed for integrated gradients.
