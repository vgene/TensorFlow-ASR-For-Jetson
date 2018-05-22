
Dataset: LibriSpeech
Training using train-100h

## Highlight:

- Using TensorFlow CudnnLSTM to train LSTMs, using warpctc-tensorflow to do ctc_loss calculation(optional).
- Simplify model to enable high performance inference on Jetson TX1