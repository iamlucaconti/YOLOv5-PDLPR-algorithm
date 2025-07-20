
## Baseline

### Detection
**TODO**: da completare

### Recognition

For the recognition task in the baseline, we employed a **CNN + BiLSTM + Linear** architecture that maps an input image to a sequence of character class logits.

The model begins with a stack of convolutional layers to extract spatial features from the input image, which is expected to have shape `[B, 3, 48, 144]`, where `B` is the batch size, and the image has 3 channels (RGB), height 48, and width 144.

The CNN backbone consists of four convolutional blocks with increasing channel dimensions (`3 → 64 → 128 → 256 → 256`). Each block uses `Conv2d + BatchNorm2d + ReLU`, and max pooling is applied after the first two blocks to reduce spatial resolution. The output of the CNN has shape `[B, 256, 12, 36]`.

To perform sequence modeling, the CNN output is reshaped and permuted so that the width dimension (36) becomes the **temporal axis**. The resulting tensor has shape `[B, 36, 3072]`, where `3072 = 256 × 12` (channels × height).

This sequence is processed by a 2-layer **Bidirectional LSTM** with hidden size 512 per direction (1024 total). Dropout is applied between the LSTM layers to prevent overfitting.

To produce a fixed-length output (7 characters for license plates), the model selects 7 time steps from the LSTM output using precomputed **fixed offsets** along the temporal axis. This selection ensures that predictions are made at consistent, evenly spaced positions across the sequence.

The selected LSTM outputs, with shape `[B, 7, 1024]`, are passed through a fully connected layer to produce the final logits of shape `[B, 7, 68]`, where 68 is the number of possible output classes (including alphanumeric characters and special symbols).

The model is trained using **Cross Entropy Loss**, which compares the predicted logits with the ground-truth character labels.