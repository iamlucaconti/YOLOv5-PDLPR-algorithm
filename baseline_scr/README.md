
## Baseline

### Detection

The goal of the detection module is to localize license plates in input images by predicting a bounding box in normalized format (cx, cy, w, h) ∈ [0,1].

We adopt a ResNet-18-based architecture with a lightweight Feature Pyramid Network (FPN) and a regression head for bounding box prediction.
The detection network is composed by CNN + FPN + MLP module:

We used a ResNet-18 pretrained on ImageNet as feature extractor, the early convolutional layer (conv1) is optionally frozen to stabilize training.
To effectively localize license plates under diverse conditions (e.g. small size, motion blur, low resolution), the model integrates multi-scale features using a **Feature Pyramid Network** (FPN),
In fact, it merges features from layer3 (higher resolution, spatially rich) and layer4 (lower resolution, semantically stronger), then these feature are aligned via upsampling and added together. This fusion enables the model to detect small or distant plates (as in CCPD-FN) by leveraging high-resolution features and handle blurred or low-quality plates (as in CCPD-DB).

The pooled features are passed through a small MLP with a hidden layer (ReLU + Dropout) and a final sigmoid layer,
the output is a vector [cx, cy, w, h] representing the predicted bounding box.

For training the license plate detector, we adopted a Complete IoU (CIoU) loss, AdamW optimizer (initial learning rate 1e-4, weight decay 1e-4),
scheduler (every 5 epochs the learning rate is reduced by a factor of 0.8 ).


### Recognition

For the recognition task in the baseline, we employed a **CNN + BiLSTM + Linear** architecture that maps an input image to a sequence of character class logits.

The model begins with a stack of convolutional layers to extract spatial features from the input image, which is expected to have shape `[B, 3, 48, 144]`, where `B` is the batch size, and the image has 3 channels (RGB), height 48, and width 144.

The CNN backbone consists of four convolutional blocks with increasing channel dimensions (`3 → 64 → 128 → 256 → 256`). Each block uses `Conv2d + BatchNorm2d + ReLU`, and max pooling is applied after the first two blocks to reduce spatial resolution. The output of the CNN has shape `[B, 256, 12, 36]`.

To perform sequence modeling, the CNN output is reshaped and permuted so that the width dimension (36) becomes the **temporal axis**. The resulting tensor has shape `[B, 36, 3072]`, where `3072 = 256 × 12` (channels × height).

This sequence is processed by a 2-layer **Bidirectional LSTM** with hidden size 512 per direction (1024 total). Dropout is applied between the LSTM layers to prevent overfitting.

To produce a fixed-length output (7 characters for license plates), the model selects 7 time steps from the LSTM output using precomputed **fixed offsets** along the temporal axis. This selection ensures that predictions are made at consistent, evenly spaced positions across the sequence.

The selected LSTM outputs, with shape `[B, 7, 1024]`, are passed through a fully connected layer to produce the final logits of shape `[B, 7, 68]`, where 68 is the number of possible output classes (including alphanumeric characters and special symbols).

The model is trained using **Cross Entropy Loss**, which compares the predicted logits with the ground-truth character labels.