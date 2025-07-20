

## YOLOvs-PDLPR

### YOLOv5

[YOLOv5](https://github.com/ultralytics/yolov5)

### PDLPR

The PDLPR model architecture is illustrated in the figure below:
![The overall framework of the license plate recognition algorithm.](/figures/pdlpr.png)
It comprises three primary modules:

1. **Improved Global Feature Extractor (IGFE)**  
   - Input: License plate images resized to `48 × 144` pixels.  
   - Process: Extracts features and converts them into a feature vector of dimensions `512 × 6 × 18`.  

2. **Encoder Module**  
   - **Position Encoder**: Encodes the position of the feature map and adds it to the image feature vector.  
   - **Multi-Head Attention**: Further encodes the combined vector to produce an output feature vector.  

3. **Parallel Decoder Module**  
   - Utilizes **Multi-Head Attention** to decode the encoder's output feature vector.  
   - Predicts the final license plate sequence.  
---
#### IGFE

The **IGFE** module consists of a Focus Structure module, two ConvDownSampling modules, and four RESBLOCK modules as shown below.
![IGFE.](/figures/igfe.png)


- **Focus Structure module**  was used to conduct picture slicing operations, and in its operation process a value was taken at each interval of one pixel in a input picture so that one picture was equally divided into four feature maps. Then, they were concatenated along the channel direction. Thus a three‑channel image became a 12‑channel feature map with half the original width and height. Finally, the obtained feature map is convolved to perform the downsampling
operation. A Focus Structure is better than other ways of downsampling because it does not lose any feature information. This means that the extracted semantic information will be more comprehensive.

- The structure of each **RESBLOCK** module consists of **two residually connected CNN BLOCK modules**. During forward inference, the residual connected structure could prevent the network’s gradient disappearance and explosion. In the CNN BLOCK module’s convolutional layer for extracting features, we utilized `Conv2d` with `stride = 1` and `kernelSize = 3` to extract features, which were then passed via the GroupNorm layer and activation function layer in order to extract the image’s visual features.The activation function made use of the SiLU . In our implementation of the RESBLOCK modules, we replaced the standard **Batch Normalization** and **LeakyReLU** (as originally used in [Tao et al.](https://www.mdpi.com/1424-8220/24/9/2791)) with Group Normalization (**GroupNorm**) and the **SiLU** activation function. This change was made to improve the model’s generalization and performance under varying data distributions and batch sizes.
Unlike BatchNorm, which depends on batch statistics and tends to be unstable when using small batch sizes, **GroupNorm** normalizes over groups of channels rather than across the batch dimension. This makes it **more robust** in scenarios with small or varying batch sizes, **more consistent** in training and inference, since statistics are independent of the batch. and **well-suited** to applications like license plate detection, where image sizes, lighting, and conditions may vary significantly. We also replaced the **LeakyReLU** activation with **SiLU** (Sigmoid-Weighted Linear Unit), also known as **Swish**. SiLU is a smooth, non-monotonic activation function defined as $\text{SiLU}(x) = x \cdot \sigma(x)$, where $\sigma(x)$ is the sigmoid of $x$. SiLU has been shown to **improve optimization** due to its smoothness, **retain small negative values** (unlike ReLU variants which zero them out), **outperform ReLU/LeakyReLU** in several vision tasks, particularly when paired with normalization techniques like GroupNorm.

- **ConvDownSampling** modules are similar in structure to the **CNN BLOCK**, but they use a `Conv2D` layer with `stride = 1` and `kernel size = 3`.The first ConvDownSampling modules has 256 output channels while the second one has 512 output channels.

---
#### Encoder
As in the original paper ([Tao et al.](https://www.mdpi.com/1424-8220/24/9/2791)), the **Encoder** consists of three encoding units connected via residual connections.

Each unit includes four submodules:

1. CNN BLOCK1
2. Multi-Head Attention (MHA)
3. NN BLOCK2
4. Add & Norm

![Encoder.](/figures/encoder.png)

The structure of the **CNN BLOCK** modules is similar to the one described previously.

A **Positional Encoding 2D** is added to the input of the encoder.

Before computing the Multi-Head Attention, **CNN BLOCK1** is applied to increase the dimensionality of the feature vectors, allowing the model to extract richer semantic representations. The configuration for CNN BLOCK1 is as follows: `stride = 1`, `kernel size = 1`, `padding = 1` and `output dimension = 1024`.

The output of CNN BLOCK1 is then passed through a **Multi-Head Attention** module with **8 attention heads**, enabling the model to capture global dependencies and contextual relationships in the feature space.

After attention is computed, **CNN BLOCK2** is used to reduce the feature dimension back to its original size to maintain consistency between the input and output of the encoder unit. The parameters of CNN BLOCK2 are `stride = 1`, `kernel size = 1`, `padding = 1` and `output dimension = 512`

Finally, the **Add & Norm** module connects the input and output of the attention mechanism via a residual connection and applies **Layer Normalization**. 

---
#### Parallel Decoder

The **CNN BLOCK3** and **CNN BLOCK4** adjust the dimensionality of the encoder’s output feature vector to `512 × 1 × 18` before decoding. This dimensionality reduction helps to lower the computational load on the parallel decoder.

CNN BLOCK3 Parameters are the following:
- **Stride**: `(3, 1)`
- **Kernel size**: `(3, 1)`
- **Padding**: `(1, 0)`
- **Output dimension**: `512`

CNN BLOCK4 Parameters are the following:
- **Stride**: `(1, 1)`
- **Kernel size**: `(2, 1)`
- **Padding**: `(0, 0)`
- **Output dimension**: `512`

The **parallel decoder** is composed of three **decoding units**, each containing the following four submodules:

1. **Multi-Head Self-Attention**
2. **Add & Norm**
3. **Multi-Head Cross Attention**
4. **Add & Norm**
5. **Feed-Forward Network (FFN)**
6. **Add & Norm**

![Dencoder.](/figures/decoder.png)

The **Multi-Head Attention** and **Add & Norm** modules follow the same structure described earlier.

Since the decoder is **not autoregressive**, we replaced the original masked Multi-Head Attention with a standard **Multi-Head Attention** module. The encoder outputs, along with **2D positional encoding**, are fed into the **Multi-Head Self-Attention**. Its output is then passed to the **Add & Norm** module, which helps prevent overfitting and accelerate model convergence.

Subsequently, the outputs of the **Add & Norm** and **CNN BLOCK4** are fed into the **Multi-Head Cross Attention** module. The output features from the cross-attention are again passed through an **Add & Norm** module and then into the **Feed-Forward Network**.

The FFN consists of two fully connected layers with a non-linear activation in between:

1. The input is passed through the first linear transformation, followed by a **ReLU** activation.
2. The result is then passed to the second linear layer.

Finally, the output of the Feed-Forward Network is passed through one last **Add & Norm** module to ensure stability and promote faster convergence during forward inference.

The decoder output is then passed through **CNN BLOCK5** and **CNN BLOCK6**, which share the same configuration as **CNN BLOCK3** and **CNN BLOCK4**, respectively. These blocks reshape the decoder output to a fixed dimensionality of `512 × 1 × 18`. Next, the tensor is reshaped and passed through the classifier. The final output has shape `(B, 18, 68)`, where:

* `B` is the batch size
* `18` represents the temporal length (used for the CTC loss)
* `68` is the total number of output classes

Thus, the final output is ready for training with a **Connectionist Temporal Classification (CTC)** loss, using a sequence length of 18 and 68 possible character classes.
Since the temporal length (18) is greater than the license plate length (7), we adopted a **greedy decoding** strategy. This method selects the most probable character (i.e., applies `argmax`) at each timestep and then performs the following post-processing steps:

* Removes repeated characters
* Removes blank symbols
