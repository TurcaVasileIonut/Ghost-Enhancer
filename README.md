# Ghost-Enhancer
Super Resolution iOS App and Super Resolution PyTorch Model


## **Generator Architecture**

### **Key Improvements**
- **Ghost Bottlenecks:** 
  - Replace Residual Blocks from SRGAN to reduce computational complexity.
  - Create a compact model without sacrificing performance.

- **Self-Attention with Multi-Head Layers (SA GhostSRGAN):**
  - Introduce Multi-head Attention Layers after each Ghost Bottleneck.
  - Improve image quality by capturing pixel-level dependencies.
  - Reduce overfitting risks using aggregated information from multiple perspectives.

---

### **Pipeline Overview**
1. **Initial Convolution:**
   - A convolutional layer with a kernel size of 9 extracts features from the input image.
   - Outputs 64 feature channels, followed by a **Parametric Rectified Linear Unit (PReLU)** layer.

2. **Core Architecture:**
   - **Ghost Bottlenecks:**
     - Efficient feature extraction with skip connections to Self-Attention Layers in SA GhostSRGAN.
   - **Multi-Head Attention Layers (SA GhostSRGAN):**
     - Aggregate data from multiple perspectives for robust feature learning.

3. **Post-Attention Layers:**
   - A sequence of:
     - **Convolutional Layer** → **Batch Normalization** → **Elementwise Sum**.
   - Ensures consistency in feature map dimensions.

4. **Upsampling:**
   - Two upsampling passes:
     - **Convolutional Layer** → **PixelShuffle** → **Parametric ReLU**.
   - A final convolution generates the output image.

---

## **Discriminator Architecture**

### **GhostSRGAN Discriminator**
1. **Initial Layers:**
   - A Convolutional Layer (kernel size = 3) followed by **Leaky ReLU activation**.

2. **Basic Blocks:**
   - Seven blocks, each containing:
     - A Convolutional Layer (kernel size = 3, stride = 2).
   - Extracts hierarchical features from the input image.

3. **Dense Layers:**
   - A Dense Layer after the final Basic Block.
   - Output processed through:
     - **Leaky ReLU activation** → **Sigmoid activation** for final classification.

---

### **SA GhostSRGAN Discriminator**
- **Enhancement:**
  - Adds a **Multi-Head Attention Layer** after every Basic Block.
  - Improves feature extraction and representation learning.

---

## **Architectural Diagram**
SA GhostSRGAN:
![Alt text]('img.png')


---

## **Summary**
- GhostSRGAN and SA GhostSRGAN improve upon SRGAN by introducing Ghost Bottlenecks and Self-Attention mechanisms.
- SA GhostSRGAN incorporates Multi-Head Attention to capture dependencies more effectively and prevent overfitting.
- The architectures are lightweight and designed for high-performance image super-resolution.
