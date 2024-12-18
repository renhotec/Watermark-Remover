# **Watermark Removal Project**

## **ģ��ˮӡ�Ƴ���Ŀ**

---

### **Project Overview | ��Ŀ����**
This project focuses on removing watermarks from images using advanced deep learning techniques. It employs a Generative Adversarial Network (GAN) architecture with residual blocks, attention mechanisms, and perceptual loss to produce clean and high-quality outputs.

����Ŀ�������ѧϰ����ʵ��ͼ��ˮӡ�Ƴ����������ɶԿ����� (GAN) �ܹ�����ϲв�顢ע���������Լ���֪��ʧ�����������Ҹ�������ͼ�������

---

### **Features | �ص�**

- **State-of-the-Art GAN Architecture**: Combines residual blocks, transformer blocks, and attention mechanisms.
  **�Ƚ�GAN�ܹ�**����ϲв�顢Transformer���ע�������ơ�
- **Comprehensive Loss Functions**: Includes adversarial loss, perceptual loss, edge-preserving loss, and laplacian loss.
  **ȫ����ʧ����**�������Կ���ʧ����֪��ʧ����Ե������ʧ��������˹��ʧ��
- **FastAPI Integration**: Provides an API for image processing via URLs or Base64 input.
  **����FastAPI**��֧��ͨ��URL��Base64�������ͼ�����API��
- **Batch Processing**: Supports batch image processing for enhanced efficiency.
  **��������**��֧������ͼ���������Ч�ʡ�

---

### **Project Structure | ��Ŀ�ṹ**

```plaintext
.
|-- main.py          # Entry point for training and testing
|-- app.py           # FastAPI server for API integration
|-- dataset.py       # Custom dataset loader for watermarked images
|-- model.py         # Definitions of generator and discriminator models
|-- train.py         # GAN training pipeline
|-- test.py          # Testing and evaluation pipeline
|-- utils.py         # Utility functions for image processing
|-- models/          # Directory for saving trained models
|-- data/            # Directory for input datasets
|-- outputs/         # Directory for processed image outputs
```

---

### **File Explanations | �ļ�˵��**

#### **main.py**
- **Description**: Entry point for training and testing the GAN model.
- **Key Parameters**:
  - `--mode`: Specifies the mode of operation (`train` or `test`).
    **ģʽ**��ָ������ģʽ��`train`ѵ����`test`���ԣ���
  - `--epochs`: Number of training iterations (default `100`).
    **��������**��Ĭ��ֵΪ`100`��
  - `--model_path`: Path to the generator model file for testing.
    **ģ��·��**������ʱ���ص�������ģ��·����
  - `--input_image` & `--output_image`: Paths for input and output images during testing.
    **��������·��**������ʱָ����ͼ��·����

#### **app.py**
- **Description**: Implements a FastAPI server for real-time image processing.
- **Key Endpoints**:
  - `POST /process-image-url-preview/`: Accepts an image URL for watermark removal.
    **����URLͼ��**����URL����ͼ���Ƴ�ˮӡ��
  - `POST /process-image-base64-preview/`: Accepts Base64-encoded image data.
    **����Base64ͼ��**������Base64�����ͼ���Ƴ�ˮӡ��
- **Example**:
  ```json
  {
    "image_url": "http://example.com/image.jpg"
  }
  ```

#### **dataset.py**
- **Description**: Custom PyTorch dataset for training and validation.
- **Key Class**: `WatermarkDataset`.
  - **Methods**:
    - `__getitem__`: Returns a paired watermarked and clean image for training.
      **����һ��ˮӡͼ��͸ɾ�ͼ������ѵ����**
    - `transform`: Resizes images to 256x256 and normalizes them to [-1, 1].
      **����ͼ���СΪ256x256���������һ����[-1, 1]��Χ�ڡ�**

#### **model.py**
- **Description**: Contains the definitions for the generator and discriminator.
- **Key Components**:
  - **Generator**: Utilizes residual blocks, attention mechanisms, and transformer layers to learn high-quality features.
    **������**��ʹ�òв�顢ע�������ƺ�Transformer����ѧϰ������������
  - **Discriminator**: Differentiates between real clean images and generated images.
    **�б���**��������ʵ����ͼ��������ͼ��
  - **Parameters**:
    - **Kernel Size**: Controls the receptive field in convolution layers.
      **����˴�С**�����������ĸ���Ұ��
    - **Channels**: Determines the depth of feature maps.
      **ͨ����**����������ͼ����ȡ�
    - **Transformer Block Size**: Default: 16. Ensures compatibility with input dimensions.
      **Transformer���С**��Ĭ��ֵΪ16��ȷ��������ά�ȼ��ݡ�

#### **train.py**
- **Description**: Implements the GAN training pipeline.
- **Key Loss Functions**:
  - **Adversarial Loss**: Ensures the generator fools the discriminator.
    **�Կ���ʧ**��ȷ��������������ƭ�б�����
  - **Perceptual Loss**: Compares feature similarities between generated and clean images.
    **��֪��ʧ**���Ƚ�����ͼ��������ͼ������������ԡ�
  - **Edge-Preserving Loss**: Focuses on retaining edges in the generated images.
    **��Ե������ʧ**����������ͼ���еı�Ե��
  - **Laplacian Loss**: Enhances high-frequency components for sharpness.
    **������˹��ʧ**����ǿ��Ƶ�ɷ������ͼ�������ȡ� 

#### **test.py**
- **Description**: Provides testing and post-processing functionalities.
- **Key Steps**:
  1. Loads the trained generator model.
  2. Processes input images with padding for transformer compatibility.
  3. Applies post-processing (denoising, sharpening).

#### **utils.py**
- **Description**: Utility functions for image handling and preprocessing.
- **Key Functions**:
  - **load_image**: Loads an image and normalizes it to PyTorch tensor format.
    **����ͼ�񲢽����һ��ΪPyTorch������ʽ��**
  - **pad_to_block_size**: Ensures input images are padded for transformer compatibility.
    **�������ͼ����ȷ����Transformer�������ݡ�**

---

### **Key Parameters Influencing Output Quality | Ӱ����������Ĺؼ�����**

1. **Epochs | ѵ���ִ�**:
   - Default: 100. More epochs improve quality but increase training time.
     **Ĭ��ֵ��100�������ִλ�������������������ѵ��ʱ�䡣**

2. **Learning Rate | ѧϰ��**:
   - Default: 0.0001. Higher rates may destabilize training; lower rates slow convergence.
     **Ĭ��ֵ��0.0001���ϸߵ�ѧϰ�ʿ��ܵ��²��ȶ����ϵ͵�ѧϰ�ʻ���������ٶȡ�**

3. **Batch Size | ������С**:
   - Default: 16. Larger sizes enhance stability but require more memory.
     **Ĭ��ֵ��16���ϴ�ֵ����ȶ��ԣ�����Ҫ�����ڴ档**

4. **Loss Weights | ��ʧȨ��**:
     g_loss = (
                    g_gan_loss +
                    9 * g_l1_loss +
                    1 * g_perceptual_loss +
                    3 * g_edge_loss +
                    1 * g_laplacian_loss +
                    1 * g_color_loss
                )
    - **GAN Loss (`g_gan_loss`) | GAN�Կ���ʧ**: Encourages realistic textures by ensuring generated images are indistinguishable from real ones.
    **����������ѧϰ��ʵ����ʹ����ͼ������ʵͼ���������֡�**

    - **L1 Loss (`g_l1_loss`) | L1��ʧ**: Focuses on pixel-level accuracy to retain global structure. Weight: `10`.
    **�����ؼ���֤׼ȷ�ԣ�������������ṹ��Ȩ�أ�`9`��**

    - **Perceptual Loss (`g_perceptual_loss`) | ��֪��ʧ**: Preserves high-level semantic features using pre-trained networks like VGG19. Weight: `1`.
    **ʹ��VGG19��Ԥѵ�����籣���߲�����������Ȩ�أ�`1`��**

    - **Edge Loss (`g_edge_loss`) | ��Ե��ʧ**: Maintains sharp boundaries and prevents blurring. Weight: `2`.
    **���ֱ߽���������ֹģ����Ȩ�أ�`3`��**

    - **Laplacian Loss (`g_laplacian_loss`) | ������˹��ʧ**: Enhances details and sharpness by focusing on high-frequency components. Weight: `1`.
    **ͨ����ǿ��Ƶ�������ϸ�ں������ȡ�Ȩ�أ�`1`��**

    - **Color Loss (`g_color_loss`) | ɫ����ʧ**: Ensures color consistency with the ground truth. Weight: `1`.
    **��֤����ͼ������ʵͼ�����ɫһ���ԡ�Ȩ�أ�`1`��**


5. **Post-Processing | ����**:
   - Denoising: Reduces residual noise.
     **ȥ��**�����ٲ���������
   - Sharpening: Enhances visual clarity.
     **��**�������Ӿ������ȡ�

---
### **Quick Start | ���ٿ�ʼ**

#### **Environment Setup | ��������**

1. **Install Required Packages | ��װ��Ҫ������**:
   Use the following command to install all required Python libraries:
   **ʹ���������װ���б�Ҫ��Python�⣺**
   ```bash
   pip install -r requirements.txt
   ```

   The `requirements.txt` file should include the following dependencies:
   **`requirements.txt` �ļ�Ӧ��������������**
   ```plaintext
   torch
   torchvision
   fastapi
   uvicorn
   pillow
   opencv-python
   opencv-python-headless
   numpy
   ```

2. **Prepare Dataset | ׼�����ݼ�**:
   - Ensure the dataset is structured as follows:
     **ȷ�����ݼ��������½ṹ��֯��**
     ```plaintext
     data/
     ������ train/
     ��   ������ watermarked/   # Folder for input watermarked images
     ��   ������ clean/         # Folder for corresponding clean images
     ������ test/              # Folder for test images
     ```

#### **Training the Model | ѵ��ģ��**
```bash
python main.py --mode train --epochs 1500
```

1. Prepare the training dataset in `data/train/`.
   **׼��ѵ�����ݼ��� `data/train/` �ļ����С�**
2. Run the above command to train the model.
   **��������������ѵ��ģ�͡�**
3. Check the `models/` directory for saved checkpoints.
   **�� `models/` Ŀ¼�м�鱣���ģ�ͼ��㡣**

#### **Testing the Model | ����ģ��**
```bash
python main.py --mode test --model_path models/generator_epoch_1500.pth --input_image input.jpg --output_image output.jpg
```
1. Ensure the trained model is in `models/`.
   **ȷ��ѵ���õ�ģ�ͱ����� `models/` �С�**
2. Provide an input image and specify the output path.
   **�ṩ����ͼ��ָ�����·����**
3. Processed images will be saved at the output path.
   **������ͼ�񽫱��������·���С�**

#### **Running the FastAPI Server | ����FastAPI����**
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```
1. Start the server using the command above.
   **ʹ������������������**
2. Access API documentation at `http://localhost:8000/docs`.
   **�� `http://localhost:8000/docs` ����API�ĵ���**

---


### **Example Results | ʾ�����**

#### Input Image (Watermarked):
![Input](data/test/sample_input.jpg)

#### Output Image (Cleaned):
![Output](outputs/sample_output.jpg)

---
