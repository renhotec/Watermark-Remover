from fastapi import FastAPI, HTTPException, Request, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List
import base64
import requests
from io import BytesIO
from PIL import Image
import torch
import time
from concurrent.futures import ThreadPoolExecutor
from scripts.model import EnhancedGenerator
import torchvision.transforms as transforms

# 初始化 FastAPI 应用
app = FastAPI(title="图片处理 API", description="支持通过图片链接和 Base64 编码处理图片，采用并行方式加速", version="1.0.0")
# 加载模板
templates = Jinja2Templates(directory="templates")
# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = EnhancedGenerator().to(device)
generator.load_state_dict(torch.load("models/generator_epoch_92.pth", map_location=device))
generator.eval()

# 图像转换函数
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
unnormalize = transforms.Compose([
    transforms.Normalize((-1, -1, -1), (2, 2, 2))
])

# 定义输入数据结构
class ImageURLRequest(BaseModel):
    urls: List[str]  # 接收图片链接列表

class ImageBase64Request(BaseModel):
    images: List[str]  # 接收 Base64 图片列表

class ImageFilesRequest(BaseModel):
    files: List[UploadFile]  # 接收图片文件列表

class ImageFileResponse(BaseModel):
    name: str  # 图片文件名
    value: str  # Base64 图片内容

class ImagesResponse(BaseModel):
    result: List[ImageFileResponse]  # 返回 Base64 图片内容

class ImagePreviewResponse(BaseModel):
    result: List[str]  # 返回 Base64 图片预览链接

# 图片处理函数
def process_image(image: Image.Image) -> str:
    original_size = image.size
    input_tensor = transform(image).unsqueeze(0).to(device)

    # 模型推理
    with torch.no_grad():
        output_tensor = generator(input_tensor)

    # 反归一化并转换为 numpy 格式
    output_tensor = unnormalize(output_tensor.squeeze(0)).clamp(0, 1)
    output_image_np = (output_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
    output_image = Image.fromarray(output_image_np).resize(original_size, Image.LANCZOS)

    # 转换为 Base64
    buffer = BytesIO()
    output_image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

headers = {
    'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
    'accept-language': 'en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7',
    'cache-control': 'no-cache',
    'dnt': '1',
    'pragma': 'no-cache',
    'priority': 'u=0, i',
    'sec-ch-ua': '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"Windows"',
    'sec-fetch-dest': 'document',
    'sec-fetch-mode': 'navigate',
    'sec-fetch-site': 'none',
    'sec-fetch-user': '?1',
    'upgrade-insecure-requests': '1',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
}

cookies = {
        'wmsc_cart_key': '784C7C7BF626CAACA28496F1DA6211563E73C581C5E0032FEC65D69C4ABEC60238BD68FDBFB938D9',
    '_gcl_au': '1.1.1051846174.1728272957',
    'tfstk': 'f2ijI4xu_sfjnbT2ROpzNvdurpq6TftEWOwtKAIVBoExBlMtI5P475uWCfHi0Pd00cG767c4goo2CGit1lrwSEV61omruGP2ilMtTlOeTH-EnxquXBREmzW0Wle9b5Ea1KtTjldea0ildaEitsACVKMJFRy1HNhOkLN8C7IYBieOwLNuBleOkGK-e-yQHsEt6Yp7I7Ftib1zCGPqhLlzOux5Z8MYNMMnV-EfoxF5WNnSPrNpc7sOX0wmETJ6dMTa92V48WhJ4iZseRG78qOCcfMtKVUSfspq9b3s680k1NNIWxun4zWOkYi_GynYPt904lwt6PgkOMyb04aIcqvGwxhUG2EmQ9OqFyg7-8URCZEq8vogJmKft7z3CbaElBsj9gSCYWwBfN67-Gw7TL95SNDzsHiNTQI24reuhk9WFsp8k827TL95SN4YE87wFL1ve',
    'LCSC_LOCALE': 'en',
    '_gid': 'GA1.2.1710230271.1734077858',
    '_clck': 'jsgevq%7C2%7Cfro%7C0%7C1741',
    '_ga_98M84MKSZH': 'GS1.1.1734077861.35.1.1734077988.48.0.1040188212',
    '_uetsid': 'b717cdd0b92a11ef948f89452ff8eaba',
    '_uetvid': 'a09a96e0514511ef90e5ef2421567922',
    '_ga': 'GA1.2.1524868691.1728272957',
}

# 并行处理图片链接
def process_url(url: str, headers: dict) -> str:
    try:
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            return f"<p>无法访问图片链接: {url}，HTTP 状态码: {response.status_code}</p>"
        image = Image.open(BytesIO(response.content)).convert("RGB")
        processed_image_base64 = process_image(image)
        return f'<p>原始链接: {url}</p><img src="data:image/jpeg;base64,{processed_image_base64}" style="max-width: 500px; max-height: 500px;"/><hr>'
    except Exception as e:
        return f"<p>处理图片失败: {url}, 错误信息: {str(e)}</p><hr>"

# 并行处理 Base64 图片
def process_base64(image_base64: str) -> str:
    try:
        image_data = base64.b64decode(image_base64)
        image = Image.open(BytesIO(image_data)).convert("RGB")
        processed_image_base64 = process_image(image)
        return f'<img src="data:image/jpeg;base64,{processed_image_base64}" style="max-width: 500px; max-height: 500px;"/><hr>'
    except Exception as e:
        return f"<p>处理图片失败，错误信息: {str(e)}</p><hr>"
    

@app.post("/process-images", response_model=ImagePreviewResponse)
async def process_images(request: ImageURLRequest):
    previews = []

    for url in request.urls:
        try:
            response = requests.get(url, headers=headers, cookies=cookies)
            if response.status_code != 200:
                raise HTTPException(
                    status_code=400,
                    detail=f"无法访问图片链接: {url}，HTTP 状态码: {response.status_code}, 响应内容: {response.text}",
                )
            image = Image.open(BytesIO(response.content)).convert("RGB")
            base64_image = process_image(image)
            # 生成预览链接
            preview = f"data:image/jpeg;base64,{base64_image}"
            previews.append(preview)
        except requests.exceptions.RequestException as e:
            raise HTTPException(
                status_code=400,
                detail=f"处理图片链接失败: {url}, 错误信息: {str(e)}"
            )
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"未知错误，无法处理图片链接: {url}, 错误信息: {str(e)}"
            )
    return ImagePreviewResponse(result=previews)

@app.post("/process-image-files", response_model=ImagesResponse)
async def process_image_files(files: List[UploadFile] = File(...)):
    previews = []

    def process_file(file: UploadFile):
        try:
            image = Image.open(BytesIO(file.file.read())).convert("RGB")
            base64_image = process_image(image)
            return ImageFileResponse(name=file.filename, value=base64_image)
        except Exception as e:
            return ImageFileResponse(name=file.filename, value=f"处理图片失败: {str(e)}")

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(process_file, files))

    return ImagesResponse(result=results)

# 根路径路由
@app.get("/")
async def root():
    return {"message": "欢迎使用图片处理 API，请访问 /docs 查看完整文档。"}

# 处理图片链接并显示图片
@app.get("/image-url-preview/", response_class=HTMLResponse)
async def image_url_preview(request: Request, image_url: str = None):
    if image_url:
        try:
            response = requests.get(image_url, headers=headers)
            if response.status_code != 200:
                raise HTTPException(
                    status_code=400,
                    detail=f"无法访问图片链接: {image_url}，HTTP 状态码: {response.status_code}, 响应内容: {response.text}",
                )
            image = Image.open(BytesIO(response.content)).convert("RGB")
            
            start_time = time.time()
            
            processing_times = []
            for i in range(1):
                start_time = time.time()
                processed_image_base64 = process_image(image)
                processing_time = time.time() - start_time
                processing_times.append(processing_time)
                print(f"第{i+1}次处理时间: {processing_time:.2f}秒")

            average_processing_time = sum(processing_times) / len(processing_times)
            processed_image_url = f"data:image/jpeg;base64,{processed_image_base64}"
        except requests.exceptions.RequestException as e:
            raise HTTPException(
                status_code=400,
                detail=f"处理图片链接失败: {image_url}, 错误信息: {str(e)}"
            )
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"未知错误，无法处理图片链接: {image_url}, 错误信息: {str(e)}"
            )
    
        return templates.TemplateResponse("show_image.html", {
            "request": request,
            "original_image_url": image_url,
            "processed_image_url": processed_image_url,
            "processing_time": f"{average_processing_time:.2f}"
        })
    
    return templates.TemplateResponse("show_image.html", {"request": request})