from inferencer.base_inferencer import (
    BaseInferencer,
    ModelFormat
)

import numpy as np
import cv2
from typing import Any
import torch
from torch import Tensor


class SingleModalInferencer(BaseInferencer):
    """
    单模态推理器（RGB 或 IR）
    """
    
    def __init__(
        self,
        model_path: str,
        input_size: int = 112,
        num_classes: int = 2,
        device: str = "cuda",
        rknn_target: str = "rv1126",
    ):
        """
        初始化单模态推理器
        
        Args:
            model_path: 模型文件路径
            input_size: 输入图像尺寸
            num_classes: 分类类别数
            device: PyTorch推理设备（cuda/cpu）
        """
        self.device: str = device
        self.rknn_target = rknn_target
        self.model: Any = None
        super().__init__(model_path, input_size, num_classes)
    
    def _load_model(self):
        """加载单模态模型"""
        if self.model_format == ModelFormat.PYTORCH:
            self._load_pytorch_model()
        elif self.model_format == ModelFormat.ONNX:
            self._load_onnx_model()
        elif self.model_format == ModelFormat.RKNN:
            self._load_rknn_model()
    
    def _load_pytorch_model(self):
        """加载PyTorch模型"""
        import torch.nn as nn
        from util.model import get_model
        
        # 创建模型
        model = get_model('MobileNetV2', num_class=self.num_classes)
        
        # 加载权重
        devices = self.device if torch.cuda.is_available() else 'cpu'
        state_dict = torch.load(self.model_path, map_location=devices)
        
        # 处理DataParallel包装的权重
        if 'module.' in list(state_dict.keys())[0]:
            model = nn.DataParallel(model)
        
        model.load_state_dict(state_dict)
        model.eval()
        
        if devices == 'cuda':
            model = model.cuda()

        self.model = model
        self.devices: str = devices
    
    def _load_onnx_model(self):
        """加载ONNX模型"""
        import onnxruntime as ort
        
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.model = ort.InferenceSession(self.model_path, providers=providers)
        self.input_name: str = self.model.get_inputs()[0].name
    
    def _load_rknn_model(self):
        """加载RKNN模型"""
        from rknnlite.api import RKNNLite # type: ignore
        
        self.rknn = RKNNLite()
        ret = self.rknn.load_rknn(self.model_path)
        if ret != 0:
            raise RuntimeError(f"加载RKNN模型失败: {self.model_path}")
        
        self.model = self.rknn
        self.rknn_initialized = False
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        预处理单模态图像
        
        Args:
            image: BGR格式图像 (H, W, 3)
            
        Returns:
            预处理后的数据
        """
        # 调整尺寸
        image = cv2.resize(image, (self.input_size, self.input_size))
        
        # 归一化到 [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # HWC -> CHW
        if self.model_format == ModelFormat.RKNN:
            # RKNN 需要 NHWC 格式
            image = np.transpose(image, (2, 0, 1))  # HWC -> CHW
            image = np.transpose(image, (1, 2, 0))  # CHW -> HWC (for RKNN)
            return image.reshape(1, self.input_size, self.input_size, 3)
        else:
            # PyTorch/ONNX 需要 NCHW 格式
            image = np.transpose(image, (2, 0, 1))
            return image.reshape(1, 3, self.input_size, self.input_size)
    
    def infer(self, input_data: np.ndarray) -> np.ndarray:
        """
        执行单模态推理
        
        Args:
            input_data: 预处理后的输入数据
            
        Returns:
            原始模型输出 (1, num_classes)
        """
        if self.model_format == ModelFormat.PYTORCH:
            input_tensor = torch.FloatTensor(input_data)
            if self.devices == 'cuda':
                input_tensor = input_tensor.cuda()
            
            with torch.no_grad():
                output: Tensor = self.model(input_tensor)
            
            return output.cpu().numpy()
        
        elif self.model_format == ModelFormat.ONNX:
            outputs: list[np.ndarray] = self.model.run(None, {self.input_name: input_data})
            return outputs[0]
        
        elif self.model_format == ModelFormat.RKNN:
            # 初始化运行时（仅第一次）
            if not self.rknn_initialized:
                ret = self.model.init_runtime(target=self.rknn_target)
                if ret != 0:
                    raise RuntimeError(
                        f"初始化RKNN运行时失败，target={self.rknn_target}"
                    )
                self.rknn_initialized = True
            
            outputs = self.model.inference(inputs=[input_data])
            return np.array(outputs[0])
        else:
            raise ValueError(f"不支持的模型格式: {self.model_format}")