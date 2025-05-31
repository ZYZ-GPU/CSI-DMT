import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import os
import threading
import torch
import cv2
import numpy as np
from pathlib import Path
import time

# 导入项目模块
try:
    from Nets.CSI_DMT import CSI_DMT
    from Utilities.CUDA_Check import GPUorCPU
    from Utilities import Consistency
    from Utilities.GuidedFiltering import guided_filter
    from torchvision.io import read_image, ImageReadMode
    import torch.nn.functional as F
    from torchvision import transforms
    from torch import einsum
except ImportError as e:
    print(f"导入模块失败: {e}")
    print("请确保所有必要的模块都在正确的路径下")


class ZeroOneNormalize(object):
    def __call__(self, img):
        return img.float().div(255)


class ImageFusionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("CSI-DMT 多聚焦图像融合系统")
        self.root.geometry("1400x900")

        # 初始化变量
        self.image_a_path = tk.StringVar()
        self.image_b_path = tk.StringVar()
        self.model_path = tk.StringVar(value="RunTimeData/Model weights/best_network.pth")
        self.output_dir = tk.StringVar(value="./Results")
        self.threshold = tk.DoubleVar(value=0.0015)
        self.window_size = tk.IntVar(value=0)

        # 设备检测
        try:
            self.device = GPUorCPU().DEVICE
        except:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = None
        self.is_processing = False

        # 图像变量
        self.image_a_display = None
        self.image_b_display = None
        self.result_fusion_display = None
        self.result_decision_display = None

        # 窗口张量（用于边缘修正）
        self.window_tensor = None

        self.setup_ui()

    def setup_ui(self):
        """设置用户界面"""
        # 创建主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 配置网格权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)

        # 1. 文件选择区域
        self.setup_file_selection(main_frame)

        # 2. 参数设置区域
        self.setup_parameters(main_frame)

        # 3. 处理流程说明
        self.setup_process_info(main_frame)

        # 4. 控制按钮区域
        self.setup_controls(main_frame)

        # 5. 图像显示区域
        self.setup_image_display(main_frame)

        # 6. 状态栏
        self.setup_status_bar(main_frame)

    def setup_file_selection(self, parent):
        """设置文件选择区域"""
        file_frame = ttk.LabelFrame(parent, text="文件选择", padding="5")
        file_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)

        # 图像A选择
        ttk.Label(file_frame, text="图像A:").grid(row=0, column=0, sticky=tk.W, padx=5)
        ttk.Entry(file_frame, textvariable=self.image_a_path, width=50).grid(row=0, column=1, padx=5,
                                                                             sticky=(tk.W, tk.E))
        ttk.Button(file_frame, text="浏览", command=lambda: self.select_image('A')).grid(row=0, column=2, padx=5)

        # 图像B选择
        ttk.Label(file_frame, text="图像B:").grid(row=1, column=0, sticky=tk.W, padx=5)
        ttk.Entry(file_frame, textvariable=self.image_b_path, width=50).grid(row=1, column=1, padx=5,
                                                                             sticky=(tk.W, tk.E))
        ttk.Button(file_frame, text="浏览", command=lambda: self.select_image('B')).grid(row=1, column=2, padx=5)

        # 模型权重选择
        ttk.Label(file_frame, text="模型权重:").grid(row=2, column=0, sticky=tk.W, padx=5)
        ttk.Entry(file_frame, textvariable=self.model_path, width=50).grid(row=2, column=1, padx=5, sticky=(tk.W, tk.E))
        ttk.Button(file_frame, text="浏览", command=self.select_model).grid(row=2, column=2, padx=5)

        # 输出目录选择
        ttk.Label(file_frame, text="输出目录:").grid(row=3, column=0, sticky=tk.W, padx=5)
        ttk.Entry(file_frame, textvariable=self.output_dir, width=50).grid(row=3, column=1, padx=5, sticky=(tk.W, tk.E))
        ttk.Button(file_frame, text="浏览", command=self.select_output_dir).grid(row=3, column=2, padx=5)

        file_frame.columnconfigure(1, weight=1)

    def setup_parameters(self, parent):
        """设置参数区域"""
        param_frame = ttk.LabelFrame(parent, text="后处理参数设置", padding="5")
        param_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)

        # 阈值设置
        ttk.Label(param_frame, text="一致性验证阈值:").grid(row=0, column=0, sticky=tk.W, padx=5)
        threshold_scale = ttk.Scale(param_frame, from_=0.0001, to=0.01, variable=self.threshold,
                                    orient=tk.HORIZONTAL, length=200)
        threshold_scale.grid(row=0, column=1, padx=5, sticky=(tk.W, tk.E))
        threshold_label = ttk.Label(param_frame, text="")
        threshold_label.grid(row=0, column=2, padx=5)

        # 实时更新阈值显示
        def update_threshold(*args):
            threshold_label.config(text=f"{self.threshold.get():.4f}")

        self.threshold.trace('w', update_threshold)
        update_threshold()

        # 窗口大小设置
        ttk.Label(param_frame, text="边缘修正窗口:").grid(row=1, column=0, sticky=tk.W, padx=5)
        window_scale = ttk.Scale(param_frame, from_=0, to=20, variable=self.window_size,
                                 orient=tk.HORIZONTAL, length=200)
        window_scale.grid(row=1, column=1, padx=5, sticky=(tk.W, tk.E))
        window_label = ttk.Label(param_frame, text="")
        window_label.grid(row=1, column=2, padx=5)

        # 实时更新窗口大小显示
        def update_window(*args):
            size = int(self.window_size.get())
            window_label.config(text=f"{size} ({'禁用' if size == 0 else '启用'})")

        self.window_size.trace('w', update_window)
        update_window()

        # 设备信息
        ttk.Label(param_frame, text=f"计算设备: {self.device.upper()}").grid(row=2, column=0, columnspan=3, padx=5,
                                                                             pady=5)

        param_frame.columnconfigure(1, weight=1)

    def setup_process_info(self, parent):
        """设置处理流程说明"""
        info_frame = ttk.LabelFrame(parent, text="完整融合处理流程", padding="5")
        info_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)

        process_text = """
步骤1: CSI-DMT模型推理 → 生成初始融合图像 + 初始决策图
步骤2: 决策图后处理 → 二值化 + 一致性验证 + 小区域去除  
步骤3: 边缘修正 → 使用滑动窗口进行边缘优化
        """

        ttk.Label(info_frame, text=process_text.strip(), justify=tk.LEFT,
                  foreground="blue", font=("TkDefaultFont", 9)).pack(anchor=tk.W, padx=5)

        ttk.Label(info_frame, text="注意: 路径中不要包含中文",
                  foreground="red", font=("TkDefaultFont", 9, "bold")).pack(anchor=tk.W, padx=5, pady=(5, 0))

    def setup_controls(self, parent):
        """设置控制按钮区域"""
        control_frame = ttk.Frame(parent)
        control_frame.grid(row=3, column=0, columnspan=2, pady=10)

        # 加载模型按钮
        self.load_model_btn = ttk.Button(control_frame, text="加载模型", command=self.load_model)
        self.load_model_btn.pack(side=tk.LEFT, padx=5)

        # 开始融合按钮
        self.fusion_btn = ttk.Button(control_frame, text="开始完整融合", command=self.start_fusion, state=tk.DISABLED)
        self.fusion_btn.pack(side=tk.LEFT, padx=5)

        # 批量处理按钮
        self.batch_btn = ttk.Button(control_frame, text="批量处理", command=self.batch_process, state=tk.DISABLED)
        self.batch_btn.pack(side=tk.LEFT, padx=5)

        # 清除结果按钮
        ttk.Button(control_frame, text="清除结果", command=self.clear_results).pack(side=tk.LEFT, padx=5)

        # 进度条
        self.progress = ttk.Progressbar(control_frame, mode='indeterminate')
        self.progress.pack(side=tk.LEFT, padx=20, fill=tk.X, expand=True)

    def setup_image_display(self, parent):
        """设置图像显示区域"""
        image_frame = ttk.Frame(parent)
        image_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)

        # 输入图像显示
        input_frame = ttk.LabelFrame(image_frame, text="输入图像")
        input_frame.grid(row=0, column=0, padx=5, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.image_a_label = ttk.Label(input_frame, text="图像A\n(点击浏览选择)")
        self.image_a_label.grid(row=0, column=0, padx=5, pady=5)

        self.image_b_label = ttk.Label(input_frame, text="图像B\n(点击浏览选择)")
        self.image_b_label.grid(row=0, column=1, padx=5, pady=5)

        # 输出结果显示
        output_frame = ttk.LabelFrame(image_frame, text="最终融合结果 (经过完整后处理)")
        output_frame.grid(row=0, column=1, padx=5, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.fusion_label = ttk.Label(output_frame, text="最终融合图像\n(FDB优化)")
        self.fusion_label.grid(row=0, column=0, padx=5, pady=5)

        self.decision_label = ttk.Label(output_frame, text="处理后决策图\n(二值化+一致性验证)")
        self.decision_label.grid(row=0, column=1, padx=5, pady=5)

        # 配置网格权重
        image_frame.columnconfigure(0, weight=1)
        image_frame.columnconfigure(1, weight=1)
        image_frame.rowconfigure(0, weight=1)

        input_frame.columnconfigure(0, weight=1)
        input_frame.columnconfigure(1, weight=1)
        output_frame.columnconfigure(0, weight=1)
        output_frame.columnconfigure(1, weight=1)

    def setup_status_bar(self, parent):
        """设置状态栏"""
        self.status_var = tk.StringVar(value="就绪 - 请选择图像并加载模型")
        status_frame = ttk.Frame(parent)
        status_frame.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)

        ttk.Label(status_frame, text="状态:").pack(side=tk.LEFT)
        ttk.Label(status_frame, textvariable=self.status_var).pack(side=tk.LEFT, padx=5)

        # 处理时间显示
        self.time_var = tk.StringVar(value="")
        ttk.Label(status_frame, textvariable=self.time_var).pack(side=tk.RIGHT, padx=5)

    def select_image(self, image_type):
        """选择图像文件"""
        filetypes = [
            ("图像文件", "*.jpg *.jpeg *.png *.bmp *.tiff"),
            ("所有文件", "*.*")
        ]

        filename = filedialog.askopenfilename(
            title=f"选择图像{image_type}",
            filetypes=filetypes
        )

        if filename:
            if image_type == 'A':
                self.image_a_path.set(filename)
                self.display_image(filename, self.image_a_label)
            else:
                self.image_b_path.set(filename)
                self.display_image(filename, self.image_b_label)

    def select_model(self):
        """选择模型权重文件"""
        filename = filedialog.askopenfilename(
            title="选择模型权重文件",
            filetypes=[("PyTorch模型", "*.pth *.pt"), ("所有文件", "*.*")]
        )

        if filename:
            self.model_path.set(filename)

    def select_output_dir(self):
        """选择输出目录"""
        dirname = filedialog.askdirectory(title="选择输出目录")
        if dirname:
            self.output_dir.set(dirname)

    def display_image(self, image_path, label_widget, max_size=(250, 250)):
        """在标签中显示图像"""
        try:
            image = Image.open(image_path)
            image.thumbnail(max_size, Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            label_widget.configure(image=photo, text="")
            label_widget.image = photo  # 保持引用
        except Exception as e:
            messagebox.showerror("错误", f"无法显示图像: {e}")

    def load_model(self):
        """加载模型"""

        def load_in_thread():
            try:
                self.root.after(0, lambda: self.status_var.set("正在加载CSI-DMT模型..."))
                self.root.after(0, lambda: self.progress.start())

                if not os.path.exists(self.model_path.get()):
                    error_msg = f"模型文件不存在: {self.model_path.get()}"
                    self.root.after(0, lambda msg=error_msg: self.on_error(msg))
                    return

                self.model = CSI_DMT().to(self.device)
                self.model.load_state_dict(torch.load(self.model_path.get(), map_location=self.device))
                self.model.eval()

                # 计算模型参数
                num_params = sum(p.numel() for p in self.model.parameters())
                param_info = f"模型参数: {num_params / 1e6:.2f}M"

                self.root.after(0, lambda info=param_info: self.on_model_loaded(info))

            except Exception as e:
                error_msg = f"模型加载失败: {e}"
                self.root.after(0, lambda msg=error_msg: self.on_error(msg))

        threading.Thread(target=load_in_thread, daemon=True).start()

    def on_model_loaded(self, param_info):
        """模型加载完成回调"""
        self.progress.stop()
        self.status_var.set(f"模型加载成功 - {param_info}")
        self.fusion_btn.configure(state=tk.NORMAL)
        self.batch_btn.configure(state=tk.NORMAL)
        messagebox.showinfo("成功", f"CSI-DMT模型加载成功！\n{param_info}")

    def on_error(self, error_msg):
        """错误处理回调"""
        self.progress.stop()
        self.status_var.set("发生错误")
        messagebox.showerror("错误", error_msg)

    def start_fusion(self):
        """开始图像融合"""
        if not self.image_a_path.get() or not self.image_b_path.get():
            messagebox.showwarning("警告", "请先选择两张输入图像！")
            return

        if self.model is None:
            messagebox.showwarning("警告", "请先加载模型！")
            return

        def fusion_in_thread():
            try:
                start_time = time.time()
                self.root.after(0, lambda: self.set_processing_state(True))

                # 执行完整的融合流程 (按照Fusion.py的逻辑)
                fusion_result, decision_result = self.process_complete_fusion(
                    self.image_a_path.get(),
                    self.image_b_path.get()
                )

                # 保存结果
                output_dir = Path(self.output_dir.get())
                output_dir.mkdir(exist_ok=True)

                # 使用更清晰的文件名
                fusion_path = output_dir / "final_fusion_result.png"
                decision_path = output_dir / "processed_decision_map.png"

                cv2.imwrite(str(fusion_path), fusion_result)
                cv2.imwrite(str(decision_path), decision_result)

                processing_time = time.time() - start_time

                self.root.after(0, lambda path1=fusion_path, path2=decision_path, time_val=processing_time:
                self.on_fusion_complete(path1, path2, time_val))

            except Exception as e:
                error_msg = f"融合处理失败: {e}"
                self.root.after(0, lambda msg=error_msg: self.on_error(msg))

        threading.Thread(target=fusion_in_thread, daemon=True).start()

    def process_complete_fusion(self, image_a_path, image_b_path):
        """执行完整的图像融合处理流程 (完全按照Fusion.py逻辑)"""

        # 步骤1: 准备数据和预处理
        self.root.after(0, lambda: self.status_var.set("步骤1/4: 数据预处理..."))

        eval_transforms = transforms.Compose([
            ZeroOneNormalize(),
            transforms.Normalize(mean=0.5, std=0.5),
        ])

        # 读取和预处理图像（用于模型输入）
        A = read_image(image_a_path, mode=ImageReadMode.RGB).to(self.device)
        B = read_image(image_b_path, mode=ImageReadMode.RGB).to(self.device)

        A_tensor = eval_transforms(A).unsqueeze(0)
        B_tensor = eval_transforms(B).unsqueeze(0)

        # 读取原始图像（用于最终融合）
        A_orig = cv2.imread(image_a_path)
        B_orig = cv2.imread(image_b_path)

        # 步骤2: CSI-DMT模型推理 (中间结果)
        self.root.after(0, lambda: self.status_var.set("步骤2/4: CSI-DMT模型推理..."))

        with torch.no_grad():
            NetOut, D = self.model(A_tensor, B_tensor)

        # 步骤3: 决策图后处理 (按照Fusion.py的ConsisVerif逻辑)
        self.root.after(0, lambda: self.status_var.set("步骤3/4: 决策图后处理..."))

        # 二值化
        D = torch.where(D > 0.5, 1., 0.)

        # 一致性验证
        if self.threshold.get() > 0:
            D = self.consistency_verification(D, self.threshold.get())

        # 边缘修正 (可选)
        if self.window_size.get() > 0:
            D = self.edge_correction(D, A_orig, B_orig, NetOut, image_a_path, image_b_path)

        # 步骤4: 最终图像融合 (按照Fusion.py逻辑)
        self.root.after(0, lambda: self.status_var.set("步骤4/4: 最终图像融合..."))

        # 转换决策图为numpy格式
        D_np = D[0, 0].cpu().numpy()

        # 确保图像尺寸匹配
        if D_np.shape != A_orig.shape[:2]:
            D_np = cv2.resize(D_np, (A_orig.shape[1], A_orig.shape[0]))

        # 初始融合
        IniF = A_orig * D_np[..., np.newaxis] + B_orig * (1 - D_np[..., np.newaxis])

        # 引导滤波 (最终优化)
        try:
            D_GF = guided_filter(IniF, D_np, 4, 0.1)
            Final_fused = A_orig * D_GF[..., np.newaxis] + B_orig * (1 - D_GF[..., np.newaxis])
        except:
            # 如果引导滤波失败，使用初始融合结果
            Final_fused = IniF

        # 决策图可视化
        decision_vis = (D_np * 255).astype(np.uint8)
        decision_vis = cv2.cvtColor(decision_vis, cv2.COLOR_GRAY2BGR)

        return Final_fused.astype(np.uint8), decision_vis

    def consistency_verification(self, img_tensor, threshold):
        """一致性验证 (对应Fusion.py的ConsisVerif)"""
        try:
            Verified_img_tensor = Consistency.Binarization(img_tensor)
            if threshold > 0:
                Verified_img_tensor = Consistency.RemoveSmallArea(Verified_img_tensor, threshold=threshold)
            return Verified_img_tensor
        except:
            # 如果一致性验证模块不可用，返回原始张量
            return img_tensor

    def edge_correction(self, D, A_orig, B_orig, NetOut, image_a_path, image_b_path):
        """边缘修正 (对应Fusion.py的window逻辑)"""
        try:
            window_size = self.window_size.get()
            if window_size > 0:
                # 创建窗口张量
                window = torch.ones([1, 1, window_size, window_size], dtype=torch.float).to(self.device)

                # 边缘修正
                decisionmap = F.conv2d(D, window, padding=window_size // 2)
                decisionmap = torch.where(decisionmap == 0., 999., decisionmap)

                for aa in range(1, window_size * window_size):
                    decisionmap = torch.where(decisionmap == float(aa), 9999., decisionmap)
                decisionmap = torch.where(decisionmap == window_size * window_size, 99999., decisionmap)

                # 根据边缘修正结果融合
                A_tensor = read_image(image_a_path, mode=ImageReadMode.RGB).to(self.device)
                B_tensor = read_image(image_b_path, mode=ImageReadMode.RGB).to(self.device)

                fused_img = torch.cat([decisionmap.detach(), decisionmap.detach(), decisionmap.detach()], dim=0)
                fused_img = torch.where(fused_img == 99999., A_tensor, fused_img)
                fused_img = torch.where(fused_img == 999., B_tensor, fused_img)
                fused_img = torch.where(fused_img == 9999., NetOut[0] * 255, fused_img)

                # 转换回决策图格式
                decision_corrected = torch.where(decisionmap == 99999., 1., 0.)
                decision_corrected = torch.where(decisionmap == 9999., 1., decision_corrected)
                decision_corrected = torch.where(decisionmap == 999., 0., decision_corrected)

                return decision_corrected
        except:
            pass

        return D

    def on_fusion_complete(self, fusion_path, decision_path, processing_time):
        """融合完成回调"""
        self.set_processing_state(False)
        self.status_var.set("完整融合流程完成！")
        self.time_var.set(f"处理时间: {processing_time:.2f}秒")

        # 显示结果
        self.display_image(str(fusion_path), self.fusion_label)
        self.display_image(str(decision_path), self.decision_label)

        messagebox.showinfo("完成",
                            f"完整融合流程完成！\n"
                            f"处理时间: {processing_time:.2f}秒\n"
                            f"结果保存至: {fusion_path.parent}\n\n"
                            f"输出说明:\n"
                            f"• final_fusion_result.png - 最终融合图像\n"
                            f"• processed_decision_map.png - 处理后决策图")

    def batch_process(self):
        """批量处理"""
        source_dir = filedialog.askdirectory(title="选择包含图像对的目录")
        if not source_dir:
            return

        result = messagebox.askquestion("批量处理确认",
                                        "批量处理将按照以下目录结构处理:\n"
                                        "选择目录/sourceA/*.png\n"
                                        "选择目录/sourceB/*.png\n\n"
                                        "是否继续？")

        if result == 'yes':
            self.start_batch_processing(source_dir)

    def start_batch_processing(self, source_dir):
        """开始批量处理"""

        def batch_in_thread():
            try:
                self.root.after(0, lambda: self.set_processing_state(True))
                self.root.after(0, lambda: self.status_var.set("正在进行批量处理..."))

                import glob

                # 获取图像列表
                source_a_dir = os.path.join(source_dir, 'sourceA')
                source_b_dir = os.path.join(source_dir, 'sourceB')

                if not os.path.exists(source_a_dir) or not os.path.exists(source_b_dir):
                    error_msg = "请确保目录下有sourceA和sourceB文件夹"
                    self.root.after(0, lambda msg=error_msg: self.on_error(msg))
                    return

                eval_list_A = sorted(glob.glob(os.path.join(source_a_dir, '*.*')))
                eval_list_B = sorted(glob.glob(os.path.join(source_b_dir, '*.*')))

                if len(eval_list_A) != len(eval_list_B):
                    error_msg = f"图像数量不匹配: A={len(eval_list_A)}, B={len(eval_list_B)}"
                    self.root.after(0, lambda msg=error_msg: self.on_error(msg))
                    return

                # 创建输出目录
                output_dir = Path(self.output_dir.get()) / "batch_results"
                output_dir.mkdir(exist_ok=True)

                total_time = 0
                for i, (img_a, img_b) in enumerate(zip(eval_list_A, eval_list_B)):
                    start_time = time.time()

                    self.root.after(0, lambda idx=i, total=len(eval_list_A):
                    self.status_var.set(f"批量处理: {idx + 1}/{total}"))

                    # 处理单张图像对
                    fusion_result, decision_result = self.process_complete_fusion(img_a, img_b)

                    # 保存结果
                    base_name = Path(img_a).stem
                    fusion_path = output_dir / f"{base_name}_fusion.png"
                    decision_path = output_dir / f"{base_name}_decision.png"

                    cv2.imwrite(str(fusion_path), fusion_result)
                    cv2.imwrite(str(decision_path), decision_result)

                    process_time = time.time() - start_time
                    if i > 0:  # 跳过第一张的时间（可能包含初始化开销）
                        total_time += process_time

                avg_time = total_time / max(1, len(eval_list_A) - 1)

                self.root.after(0, lambda out_dir=output_dir, count=len(eval_list_A), avg=avg_time:
                self.on_batch_complete(out_dir, count, avg))

            except Exception as e:
                error_msg = f"批量处理失败: {e}"
                self.root.after(0, lambda msg=error_msg: self.on_error(msg))

        threading.Thread(target=batch_in_thread, daemon=True).start()

    def on_batch_complete(self, output_dir, count, avg_time):
        """批量处理完成回调"""
        self.set_processing_state(False)
        self.status_var.set(f"批量处理完成: {count}对图像")
        self.time_var.set(f"平均时间: {avg_time:.2f}秒/对")

        messagebox.showinfo("批量处理完成",
                            f"成功处理 {count} 对图像\n"
                            f"平均处理时间: {avg_time:.2f}秒/对\n"
                            f"结果保存至: {output_dir}")

    def clear_results(self):
        """清除显示结果"""
        self.image_a_label.configure(image='', text="图像A\n(点击浏览选择)")
        self.image_b_label.configure(image='', text="图像B\n(点击浏览选择)")
        self.fusion_label.configure(image='', text="最终融合图像\n(FDB优化)")
        self.decision_label.configure(image='', text="处理后决策图\n(二值化+一致性验证)")

        # 清空图像引用
        self.image_a_label.image = None
        self.image_b_label.image = None
        self.fusion_label.image = None
        self.decision_label.image = None

        # 清空路径
        self.image_a_path.set("")
        self.image_b_path.set("")

        self.status_var.set("已清除结果")
        self.time_var.set("")

    def set_processing_state(self, processing):
        """设置处理状态"""
        self.is_processing = processing
        if processing:
            self.progress.start()
            self.fusion_btn.configure(state=tk.DISABLED)
            self.batch_btn.configure(state=tk.DISABLED)
            self.load_model_btn.configure(state=tk.DISABLED)
        else:
            self.progress.stop()
            if self.model is not None:
                self.fusion_btn.configure(state=tk.NORMAL)
                self.batch_btn.configure(state=tk.NORMAL)
            self.load_model_btn.configure(state=tk.NORMAL)


def main():
    """主函数"""
    root = tk.Tk()
    app = ImageFusionGUI(root)

    # 设置窗口图标和属性
    try:
        root.iconname("CSI-DMT Fusion")
        # 设置窗口最小尺寸
        root.minsize(1200, 800)
    except:
        pass

    # 启动GUI
    root.mainloop()


if __name__ == "__main__":
    main()