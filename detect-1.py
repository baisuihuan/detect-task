"""
Run YOLOv5 detection inference on  webcam
Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam                                                    
Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
"""
#导入所需的各种库
import argparse
import os
import sys
from pathlib import Path
import torch
import cv2
import numpy as np
from typing import List,Tuple

#设置项目根目录
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

#导入YOLOv5的模板
from utils.dataloaders import LoadStreams
from utils.torch_utils import select_device, smart_inference_mode
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from ultralytics.utils.plotting import Annotator, colors
#类的构造
class CameraDetector:
    def __init__(self, weights: Path = ROOT / "yolov5s.pt", conf_thres: float = 0.25, iou_thres: float=0.45):
        self.conf_thres=conf_thres
        self.iou_thres=iou_thres
        self.device=select_device("")
        self.model=DetectMultiBackend(weights,device=self.device)
    #预处理
    def _process_detections(self, det: torch.Tensor, im0: np.ndarray, im: torch.Tensor) -> Tuple[np.ndarray, List]:
        detection_info = []
        if len(det):
         #如果有检测到的对象，将边框恢复到原始尺寸大小
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
            annotator = Annotator(im0, line_width=3, example=str(self.model.names))
            for *xyxy, conf, cls in reversed(det):
                c = int(cls) 
                label = f"{self.model.names[c]} {conf:.2f}"
                annotator.box_label(xyxy, label, color=colors(c))
                center_x=int((xyxy[0]+xyxy[2]) / 2)
                center_y=int((xyxy[1]+xyxy[3]) / 2)
                color=colors(c)
                cv2.circle(im0, (center_x,center_y), 5, color, -1)
                coord_text=f"({center_x},{center_y})"
                cv2.putText(im0, coord_text, (center_x+10,center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
                detection_info.append((self.model.names[c],center_x,center_y))
        return im0, detection_info        

    @smart_inference_mode()
    def run(self, source: str='0') ->None:
        dataset = LoadStreams(source, img_size=(640,640), stride=self.model.stride, auto=self.model.pt, vid_stride=1)
        for path, im, im0s, vid_cap, s in dataset:
            im = torch.from_numpy(im).to(self.model.device)
            im = im.half() if self.model.fp16 else im.float()
            im /= 255
            if len(im.shape)==3:
                im=im[None]
            pred=self.model(im)
            pred=non_max_suppression(pred, self.conf_thres, self.iou_thres)
            for i, det in enumerate(pred):
                im0=im0s[i].copy()
                im0, detection_info=self._process_detections(det,im0,im)
                cv2.imshow('YOLOv5 Detection', im0)
                if detection_info:
                    print(f"Detected {len(detection_info)} objects")
                    for name, cx, cy in detection_info:
                        print(f" -{name} at center ({cx},{cy})")
                else:
                    print("No objects detected.")
                if cv2.waitKey(1) == ord('q'):
                    cv2.destroyAllWindows()
                    return
#运行检测器
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default='0', help="source of the video steam(default:0 for webcam)")
    opt=parser.parse_args()

    detector=CameraDetector()
    detector.run(source=opt.source)

if __name__=="__main__":
    main()