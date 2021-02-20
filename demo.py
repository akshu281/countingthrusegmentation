import tkinter as tk
from PIL import ImageTk, Image
from tkinter import filedialog
import numpy as np
import torch
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt
import cv2
from torchvision import transforms
import random

#loading pre-trained model. Using Mask R-CNN
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

#From PyTorch documentation for the Mask R-CNN classes
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic', 'light', 'fire', 'hydrant', 'N/A', 'stop',
    'sign', 'parking', 'meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports', 'ball',
    'kite', 'baseball', 'bat', 'baseball', 'glove', 'skateboard', 'surfboard', 'tennis',
    'racket', 'bottle', 'N/A', 'wine', 'glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot', 'dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted', 'plant', 'bed', 'N/A', 'dining', 'table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell',
    'phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy', 'bear', 'hair', 'drier', 'toothbrush'
]

output_dir = './output/output.mp4'
filename = ''
i_ = 0

def get_coloured_mask(mask):
  """
  random_colour_masks
    parameters:
      - image - predicted masks
    method:
      - the masks of each predicted object is given random colour for visualization
  """
  r = np.zeros_like(mask).astype(np.uint8)
  g = np.zeros_like(mask).astype(np.uint8)
  b = np.zeros_like(mask).astype(np.uint8)
  r[mask == 1], g[mask == 1], b[mask == 1] = [0,255,0]
  coloured_mask = np.stack([r, g, b], axis=2)
  return coloured_mask

def load_img():
    global image_data
    image_data = filedialog.askopenfilename(initialdir="./", title="Choose the video file",
                                       filetypes=(("Text files", "*.txt"),
                                       ("All files", "*.*") ))

    global filename
    filename = image_data.split("/")[-1]

def process():
    cap = cv2.VideoCapture(filename)
    vid_path, vid_writer = None, None
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vid_writer = cv2.VideoWriter(output_dir, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    font = cv2.FONT_HERSHEY_SIMPLEX

    ret = True
    cuda = True
    frame_count = 1
    font = cv2.FONT_HERSHEY_SIMPLEX

    while ret:
        ret,img=cap.read()
        
        confidence = 0.5
        n_people = 0

        if ret == False:
            root.destroy()
            break
        
        model.eval()
        
        draw = img.copy()     
        img_pil = Image.fromarray(img)
        transform = T.Compose([T.ToTensor()])
        img_tensor = transform(img_pil)

        if cuda:
            img_tensor = img_tensor.cuda()
            model.cuda()
        else:
            img_tensor = img_tensor.cpu()
            model.cpu()

        predictions = model([img_tensor])
        pred_score = list(predictions[0]['scores'].detach().cpu().numpy())
        pred_t = [pred_score.index(x) for x in pred_score if x>confidence][-1]
        masks = (predictions[0]['masks']>0.5).squeeze().detach().cpu().numpy()
        pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(predictions[0]['labels'].detach().cpu().numpy())]
        pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(predictions[0]['boxes'].detach().cpu().numpy())]
        
        pred_masks = masks[:pred_t+1]
        pred_boxes = pred_boxes[:pred_t+1]
        pred_class = pred_class[:pred_t+1]

        # print("Stats..")
        # print(len(pred_masks))
        # print(len(pred_boxes))
        # print(len(pred_class))

        for k in range(len(pred_class)):
            if pred_class[k] == 'person':
                n_people += 1
                rgb_mask = get_coloured_mask(pred_masks[k])
                draw = cv2.addWeighted(draw, 1, rgb_mask, 0.5, 0)

        cv2.putText(draw,'People Count:'+str(n_people),(10,50), font, 1,(0,0,255),3)
        vid_writer.write(draw)

        ### UI part ###

        if frame_count % fps == 0:
            # cv2.imwrite(str(frame_count)+'.jpg',draw)           
            # print("Frame number displaying..",frame_count)
            draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
            temp = Image.fromarray(draw)
            basewidth =800
            wpercent = (basewidth / float(temp.size[0]))
            hsize = int((float(temp.size[1]) * float(wpercent)))
            temp = temp.resize((basewidth, hsize), Image.ANTIALIAS)
            temp = ImageTk.PhotoImage(temp)
            panel = tk.Label(frame, text=str("Total People Count:")+str(n_people),anchor = "w").pack()
            panel1 = tk.Label(frame, text=str("People Count (Foreground):")+str(n_people),anchor = "w").pack()
            panel2 = tk.Label(frame, text=str("People Count (Background):")+str("NA"),anchor = "w").pack()
            panel_image = tk.Label(frame, image=temp).pack()
            root.update()
            # print(frame.winfo_children())
            # print(f"People Count at one second interval: {n_people}")

        if frame_count % fps != 0:
            for widget in frame.winfo_children():
                widget.destroy()

        frame_count = frame_count +1 

if __name__ == '__main__':
    root = tk.Tk()
    root.title('People Counting')
    root.resizable(False, False)
    tit = tk.Label(root, text="People Counting", padx=25, pady=6, font=("", 12)).pack()
    canvas = tk.Canvas(root, height=800, width=1000, bg='grey')
    canvas.pack()
    frame = tk.Frame(root, bg='white')
    frame.place(relwidth=0.8, relheight=0.8, relx=0.1, rely=0.1)
    chose_image = tk.Button(root, text='Choose Video',
                            padx=35, pady=10,
                            fg="white", bg="grey", command=load_img)
    chose_image.pack(side=tk.LEFT)
    class_image = tk.Button(root, text='Process Video',
                            padx=35, pady=10,
                            fg="white", bg="grey", command=process)
    class_image.pack(side=tk.RIGHT)
    root.mainloop()





