import streamlit as st 
from streamlit_option_menu import option_menu
import os
import time
import torch
from models.common import DetectMultiBackend
from PIL import Image
import numpy as np
import tempfile
import imageio
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.torch_utils import select_device, smart_inference_mode
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
import requests
from streamlit_lottie import st_lottie


def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def detectImage(image, imgsz, stride, pt, model, conf_index, iou_index, option):
    imageName = image.name
    image = Image.open(image)
    image=image.convert('RGB')
    path = f"./data/images/{imageName}"
    image.save(path)
    dataset = LoadImages(path, img_size=imgsz, stride=stride, auto=pt, vid_stride=1)
    
    bs = 1 
    model.warmup(imgsz=(1, 3, *imgsz))  
    for path, im, im0s, vid_cap, s in dataset:
        im = torch.from_numpy(im).to(model.device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]
        pred = model(im)
        # conf_thres, iou_thres = 0.25,0.25
        pred = non_max_suppression(pred,conf_index, iou_index)
    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
    for i, det in enumerate(pred):
        if len(det):
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

    img= Image.open(path)

    img = np.array(img)
    a, b = st.columns(2)
    with b:
        i = 0
        for inf in det : 
            c1, c2 = (int( inf[0]), int( inf[1])), (int(inf[2]), int(inf[3]))
            # st.write(inf)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            font_color = (255, 0, 0) 
            thickness = 2
            label= "TS {:.2%}".format(inf[-2])
            cv2.putText(img,label , (int( inf[0]), int( inf[1])-10), font, font_scale, font_color, thickness, cv2.LINE_AA)
            cv2.rectangle(img, c1, c2, [255,0,0], thickness=2, lineType=cv2.LINE_AA)
            st.write("Traffic Signs")
            st.write("Precision: {:.2%}".format(inf[-2]))
            imagecrop = image.crop((int(inf[0]), int(inf[1]), int(inf[2]), int(inf[3])))
            st.image(imagecrop, width=300)
            # st.divider()
            if option == "Oui":
                folder = "./media/ts"
                os.makedirs(folder, exist_ok=True)
                imagecrop.save(f"./media/ts/TS-{imageName}-{i}.jpg")
                i += 1

    # img.save("output.jpeg") 
    # os.remove(path)
    with a:
        st.image(img)
    
 
def detectVideo(video, imgsz, stride, pt, model, result, conf_index, iou_index, option):
    videoName = video.name
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video.read())
    cap = cv2.VideoCapture(tfile.name)
    
    writer = imageio.get_writer(f'./data/videos/{videoName}', fps=cap.get(cv2.CAP_PROP_FPS), codec='libx264')
    
    max_imgsz = 2000  # Maximum allowed imgsz value (adjust if needed)
    if imgsz[0] > max_imgsz or imgsz[1] > max_imgsz:
        ratio = max(imgsz[0] / max_imgsz, imgsz[1] / max_imgsz)
        imgsz = (int(imgsz[0] / ratio), int(imgsz[1] / ratio))
    inc = 0   
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            
            frame = cv2.resize(frame, (imgsz[0], imgsz[1]))
            
            cv2.imwrite("input3.jpg",frame)
            path = "input3.jpg"
            dataset = LoadImages(path, img_size=imgsz, stride=stride, auto=pt, vid_stride=1)
            bs = 1 
            model.warmup(imgsz=(1, 3, *imgsz))  
            for path, im, im0s, vid_cap, s in dataset:
                im = torch.from_numpy(im).to(model.device)
                im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]
                pred = model(im)
                # conf_thres, iou_thres = conf_index, iou_index
                pred = non_max_suppression(pred,conf_index, iou_index)
            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
            for i, det in enumerate(pred):
                if len(det):
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

            img= Image.open(path)
            img = np.array(img)
            for inf in det : 
                c1, c2 = (int( inf[0]), int( inf[1])), (int(inf[2]), int(inf[3]))
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1
                font_color = (255, 0, 0) 
                thickness = 2
                label= "TS {:.2%}".format(inf[-2]) 
                cv2.putText(img,label , (int( inf[0]), int( inf[1])-10), font, font_scale, font_color, thickness, cv2.LINE_AA)
                cv2.rectangle(img, c1, c2, [255,0,0], thickness=2, lineType=cv2.LINE_AA)
                
            # writer.append_data(img)
            # i += 1
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result.image(img, width=500)
            if option == "Oui" and len(det) > 0:
                folder = "./media/frames"
                folderVideo = f"{folder}/{videoName}"
                os.makedirs(folder, exist_ok=True)
                os.makedirs(folderVideo, exist_ok=True)
                image_path = os.path.join(folderVideo, f"image_{inc}.jpg")
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                cv2.imwrite(image_path, img_rgb)
                inc += 1
                # img_result = Image.open(img)
                # img_result.save(f"./media/frames/TS-{videoName}-{i}.jpg")
            # TSimage = st.empty()
            # TSimage.image(img)
            # time.sleep(1)
        else:
            break
    writer.close()

def main():
    
    st.set_page_config(page_title = "Traffic Signs", page_icon="🛑", layout="wide")

    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
            width: 340px;
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
            width: 340px;
            margin-left: -340px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    #################### Title #####################################################

    st.markdown("<h2 style='text-align: center; color: red; font-weight: bold; font-family: font of choice,    fallback font no1, sans-serif;'>Détection des panneaux de signalisation</h2>", unsafe_allow_html=True)
    a, b, c = st.columns((1,2,1))
    try:
        traffic_lottie = load_lottieurl('https://assets8.lottiefiles.com/packages/lf20_SDVHP8.json')
    except:
        traffic_lottie = ""
        
    with a:
        st_lottie(traffic_lottie, height=0, key="Traffic Signs")
    with b:
        st.markdown("<h4 style='text-align: center; color: grey; font-family: font of choice, fallback font no1, sans-serif;'>Projet de fin d'étude 2022/2023</h4>", unsafe_allow_html=True)
    with c:
        st_lottie(traffic_lottie, height=0, key="Traffic signs")

        
    device = select_device("cpu")
    with st.spinner("Veuillez patienter quelques instants..."):
        model = DetectMultiBackend("./runs/train/yolov5s_results3/weights/best.pt")
    stride, names, pt = model.stride, model.names, model.pt
    imgsz=(640, 640)
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    
    
    with st.expander("A PROPOS DE NOUS", expanded=False):
        st.info("Nous sommes Aymane Fadili et Oualid Ait Aissa, étudiants en troisième année de licence à l'université Samlalia, dans la filière Science Mathématique et Informatique (SMI) pour l'année universitaire 2022/2023. Encadrés par Jihad Zahir et co-encadrés par Ouassine Younes, nous avons réalisé un stage de deux mois, du 08/05/2023 au 08/07/2023, dans le but de valider notre licence et de développer notre expertise en vision par ordinateur. Ce projet revêt également une importance particulière pour nous, car il nous permet de nous préparer en vue de notre future poursuite d'études en mastère, avec une orientation vers la science des données. En nous impliquant activement dans la détection des panneaux de signalisation routière en utilisant YOLOv5 et en le déployant sur la plateforme Streamlit, nous avons pu acquérir des compétences essentielles dans le domaine de l'analyse visuelle des données, renforçant ainsi notre parcours académique et notre préparation pour le mastère.")
        if st.button("Contact"):
            st.warning("FADILI AYMANE")
            st.write("Email: aymanefdl1@gmail.com")
            st.write("LinkedIn: https://www.linkedin.com/in/fadili-aymane")
            
            st.warning("AIT AISSA OUALID")
            st.write("Email: oualid.aitaissa@gmail.com")
            st.write("LinkedIn: https://www.linkedin.com/in/oualid-ait-aissa-914643239")
        
    with st.expander("Description de l'outil", expanded=False):
        st.info("DESCRIPTION: Le projet vise à développer un système de détection des panneaux de signalisation au Maroc en utilisant le modèle pré-entraîné YOLOv5. L'objectif principal est d'obtenir une précision élevée dans la détection et la reconnaissance des panneaux de signalisation sur les routes. Ce système permettra d'améliorer la sécurité routière en identifiant de manière précise et fiable les panneaux de signalisation.")
        st.info("BASE DE DONNEES: Le modèle est entraîné en utilisant un vaste ensemble d'images annotées représentant différents types de panneaux de signalisation routière, accompagnées d'informations de localisation correspondantes. En analysant ces images étiquetées, le modèle acquiert la capacité de reconnaître les caractéristiques visuelles spécifiques des panneaux de signalisation et d'appliquer ces connaissances pour détecter de nouveaux panneaux dans des images inédites. Ce processus d'entraînement permet au modèle de devenir compétent dans la reconnaissance précise des différents types de panneaux de signalisation, ce qui contribue grandement à l'efficacité globale du projet.")
        st.info("UTILISATION: Notre projet utilise ce modèle pour détecter les panneaux de signalisation routière dans des images ou des vidéos. Il affiche des encadrés autour des panneaux détectés, fournissant ainsi une indication visuelle claire de leur emplacement.")
        # st.success("In summary, the coral detection model using YOLOv5 is a powerful artificial intelligence tool that automates the detection and identification of corals in images or videos, thus contributing to the preservation and protection of these critical marine ecosystems.")
    
    
    # with st.expander("DEPLOIMENT DU PROJET", expanded=True):
    selected = option_menu(
        menu_title=None,
        options=["Détection à partir d'images", "Détection à partir de Vidéos"],
        icons=["image-fill", "camera-video-fill"],
        orientation="horizontal",
    )
    if selected == "Detection des Images":
        image = st.file_uploader("Upload image")
        if image:
            left, right = st.columns(2)
            with left:
                st.image(image,width=400)
            with right:
                conf_threshold = st.slider("Confidence Index", 0.0, 1.0, 0.25, 0.01)
                iou_threshold = st.slider("IOU Index", 0.0, 1.0, 0.45, 0.01)
                option = st.radio('Voulez-vous enregistrer les images des panneaux de signalisation ?', ['Oui', 'Non'])
                st.info("Le chemin d'enregistrement est:   yolov5/media/ts")
            with st.spinner("Veuillez patienter quelques instants..."):
                if st.button('Submit'):
                    # result = st.empty()
                    detectImage(image, imgsz, stride, pt, model, conf_threshold, iou_threshold, option)
        # else:
        #     st.markdown(f'<h1 style="color:#ff0000;font-size:24px;">{"First, you need to upload an image"}</h1>', unsafe_allow_html=True)
                    
    elif selected == "Detection des Videos":
        video = st.file_uploader("Sélectionnez une vidéo", type=['mp4', 'mov'])
        with st.spinner("Veuillez patienter quelques instants..."):
            if video:    
                left, right = st.columns(2)
                with right:
                    conf_threshold = st.slider("Confidence Index", 0.0, 1.0, 0.25, 0.01)
                    iou_threshold = st.slider("IOU Index", 0.0, 1.0, 0.45, 0.01)
                    option = st.radio('Voulez-vous enregistrer les images des panneaux de signalisation ?', ['Oui', 'Non'])
                    st.info("Le chemin d'enregistrement est: yolov5/media/frame")
                with left:
                    st.video(video)
                if st.button('Submit'):
                    col = st.columns(3)
                    with col[1]:
                        result = st.empty()
                        detectVideo(video, imgsz, stride, pt, model, result, conf_threshold, iou_threshold, option)                         
                        
    if st.button("Afficher les photo du panneaux de signalisation"):
        with st.expander("TRAFFIC SIGNS"):
            url = "./media/ts"
            columnNumber = 8
            col = st.columns(columnNumber)
            folder = os.listdir(url)
            for i, img in enumerate(os.listdir(url)):
                file_path = os.path.join(url, img)
                col[i%columnNumber].image(file_path, width=80)
                        

    with st.expander("RESULTATS", expanded=False): 
            
        a, b = st.columns(2)
        with a:
            st.image("./runs/train/yolov5s_results3/confusion_matrix.png",caption="Matrice de Confusion")
            st.image("./runs/train/yolov5s_results3/R_curve.png")
            st.write("Ce graphe montre la variation du rappel (Recall) en fonction de la confiance. Il permet de quantifier la capacité du modèle à trouver tous les exemples positifs présents dans les données.")
            st.info("À mesure que la confiance augmente, moins de prédictions sont faites, ce qui entraîne une précision plus élevée mais éventuellement un rappel (Recall) plus faible. ")
            # st.divider()
            # st.image("./runs/train/yolov5s_results3/PR_curve.png")
        with b:
            st.image("./runs/train/yolov5s_results3/P_curve.png")
            st.write("Ce graphe montre la variation du precision en fonction de la confaiance pour évaluer les performances du modèle à différents seuils de confiance.")
            st.divider()
            st.image("./runs/train/yolov5s_results3/F1_curve.png")
            st.write("Ce graphe montre la variation du F1-score en fonction de la confiance pour évaluer les performances d'un modèle de détection d'objets à différents seuils de confiance.")
            st.info("Ce graphe indique que le modèle de détection d'objets est relativement performant dans la tâche à laquelle il est confronté [presque 80%].")
        
        
        
if __name__ == '__main__':
    main()
