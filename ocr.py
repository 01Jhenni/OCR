import cv2
import pytesseract
import numpy as np
from tkinter import Tk, filedialog, Button, Canvas, NW, Label, Scale, HORIZONTAL, Toplevel, Menu
from PIL import Image, ImageTk
import fitz  # PyMuPDF para suporte a PDF
import pandas as pd


pytesseract.pytesseract.tesseract_cmd = r'C:\\Users\\jhennifer.nascimento\\Downloads\\5.4.1 source code'

# Inicialização da janela Tkinter
root = Tk()
root.title("OCR NFS")


rects = []
start_x, start_y = None, None
selected_rois = []
img, tk_img, original_img = None, None, None
brightness_value, contrast_value = 1, 1

def load_image(file_path=None):
    """Carrega uma imagem do sistema de arquivos ou PDF."""
    global img, tk_img, canvas, original_img

    if not file_path:
        # Selecionar o arquivo de imagem ou PDF
        file_path = filedialog.askopenfilename(filetypes=[("Image/Document Files", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff;*.pdf")])
        if not file_path:
            return

    if file_path.endswith(".pdf"):
        # Carregar PDF e converter a primeira página para imagem
        doc = fitz.open(file_path)
        page = doc.load_page(0)
        pix = page.get_pixmap()
        img = cv2.cvtColor(np.array(Image.frombytes("RGB", [pix.width, pix.height], pix.samples)), cv2.COLOR_RGB2BGR)
    else:
        # Carregar imagem com OpenCV
        img = cv2.imread(file_path)

    original_img = img.copy()  # Manter uma cópia original para ajustes
    display_image(img)

def display_image(image):
    """Exibe a imagem no canvas."""
    global tk_img, canvas

    # Converter imagem para formato Tkinter
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tk_img = ImageTk.PhotoImage(image=Image.fromarray(img_rgb))

    # Mostrar imagem no canvas
    canvas.create_image(0, 0, anchor=NW, image=tk_img)
    canvas.config(scrollregion=canvas.bbox("all"))

def start_select(event):
    """Inicia a seleção do texto."""
    global start_x, start_y
    start_x, start_y = event.x, event.y

def select_roi(event):
    """Desenha o texto selecionado."""
    global rects, start_x, start_y, canvas

    end_x, end_y = event.x, event.y

    # Desenhar retângulo
    rect = canvas.create_rectangle(start_x, start_y, end_x, end_y, outline='red')
    rects.append(rect)

    # Armazenar a posição do texto
    selected_rois.append((start_x, start_y, end_x, end_y))

def adjust_image():
    """Ajusta brilho e contraste da imagem."""
    global img, brightness_value, contrast_value, original_img

    # Aplicar ajustes de brilho e contraste
    adjusted_img = cv2.convertScaleAbs(original_img, alpha=contrast_value, beta=brightness_value)
    display_image(adjusted_img)

def change_brightness(val):
    """Callback para alterar o brilho."""
    global brightness_value
    brightness_value = int(val)
    adjust_image()

def change_contrast(val):
    """Callback para alterar o contraste."""
    global contrast_value
    contrast_value = float(val)
    adjust_image()

def remove_last_roi():
    """Remove o último texto selecionado."""
    global rects, selected_rois, canvas

    if rects:
        canvas.delete(rects.pop())
        selected_rois.pop()

def extract_text():
    """Extrai texto das ROIs selecionadas."""
    global img, selected_rois

    # Resultados da extração de texto
    extracted_data = []

    # Pré-processamento da imagem para OCR
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    for idx, (x1, y1, x2, y2) in enumerate(selected_rois):
        # Extrair texto da imagem binária
        roi = binary[y1:y2, x1:x2]
        text = pytesseract.image_to_string(roi)
        extracted_data.append((f'Campo_{idx + 1}', text.strip()))

    # Mostrar resultados
    result_text = "\n".join([f"{campo}: {texto}" for campo, texto in extracted_data])
    label_result.config(text=result_text)

    return extracted_data

def export_results():
    """Exporta os resultados para um arquivo CSV."""
    extracted_data = extract_text()
    if not extracted_data:
        return

    # Criar DataFrame
    df = pd.DataFrame(extracted_data, columns=["Campo", "Texto"])

    # Salvar como CSV
    export_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
    if export_path:
        df.to_csv(export_path, index=False)

# Configurar Canvas para mostrar imagem
canvas = Canvas(root, width=800, height=600)
canvas.pack()

# Botões da interface
btn_load = Button(root, text="Carregar Imagem/PDF", command=load_image)
btn_load.pack()

btn_remove_roi = Button(root, text="Remover Ultima Seleção", command=remove_last_roi)
btn_remove_roi.pack()

btn_extract = Button(root, text="Extrair Texto", command=extract_text)
btn_extract.pack()

btn_export = Button(root, text="Exportar Resultados", command=export_results)
btn_export.pack()

# Sliders para ajuste de brilho e contraste
brightness_slider = Scale(root, from_=-100, to=100, orient=HORIZONTAL, label="Brilho", command=change_brightness)
brightness_slider.pack()
contrast_slider = Scale(root, from_=0.5, to=3.0, orient=HORIZONTAL, resolution=0.1, label="Contraste", command=change_contrast)
contrast_slider.pack()

label_result = Label(root, text="", justify="left")
label_result.pack()

# Eventos de mouse para seleção
canvas.bind("<ButtonPress-1>", start_select)
canvas.bind("<ButtonRelease-1>", select_roi)


root.mainloop()
