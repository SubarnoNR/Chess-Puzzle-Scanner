from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog
import Piece_Classifier
import webbrowser


def openfilename():
    filename = filedialog.askopenfilename(title ='Upload Puzzle')
    return filename

def callback(url):
    webbrowser.open_new(url)

def open_img():
    x = openfilename()
    fen = Piece_Classifier.get_prediction(x,show=False)
    img = Image.open('results.png')
    img = img.resize((450, 250), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    panel = Label(root, image = img)
    panel.image = img
    panel.place(x=8,y=45)
    get_link(root,fen)

def get_link(root,fen):
    link1 = Label(root, text="Link to Analysis Board", fg="blue", cursor="hand2")
    link1.pack()
    link1.bind("<Button-1>", lambda e: callback("https://lichess.org/analysis/{}".format(fen)))

    link1.place(x=8,y=320)

root = Tk()
root.title("Chess Puzzle Scanner")
root.geometry("480x350")
root.resizable(width = True, height = True)

btn = Button(root, text ='Upload image', command = open_img).place(x=205,y=15)
root.mainloop()