import tkinter as tk
import cv2
from PIL import ImageTk
from PIL import Image




class VideoCapture:

    def __init__(self , video_source = 0):

        self._init_video(video_source)
        self._init_size()


    def _init_video(self , video_source):


        self.vid = cv2.VideoCapture(video_source)

        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)


    def _init_size(self):

        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height= self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)


    def get_frame(self):

        if self.vid.isOpened():
            frame_loaded, frame = self.vid.read()

            if frame_loaded:
                return (frame_loaded, cv2.cvtColor(frame , cv2.COLOR_BGR2RGB))


        return (False , None)




    def __del__(self):

        if self.vid.isOpened():
            self.vid.release()


class App:

    def __init__(self , window , window_title , video_source = 0):

        self.window = window
        self.window.title(window_title)
        self.vid = VideoCapture(video_source)

        self.delay = 15
        self.canvas = tk.Canvas(window , width = self.vid.width , height = self.vid.height)
        self.canvas.pack()


        self._init_buttoms()
        self._init_label()


        self.update()

        self.window.mainloop()


    def update(self):

        frame_loaded , frame = self.vid.get_frame()

        if frame_loaded:

            self.photo = ImageTk.PhotoImage ( image = Image.fromarray(frame))
            self.canvas.create_image(0 , 0 , image = self.photo , anchor = tk.NW)


        self.window.after(self.delay , self.update)


    def _init_label(self):

        self.label = tk.Label(self.window , text = "")
        self.label.pack()
        self.show_introduction()


    def _init_buttoms(self):

        self.add_label_button = tk.Button(self.window , text = "Add to model" , state = tk.NORMAL,  width = 20 , command = self.add_model_action)
        self.add_label_button.pack(anchor = tk.W)


        #self.add_label_button.config(state = "activate" )
        #self.add_label_button.config(state = "disable" )


        self.update_label_button = tk.Button(self.window , text = "Update to model" , state = tk.NORMAL, width = 20, command = self.update_model_action)
        self.update_label_button.pack(anchor = tk.W)



    def add_model_action(self):
        pass


    def update_model_action(self):
        pass

    def show_introduction(self):

        self.label["text"] = "If user detects a face,click the add model button."




def main():

    App(tk.Tk() , "Tkinter in Opencv")

if __name__ == "__main__":

    main()
