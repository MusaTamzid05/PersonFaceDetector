import tkinter as tk
import cv2
from PIL import ImageTk
from PIL import Image
import os
import bina_ascii_utils
import numpy as np


class InteractiveRecognizer:

    def __init__(self ,  recognizer_path , cascade_path , scale_factor = 1.3 , min_neighbors = 4 , min_size_proportional = (0.25 , 0.25) , flags = cv2.CASCADE_SCALE_IMAGE , rect_color = (0 , 255 , 0) , video_source = 0 ):


        self._mirror = True
        self._running = True

        self._recognizer_path = recognizer_path
        self._cascade_path = cascade_path
        self._curr_detected_object = None
        self._init_detector()
        self._init_recognizer()

        self._scale_factor = scale_factor
        self._min_neighbors = min_neighbors

        self._flags = flags
        self._rect_color = rect_color

        self._init_video(video_source)
        self._init_size()

        self._min_size = (int(self._width * min_size_proportional[0]),
                        int(self._height * min_size_proportional[1]))

        self.current_message = self.get_instructions()


    def _init_video(self , video_source):


        self.capture= cv2.VideoCapture(video_source)

        if not self.capture.isOpened():
            raise ValueError("Unable to open video source", video_source)


    def _init_size(self):

        self._width = self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        self._height= self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)


    def _init_recognizer(self):

        self._recognizer = cv2.face.LBPHFaceRecognizer_create()

        if os.path.isfile(self._recognizer_path):

            self._recognizer.read(self._recognizer_path)
            self._recognizer_trained = True

        else:
            self._recognizer_trained = False

    def _init_detector(self):


        self._detector = cv2.CascadeClassifier(self._cascade_path)

        if self._detector.empty():
            print("cascade path {} is invalid.".format(self._cascade_path))


    def get_frame(self):

        if self.capture.isOpened():
            frame_loaded, frame = self.capture.read()

            if frame_loaded:
                image , detected_object = self.detect_and_recognized(frame)
                return (frame_loaded, image , detected_object)

        return (False , None)

    def detect_and_recognized(self , image):


        gray_image = cv2.cvtColor(image , cv2.COLOR_BGR2GRAY)
        equalized_gray_image = cv2.equalizeHist(gray_image)

        rects = self._detector.detectMultiScale(equalized_gray_image , scaleFactor = self._scale_factor,
                                               minNeighbors = self._min_neighbors , minSize = self._min_size)

        for x , y , w , h in rects:

            cv2.rectangle(image , (x , y) , (x + w , y + h) , self._rect_color , 1)


        if  len(rects) > 0:
            self.update_model(rects , gray_image)

        else:

            self._curr_detected_object = None

            if self._recognizer_trained:
                self.current_message = "\n"
            else:
                self.current_message = self.get_instructions()


        return image , len(rects)

    def update_model(self , rects , gray_image):

        x , y , w , h = rects[0]
        self._curr_detected_object = cv2.equalizeHist(gray_image[y : y + h , x : x + w])

        if self._recognizer_trained:
            self._show_current_object_info()

        else:
            self.current_message = self.get_instructions()



    def get_current_message(self):

        return self.current_message


    def _show_current_object_info(self):

        try:

            label_as_int , distance = self._recognizer.predict(self._curr_detected_object)
            label_as_string = bina_ascii_utils.int_to_four_chars(label_as_int)
            self.current_message = "It looks most like {}.\nThis distance is {}".format(label_as_string , distance)

        except cv2.error:

            self.current_message = self.get_instructions()
            print("Recreating model duo to error.")
            self.clear_model()


    def get_instructions(self):

        return "If user detects a face,click the add model button."


    def clear_model(self):

        self._recognizer_trained = False

        if os.path.isfile(self._recognizer_path):
            os.remove(self._recognizer_path)


        self._recognizer = cv2.face.LBPHFaceRecognizer_create()


    def is_recognizer_trained(self):

        return self._recognizer_trained

    def update_recognizer(self , face_label):

        label_as_int = bina_ascii_utils.four_char_to_int(face_label)
        labels = np.array([ label_as_int ])
        src =[self._curr_detected_object]

        if self._recognizer_trained:

            self._recognizer.update(src , labels)

        else:

            self._recognizer.train(src , labels)
            self._recognizer_trained = True


    def __del__(self):

        if self._recognizer_trained:
            self._save_recognizer()

        if self.capture.isOpened():
            self.capture.release()


    def _save_recognizer(self):

        model_dir = os.path.dirname(self._recognizer_path)

        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)


        self._recognizer.save(self._recognizer_path)
        print("Recognizer saved.")


class App:

    def __init__(self , window , window_title , cascade_path  , recognizer_path , video_source = 0):

        self.window = window
        self.window.title(window_title)
        self.recognizer = InteractiveRecognizer(cascade_path = cascade_path , recognizer_path = recognizer_path)

        self.delay = 15
        self.canvas = tk.Canvas(window , width = self.recognizer._width , height = self.recognizer._height)
        self.canvas.pack()


        self._init_buttoms()
        self._init_label()
        self._init_text_field()


        self.update()

        self.window.mainloop()



    def update(self):


        frame_loaded , frame , detected_object = self.recognizer.get_frame()

        if frame_loaded:

            self.photo = ImageTk.PhotoImage ( image = Image.fromarray(frame))
            self.canvas.create_image(0 , 0 , image = self.photo , anchor = tk.NW)

        message = self.recognizer.get_current_message()

        self.label["text"] =  message
        self._enable_disable_button(detected_object , self.recognizer.is_recognizer_trained())
        self.window.after(self.delay , self.update)




    def _init_label(self):

        self.label = tk.Label(self.window , text = "")
        self.label.pack()


    def _init_buttoms(self):



        #self.add_label_button.config(state = "activate" )
        #self.add_label_button.config(state = "disable" )


        self.update_label_button = tk.Button(self.window , text = "Update to model" , state = tk.NORMAL, width = 20, command = self.update_model_action)
        self.update_label_button.pack(anchor = tk.W)



        self.clear_label_button= tk.Button(self.window , text = "Clear model" , state = tk.NORMAL,  width = 20 , command = self.clear_model_action)
        self.clear_label_button.pack(anchor = tk.W)


    def _init_text_field(self):

        self.entry = tk.Entry(self.window)
        self.entry.pack()


    def clear_model_action(self):


        self.recognizer.clear_model()


    def update_model_action(self):

        face_label = self.entry.get()
        self.recognizer.update_recognizer(face_label)


    def _enable_disable_button(self , detected_object , recognizer_trained):

        if  recognizer_trained:
            self.clear_label_button.config(state = "active")

        else:
            self.clear_label_button.config(state = "disabled")


        text =  self.entry.get()[0:4]

        if detected_object == 0 or len(text) == 0:

            self.update_label_button.config(state = "disabled")
            return

        self.update_label_button.config(state = "active")






def main():

    recognizer_path = './recognizers/python_human_face_recognizer.xml'
    cascade_path = "./cascades/haarcascade_frontalface_alt.xml"

    try:

        App(tk.Tk() , window_title =  "Tkinter in Opencv" , cascade_path = cascade_path , recognizer_path = recognizer_path)

    except KeyboardInterrupt as e:
        print(e)

if __name__ == "__main__":

    main()
