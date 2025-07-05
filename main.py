import time
import os

os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
from kivy.clock import Clock
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.camera import Camera
from kivy.core.window import Window
from kivy.uix.popup import Popup
from kivy.uix.button import Button
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.lang import Builder
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import cv2
import json
from kivymd.app import MDApp
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.floatlayout import MDFloatLayout
from kivymd.uix.button import MDIconButton
from kivymd.uix.label import MDLabel
from kivy.utils import get_color_from_hex
from kivymd.uix.toolbar import MDTopAppBar
from kivymd.uix.toolbar import MDBottomAppBar
from kivymd.uix.progressbar import MDProgressBar
from kivymd.uix.menu import MDDropdownMenu
from kivymd.uix.list import OneLineIconListItem
from kivy.metrics import dp
from kivy.lang import Builder
from kivy.metrics import dp

from kivymd.app import MDApp
from kivymd.uix.menu import MDDropdownMenu
from kivymd.uix.snackbar import Snackbar
import tensorflow as tf
import pyttsx3
import threading

Window.size = (360, 620)
KV = '''
ScreenManager:
    MainScreen:
    ReadMoreScreen:

<MainScreen>:
    name: 'main'
    MDBoxLayout:
        id: main_screen
        orientation: 'vertical'
        MDTopAppBar:
            left_action_items: [["sprout", lambda x: None]]
            right_action_items: [["dots-vertical", lambda x: app.callback(x)]]
            title: 'Lettuce Care'
            elevation: 2
            pos_hint: {"center_x": 0.5, "center_y": 0.5}

        MDFloatLayout:
            Image:
                id: logo_image
                source: 'logo.png'
                size_hint: None, None
                size: dp(300), dp(300)
                allow_stretch: True
                keep_ratio: True
                pos_hint: {"center_x": 0.5, "center_y": 0.7}

            MDLabel:
                id: result_label
                text: "Lettuce Care is an AI-Based Disease Detector using Model trained through Convolutional Neural Network"
                halign: "center"
                pos_hint: {"center_x": 0.5, "center_y": 0.44}
                padding: (10, 10, 10, 10)

            MDLabel:
                id: desc_label
                text: ""
                halign: "center"
                pos_hint: {"center_x": 0.5, "center_y": 0.22}
                padding: (10, 10, 10, 10)

        MDBottomAppBar:
            MDTopAppBar:
                left_action_items: [["file-image", lambda x: app.show_file_chooser()]]
                right_action_items: [["restore", lambda x: app.restart()]]
                icon: "camera"
                icon_size:"100sp"
                padding: (10, 10, 10, 10)
                type: "bottom"
                on_action_button: app.capture()

<ReadMoreScreen>:
    name: 'read_more'
    MDBoxLayout:
        orientation: 'vertical'
        MDTopAppBar:
            id: top_bar
            left_action_items: [["arrow-left", lambda x: app.go_back()]]
            title: 'Read More'
            elevation: 2
            pos_hint: {"center_x": 0.5, "center_y": 0.5}
            size_hint_y: None
            height: dp(56)  # Adjust the height as needed
        MDFloatLayout:
            Image:
                id: logo
                source: 'logo.png'
                size_hint: None, None
                size: dp(300), dp(300)
                allow_stretch: True
                keep_ratio: True
                pos_hint: {"center_x": 0.5, "center_y": 0.45}
                padding: (10, 10, 10, 10)

        ScrollView:
            size_hint: (1, 1)  # Take up the remaining space
            do_scroll_x: False
            do_scroll_y: True
            MDLabel:
                id: read_more_label
                text: ""
                halign: "justify"
                text_size: (self.width - 20, None)  # Width of label minus padding for justification
                size_hint_y: None
                height: self.texture_size[1]  # Height based on the text content
                padding: (10, 10, 10, 10)

'''


class MainScreen(Screen):
    def __init__(self, **kwargs):
        super(MainScreen, self).__init__(**kwargs)
        self.camera = None
        self.img = None
        self.label = None
        self.disease_button = None
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.class_names = None
        self.popup = None
        self.disease_name = "Lettuce Care"
        self.img_path = "logo.png"
        # Load the TFLite model and class names
        self.load_model_and_classes()

    def getDiseaseName(self):
        return self.disease_name

    def getImgPath(self):
        return self.img_path

    def restart(self):
        self.ids.logo_image.source = "logo.png"
        self.ids.result_label.text = "Lettuce Care is an AI-Based Disease Detector using a model trained through Convolutional Neural Network"
        self.ids.desc_label.text = ""
        self.img_path = "logo.png"
        self.disease_name = "lettuce Care"

    def load_model_and_classes(self):
        try:
            # Load the TFLite model
            self.interpreter = tf.lite.Interpreter(
                model_path='model_quantized.tflite')
            self.interpreter.allocate_tensors()

            # Get input and output details
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()

            # Load the class names
            with open(r'D:\Old Files\Downloads\class_indices (1).json') as f:
                self.class_names = json.load(f)
        except Exception as e:
            print(f"Error loading model or classes: {e}")

    def capture(self, *args):
        content = BoxLayout(orientation='vertical', padding=10, spacing=10)
        self.camera = Camera(play=True, resolution=(640, 480))
        content.add_widget(self.camera)

        save_button = MDIconButton(
            icon="camera",
            size_hint=(1, .1),
            on_press=self.capture_image,
            pos_hint={"center_x": 0.5}
        )
        content.add_widget(save_button)

        self.popup = Popup(title="Camera", content=content, size_hint=(0.9, 0.9))
        self.popup.open()

    def capture_image(self, instance):
        cam = None
        try:
            # Open the default camera with better settings
            cam_port = 0
            cam = cv2.VideoCapture(cam_port, cv2.CAP_DSHOW)

            # Set the resolution of the camera
            cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

            # Allow the camera to warm up for a short period
            time.sleep(1)

            # Capture a single frame
            result, image = cam.read()

            if not result or image is None:
                print("Error: Unable to capture image from the camera.")
                cam.release()
                return

            img_path = "captured_image.png"
            self.img_path = img_path

            # Save the image with high quality
            cv2.imwrite(img_path, image)

            # Update image source in UI
            self.ids.logo_image.source = img_path
            self.ids.logo_image.reload()

            # Release the camera
            cam.release()

            # Classify the flower using the captured image
            self.detect_disease(img_path)
            self.popup.dismiss()
        except Exception as e:
            print(f"Exception occurred: {e}")
            if cam.isOpened():
                cam.release()

    def show_file_chooser(self, *args):
        layout = BoxLayout(orientation='vertical')

        # Create FileChooser
        filechooser = FileChooserIconView()
        filechooser.bind(on_submit=self.selected)

        # Create Buttons
        button_layout = BoxLayout(size_hint_y=None, height=50)
        back_button = Button(text='Back')
        ok_button = Button(text='OK')

        # Bind button actions
        back_button.bind(on_release=self.on_back)
        ok_button.bind(on_release=lambda btn: self.selected(filechooser, filechooser.selection))

        # Add widgets to layout
        button_layout.add_widget(back_button)
        button_layout.add_widget(ok_button)
        layout.add_widget(filechooser)
        layout.add_widget(button_layout)

        # Create and open Popup
        self.popup = Popup(title="Select Image", content=layout, size_hint=(0.9, 0.9))
        self.popup.open()

    def on_back(self, instance):
        # Handle back button action
        self.popup.dismiss()

    def selected(self, filechooser, selection, *args):
        if selection:
            self.ids.logo_image.source = selection[0]
            self.img_path = selection[0]
            self.ids.logo_image.reload()
            self.detect_disease(selection[0])
            self.popup.dismiss()

    def detect_disease(self, img_path):
        try:
            # Load and preprocess image
            img = image.load_img(img_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array.astype('float32') / 255.0

            # Set the tensor to the interpreter
            self.interpreter.set_tensor(self.input_details[0]['index'], img_array)

            # Run the inference
            self.interpreter.invoke()

            # Get the results
            predictions = self.interpreter.get_tensor(self.output_details[0]['index'])
            predicted_class = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class]
            disease_name = self.class_names[str(predicted_class)]  # Ensure correct indexing
            self.disease_name = disease_name

            # Define a confidence threshold
            confidence_threshold = 0.7  # Adjust this value as needed

            # Check if the confidence is below the threshold
            if confidence < confidence_threshold:
                disease_name = 'Non-lettuce'

            # Display results
            if disease_name == 'Healthy':
                self.ids.result_label.text = f'Lettuce is: {disease_name}'
                self.ids.desc_label.text = 'A healthy lettuce plant appears vibrant and green, with no visible signs of disease or stress. The leaves are firm and free from spots, lesions, or discoloration. Healthy lettuce shows normal growth and development, indicating that it is free from common pests and diseases.'
            elif disease_name == 'Downy_mildew_on_lettuce':
                self.ids.result_label.text = f'Detected disease: {disease_name}'
                self.ids.desc_label.text = 'Downy mildew is a fungal disease that affects lettuce and other leafy greens. Symptoms include yellowing of the leaves, often starting at the base of the plant, followed by the appearance of white, powdery fungal growth on the underside of the leaves. The disease can lead to significant yield loss if not managed promptly.'
            elif disease_name == 'Septoria_blight_on_lettuce':
                self.ids.result_label.text = f'Detected disease: {disease_name}'
                self.ids.desc_label.text = 'Septoria blight is a fungal infection that causes dark, round spots with grayish centers on lettuce leaves. These spots often have a yellow halo and can merge to form larger areas of decay. Over time, the affected leaves may die and drop off, reducing the plants overall productivity. Proper management and treatment are essential to control the spread of this disease.'
            else:
                self.ids.result_label.text = 'Unknown Object Detected'
                self.ids.desc_label.text = 'Model could not identify the object. Please upload or capture a clear image of a lettuce.'

        except Exception as e:
            self.ids.result_label.text = f"Error detecting disease: {str(e)}"

    def show_disease_info(self, instance):
        if instance.text:
            app = MDApp.get_running_app()
            app.root.current = 'disease_info'
            disease_info_screen = app.root.get_screen('disease_info').children[0]
            disease_info_screen.update_info(instance.text)


class ReadMoreScreen(Screen):
    def __init__(self, **kwargs):
        super(ReadMoreScreen, self).__init__(**kwargs)
        self.engine = pyttsx3.init()  # Initialize the engine once
        self.speech_thread = None  # To keep track of the speech thread
        self.stop_flag = threading.Event()  # Use an event to signal stopping

    def update_text(self, text):
        self.ids.read_more_label.text = text

    def update_title(self, title):
        self.ids.top_bar.title = title

    def update_image(self, img_path):
        self.ids.logo.source = img_path

    def reset(self):
        self.ids.logo.source = "logo.png"

    def read_aloud(self, text):
        if text:
            # Reset the stop flag
            self.stop_flag.clear()
            # Run the speech in a separate thread
            self.speech_thread = threading.Thread(target=self._read_aloud_thread, args=(text,))
            self.speech_thread.start()

    def _read_aloud_thread(self, text):
        self.engine.stop()  # Stop any ongoing speech
        self.engine.say(text)

        # Start the engine and wait for it to finish speaking or for the stop flag
        while not self.stop_flag.is_set():
            try:
                self.engine.runAndWait()
                break  # Speech completed
            except Exception as e:
                print(f"Error: {e}")
                break  # Exit if there's an error

    def stop_reading(self):
        # Set the stop flag to signal the thread
        self.stop_flag.set()
        # Stop the engine in a separate thread to avoid blocking
        if self.speech_thread and self.speech_thread.is_alive():
            threading.Thread(target=self._stop_thread).start()

    def _stop_thread(self):
        if self.engine:
            self.engine.stop()  # Stop any ongoing speech


class LettuceCareApp(MDApp):
    icon_color = get_color_from_hex("#74C365")

    def build(self):
        self.theme_cls.theme_style = "Light"
        self.theme_cls.primary_palette = "Green"
        self.theme_cls.material_style = "M2"
        menu_items = [
            {
                "text": "Read More",
                "on_release": lambda x="Read More": self.menu_callback(x),
            },
            {
                "text": "About",
                "on_release": lambda x="About": self.menu_callback(x),
            },
            {
                "text": "Version",
                "on_release": lambda x="Version": self.menu_callback(x),
            }
        ]
        self.menu = MDDropdownMenu(items=menu_items)
        screen = Builder.load_string(KV)
        return screen

    def show_file_chooser(self):
        main_screen = self.root.get_screen('main')
        main_screen.show_file_chooser()

    def capture(self):
        main_screen = self.root.get_screen('main')
        main_screen.capture()

    def go_back(self):
        # read_screen = self.root.get_screen('read_more')
        # read_screen.stop_reading()
        self.root.current = 'main'

    def restart(self):
        main_screen = self.root.get_screen('main')
        main_screen.restart()
        read_screen = self.root.get_screen('read_more')
        read_screen.reset()

    def create_dropdown_menu(self):
        main_screen = self.root.get_screen('main')
        main_screen.create_dropdown_menu()

    def callback(self, button):
        self.menu.caller = button
        self.menu.open()

    def menu_callback(self, text_item):
        main_screen = self.root.get_screen('main')
        disease_name = main_screen.getDiseaseName()
        self.menu.dismiss()
        self.root.current = 'read_more'

        # Update the text based on the menu selection
        read_more_screen = self.root.get_screen('read_more')

        if text_item == "Read More":
            read_more_screen.update_title(disease_name)
            # Determine the filename based on the disease name
            filename = None
            main_screen = self.root.get_screen('main')
            image_path = main_screen.getImgPath()
            read_screen = self.root.get_screen('read_more')
            if disease_name == "Healthy":
                filename = "Healthy.txt"
                read_screen.update_image(image_path)
            elif disease_name == "Septoria_blight_on_lettuce":
                filename = "Septoria.txt"
                read_screen.update_image(image_path)
            elif disease_name == "Downy_mildew_on_lettuce":
                filename = "DownyMildew.txt"
                read_screen.update_image(image_path)
            else:
                filename = "LettuceCare.txt"
                read_screen.update_image(image_path)

            # Read the content from the file and update the text
            if filename:
                try:
                    with open(filename, 'r') as file:
                        content = file.read()
                        read_more_screen.update_text(content)
                    # threading.Thread(target=lambda: read_screen.read_aloud(content)).start()
                except FileNotFoundError:
                    read_more_screen.update_text(f"File {filename} not found.")
                except Exception as e:
                    read_more_screen.update_text(f"An error occurred: {e}")
            else:
                read_more_screen.update_text("No information available for this disease.")

        elif text_item == "About":
            read_more_screen.update_text(
                "\nDescription:\n\nThis application is developed as part of a thesis research project for BSCS 4th-year students of ASU-Ibajay Campus. It aims to leverage artificial intelligence (AI) technology to assist in the detection and diagnosis of common diseases affecting lettuce plants. By employing machine learning algorithms to analyze images of lettuce leaves, the app provides accurate identification of disease symptoms and offers actionable recommendations for disease management. The research focuses on enhancing agricultural productivity and sustainability through the integration of AI-driven diagnostic tools, reducing reliance on traditional methods and improving early intervention strategies.\n\n\nDeveloper: BSCS Students | ASU-Ibajay")
            read_more_screen.update_title("About : Lettuce Care")

        elif text_item == "Version":
            read_more_screen.update_text(
                "\nVersion: 1.0.0\nRelease Date: 2024-08-18\nDescription: Initial release with core features.\nDeveloped by: ASU-Ibajay, BSCS Students")
            read_more_screen.update_title("Version")


if __name__ == '__main__':
    import logging

    logging.basicConfig(level=logging.DEBUG)
    LettuceCareApp().run()
