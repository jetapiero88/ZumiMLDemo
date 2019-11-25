import numpy as np
import time
import cv2
import socket
import os


class ClassiffierUtilities():
    class Predictions():
        def __init__(self, outer_instance=None):
            self.outer_instance = outer_instance

        def set_value(self, predictions):
            self.predicts = predictions
            self.pred_index = np.argmax(self.predicts[0])
            self.pred_label = self.outer_instance.labels.get(self.pred_index)
            self.pred_confidence = self.predicts[0][self.pred_index]

        def get_readable_predictions(self):
            string = ""
            for index in self.outer_instance.labels:
                string = string + self.outer_instance.labels.get(index) + " {:.2%}".format(
                    self.predicts[0][index]) + "  "
            return string

    def __init__(self, demo_name, folder, labels):
        self.demo_name = demo_name
        self.folder = folder
        self.model = None
        self.labels = labels
        self.commands = {}
        self.predictions = self.Predictions(self)
        self.current_image = None

    def load_model(self):
        import json
        from tensorflow.keras.models import model_from_json
        from tensorflow.keras.models import load_model

        with open(self.folder + "/" + self.demo_name + ".json") as f:
            json_string = json.load(f)

        self.model = model_from_json(json_string)
        self.model.load_weights(self.folder + "/" + self.demo_name + "weights.h5")

    def predict(self, image):
        self.current_image = image
        # if self.color :
        image = image[None, :, :, :]
        # else :
        #    print("gray")
        #    image = Image.fromarray(image)
        #    image = np.expand_dims(image, axis = 0)
        self.predictions.set_value(self.model.predict(image))

        print("found " + self.predictions.pred_label + " with confidence: " + str(self.predictions.pred_confidence))

        return self.predictions

    def save_image(self, image, data_type, label, extension):
        file_name = self.folder + '/' + self.demo_name + '/' + data_type + '/' + label + '/' + socket.gethostname()  + str(
            time.time()) + extension
        cv2.imwrite(file_name, image)

    def take_images(self, camera):

        extension ='.png'
        camera.start_camera()
        try:
            while True:
                data_type = input("Which data will you collect? (Type train / test) \n: ")
                if data_type == "train" or data_type == "test":
                    break

            print("Press label number (starting from zero) and Enter for take images.\n")
            while True:
                command = input("")
                commandint = int(command)

                if commandint < len(self.labels):
                    
                    image = camera.capture()
                    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                    #self.current_image = image
                    self.save_image(image, data_type, self.labels[commandint], extension)
                    print(self.labels[commandint])
                else:
                    print("# bad label index: " + commandint)
                if command == 'q':
                    break
                   
        finally:
            print("Error")
            camera.close()

    def create_folders(self):
        try:
            dic = self.demo_name
            path = self.folder + '/' + dic
            
            os.mkdir(path)
            os.mkdir(path + '/train')
            os.mkdir(path + '/test')
            for label in self.labels:
                os.mkdir(path + '/train/' + label)
                os.mkdir(path + '/test/' + label)

        except OSError as e:
            if e.errno != errno.EEXIST:
                print("Failed to create directory!!!!!")
                raise
