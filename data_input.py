import csv
from PIL import Image
import random

image_source_path = '../../LEAF_DATA/IMAGES/images_square_scaled_32_32/'
csv_path = '../../LEAF_DATA/NUMERICAL/'
submission_file_path = 'submission_file/'

csv_train_source = csv_path + "train.csv"
csv_test_source = csv_path + "test.csv"
csv_sample_submission_source = csv_path + "sample_submission.csv"
csv_submission_file = submission_file_path + "submission_file_augmented.csv"

n_images = 1584

NUM_CLASSES = 99

class LeafData(object):

    train_images = []
    train_labels = []
    train_statistics = []
    test_images = []
    test_labels = []
    test_statistics = []

    train_images_labels = []

    randomization_list = []

    #following list will be used to convert from images position in test_images
    #to their file id's. Will be used to create output file for submission:
    test_position_to_id = []

    train_position_to_id = []


    #dictionary that will associate species target label with species name
    label_to_species = dict()

    def scale_pixel_value(self, initial, divisor):

        scaled = initial

        if divisor != 0.0:
            scaled = (2.0 * initial / divisor) - 1.0

        return scaled

    def import_data(self):

        # dictionary that will associate species names with their target classification label:
        species_to_label = dict()

        # dictionary that will associate species target label with species name
        self.label_to_species = dict()

        # dictionary that will associate species id number with species name:
        id_to_species_name = dict()

        with open(csv_sample_submission_source, 'rb') as f_sample:
            reader = csv.reader(f_sample)
            i = 0
            for row in reader:
                for (index_label, name) in enumerate(row):

                    #columns with species name begins at 1, but index
                    #for labels starts at zero, so subtract 1
                    index_label -= 1

                    if(index_label >= 0):

                        if(species_to_label.has_key(name)) != True:
                            species_to_label[name] = index_label
                            self.label_to_species[index_label] = name

                #only need to read in first line from sample_submission file, so break
                break

            f_sample.close()

        with open(csv_train_source, 'rb') as f:
            reader = csv.reader(f)
            row = reader.next()

            for row in reader:
                id_to_species_name[row[0]] = row[1]
                self.train_statistics.append(row[2:])

            #dictionary that will be used to count number_of_examples:
            counter = species_to_label.copy()

            #number of examples of each species type to be included in
            #training file:

            #There are only 16 examples of each species, so if you
            #set number_of_examples to 16 or more, all training
            #images will be used.

            #have now added rotated images, so total will be 4 * 16
            number_of_examples = 4 * 16

            keys = counter.keys()
            for k in keys:
                counter[k] = number_of_examples

            f.seek(0)

            n_image_count = 0

            train_image_data = []
            train_labels = []

            for m in range(1, n_images + 1):

                #conditional that stops loop when you have correct number_of_examples:
                if len(counter) > 0:
                    #get species name associated with jth image
                    id = str(m)
                    if id in id_to_species_name:
                        species_name = id_to_species_name[id]

                        self.train_position_to_id.append(id)

                        if species_name in counter:

                            species_label = species_to_label[species_name]

                            source_file_name = image_source_path + id + '.bmp'


                        with Image.open(source_file_name) as im:

                            width, height = im.size

                            pixels = []
                            label = [0] * NUM_CLASSES

                            #using "one hot" target labels, set element associated with this species to 1:
                            label[species_label] = 1


                            for n in range(0, width):
                                for p in range(0, height):
                                    xy = (n,p)

                                    #values are stored in image file in integers range [0, 255]
                                    #but Tensorflow CNN is set up to accept float inputs in range of [0, 1]
                                    #so divide by 255
                                    div = 255.0

                                    scaled_pixel_value = self.scale_pixel_value(im.getpixel(xy), div)
                                    pixels.append(scaled_pixel_value)

                        self.randomization_list.append(n_image_count)

                        n_image_count += 1

                        train_image_data.append(pixels)
                        train_labels.append(label)

                        #decrement count for species name in counter
                        #if count <= 0, pop species out of counter
                        counter[species_name] -= 1
                        if counter[species_name] <= 0:
                            counter.pop(species_name)
                else:
                    break

            print "total number of images loaded into training array:", n_image_count
            print "size of train image data: ", len(train_image_data)

        #NOW READ IN TEST DATA:

        with open(csv_test_source, 'rb') as f:
            reader = csv.reader(f)
            row = reader.next()

            n_test_image_count = 0

            #TEST ARRAYS:
            test_image_data = []
            test_labels = []


            for row in reader:
                self.test_statistics.append(row[1:])
                id = row[0]
                self.test_position_to_id.append(id)

                source_file_name = image_source_path + id + ".bmp"

                with Image.open(source_file_name) as im:
                    n_test_image_count += 1

                    width, height = im.size

                    pixels = []

                    #don't have labels for training data, will be evaluated
                    #by submission to kaggle. Just use label of all 0's
                    label = [0] * NUM_CLASSES

                    for n in range(0, width):
                        for p in range(0, height):
                            xy = (n, p)

                            # values are stored in image file in integers range [0, 255]
                            # Tensorflow CNN is set up to accept float inputs in range of [0, 1]

                            div = 255.0

                            scaled_pixel_value = self.scale_pixel_value(im.getpixel(xy), div)

                            pixels.append(scaled_pixel_value)

                    test_image_data.append(pixels)
                    test_labels.append(label)

            f.close()

            print "total number of images loaded into testing array:", n_test_image_count

        self.train_images = train_image_data
        self.train_labels = train_labels
        self.test_images = test_image_data
        self.test_labels = test_labels

        self.train_images_labels = [train_image_data, train_labels]

        return 0

    def create_output_file(self, results, file_name):

        # open file first as w, to clear it

        sf = open(csv_submission_file, 'w')
        sf.close()

        sf = open(csv_submission_file, 'ab')

        output_line = "id"
        for i in range(NUM_CLASSES):
            output_line = output_line + "," + self.label_to_species[i]

        output_line += "\n"

        sf.write(output_line)


        for j in range(len(results)):
            output_line = self.test_position_to_id[j]

            rj = results[j]

            for k in range(len(rj)):
                val = str("%11.10f" % rj[k])
                output_line = output_line + "," + val

            output_line += "\n"
            sf.write(output_line)
            output_line = ""

        sf.close()
        return 0

    def get_next_batch(self, batch_size):

        if(batch_size > len(self.randomization_list)):
            batch_size = len(self.randomization_list)

        ids = random.sample(self.randomization_list, batch_size)

        x = []
        y = []
        statistics = []

        for i in range(0, batch_size):
            x.append(self.train_images[ids[i]])
            y.append(self.train_labels[ids[i]])
            statistics.append(self.train_statistics[ids[i]])

        return x, y, statistics