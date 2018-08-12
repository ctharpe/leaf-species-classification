#Implements K-Nearest Neighbor for leaf project

import csv
import math
import time

csv_path = '../../LEAF_DATA/NUMERICAL/'
submission_file_path = 'submission_file/'

csv_train_source = csv_path + "train_std_scaler_OF_min_max.csv"
csv_test_source = csv_path + "test_std_scaler_OF_min_max.csv"
csv_sample_submission_source = csv_path + "sample_submission.csv"
csv_submission_file = submission_file_path + "KNN_submission_file.csv"

train_data = []
test_data = []

MAXIMUM_DISTANCE = 0.0

submission_file_header = ""

name_to_column = dict()
id_to_species_name = dict()

#read in train data

def read_train_data():

    # dictionary that will associate species id number with species name:
    global id_to_species_name

    with open(csv_train_source, 'rb') as f:
        reader = csv.reader(f)
        row = reader.next()

        for row in reader:
            id_to_species_name[row[0]] = row[1]
            train_data.append(row)

        f.close()

    return

def read_test_data():

    with open(csv_test_source, 'rb') as f:
        reader = csv.reader(f)
        row = reader.next()

        for row in reader:
            test_data.append(row)

        f.close()

def calculate_distance(list1, list2):

    if len(list1) != len(list2):
        print "ERROR: list lengths do not match."
        return -1

    distance = 0.0

    for i in range(len(list1)):
        x = float(list1[i])
        y = float(list2[i])
        distance += ((x - y) * (x - y))

    distance = math.sqrt(distance)

    global MAXIMUM_DISTANCE

    if distance > MAXIMUM_DISTANCE:
        MAXIMUM_DISTANCE = distance


    return distance

def loop_through_train():
    for i in range(len(train_data)):
        r = train_data[i]
        print r

def sort_distance_list(d_list):
    # sorts a list with format [[test_id, distance] , [test_id_2, distance_2]] ...
    #sorts from low to high based on distance

    d_list.sort(key=lambda x: float(x[1]))

def calculate_distances_for_all_test_data(k, result_list):

    i = 0

    for td in test_data:

        r_test = td
        i = i + 1

        # remove first column. First column is id number
        test_id = r_test[0]

        r_test = r_test[1:]

        distance_list = []

        for tr_d in train_data:
            r = tr_d

            #remove first two columns. First column is id number. Second column is species name.
            train_id = r[0]
            r = r[2:]
            distance = calculate_distance(r, r_test)

            distance_list.append([train_id, distance])

            sort_distance_list(distance_list)

            if(len(distance_list) > k):
                distance_list = distance_list[0 : k]

        temp_list = [test_id]
        temp_list.append(distance_list)

        result_list.append(temp_list)

def convert_item_to_probability(item):
    #subtract each distance in item from MAXIMUM_DISTANCE
    #convert to percentages
    for i in item[1]:
        i[1] = MAXIMUM_DISTANCE - i[1]

    sum = 0.0
    for i in item[1]:
        j = i[1]
        sum = sum + j

    if sum != 0:
        for i in item[1]:
            i[1] = i[1] / sum
    else:
        for i in item[1]:
            i[1] = 1.0


def convert_result_list_entries_to_probabilities(result_list):

    for item in result_list:
        convert_item_to_probability(item)

def read_in_submission_headers():

    global submission_file_header
    submission_file_header = ""

    with open(csv_sample_submission_source, 'rb') as f_sample:
        reader = csv.reader(f_sample)

        for row in reader:
            submission_file_header = row

            # only need to read in first line from sample_submission file, so break
            break

        f_sample.close()

def setup_name_to_column_dictionary():
    global submission_file_header
    global name_to_column
    i = 0
    for j in submission_file_header:
        name_to_column[j] = i
        i = i + 1

def convert_to_pure_one_hot(probability_list):
    # sets highest probability in probability_list = 1
    # all others = 0 (so "pure" one hot - only one output = 1
    max_entry_index = 0
    max_entry_value = 0.0

    for i in range(0, len(probability_list)):
        if probability_list[i] > max_entry_value:
            max_entry_value = probability_list[i]
            max_entry_index = i

    for i in range(0, len(probability_list)):
        if i == max_entry_index:
            probability_list[i] = 1.0
        else:
            probability_list[i] = 0.0

def create_output_file(result_list):

    sf = open(csv_submission_file, 'w')
    sf.close()

    sf = open(csv_submission_file, 'ab')

    # the '- 1' below is to account for the 'id' column in header:
    number_of_species = len(submission_file_header) - 1

    output_line = ""
    for i in submission_file_header:
        output_line = output_line + i + ','

    #remove the final ","
    output_line = output_line[0 : len(output_line) - 1]
    output_line += "\n"

    sf.write(output_line)
    global id_to_species_name
    global name_to_column

    total_multiple_guesses = 0

    #sum the probabilities for each species in header file, for each test id:
    for result in result_list:
        outputs = number_of_species * [0.0]
        probability_list = result[1]

        for entry in probability_list:
            index = name_to_column[id_to_species_name[entry[0]]]

            index = index - 1

            outputs[index] += entry[1]

        test_species_id_number = result[0]

        output_line = test_species_id_number

        entry_greater_than_zero_count = 0

        for probability in outputs:
            if probability > 0.0:
                entry_greater_than_zero_count += 1

            output_line = output_line + ',' + str(probability)

        if entry_greater_than_zero_count > 1:
            print "multiple guesses for test species id ", test_species_id_number
            total_multiple_guesses += 1

        output_line += "\n"
        sf.write(output_line)

    print "total multiple guesses = ", total_multiple_guesses
    sf.close()

def main():

    read_in_submission_headers()
    setup_name_to_column_dictionary()

    t1 = time.time()

    # k = number of nearest neighbors to use
    k = 5
    result_list = []

    read_train_data()
    read_test_data()

    calculate_distances_for_all_test_data(k, result_list)

    convert_result_list_entries_to_probabilities(result_list)

    t2 = time.time()

    print("KNN run time(in seconds) =", t2 - t1)

    create_output_file(result_list)

if __name__== "__main__": main()

