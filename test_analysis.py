import sys
import os


from data_analysis_regression import *



if __name__ == "__main__":

    model_file_path = sys.argv[1]
    data_file_path = sys.argv[2]

    conduct_analysis(model_file_path, data_file_path)

    