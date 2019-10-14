import PySimpleGUI as sg
import pandas as pd
from pystreamfs.pipeline import Pipeline
from pystreamfs.feature_selector import FeatureSelector
from pystreamfs.data_generator import DataGenerator
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score


def convert_input_true_false(conv_val):
    """
    Converts the user input into a string of either True or False

    :param conv_val: string: Userinput either true or false
    :return:
    """
    if conv_val.lower() in ['true', '1', 't', 'y', 'yes', 'yeah', 'yup']:
        ret_val = True
    else:
        ret_val = False
    return str(ret_val)


def create_dataset_input_path(dataset_name):
    """
    Returns to path to the saved CSV files by inputting the name of the file

    :param dataset_name: Takes string of the chosen dataset name
    :return: string: path to the CSV file
    """
    switcher = {

        'Credit': 'credit.csv',

        'Drift': 'drift.csv',

        'Har': 'har_binary.csv',

        'KDD': 'kdd.csv',

        'MOA': 'moa.csv',

        'Spambase': 'spambase.csv',

        'Usenet': 'usenet.csv'
    }
    file_path = '../datasets/' + switcher.get(dataset_name)
    return file_path

class GUI:
    def create_gui(self):
        """
        Creates the GUI and reads in the chosen values

        :return: dict parameters: returns a dict with the chosen parameters
        """

        sg.ChangeLookAndFeel('GreenTan')

        # ------ Menu Definition ------ #
        menu_def = [['File', ['Open file with parameters', 'Save parameters', 'Exit']],
                    ['Help', 'About...'], ]

        layout = \
            [[sg.Image(r'../logo_scaled.png')],
             [sg.Menu(menu_def, tearoff=True, key='menu_action')],
             [sg.Text('Chose your FS algorithms and parameters', justification='center', font=("Helvetica", 16))],

             # Select FS algorithm
             [sg.Text('FS algorithms:')],
             [sg.InputCombo(('Cancleout', 'EFS', 'FSDS', 'IUFES', 'MCNN', 'OFS'), size=(30, 1), key='_fs_algorithm_'),
              sg.Button('Edit parameters'), ],
             [sg.Text('Your chosen parameters are:'),sg.Text('                               -                         '
                                                             '                           '
                                                             '                                      ', key='_OUTPUT_')],
             [sg.Text(' '), sg.Text('                                                         '
                                                             '                           '
                                                             '                                     ', key='_OUTPUT_2')],

             # Begin frame for parameters
             [sg.Frame(layout=[
                 [sg.Text('Number of  epochs:     '), sg.Input(default_text=5, key='_epochs_', size=(6,1))],
                 [sg.Text('Batch size:                 '), sg.Input(default_text=50, key='_batch_size_', size=(6,1))],
                 [sg.Text('Shifting window range: '), sg.Input(default_text=20, key='_shifting_window_range_', size=(6,1))],
             ],
                 title='Feature selection parameters', title_color='red', relief=sg.RELIEF_SUNKEN)],
             # End frame for data options

             # Begin frame for data options
             [sg.Frame(layout=[
                 [sg.Radio('Enter a CSV with your data', "RADIO1", default=True, size=(40, 1),key='_load_data_path_'),
                  sg.Input(), sg.FileBrowse(key='_file_path_'), ],
                 [sg.Radio('Chose a generator to create data', "RADIO1", size=(40, 1), key='_use_generator_'),
                  sg.InputCombo(('Agrawal', 'Rbf', 'Sine'), size=(30, 1), key='_data_generator_')],
                 [sg.Radio('Use one of the existing datasets', "RADIO1", size=(40, 1), key='_load_data_CSV_'),
                  sg.InputCombo(('Credit', 'Drift', 'Har', 'KDD', 'MOA', 'Spambase', 'Usenet'), size=(30, 1),
                                key='_use_dataset_path_')],
                 [sg.Checkbox('Shuffle dataset', size=(25, 1), default=False, key='_shuffle_data_')],
             ],
                 title='Data', title_color='red', relief=sg.RELIEF_SUNKEN,
                 tooltip='Chose one of these options')],
             # End frame for data options

             # Begin frame for output options
             [sg.Frame(layout=[
                 [sg.Checkbox('Sum up results', size=(20, 1), default=True, key='_summation_'),
                  sg.Checkbox('Live visualization', default=True, key='_life_visualization_')], ],
                 title='Output options', title_color='red', relief=sg.RELIEF_SUNKEN,
                 tooltip='Set these options to customize your output')],
             # End frame for output options

             # Submit to start program, cancle to stop
             [sg.Submit(), sg.Cancel()]]

        # Set window to display your options
        window = sg.Window('PystreamFS', layout)

        # Check for the different events and perform the needed actions
        while True:
            event, values = window.Read()
            if event is None or event == 'Cancel':
                break
            if event == 'Submit':
                print('Chosen FS algorithm is: ' + values['_fs_algorithm_'] + '\nSummation of the resuls: ' + str(
                    values['_summation_'])
                      + '\nLife visualization: ' + str(values['_life_visualization_']))
                break
            if event == 'About...':
                sg.PopupOK("Chose your parameter you want to use for the selected feature selection algorithm")

            # Events for fs_selection
            if event == 'Edit parameters' and values['_fs_algorithm_'] == 'Cancleout':
                values['_cancleout_shifting_window_range'] = sg.PopupGetText(
                    'Parameter cancleout shifting window range: ', default_text="20")
                window.Element('_OUTPUT_').Update(
                    'Shifting window range:  ' + values['_cancleout_shifting_window_range'])
                window.Element('_OUTPUT_2').Update(' ')

            if event == 'Edit parameters' and values['_fs_algorithm_'] == 'EFS':
                values['_efs_alpha'] = sg.PopupGetText(
                    'Parameter EFS alpha: ', default_text="1.5")
                values['_efs_beta'] = sg.PopupGetText(
                    'Parameter EFS beta: ', default_text="0.5")
                values['_efs_threshold'] = sg.PopupGetText(
                    'Parameter EFS threshold: ', default_text="1")
                values['_efs_margin'] = sg.PopupGetText(
                    'Parameter EFS margin: ', default_text="1")
                window.Element('_OUTPUT_').Update(
                    'Alpha: ' + values['_efs_alpha'] + ', Beta: ' + values['_efs_beta'] + ', Thresh.: ' +
                    values['_efs_threshold'] + ', Margin: ' + values['_efs_margin'])
                window.Element('_OUTPUT_2').Update(' ')

            if event == 'Edit parameters' and values['_fs_algorithm_'] == 'FSDS':
                values['_fsds_ell'] = sg.PopupGetText(
                    'Parameter FSDS initial sketch size ell: ', default_text="0")
                values['_fsds_k'] = sg.PopupGetText(
                    'Parameter FSDS no. of singular values: ', default_text="2")
                window.Element('_OUTPUT_').Update(
                    'Init. sketch size: ' + values['_fsds_ell'] + ', No. singular values: ' + values['_fsds_k'])
                window.Element('_OUTPUT_2').Update(' ')

            # TODO: Insert rest of params, maybe even on output2
            if event == 'Edit parameters' and values['_fs_algorithm_'] == 'IUFES':
                values['_iufes_drift_check'] = sg.PopupGetText(
                    'Parameter IUFES check drift: ', default_text="False")
                values['_iufes_drift_check'] = convert_input_true_false(values['_iufes_drift_check'])
                values['_iufes_range'] = sg.PopupGetText(
                    'Parameter IUFES range of last t to check drift: ', default_text="2")
                values['_iufes_drift_basis'] = sg.PopupGetText(
                    'Parameter IUFES drift basis (mu): ', default_text="mu")
                window.Element('_OUTPUT_').Update(
                    'Drift check: ' + values['_iufes_drift_check'] + ', Range: ' +
                    values['_iufes_range'] + ', Drift basis: ' + values['_iufes_drift_basis'])
                window.Element('_OUTPUT_2').Update(' ')

            if event == 'Edit parameters' and values['_fs_algorithm_'] == 'MCNN':
                values['_mcnn_max_n'] = sg.PopupGetText(
                    'Parameter MCNN Max. num. of saved instances per cluster: ', default_text="100")
                values['_mcnn_e_threshold'] = sg.PopupGetText(
                    'Parameter MCNN Treshold for splitting a cluster: ', default_text="3")
                values['_mcnn_max_out_of_var_bound'] = sg.PopupGetText(
                    'Parameter MCNN Max out of var bound: ', default_text="0.3")
                values['_mcnn_p_diff_threshold'] = sg.PopupGetText(
                    'Parameter MCNN Threshold of perc. diff. for split/death rate: ', default_text="50")
                window.Element('_OUTPUT_').Update(
                    'Max saved distances: ' + values['_mcnn_max_n'] + ', Splitting treshold: ' +
                    values['_mcnn_e_threshold'] + ', Max out of var bound: ' + values['_mcnn_max_out_of_var_bound'])
                window.Element('_OUTPUT_2').Update(
                    '                                         P diff treshold: ' + values['_mcnn_p_diff_threshold'])

            if event == 'Edit parameters' and values['_fs_algorithm_'] == 'OFS':
                sg.PopupOK('No additional parameters needed for this algorithm')
                window.Element('_OUTPUT_').Update('                   ')
                window.Element('_OUTPUT_2').Update('                  ')

        window.Close()

        # Print all the chosen stats
        print(event, values)

        return values

    def run_pipeline(self, gui_dict):
        """
        Run the pipeline with the selected parameter from the gui

        :param gui_dict: dict parameters: parameters from the gui
        """

        # dictionary which takes all the parameters for the pipeline, most out of the gui dictionary gui_dict
        param = dict()

        # Generate data TODO: Impletment functionalities to pass the needed data from the dict
        # Check if the dataset has to be loaded from a path or from a existing CSV or created
        # Generator has to create data
        if gui_dict['_use_generator_']:
            generator = DataGenerator(gui_dict['_data_generator_'].lower())

        # User passes his own CSV file
        elif gui_dict['_load_data_path_']:
            dataset = pd.read_csv(gui_dict['_file_path_'])

        # Userer choses one of the existing files
        else:
            dataset = pd.read_csv(create_dataset_input_path(gui_dict['_use_dataset_path_']))

        param['shuffle_data'] = gui_dict['_shuffle_data_']





        # fs_algorithm = FeatureSelector('iufes', param)

        # pipe = Pipeline(None, generator, fs_algorithm, Perceptron(), accuracy_score, param)

        # Start Pipeline
        # pipe.start()

        # Plot results
        # pipe.plot()
        return ' '

# Create the GUi object and collect the parameter as a dict
test_gui = GUI()
params = test_gui.create_gui()
print('The chosen parameters are: ' + str(params))
#print(params['_use_dataset_path_'])

print('Pathhhhhhhhh:' + create_dataset_input_path(params['_use_dataset_path_']))


#print(params['_data_generator_'].lower())
test_gui.run_pipeline(params)






