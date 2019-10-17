import PySimpleGUI as sg
import pandas as pd
import numpy as np
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
    def __init__(self):
        # Create the dict with all default values
        self.values = {
            '_fs_algorithm_': 'Cancleout',
            '_no_features_': 5,
            '_batch_size_': 50,
            '_shifting_window_range_': 20,
            '_file_path_': '../datasets/credit.csv',
            '_data_generator_': 'Agrawal',
            '_use_dataset_path_': '../datasets/credit.csv',
            '_shuffle_data_': False,
            '_label_index_': 0,
            '_summation_': True,
            '_life_visualization_': True,
            '_efs_u_': None,
            '_efs_v_': None,
            '_efs_alpha_': 1.5,
            '_efs_beta_': 0.5,
            '_efs_threshold_': 1.0,
            '_efs_margin_:': 1.0,
            '_fsds_b_': [],
            '_fsds_ell_': 0,
            '_fsds_k_': 2,
            '_fsds_m_': None,
            '_iufes_epochs_': 5,
            '_iufes_mini_batch_size_': 25,
            '_iufes_lr_mu_': 0.1,
            '_iufes_lr_sigma_': 0.1,
            '_iufes_init_sigma_': 1.0,
            '_iufes_lr_w_': 0.1,
            '_iufes_lr_lambda_': 0.1,
            '_iufes_init_lambda_': 1.0,
            '_iufes_drift_check_': False,
            '_iufes_range_': 2,
            '_iufes_drift_basis_': 'mu',
            '_mcnn_max_n_': 100,
            '_mcnn_e_threshold_': 3,
            '_mcnn_max_out_of_var_bound_': 0.3,
            '_mcnn_p_diff_threshold_': 50
        }

    def create_gui(self):
        """
        Creates the GUI and reads in the chosen values

        :return: dict parameters: returns a dict with the chosen parameters
        """

        # Choose the
        sg.ChangeLookAndFeel('GreenTan')

        # ------ Menu Definition ------ #
        menu_def = [['File', ['Open file with parameters', 'Save parameters', 'Exit']],
                    ['Help', 'About...'], ]

        layout = \
            [[sg.Image(r'../logo_scaled.png')],
             [sg.Menu(menu_def, tearoff=True, key='menu_action')],
             [sg.Text('Chose your FS algorithms and parameters', justification='center', font=("Helvetica", 16))],

             # Select FS algorithm
             [sg.Text('FS algorithm:')],
             [sg.InputCombo(('Cancleout', 'EFS', 'FSDS', 'IUFES', 'MCNN', 'OFS'), size=(30, 1), key='_fs_algorithm_'),
              sg.Button('Edit parameters'), ],
             [sg.Text('Your chosen parameters are:')],
             [sg.Text('                                                            '
                      '                           '
                      '                                      '
                      '                                       ', key='_OUTPUT_')],
             [sg.Text('                                                         '
                      '                           '
                      '                                     '
                      '                                     ', key='_OUTPUT_2_')],

             # Begin frame for parameters
             [sg.Frame(layout=[
                 [sg.Text('Number of  features:     '), sg.Input(default_text=5, key='_no_features_', size=(6, 1))],
                 [sg.Text('Batch size:                 '), sg.Input(default_text=50, key='_batch_size_', size=(6, 1))],
                 [sg.Text('Shifting window range: '),
                  sg.Input(default_text=20, key='_shifting_window_range_', size=(6, 1))],
             ],
                 title='Feature selection parameters', title_color='red', relief=sg.RELIEF_SUNKEN)],
             # End frame for data options

             # Begin frame for data options
             [sg.Frame(layout=[
                 [sg.Radio('Enter a CSV with your data', "RADIO1", default=True, size=(40, 1), key='_load_data_path_'),
                  sg.Input(), sg.FileBrowse(key='_file_path_'), ],
                 [sg.Radio('Chose a generator to create data', "RADIO1", size=(40, 1), key='_use_generator_'),
                  sg.InputCombo(('Agrawal', 'Rbf', 'Sine'), size=(30, 1), key='_data_generator_')],
                 [sg.Radio('Use one of the existing datasets', "RADIO1", size=(40, 1), key='_load_data_CSV_'),
                  sg.InputCombo(('Credit', 'Drift', 'Har', 'KDD', 'MOA', 'Spambase', 'Usenet'), size=(30, 1),
                                key='_use_dataset_path_')],
                 [sg.Checkbox('Shuffle dataset', size=(25, 1), default=False, key='_shuffle_data_')],
                 [sg.Text('Label index: '), sg.Input(default_text=0, key='_label_index_', size=(6, 1))],
             ],
                 title='Data', title_color='red', relief=sg.RELIEF_SUNKEN)],
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
        values = dict()
        while True:
            event, w_input = window.Read()
            if event is None or event == 'Cancel':
                break
            if event == 'Submit':
                break
            if event == 'About...':
                sg.PopupOK("Chose your parameter you want to use for the selected feature selection algorithm")

            # Events for fs_selection
            if event == 'Edit parameters' and w_input['_fs_algorithm_'] == 'Cancleout':
                sg.PopupOK('No additional parameters needed for this algorithm')
                window.Element('_OUTPUT_').Update('No additional parameters required for Cancleout')
                window.Element('_OUTPUT_2_').Update('                  ')

            # Selected FS algorithm is EFS
            if event == 'Edit parameters' and w_input['_fs_algorithm_'] == 'EFS':
                self.values['_efs_alpha_'] = float(sg.PopupGetText(
                    'Parameter EFS alpha: ', default_text="1.5"))
                self.values['_efs_beta_'] = float(sg.PopupGetText(
                    'Parameter EFS beta: ', default_text="0.5"))
                self.values['_efs_threshold_'] = float(sg.PopupGetText(
                    'Parameter EFS threshold: ', default_text="1"))
                self.values['_efs_margin_'] = float(sg.PopupGetText(
                    'Parameter EFS margin: ', default_text="1"))
                window.Element('_OUTPUT_').Update('Alpha: {0:.2f}, Beta: {1:.2f}, Thresh: {2:.2f}, '
                                                  'Margin: {3:.2f}'.format(self.values['_efs_alpha_'],
                                                                           self.values['_efs_beta_'],
                                                                           self.values['_efs_threshold_'],
                                                                           self.values['_efs_margin_']))
                window.Element('_OUTPUT_2_').Update(' ')

            # Selected FS algorithm is FSDS
            if event == 'Edit parameters' and w_input['_fs_algorithm_'] == 'FSDS':
                self.values['_fsds_ell_'] = int(sg.PopupGetText(
                    'Parameter FSDS initial sketch size: ', default_text="0"))
                self.values['_fsds_k_'] = int(sg.PopupGetText(
                    'Parameter FSDS no. of singular values: ', default_text="2"))
                window.Element('_OUTPUT_').Update(
                    'Init. sketch size: {0:d}, No. singular values: {1:d}'.format(self.values['_fsds_ell_'],
                                                                                  self.values['_fsds_k_']))
                window.Element('_OUTPUT_2_').Update(' ')

            # Selected FS algorithm is IUFES
            if event == 'Edit parameters' and w_input['_fs_algorithm_'] == 'IUFES':
                self.values['_iufes_epochs_'] = int(sg.PopupGetText(
                    'Parameter IUFES epochs: ', default_text="5"))
                self.values['_iufes_mini_batch_size_'] = int(sg.PopupGetText(
                    'Parameter IUFES Mini batch size: ', default_text="25"))
                self.values['_iufes_lr_mu_'] = float(sg.PopupGetText(
                    'Parameter IUFES Mean learning rate: ', default_text="0.1"))
                self.values['_iufes_lr_sigma_'] = float(sg.PopupGetText(
                    'Parameter IUFES standard deviation learning rate: ', default_text="0.1"))
                self.values['_iufes_init_sigma_'] = float(sg.PopupGetText(
                    'Parameter IUFES initial sigma: ', default_text="1.0"))
                self.values['_iufes_lr_w_'] = float(sg.PopupGetText(
                    'Parameter IUFES weight learning rate: ', default_text="0.1"))
                self.values['_iufes_lr_lambda_'] = float(sg.PopupGetText(
                    'Parameter IUFES lambda learning rate: ', default_text="0.1"))
                self.values['_iufes_init_lambda_'] = float(sg.PopupGetText(
                    'Parameter IUFES initial lambda: ', default_text="1.0"))
                self.values['_iufes_drift_check_'] = sg.PopupGetText(
                    'Parameter IUFES check drift: ', default_text="False")
                self.values['_iufes_drift_check_'] = convert_input_true_false(self.values['_iufes_drift_check_'])
                self.values['_iufes_range_'] = int(sg.PopupGetText(
                    'Parameter IUFES range of last t to check drift: ', default_text="2"))
                self.values['_iufes_drift_basis_'] = sg.PopupGetText(
                    'Parameter IUFES drift basis (mu): ', default_text="mu")
                window.Element('_OUTPUT_').Update('Epochs: {0:d}, Mini Batch Size: {1:d}, Mean LR: {2:.2f}, '
                                                  'Sigma LR: {3:.2f}, Init. Sigma: {4:0.2f}'
                                                  .format(self.values['_iufes_epochs_'],
                                                          self.values['_iufes_mini_batch_size_'],
                                                          self.values['_iufes_lr_mu_'],
                                                          self.values['_iufes_lr_sigma_'],
                                                          self.values['_iufes_init_sigma_']))
                window.Element('_OUTPUT_2_').Update('Weight LR: {0:.2f}, Lambda LR: {1:.2f}, Init. Lambda: {2:.2f}, '
                                                    'Drift check: {3:s}, Range: {4:d}, Drift Basis: {5:s}'
                                                    .format(self.values['_iufes_lr_w_'],
                                                            self.values['_iufes_lr_lambda_'],
                                                            self.values['_iufes_init_lambda_'],
                                                            self.values['_iufes_drift_check_'],
                                                            self.values['_iufes_range_'],
                                                            self.values['_iufes_drift_basis_']))

            # Selected FS algorithm is MCNN
            if event == 'Edit parameters' and w_input['_fs_algorithm_'] == 'MCNN':
                self.values['_mcnn_max_n_'] = int(sg.PopupGetText(
                    'Parameter MCNN Max. num. of saved instances per cluster: ', default_text="100"))
                self.values['_mcnn_e_threshold_'] = int(sg.PopupGetText(
                    'Parameter MCNN Threshold for splitting a cluster: ', default_text="3"))
                self.values['_mcnn_max_out_of_var_bound_'] = float(sg.PopupGetText(
                    'Parameter MCNN Max out of var bound: ', default_text="0.3"))
                self.values['_mcnn_p_diff_threshold_'] = int(sg.PopupGetText(
                    'Parameter MCNN Threshold of perc. diff. for split/death rate: ', default_text="50"))
                window.Element('_OUTPUT_').Update('Max. num. saved instances: {0:d}, Split thresh.: {1:d}, '
                                                  'Max out of var bound: {2:.2f}, Perc. thresh. rate: {3:.2f}'
                                                  .format(self.values['_mcnn_max_n_'],
                                                          self.values['_mcnn_e_threshold_'],
                                                          self.values['_mcnn_max_out_of_var_bound_'],
                                                          self.values['_mcnn_p_diff_threshold_']))
                window.Element('_OUTPUT_2_').Update(' ')

            # Selected FS algorithm is OFS
            if event == 'Edit parameters' and w_input['_fs_algorithm_'] == 'OFS':
                sg.PopupOK('No additional parameters needed for this algorithm')
                window.Element('_OUTPUT_').Update('No additional parameters required for OFS')
                window.Element('_OUTPUT_2_').Update('                  ')

        window.Close()

        # Add the last event to the dict to check whether the pipe should be started or not
        self.values['_final_event_'] = event

        # Update the values dict with the window input dict
        self.values.update(w_input)
        print('Update values are: ' + str(self.values))

    def run_pipeline(self):
        """
        Run the pipeline with the selected parameter from the gui

        """

        # dictionary which takes all the parameters for the pipeline, most out of the gui dictionary gui_dict
        param = dict()


        # Generate data
        # Check if the dataset has to be loaded from a path or from a existing CSV or created
        # Generator has to create data
        if self.values['_use_generator_']:
            generator = DataGenerator(self.values['_data_generator_'].lower())
        # User passes his own CSV file
        elif self.values['_load_data_path_']:
            dataset = pd.read_csv(self.values['_file_path_'])
        # User chooses one of the existing files
        else:
            dataset = pd.read_csv(create_dataset_input_path(self.values['_use_dataset_path_']))

        # Parameters for the dataset
        param['shuffle_data'] = self.values['_shuffle_data_']
        param['label_idx'] = self.values['_label_index_']

        ###################################################################################################
        # Hand over all FS properties from the self.values dict
        fs_prop = dict()  # FS Algorithm properties

        # Properties EFS:
        param['u'] = np.ones(num_features) * 2  # initial positive model with weights 2
        param['v'] = np.ones(num_features)  # initial negative model with weights 1
        param['alpha'] = 1.5  # promotion parameter
        param['beta'] = 0.5  # demotion parameter
        param['threshold'] = 1  # threshold parameter
        param['M'] = 1  # margin

        fs_prop['epochs'] = self.values['_iufes_drift_check']  # iterations over curr. batch during one execution IUFES
        fs_prop['mini_batch_size'] = self.values['_iufes_mini_batch_size_']  # must be smaller than batch_size
        fs_prop['lr_mu'] = self.values['_iufes_lr_mu_']  # learning rate for mean
        fs_prop['lr_sigma'] = 0.1  # learning rate for standard deviation
        fs_prop['init_sigma'] = 1
        fs_prop['lr_w'] = 0.1  # learning rate for weights
        fs_prop['lr_lambda'] = 0.1  # learning rate for lambda
        fs_prop['init_lambda'] = 1

        # Parameters for concept drift detection
        fs_prop['check_drift'] = False  # indicator whether to check drift or not
        fs_prop['range'] = 2  # range of last t to check for drift
        fs_prop['drift_basis'] = 'mu'  # basis parameter on which we perform concept drift detection

        # fs_algorithm = FeatureSelector('iufes', param)

        # pipe = Pipeline(None, generator, fs_algorithm, Perceptron(), accuracy_score, param)

        # Start Pipeline
        # pipe.start()

        # Plot results
        # pipe.plot()
        return ' '


# Create the GUi object and collect its parameters as a dict
test_gui = GUI()
test_gui.create_gui()

# Check whether the pipeline should be started and print the parameters
if test_gui.values['_final_event_'] == 'Submit':
    print('The chosen parameters are: ')
    for k, v in test_gui.values.items():
        print('Keys: ' + str(k) + ', values: ' + str(v))
        print('Starting the pipeline.')
        # test_gui.run_pipeline()
