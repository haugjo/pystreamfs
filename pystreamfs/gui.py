import PySimpleGUI as sg
import pandas as pd
import numpy as np
from pystreamfs.pipeline import Pipeline
from pystreamfs.feature_selector import FeatureSelector
from pystreamfs.data_generator import DataGenerator
from pystreamfs.visualizer import Visualizer
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
            '_fs_algorithm_': 'Cancelout',
            '_delay_': 1.0,
            '_no_features_': 5,
            '_batch_size_': 50,
            '_max_timesteps_': 10,
            '_shifting_window_range_': 20,
            '_file_path_': '../datasets/credit.csv',
            '_file_path_saving_': '../pystreamfs/output_results',
            '_data_generator_': 'Agrawal',
            '_use_dataset_path_': '../datasets/credit.csv',
            '_shuffle_data_': False,
            '_label_index_': 0,
            '_font_scale_': 0.8,
            '_save_results_': True,
            '_live_visualization_': True,
            '_efs_u_': None,
            '_efs_v_': None,
            '_efs_alpha_': 1.5,
            '_efs_beta_': 0.5,
            '_efs_threshold_': 1.0,
            '_efs_margin_': 1.0,
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
            '_mcnn_p_diff_threshold_': 50,
            '_use_param_file_': False
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
             [sg.Text('Choose your FS algorithms and parameters', justification='center', font=("Helvetica", 16))],

             # Select FS algorithm
             [sg.Text('FS algorithm:')],
             [sg.InputCombo(('Cancelout', 'EFS', 'FSDS', 'IUFES', 'MCNN', 'OFS'), size=(30, 1), key='_fs_algorithm_'),
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
                 [sg.Text('Number of  features:      '), sg.Input(default_text=5, key='_no_features_', size=(6, 1))],
                 [sg.Text('Batch size:                   '), sg.Input(default_text=50, key='_batch_size_', size=(6, 1))],
                 [sg.Text('Shifting window range:   '),
                  sg.Input(default_text=20, key='_shifting_window_range_', size=(6, 1))],
             ],
                 title='Feature selection parameters', title_color='red', relief=sg.RELIEF_SUNKEN)],
             # End frame for data options

             # Begin frame for data options
             [sg.Frame(layout=[
                 [sg.Radio('Chose a generator to create data', "RADIO1", default=True, size=(24, 1), key='_use_generator_'),
                  sg.InputCombo(('Agrawal', 'Rbf', 'Sine'), size=(14, 1), key='_data_generator_')],
                 [sg.Radio('Use one of the existing datasets', "RADIO1", size=(24, 1), key='_load_data_CSV_'),
                  sg.InputCombo(('Credit', 'Drift', 'Har', 'KDD', 'MOA', 'Spambase', 'Usenet'), size=(30, 1),
                                key='_use_dataset_path_')],
                 [sg.Radio('Enter a CSV with your data', "RADIO1", size=(24, 1), key='_load_data_path_'),
                  sg.Input(), sg.FileBrowse(key='_file_path_'), ],
                 [sg.Checkbox('Shuffle dataset', size=(19, 1), default=False, key='_shuffle_data_')],
                 [sg.Text('Timesteps generator:'),
                  sg.Input(default_text=10, key='_max_timesteps_', size=(6, 1))],
                 [sg.Text('Label index:             '), sg.Input(default_text=0, key='_label_index_', size=(6, 1))],
             ],
                 title='Data', title_color='red', relief=sg.RELIEF_SUNKEN)],
             # End frame for data options

             # Begin frame for output options
             [sg.Frame(layout=[
                 [sg.Text('Font scale:                           '),
                  sg.Input(default_text=0.8, key='_font_scale_', size=(6, 1))],
                 [sg.Text('Delay live visualization:          '),
                  sg.Input(default_text=1.0, key='_delay_', size=(6, 1))],
                 [sg.Checkbox('Live visualization', default=True, key='_live_visualization_')],
                 [sg.Checkbox('Save results:', size=(19, 1), default=True,
                              key='_save_results_'),
                  sg.Input('../pystreamfs/output_results', key='_file_path_saving_'), sg.FolderBrowse(),],],
                 title='Output options', title_color='red', relief=sg.RELIEF_SUNKEN,
                 tooltip='Set these options to customize your output')],
             # End frame for output options

             # Submit to start program, cancel to stop
             [sg.Submit(), sg.Exit()]]

        # Set window to display your options
        window = sg.Window('PystreamFS', layout)
        # Check for the different events and perform the needed actions
        while True:
            event, w_input = window.Read()
            if event is None or event == 'Cancel' or event == 'Exit':
                break
            if event == 'Submit':
                break
            if event == 'About...':
                sg.PopupOK('Chose the parameters you want to use for the selected feature selection algorithm.')

            # Save parameters
            if event == 'Open file with parameters':
                try:
                    f = open('param.txt', 'r').read()
                    self.values = eval(f)
                    self.values['_use_param_file_'] = True
                    # event = 'Submit'
                    break
                except IOError:
                    sg.Popup('File with parameters is not accessible')
            if event == 'Save parameters':
                try:
                    f = open('param.txt', 'w')
                    f.write(str(self.values))
                    f.close()
                    sg.Popup('Parameters written to param.txt')
                except IOError:
                    print('File not accessible')

            # Events for fs_selection
            if event == 'Edit parameters' and w_input['_fs_algorithm_'] == 'Cancelout':
                sg.PopupOK('No additional parameters needed for this algorithm')
                window.Element('_OUTPUT_').Update('No additional parameters required for Cancelout')
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

        # Update the values dict with the window input dict if no param file is used
        if not self.values['_use_param_file_']:
            self.values.update(w_input)
        print('Values are: ' + str(self.values))

    def run_pipeline(self):
        """
        Run the pipeline with the selected parameters from the gui

        """

        # dictionary which takes all the parameters for the pipeline, most out of the gui dictionary gui_dict
        param = dict()

        # Generate empty generator and dataset if either is not used
        generator = None
        dataset = None

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
        param['label_idx'] = int(self.values['_label_index_'])

        ###################################################################################################
        # Hand over all FS properties from the self.values dict
        fs_prop = dict()  # FS Algorithm properties

        # Properties EFS:
        if self.values['_fs_algorithm_'] == 'EFS':
            # get u and v using a generator
            if self.values['_use_generator_']:
                fs_prop['u'] = np.ones(generator.no_features) * 2  # initial positive model with weights 2
                fs_prop['v'] = np.ones(generator.no_features)  # initial negative model with weights 1
            # get u and v using a dataset
            else:
                fs_prop['u'] = np.ones(dataset.shape[1] - 1) * 2  # initial positive model with weights 2
                fs_prop['v'] = np.ones(dataset.shape[1] - 1)  # initial negative model with weights 1
            fs_prop['alpha'] = self.values['_efs_alpha_']  # promotion parameter
            fs_prop['beta'] = self.values['_efs_beta_']  # demotion parameter
            fs_prop['threshold'] = self.values['_efs_threshold_']  # threshold parameter
            fs_prop['M'] = self.values['_efs_margin_']  # margin

        # Properties FSDS:
        if self.values['_fs_algorithm_'] == 'FSDS':
            fs_prop['B'] = []  # initial sketch matrix
            fs_prop['ell'] = self.values['_fsds_ell_']  # initial sketch size
            fs_prop['k'] = self.values['_fsds_k_']  # no. of singular values (can be equal to no. of clusters/classes -> 2 for binary class.)
            # get m using a generator
            if self.values['_use_generator_']:
                fs_prop['m'] = generator.no_features  # no. of original features
            else:
                fs_prop['m'] = dataset.shape[1] - 1

        # Properties IUFES:
        if self.values['_fs_algorithm_'] == 'IUFES':
            fs_prop['epochs'] = self.values['_iufes_epochs_']  # iterations over curr. batch
            fs_prop['mini_batch_size'] = self.values['_iufes_mini_batch_size_']  # must be smaller than batch_size
            fs_prop['lr_mu'] = self.values['_iufes_lr_mu_']  # learning rate for mean
            fs_prop['lr_sigma'] = self.values['_iufes_lr_sigma_']  # learning rate for standard deviation
            fs_prop['init_sigma'] = self.values['_iufes_init_sigma_']
            fs_prop['lr_w'] = self.values['_iufes_lr_w_']  # learning rate for weights
            fs_prop['lr_lambda'] = self.values['_iufes_lr_lambda_']  # learning rate for lambda
            fs_prop['init_lambda'] = self.values['_iufes_init_lambda_']
            fs_prop['check_drift'] = self.values['_iufes_drift_check_']  # indicator whether to check drift or not
            fs_prop['range'] = self.values['_iufes_range_']  # range of last t to check for drift
            fs_prop['drift_basis'] = self.values['_iufes_drift_basis_']  # basis param to perform concept drift det.

        # Properties MCNN:
        if self.values['_fs_algorithm_'] == 'MCNN':
            fs_prop['max_n'] = self.values['_mcnn_max_n_']  # maximum number of saved instances per cluster
            fs_prop['e_threshold'] = self.values['_mcnn_e_threshold_']  # error threshold for splitting of a cluster
            # Additional parameters
            # percentage of variables that can at most be outside of variance boundary before new cluster is created
            fs_prop['max_out_of_var_bound'] = self.values['_mcnn_max_out_of_var_bound_']
            # threshold of perc. diff. for split/death rate when drift is assumed
            fs_prop['p_diff_threshold'] = self.values['_mcnn_p_diff_threshold_']

        ###################################################################################################
        # General parameters
        param['batch_size'] = int(self.values['_batch_size_'])
        param['num_features'] = int(self.values['_no_features_'])
        param['max_timesteps'] = int(self.values['_max_timesteps_'])
        param['font_scale'] = float(self.values['_font_scale_'])
        param['r'] = int(self.values['_shifting_window_range_'])
        param['is_live'] = self.values['_live_visualization_']
        param['time_delay'] = float(self.values['_delay_'])
        param['save_results'] = self.values['_save_results_']
        param['save_path'] = self.values['_file_path_saving_']
        param['fs_algo'] = self.values['_fs_algorithm_']

        # Use the feature selector on the chosen algorithm
        fs_algorithm = FeatureSelector(self.values['_fs_algorithm_'].lower(), fs_prop)

        # Generate Visualizer
        visual = Visualizer(self.values['_live_visualization_'])

        # Create the pipeline with the necessary parameters and options
        pipe = Pipeline(dataset, generator, fs_algorithm, visual, Perceptron(), accuracy_score, param)

        # Start Pipeline
        pipe.start()

        # Plot results
        pipe.plot()
        return ' '


# Create the GUi object and collect its parameters as a dict
pysteamfs_gui = GUI()
pysteamfs_gui.create_gui()

# Check whether the pipeline should be started and print the parameters
if pysteamfs_gui.values['_final_event_'] == 'Submit':
    print('The chosen parameters are: ')
    for k, v in pysteamfs_gui.values.items():
        print('Keys: ' + str(k) + ', value: ' + str(v))
    print('Starting the pipeline.')
    pysteamfs_gui.run_pipeline()
