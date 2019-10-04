import PySimpleGUI as sg
from pystreamfs.pipeline import Pipeline
from pystreamfs.feature_selector import FeatureSelector
from pystreamfs.data_generator import DataGenerator
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

class GUI:
    def create_gui(self):
        """
        Creates the GUI and reads in the chosen values

        :return: dict parameters: returns a dict with the chosen parameters
        """

        sg.ChangeLookAndFeel('GreenTan')

        # ------ Menu Definition ------ #
        menu_def = [['File', ['Open', 'Save results', 'Exit']],
                    ['Help', 'About...'], ]

        layout = \
            [[sg.Menu(menu_def, tearoff=True, key='menu_action')],
             [sg.Text('Chose your FS algorithms and parameters', justification='center', font=("Helvetica", 16))],

             # Select FS algorithm
             [sg.Text('FS algorithms:')],
             [sg.InputCombo(('Cancleout', 'EFS', 'FSDS', 'IUFES', 'MCNN', 'OFS'), size=(30, 1), key='_fs_algorithm_'),
              sg.Button('Edit parameters'), ],
             [sg.Text('Your chosen parameters are:'),sg.Text('                   -                  ', key='_OUTPUT_')],


             # Begin frame for parameters
             [sg.Frame(layout=[
                 [sg.Text('Number of  epochs:     '), sg.Input(default_text=5, key='_epochs_', size=(6,1))],
                 [sg.Text('Batch size:                 '), sg.Input(default_text=50, key='_batch_size_', size=(6,1))],
                 [sg.Text('Shifting window range: '), sg.Input(default_text=20, key='_shifting_window_range_', size=(6,1))],
             ],
                 title='Feature selection parameters', title_color='red', relief=sg.RELIEF_SUNKEN)],
             # End frame for data options

             # Begin frame for data options TODO: Check if not set radio button options are still passed to dict
             [sg.Frame(layout=[
                 [sg.Radio('Enter a CSV with your data', "RADIO1", default=True, size=(25, 1)), sg.Input(),
                  sg.FileBrowse(), ],
                 [sg.Radio('Chose a generator to create data', "RADIO1", size=(25, 1), key='_create_data_'),
                  sg.InputCombo(('Gen1', 'Gen2', 'Gen3'), size=(25, 1), key='_create_dataset_')],
                 [sg.Radio('Use one of the existing datasets', "RADIO1", size=(25, 1), key='_have_data_'),
                  sg.InputCombo(('Credit', 'Drift', 'Har', 'KDD', 'MOA', 'Spambase', 'Usenet'), size=(25, 1),
                                key='_use_dataset_')],
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

            # Events for fs_selection TODO: Complete
            if event == 'Edit parameters' and values['_fs_algorithm_'] == 'Cancleout':
                values['_cancleout_shifting_window_range'] = sg.PopupGetText('Parameter cancleout shifting window range: ', default_text="20")
                window.Element('_OUTPUT_').Update('Shifting window range:  ' + values['_cancleout_shifting_window_range'])

            if event == 'Edit parameters' and values['_fs_algorithm_'] == 'EFS':
                sg.PopupYesNo()

            if event == 'Edit parameters' and values['_fs_algorithm_'] == 'FSDS':
                sg.PopupYesNo()

            if event == 'Edit parameters' and values['_fs_algorithm_'] == 'IUFES':
                sg.PopupYesNo()

            if event == 'Edit parameters' and values['_fs_algorithm_'] == 'MCNN':
                sg.PopupYesNo()

            if event == 'Edit parameters' and values['_fs_algorithm_'] == 'OFS':
                sg.PopupYesNo()

        window.Close()

        # Print all the chosen stats
        print(event, values)

        return values

    def run_pipeline(self, param):
        """
        Run the pipeline with the selected parameter from the gui

        :param param: dict parameters: parameters from the gui
        """

        # Generate data TODO: Impletment functionalities to pass the needed data from the dict
        generator = DataGenerator('agrawal')

        fs_algorithm = FeatureSelector('iufes', param)

        pipe = Pipeline(None, generator, fs_algorithm, Perceptron(), accuracy_score, param)

        # Start Pipeline
        pipe.start()

        # Plot results
        pipe.plot()

# Create the GUi object and collect the parameter as a dict
test_gui = GUI()
params = test_gui.create_gui()
print('The chosen parameters are: ' + str(params))

# TODO: create the pipe with run_pipline





