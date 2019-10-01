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
             [sg.InputCombo(('Cancleout', 'EFS', 'FSDS', 'MCNN', 'OFS', 'UBFS'), size=(30, 1), key='_fs_algorithm_'),
              sg.Slider(range=(1, 500), size=(10, 5), default_value=50, key='_batchsize_'), ],

             # Begin frame for data options
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

        # while True:
        #     event, values = window.Read()
        #     if event is None:
        #         break
        # window.Close()

        event, values = window.Read()
        window.Close()

        # Print all the chosen stats
        print(event, values)
        print('')
        print('Chosen FS algorithm is: ' + values['_fs_algorithm_'] + '\nSummation of the resuls: ' + str(
            values['_summation_'])
              + '\nLife visualization: ' + str(values['_life_visualization_']))

        return values

    def run_pipeline(self, param):
        """
        Run the pipeline with the selected parameter from the gui

        :param param: dict parameters: parameters from the gui
        """

        # Generate data
        generator = DataGenerator('agrawal')

        fs_algorithm = FeatureSelector('iufes', param)

        pipe = Pipeline(None, generator, fs_algorithm, Perceptron(), accuracy_score, param)

        # Start Pipeline
        pipe.start()

        # Plot results
        pipe.plot()


test_gui = GUI()
params = test_gui.create_gui()

print('The chosen parameters are: ' + str(params))





