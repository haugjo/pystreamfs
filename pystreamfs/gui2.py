import sys
from PySide2.QtCore import Qt, Slot
from PySide2.QtGui import QPainter
from PySide2.QtWidgets import (QAction, QApplication, QHeaderView, QHBoxLayout, QLabel, QLineEdit,
                               QMainWindow, QPushButton, QTableWidget, QTableWidgetItem,
                               QVBoxLayout, QWidget)
from PySide2.QtCharts import QtCharts

class Widget(QWidget):
    def __init__(self):
        QWidget.__init__(self)
        self.items = 0

        # Example data
        self._data = {"Water": 24.5, "Electricity": 55.1, "Rent": 850.0,
                      "Supermarket": 230.4, "Internet": 29.99, "Bars": 21.85,
                      "Public transportation": 60.0, "Coffee": 22.45, "Restaurants": 120}

        # Left
        self.table = QTableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["Description", "Price"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        # Chart
        self.chart_view = QtCharts.QChartView()
        self.chart_view.setRenderHint(QPainter.Antialiasing)

        # Right
        self.description = QLineEdit()
        self.price = QLineEdit()
        self.add = QPushButton("Add")
        self.clear = QPushButton("Clear")
        self.quit = QPushButton("Quit")
        self.plot = QPushButton("Plot")

        # Disabling 'Add' button
        self.add.setEnabled(False)

        self.right = QVBoxLayout()
        self.right.setMargin(10)
        self.right.addWidget(QLabel("Description"))
        self.right.addWidget(self.description)
        self.right.addWidget(QLabel("Price"))
        self.right.addWidget(self.price)
        self.right.addWidget(self.add)
        self.right.addWidget(self.plot)
        self.right.addWidget(self.chart_view)
        self.right.addWidget(self.clear)
        self.right.addWidget(self.quit)

        # QWidget Layout
        self.layout = QHBoxLayout()

        #self.table_view.setSizePolicy(size)
        self.layout.addWidget(self.table)
        self.layout.addLayout(self.right)

        # Set the layout to the QWidget
        self.setLayout(self.layout)

        # Signals and Slots
        self.add.clicked.connect(self.add_element)
        self.quit.clicked.connect(self.quit_application)
        self.plot.clicked.connect(self.plot_data)
        self.clear.clicked.connect(self.clear_table)
        self.description.textChanged[str].connect(self.check_disable)
        self.price.textChanged[str].connect(self.check_disable)

        # Fill example data
        self.fill_table()

    @Slot()
    def add_element(self):
        des = self.description.text()
        price = self.price.text()

        self.table.insertRow(self.items)
        description_item = QTableWidgetItem(des)
        price_item = QTableWidgetItem("{:.2f}".format(float(price)))
        price_item.setTextAlignment(Qt.AlignRight)

        self.table.setItem(self.items, 0, description_item)
        self.table.setItem(self.items, 1, price_item)

        self.description.setText("")
        self.price.setText("")

        self.items += 1

    @Slot()
    def check_disable(self, s):
        if not self.description.text() or not self.price.text():
            self.add.setEnabled(False)
        else:
            self.add.setEnabled(True)

    @Slot()
    def plot_data(self):
        # Get table information
        series = QtCharts.QPieSeries()
        for i in range(self.table.rowCount()):
            text = self.table.item(i, 0).text()
            number = float(self.table.item(i, 1).text())
            series.append(text, number)

        chart = QtCharts.QChart()
        chart.addSeries(series)
        chart.legend().setAlignment(Qt.AlignLeft)
        self.chart_view.setChart(chart)

    @Slot()
    def quit_application(self):
        QApplication.quit()

    def fill_table(self, data=None):
        data = self._data if not data else data
        for desc, price in data.items():
            description_item = QTableWidgetItem(desc)
            price_item = QTableWidgetItem("{:.2f}".format(price))
            price_item.setTextAlignment(Qt.AlignRight)
            self.table.insertRow(self.items)
            self.table.setItem(self.items, 0, description_item)
            self.table.setItem(self.items, 1, price_item)
            self.items += 1

    @Slot()
    def clear_table(self):
        self.table.setRowCount(0)
        self.items = 0


class MainWindow(QMainWindow):
    def __init__(self, widget):
        QMainWindow.__init__(self)
        self.setWindowTitle("PystreamFS")

        # Menu
        self.menu = self.menuBar()
        self.file_menu = self.menu.addMenu("File")

        # Exit QAction
        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.exit_app)

        self.file_menu.addAction(exit_action)
        self.setCentralWidget(widget)

    @Slot()
    def exit_app(self, checked):
        QApplication.quit()


if __name__ == "__main__":
    # Qt Application
    app = QApplication(sys.argv)
    # QWidget
    widget = Widget()
    # QMainWindow using QWidget as central widget
    window = MainWindow(widget)
    window.resize(800, 600)
    window.show()

    # Execute application
    sys.exit(app.exec_())

# class GUI:
#     def __init__(self):
#         self.layout = [[sg.Text('Window 1'), ],
#                        [sg.Input(do_not_clear=True)],
#                        [sg.Text('', key='_OUTPUT_')],
#                        [sg.Text('', key='_OUTPUT2_')],
#                        [sg.Text('', key='_OUTPUT3_')],
#                        [sg.Text('', key='_OUTPUT4_')],
#                        [sg.Button('Launch 2'), sg.Button('Exit')]]
#         self.win1 = sg.Window('Window 1').Layout(self.layout)
#         self.vars = {}
#         self.ev1 = {}
#         self.vals1 = {}
#         self.win2_active = False
#         self.win2 = None
#         self.mode = ''
#
#     def close_main_window(self):
#         self.win1.Close()
#
#     def check_events(self):
#         while True:
#             self.ev1, self.vals1 = self.win1.Read(timeout=100)
#             self.win1.FindElement('_OUTPUT_').Update(self.vals1)
#             if self.ev1 is None or self.ev1 == 'Exit':
#                 break
#
#             if self.ev1 == 'Launch 2':
#                 self.mode = 'Launch 2'
#
#             self.win2, self.win2_active = self.window2()
#             self.mode = ''
#
#     def window2(self):
#         if not self.win2_active and self.mode == 'Launch 2':
#             self.win2_active = True
#             layout2 = [[sg.Text('Window 2')],
#                        [sg.Input(do_not_clear=True)],
#                        [sg.Input(do_not_clear=True)],
#                        [sg.Input(do_not_clear=True)],
#                        [sg.Button('Exit')]]
#
#             self.win2 = sg.Window('Window 2').Layout(layout2)
#
#         if self.win2_active:
#             ev2, vals2 = self.win2.Read(timeout=100)
#             if ev2 is None or ev2 == 'Exit':
#                 self.win2_active = False
#                 self.win2.Close()
#             else:
#                 self.win1.FindElement('_OUTPUT2_').Update(vals2[0])
#                 self.win1.FindElement('_OUTPUT3_').Update(vals2[1])
#                 self.win1.FindElement('_OUTPUT4_').Update(vals2[2])
#
#         return self.win2, self.win2_active
#
#
# test_gui = GUI()
#
# test_gui.check_events()
# test_gui.close_main_window()

