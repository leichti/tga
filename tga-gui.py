import sys
from copy import copy

import matplotlib
from PyQt5.QtWidgets import QFileDialog, QMessageBox

from gui.gui import Ui_MainWindow
from PyQt5 import QtWidgets
from dataprep import TgaFile, ReductionTrial, ReductionReference, TgaFolder
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

class TgaGui:

    def __init__(self, main_window):
        ui = Ui_MainWindow()
        ui.setupUi(main_window)
        self.ui = ui
        self.tabTrial = TabTrial(self)
        self.tabEvaluate = TabEvaluate(self)
        self.tabFiles = TabFiles(self)
        self.connect()

    def connect(self):
        self.ui.loadReference.clicked.connect(self.tabTrial.loadReference)
        self.ui.detectReduction.clicked.connect(self.tabTrial.detectReduction)
        self.ui.trimReduction.clicked.connect(self.tabTrial.trimReduction)
        self.ui.saveReduction.clicked.connect(self.tabTrial.save_reduction)
        self.ui.adjustWeight.clicked.connect(self.tabTrial.adjustWeight)
        self.ui.openFolderBtn.clicked.connect(self.tabFiles.openFolder)
        self.ui.investigateBtn.clicked.connect(self.tabFiles.investigate)
        self.ui.exportAllBtn.clicked.connect(self.tabFiles.exportAll)


    def directExport(self, trialname=None, experiment=None):

        if trialname:
            file = self.tabFiles.folder.select(trialname)
            experiment = ReductionTrial(file)
        if experiment:
            experiment = experiment
            file = experiment.file

        if self.ui.setInitialWeightCheckBox.isChecked():
            experiment.adjust_weight()

        df = experiment.export(trimmed_export=self.ui.trimReductionCheckBox.isChecked())
        if self.ui.directExportCheckBox.isChecked():
            save_path = f"{file.folder.path}/export/{file}"
        else:
            save_path = QFileDialog.getSaveFileName(directory=f"{file.folder.path}/export/", caption="Save as")[0]

        df.to_csv(save_path)



class Figure:

    def __init__(self):
        self.figure, self.axes = plt.subplots(2,1,dpi=45)
        self.ax = self.axes[0]
        self.canvas = FigureCanvas(self.figure)

    def reset(self, ax=None):
        if type(ax) is int:
            ax = self.axes[ax]

        if hasattr(ax, "clear"):
            ax.clear()
            self.canvas.draw()
            return True

        for ax in self.axes:
            self.reset(ax)


    def plot(self, *args, **kwargs):

        if "ax" in kwargs:
            ax = kwargs.pop("ax")
        else:
            ax = self.ax

        if type(ax) is int:
            self.ax = self.axes[ax]

        elif not hasattr(ax, "plot"):
            raise TypeError(f"Parameter ax is of type {type(ax)}")

        self.ax.plot(*args, **kwargs)

        self.update()

    def update(self):
        self.canvas.draw()

class TabTrial:
    def __init__(self, main):
        self.main = main

        self.experiment = None
        self.reference = None

        # conditions
        self.reduction_detected = False
        self.reference_loaded = False

        self.figure = Figure()
        self.main.ui.horizontalLayout.addWidget(self.figure.canvas)

    def adjustWeight(self):
        if not self.experiment:
            return False

        self.experiment.adjust_weight()
        self.update_plot()

    def loadFile(self, trial = None):
        if trial is None:
            self.path = QFileDialog.getOpenFileName(filter="Exported TGA Files (*.txt)")[0]

            if not self.path:
                return False

            self.file = TgaFile(self.path)
        else:
            self.file = trial

        self.figure.reset()

        self.experiment = ReductionTrial(self.file)
        self.experiment.t_dimension = "min"

        self.main.ui.initialWeight.setText(f"{self.experiment.weight} mg")
        self.main.ui.labelFilename.setText(f"{self.experiment.file}")

        self.update_plot(cropped=False)
        #self.figure.plot([min(args[0]), max(args[0])], [0,0])

    def loadReference(self):
        if not self.experiment:
            QMessageBox.about(self.main.ui.centralwidget, "Info", "Can't load reference before measurement")
            return False
        if self.reference:
            QMessageBox.about(self.main.ui.centralwidget, "Info", "Reference already loaded")
            return False

        self.reference_path = QFileDialog.getOpenFileName(filter="Exported TGA Files (*.txt)")[0]

        if not self.reference_path:
            return False

        self.reference_file = TgaFile(self.reference_path)
        self.reference = ReductionReference(self.reference_file)
        self.reference.t_dimension = "min"
        self.experiment.add_ref(self.reference, apply=True)
        self.update_plot()

    def detectReduction(self):
        if self.reduction_detected is True:
            QMessageBox.about(self.main.ui.centralwidget, "Info", "Reduction already detected")
            return

        if not self.experiment:
            QMessageBox.about(self.main.ui.centralwidget, "Info", "No data loaded")
            return False

        self.experiment.detect()
        self.update_plot()
        self.reduction_detected = True

    def update_plot(self, *args, **kwargs):
        self.figure.plot(*self.experiment.get("m", "t", **kwargs), ax=0)
        self.figure.plot(*self.experiment.get("dmdt", "dm_remaining", **kwargs), ax=1)

    def trimReduction(self):
        self.figure.reset()
        if not self.experiment:
            QMessageBox.about(self.main.ui.centralwidget, "Info", "No data loaded")
            return False

        self.experiment.shift_to_start()
        self.update_plot()

    def save_reduction(self):
        if not self.experiment:
            QMessageBox.about(self.main.ui.centralwidget, "Info", "No data loaded")
            return False
        self.main.directExport(experiment=self.experiment)


    def reset(self):
        self.reduction_detected = False
        self.reference_loaded = False
        self.figure.reset()


class TabEvaluate():

    def __init__(self, main):
        self.main = main
        self.figure = Figure()
        self.main.ui.horizontalLayout_3.addWidget(self.figure.canvas)
        self.folder = None
        self.i = -1


class TabFiles:

    def __init__(self, main):
        self.main = main
        self.figure = Figure()
        self.main.ui.horizontalLayout_3.addWidget(self.figure.canvas)
        self.folder = None


    def openFolder(self):
        path = QFileDialog.getExistingDirectory()
        self.folder = TgaFolder(path+"/")
        trial_names = sorted(self.folder.trial_names())
        self.main.ui.filesListWidget.insertItems(0, trial_names)

    def investigate(self):
        trialname = self.main.ui.filesListWidget.currentItem().text()
        trial = self.folder.select(trialname)

        self.main.ui.tabWidget.setCurrentWidget(self.main.ui.trial_tab)
        self.main.tabTrial.loadFile(trial)
        ## load trial

    def exportAll(self):
        listWidget = self.main.ui.filesListWidget
        trial_list = []
        for x in  range(0, listWidget.count()-1):
            trial_list.append(listWidget.item(x).text())

        for trialname in trial_list:
            self.main.directExport(trialname=trialname)


    # def plot(self):
    #     if not self.folder:
    #         QMessageBox().about(self.main.ui.centralwidget, "Info", "No Folder loaded")
    #         return False
    #
    #     self.i += 1
    #     if self.i+1 >= len(self.folder.files):
    #         QMessageBox().about(self.main.ui.centralwidget, "Info", "No more files found")
    #         return False
    #
    #     experiment = ReductionTrial(self.folder.files[self.i])
    #
    #     self.figure.plot(*experiment.get("m", "t", cropped=False))
    #     self.figure.plot(*experiment.get("m", "t", cropped=True))

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    main_window = QtWidgets.QMainWindow()
    gui = TgaGui(main_window)
    main_window.show()
    sys.exit(app.exec_())
