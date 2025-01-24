import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QDialog, QMessageBox
from PyQt5.QtCore import pyqtSignal
import dtw_slot


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('Audio Matching')
        self.setGeometry(1000, 500, 640, 480)

        # Create buttons
        self.start_button = QPushButton('Start', self)
        self.exit_button = QPushButton('Exit', self)

        # Set button actions
        self.start_button.clicked.connect(self.open_second_window)
        self.exit_button.clicked.connect(self.close_application)

        # Layout setup
        layout = QVBoxLayout()
        layout.addWidget(self.start_button)
        layout.addWidget(self.exit_button)
        self.setLayout(layout)

    def open_second_window(self):
        self.second_window = SecondWindow()
        self.second_window.exec_()

    def close_application(self):
        QApplication.quit()


class SecondWindow(QDialog):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('Processing... ')
        self.setGeometry(1000, 500, 640, 480)

        # Create button in the second window
        self.push_button = QPushButton('Click to Start Recording Templates', self)
        self.push_button.clicked.connect(self.template_record)

        # Layout setup for the second window
        layout = QVBoxLayout()
        layout.addWidget(self.push_button)
        self.setLayout(layout)

    def close_window(self):
        self.accept()

    def template_record(self):
        num_records = 10  # Number of recordings
        template_dir = '../templates/'

        for i in range(num_records):
            self.show_message(f"Recording {i + 1}/{num_records}...")
            recorded_audio = dtw_slot.record_audio()
            dtw_slot.save_audio(recorded_audio, i, template_dir)

        self.show_message("Recording complete! 10 files saved.")

        self.ask_to_record()

    def ask_to_record(self):
        """Show a new window reminding user to start recording sample voice"""
        reply = QMessageBox.question(self, 'Record Sample',
                                     'Now start recording the sample, do you want to proceed?',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.record_new_sample()
        else:
            self.close_window()

    def ask_to_record_again(self):
        """Show a new window asking if the user wants to record again."""
        reply = QMessageBox.question(self, 'Record Another Sample',
                                     'Do you want to record another sample and match it?',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.record_new_sample()
        else:
            self.close_window()

    def record_new_sample(self):
        """Allow the user to record another sample."""
        self.show_message("Recording new sample...")
        sample_dir='../sample/'

        # Record audio for 2 seconds
        recorded_audio = dtw_slot.record_audio()
        dtw_slot.save_audio(recorded_audio, 0, sample_dir)

        # Match the newly recorded sample
        dtw_slot.match_audio(recorded_audio)

        # return to asking
        self.ask_to_record_again()

    def show_message(self, message):
        """Utility function to display messages to the user."""
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Information")
        msg_box.setText(message)
        msg_box.setIcon(QMessageBox.Information)
        msg_box.exec_()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
