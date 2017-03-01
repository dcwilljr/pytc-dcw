from qtpy.QtGui import *
from qtpy.QtCore import *
from qtpy.QtWidgets import *

import pytc
import inspect

class Sliders(QWidget):
	"""
	create sliders for an experiment
	"""

	def __init__(self, param_name, parent):
		super().__init__()

		self._exp = parent._exp
		self._param_name = param_name
		self._fitter = parent._fitter
		self._fit_run = parent._fit_run
		self._plot_frame = parent._plot_frame

		self.layout()

	def bounds(self):
		"""
		"""
		pass

	@property
	def name(self):
		"""
		"""
		return self._param_name


	def layout(self):
		"""
		"""
		self._main_layout = QGridLayout(self)
		self._main_layout.setVerticalSpacing(40)

		self._name_label = QLabel(self._param_name, self)
		self._main_layout.addWidget(self._name_label, 0, 0, 0, 2)

		self._fix = QCheckBox("Fix?", self)
		self._fix.toggle()
		self._fix.setChecked(False)
		self._fix.stateChanged.connect(self.fix_layout)
		self._main_layout.addWidget(self._fix, 1, 0)

		self._slider = QSlider(Qt.Horizontal)
		self._slider.sliderReleased.connect(self.update_val)
		self._main_layout.addWidget(self._slider, 1, 1)

		self._param_guess_label = QLabel("", self)
		self._main_layout.addWidget(self._param_guess_label, 1, 3)

		self.bounds()

		self._min = self._slider.minimum()
		self._max = self._slider.maximum()

		self._fix_int = QLineEdit(self)
		self._main_layout.addWidget(self._fix_int, 1, 4)
		self._fix_int.setText(str(1))
		self._fix_int.textChanged[str].connect(self.fix)
		self._fix_int.hide()

		self._update_min_label = QLabel("min: ", self)
		self._main_layout.addWidget(self._update_min_label, 1, 5)

		self._update_min = QLineEdit(self)
		self._main_layout.addWidget(self._update_min, 1, 6)
		self._update_min.textChanged[str].connect(self.min_bounds)

		self._update_max_label = QLabel("max: ", self)
		self._main_layout.addWidget(self._update_max_label, 1, 7)

		self._update_max = QLineEdit(self)
		self._main_layout.addWidget(self._update_max, 1, 8)
		self._update_max.textChanged[str].connect(self.max_bounds)

		# if parameter is K, be able to manually enter/change the value of the slider
		#if "K" in self._param_name:
		#	self._update_K = QLineEdit(self)
		#	self._update_K.textChanged[str].connect(self.slider_link)
		#	self._main_layout.addWidget(self._update_K, 1, 2)

	def slider_link(self, value):
		"""
		if K parameter, be able to update slider value with text box
		"""
		if "." not in value:
			new_val = int(value)
			self._slider.setValue(new_val)
			self._param_guess_label.setText(value)
			print(value)
		else:
			print("can't update slider to float")

	def check_if_fit(self):
		"""
		if a fit has been run, and a slider is changed, change all parameters back to guesses in slider widgets
		"""
		if self._fit_run:
			self._fitter.guess_to_value()
			self._fit_run = False

		print("fit has been run: " + str(self._fit_run))

	def fix_layout(self, state):
		"""
		initial parameter fix and updating whether slider/fixed int is hidden or shown
		"""
		if state == Qt.Checked:
			# change widget views
			self._fix_int.show()
			self._slider.hide()
			self._fitter.update_fixed(self._param_name, int(self._fix_int.text()), self._exp)
			self.check_if_fit()

			self._plot_frame.update()

			print(self._fix_int.text())
		else:
			#change widget views
			self._fix_int.hide()
			self._slider.show()

			self._fitter.update_fixed(self._param_name, None, self._exp)
			print('unfixed')

	def fix(self, value):
		"""
		update fixed value if changed
		"""
		if value:
			self._fitter.update_fixed(self._param_name, int(value), self._exp)
			self.check_if_fit()
			self._plot_frame.update()
		else:
			print("unabled to fix")

		print("fixed to value " + value)

	def update_val(self):
		"""
		update value for paremter based on slider value
		"""

		value = int(self._slider.value())

		# if guess update, update parameter as well for plot
		self._fitter.update_guess(self._param_name, value, self._exp)
		self._fitter.update_value(self._param_name, value, self._exp)
		self._param_guess_label.setText(str(value))

		self.check_if_fit()
		self._plot_frame.update()

	def min_bounds(self, value):
		"""
		update the minimum bounds
		"""
		try:
			self._min = int(value)
			self.update()
		except:
			pass

		print("min bound updated: " + value)

	def max_bounds(self, value):
		"""
		update maximum bounds
		"""
		try:
			self._max = int(value)
			self.update_bounds()
		except:
			pass

		print("max bound updated: " + value)

	def update_bounds(self):
		"""
		update min/max bounds and check if range needs to be updated as well
		"""
		pass