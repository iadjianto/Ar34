import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sympy.solvers import solve
from sympy import Symbol

abso_path = "C:/Users/Isaiah/Documents/Python Scripts/"
name = "drift_pid_0.txt"

# Script is meant to onserve effects of detector energy drifting over time/runs and fit coefficients to correct energy drift.

def thefunc (x,a0,a1,a2,a3,a4,a5):
	return a0 + a1*x + a2*x**2 + a3*x**3 + a4*x**4 + a5*x**5

class coefficients: #used to calculate coefficients to modify energy based on peaks fitted in C++/ROOT

	def __init__(self,fitted,reffed):
		self.x = fitted
		self.y = reffed 

		self.coeff,self.coeff_cov = curve_fit(thefunc,self.x,self.y) # (f(x),x,y) # Find coefficients to calibrate peak energies relative to referenced energies

	def plot(self): #Plot fitted peaks
		domain = np.linspace(5000,0,5000)

		print(self.coeff)

		fig = plt.figure()
		fig.suptitle('Fitting Peaks', fontsize=14, fontweight='bold')

		ax = fig.add_subplot(111)
		fig.subplots_adjust(top=0.85)
		# ax.set_title('axes title')

		ax.set_xlabel(r"Off")
		ax.set_ylabel(r"ref")

		plt.plot(self.x,self.y,linestyle="",marker="o")
		plt.plot(domain,thefunc(domain,*self.coeff))
		plt.show()

	def write_coeff(self,path,run,detector = 16): # Fitted peaks will be written to a lookup table along with their coefficients.
		# print("{}".format(int(run)))
		with open(path+"LUT_GALILEO_{0}.dat".format(int(run)),"w") as new_file:
			with open(path+"LUT_GALILEO.dat","r") as old_file:
				for line in old_file:
					if line.split(" ")[0] == str(detector):
						edit_line = line
						for i in range(6):
							edit_line = edit_line.replace\
								(line.split("  ")[i-6], str(self.coeff[i-6]))
						new_file.write(edit_line+"\n")
					else:
						new_file.write(line)


class data: # Used to show fitted peaks and energy calibrated peaks against run number

	def __init__(self,path):
		self.run = []
		self.peaks = []
		self.list = np.loadtxt(path, delimiter='\t')
		self.run = self.list[:,0]
		self.peaks = self.list[:,1:].transpose(1,0)

	def plot(self,i,title='Drift of Peak Over Runs'):
		fig = plt.figure()
		fig.suptitle(title, fontsize=14, fontweight='bold')

		ax = fig.add_subplot(111)
		fig.subplots_adjust(top=0.85)
		# ax.set_title('axes title')

		ax.set_xlabel(r"Runs")
		ax.set_ylabel(r"Energy")
		plt.plot(self.run,self.peaks[i],marker="")
		ax.legend()

		plt.show()

	def plot_all(self,title='Drift of Peaks Over Runs'):
		fig = plt.figure()
		fig.suptitle(title, fontsize=14, fontweight='bold')

		ax = fig.add_subplot(111)
		fig.subplots_adjust(top=0.85)
		# ax.set_title('axes title')

		ax.set_xlabel(r"Runs")
		ax.set_ylabel(r"Energy")
		for i in range(len(self.peaks)):
			plt.plot(self.run,self.peaks[i],marker="")
			# plt.plot(self.run,np.array(self.peaks[i])\
				# -np.average(self.peaks[i]),marker="")
		ax.legend()

		plt.show()

	def peaks_in_run(self,run_wanted):
		self.runpeak = (self.peaks.transpose()[i] for i in \
			range(len(self.run)) if d.run[i] in [run_wanted])
		return next(self.runpeak)

# ref_peak = [2127,491,1001,1066,1177,\
# 	1266,1320,2229,2561,3306]

ref_peak = [461,491,511,572,666,709,841,879,967,1001,1066,1137\
	,1177,1266,1320,1779,1935,1967,2139,2230,2561,2838,3040,3308]

# ref_peak = [491,2561]

d = data(abso_path + name)

# d.plot(-1)
d.plot_all()
# runpeaks = d.peaks_in_run(1351)

# c = coefficients([0,*runpeaks],[0,*ref_peak])
# c.plot()
# print(c.coeff)

# c.write_coeff(abso_path,1351)

list_coeff = []

for i in range(len(d.run)):
	peaks_run = coefficients([0,*d.peaks_in_run(d.run[i])],[0,*ref_peak])
	list_coeff.append(peaks_run.coeff)
	# peaks_run.plot()
	# peaks_run.write_coeff(abso_path,d.run[i])

fig = plt.figure()
fig.suptitle("Coefficients", fontsize=14, fontweight='bold')

ax = fig.add_subplot(111)
fig.subplots_adjust(top=0.85)
# ax.set_title('axes title')

ax.set_xlabel(r"Runs")
ax.set_ylabel(r"log(coeff)")

# plt.plot(d.run,np.array(list_coeff).transpose()[1],marker="")
for i in range(len(np.array(list_coeff).transpose())):
	plt.plot(d.run,np.log10(np.abs(np.array(list_coeff).transpose()[i])),marker="")
ax.legend()

plt.show()
