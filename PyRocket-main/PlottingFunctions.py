#####################################################################
#                           PyRocket								#
# 2D Regenertive Cooling Simulation for Bipropellant Rocket Engines #
#                                                                   #
# Creator:  Joanthan Neeser                                         #
# Date:     15.12.2022                                              #
# Version:  2.3  													#
# License:	GNU GENERAL PUBLIC LICENSE V3			                #	 
#                                                                   #
#####################################################################

import matplotlib.pyplot as plt
import numpy as np

import config
from StressAnalysis import REGIMES, bree_regime

class Plotting1D():
    def __init__(self, save_path, save=True, show=False):
        self.save_path = save_path
        self.save = save
        self.show = show
        self.data = np.genfromtxt(save_path+'/sim_data.csv', delimiter=",", skip_header=1)


    def temperature_plot(self):
        plt.plot(config.geometry[:,0]*1e3, self.data[:,7]-273.15, color='b', label='Coolant bulk temperature')
        plt.plot(config.geometry[:,0]*1e3, self.data[:,6]-273.15, color='r', linestyle='--', label='Inner wall temperature')
        if config.boiling and self.data.shape[1] > 12:
            plt.plot(config.geometry[:,0]*1e3, self.data[:,12]-273.15, color='g', linestyle='-.', label='Cooling channel wall temperature')
            plt.plot(config.geometry[:,0]*1e3, self.data[:,13]-273.15, color='k', linestyle=':', label='Coolant saturation temperature')
            # sections above the critical heat flux are solved without nucleate boiling, the wall temperatures there are not a prediction
            x   = config.geometry[:,0]*1e3
            dx  = x[1] - x[0]
            chf = np.where((self.data[:,14] > self.data[:,15]) & (self.data[:,12] > self.data[:,13]))[0]
            for i in chf:
                plt.axvspan(x[i] - dx/2, x[i] + dx/2, color='r', alpha=0.15, linewidth=0, label='CHF exceeded, solved without boiling' if i == chf[0] else None)
        plt.xlabel('x coordinate [mm]')
        plt.ylabel('Temperature [$^\circ$C]')
        plt.legend(loc='best')

        if self.save:
            plt.savefig(self.save_path + '/temperature', dpi=600)
            plt.close()
            
        if self.show:
            plt.show()
        

    def pressure_plot(self):
        plt.plot(config.geometry[:,0]*1e3, self.data[:,8]/1e5, color='b', label='Coolant pressure')
        plt.xlabel('x coordinate [mm]')
        plt.ylabel('Pressure [bar]')
        plt.legend(loc='best')

        if self.save:
            plt.savefig(self.save_path + '/pressure', dpi=600)
            plt.close()
        
        if self.show:
            plt.show()
        

    def heat_transfer_coeff_plot(self):
        plt.plot(config.geometry[:,0]*1e3, self.data[:,2], color='b', label='$h_{alpha}$ gas')
        plt.plot(config.geometry[:,0]*1e3, self.data[:,3], color='r', linestyle='--',  label='$h_{alpha}$ coolant')
        plt.xlabel('x coordinate [mm]')
        plt.ylabel('Heat transfer coefficient [W/m^2/K]')
        plt.legend(loc='best')
        
        if self.save:
            plt.savefig(self.save_path + '/halpha', dpi=600)
            plt.close()

        if self.show:
            plt.show()
    

    def heat_flux_plot(self):
        plt.plot(config.geometry[:,0]*1e3, self.data[:,4]/1e6, color='b', label='q radiation')
        plt.plot(config.geometry[:,0]*1e3, self.data[:,5]/1e6, color='r', linestyle='--', label='q total')
        plt.xlabel('x coordinate [mm]')
        plt.ylabel('Heat flux [MW/m^2]')
        plt.legend(loc='best')
        
        if self.save:
            plt.savefig(self.save_path + '/heat_flux', dpi=600)
            plt.close()
        
        if self.show:
            plt.show()
    

    def chf_plot(self):
        plt.plot(config.geometry[:,0]*1e3, self.data[:,14]/1e6, color='b', label='Highest heat flux at the channel wall')
        plt.plot(config.geometry[:,0]*1e3, self.data[:,15]/1e6, color='r', linestyle='--', label='Critical heat flux')
        plt.plot(config.geometry[:,0]*1e3, self.data[:,15]/1e6 * 1.5, color='g', linestyle='--', label='1.5x Critical heat flux')
        plt.xlabel('x coordinate [mm]')
        plt.ylabel('Heat flux [MW/m^2]')
        plt.legend(loc='best')

        if self.save:
            plt.savefig(self.save_path + '/chf', dpi=600)
            plt.close()

        if self.show:
            plt.show()


    def wall_delta_plot(self):
        # temperature differences in the wall which drive the thermal stresses
        if self.data.shape[1] <= 19:
            return

        plt.plot(config.geometry[:,0]*1e3, self.data[:,16], color='r', label='Through the inner wall (channel centre line)')
        plt.plot(config.geometry[:,0]*1e3, self.data[:,17]-self.data[:,19], color='b', linestyle='--', label='Inner wall mean - outer wall mean')
        plt.plot(config.geometry[:,0]*1e3, self.data[:,18]-self.data[:,19], color='g', linestyle='-.', label='Rib mean - outer wall mean')
        plt.xlabel('x coordinate [mm]')
        plt.ylabel('Temperature difference [K]')
        plt.legend(loc='upper left')

        if self.save:
            plt.savefig(self.save_path + '/wall_delta_T', dpi=600)
            plt.close()

        if self.show:
            plt.show()


    def stress_plot(self):
        # elastic wall stresses of the stress analysis compared to the yield strength at wall temperature
        if self.data.shape[1] <= 29 or np.all(np.isnan(self.data[:,26])):
            return

        x = config.geometry[:,0]*1e3
        plt.plot(x, self.data[:,26]/1e6, color='k', label='Yield strength at wall temperature')
        plt.plot(x, self.data[:,25]/1e6, color='r', linestyle='--', label='Secondary stress (temperature), hot gas side')
        plt.plot(x, self.data[:,22]/1e6, color='orange', linestyle=':', label='of which temperature gradient through the inner wall')
        plt.plot(x, (self.data[:,20]+self.data[:,21])/1e6, color='b', linestyle='-.', label='Primary stress (pressure)')
        plt.xlabel('x coordinate [mm]')
        plt.ylabel('Stress [MPa]')
        plt.ylim(0, 1.45 * np.nanmax(self.data[:,[25, 26]]) / 1e6)      # room for the legend above the curves
        plt.legend(loc='upper left')

        if self.save:
            plt.savefig(self.save_path + '/stress', dpi=600)
            plt.close()

        if self.show:
            plt.show()


    def bree_plot(self):
        # Bree diagram with one point for every section of the chamber, and the same data along the chamber contour
        if self.data.shape[1] <= 29 or np.all(np.isnan(self.data[:,27])):
            return

        x      = config.geometry[:,0]*1e3
        X, Y   = self.data[:,27], self.data[:,28]
        regime = [bree_regime(X[i], Y[i]) for i in range(len(x))]
        Y_max  = max(4.5, 1.15 * np.nanmax(Y))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

        # boundaries between the regimes
        hyperbola = np.linspace(1 / Y_max, 0.5, 100)
        ax1.plot([0, 1], [1, 0], color='k', linewidth=1)
        ax1.plot([0, 0.5], [2, 2], color='k', linewidth=1)
        ax1.plot([0.5, 1], [2, 0], color='k', linewidth=1)
        ax1.plot(hyperbola, 1 / hyperbola, color='k', linewidth=1)
        ax1.plot([1, 1], [0, Y_max], color='k', linewidth=1)
        ax1.text(0.45, 0.22, 'E\nelastic', ha='center', va='center', color='dimgray')
        ax1.text(0.5, 1.25, 'S\nshakedown', ha='center', va='center', color='dimgray')
        ax1.text(0.12, 2.0 + 0.3 * (Y_max - 2), 'P\nplastic\ncycling', ha='center', va='center', color='dimgray')
        ax1.text(0.72, 2.0 + 0.3 * (Y_max - 2), 'R\nratcheting', ha='center', va='center', color='dimgray')

        # sections of the chamber, connected in the order of the x coordinate
        ax1.plot(X, Y, color='gray', linewidth=0.8, zorder=2)
        points = ax1.scatter(X, Y, c=x, cmap='viridis', s=28, edgecolors='w', linewidths=0.5, zorder=3)
        fig.colorbar(points, ax=ax1, label='x coordinate [mm]')
        for i, name in ((0, 'injector'), (int(np.argmin(config.geometry[:,1])), 'throat'), (len(x) - 1, 'exit')):
            ax1.annotate(name, (X[i], Y[i]), xytext=(40, 6), textcoords='offset points', fontsize=9, arrowprops=dict(arrowstyle='-', color='gray', linewidth=0.8))
        ax1.set_xlim(0, 1.1)
        ax1.set_ylim(0, Y_max)
        ax1.set_xlabel('X = primary stress / yield strength')
        ax1.set_ylabel('Y = secondary stress / yield strength')
        ax1.set_title('Bree diagram')

        # X and Y along the chamber contour, the background shows the regime of every section
        dx = x[1] - x[0]
        for key, color in (('E', 'g'), ('S', 'b'), ('P', 'orange'), ('R', 'r'), ('C', 'k')):
            sections = [i for i in range(len(x)) if regime[i] == key]
            for i in sections:
                ax2.axvspan(x[i] - dx/2, x[i] + dx/2, color=color, alpha=0.15, linewidth=0, label=key + ': ' + REGIMES[key] if i == sections[0] else None)
        ax2.plot(x, Y, color='r', label='Y (secondary stress)')
        ax2.plot(x, X, color='b', linestyle='--', label='X (primary stress)')
        ax2.set_ylim(bottom=0)
        ax2.set_xlabel('x coordinate [mm]')
        ax2.set_ylabel('Stress / yield strength')
        ax2.set_title('Along the chamber')
        ax2.legend(loc='best')

        fig.tight_layout()

        if self.save:
            plt.savefig(self.save_path + '/bree', dpi=300)
            plt.close()

        if self.show:
            plt.show()


    def reynolds_plot(self):
        plt.plot(config.geometry[:,0]*1e3, self.data[:,9], color='b', label='Coolant Re')
        plt.xlabel('x coordinate [mm]')
        plt.ylabel('Re')
        plt.legend(loc='best')

        if self.save:
            plt.savefig(self.save_path + '/reynolds', dpi=600)
            plt.close()
    
        if self.show:
            plt.show()
        

def multi_plot(data1, data2, data3, data4, label1, label2, label3, label4):
    f, axes = plt.subplots(4, 1)
    #f.subplots_adjust(hspace = 0)

    axes[0].plot(config.geometry[:,0], data1)
    axes[0].set_ylabel(label1)

    axes[1].plot(config.geometry[:,0], data2)
    axes[1].set_ylabel(label2)

    axes[2].plot(config.geometry[:,0], data3)
    axes[2].set_ylabel(label3)

    axes[3].plot(config.geometry[:,0], data4)
    axes[3].set_ylabel(label4)

    plt.xlabel('x coordinate [m]')
    plt.show()


if __name__ == "__main__":
    plot = Plotting1D(save_path=config.save_path, save=True, show=True)
    plot.temperature_plot()
    plot.pressure_plot()
    plot.heat_transfer_coeff_plot()
    plot.heat_flux_plot()
    plot.reynolds_plot()
    plot.chf_plot()
    plot.wall_delta_plot()
    plot.stress_plot()
    plot.bree_plot()
