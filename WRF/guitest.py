import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLabel,
    QFileDialog, QCheckBox, QMessageBox)
import matplotlib.pyplot as plt
from wrf import (to_np,interplevel, getvar, get_cartopy, cartopy_xlim, cartopy_ylim, latlon_coords)
from netCDF4 import Dataset
import cartopy.crs as crs
import cartopy.feature as cfeature
from cartopy.feature import NaturalEarthFeature

class PlotApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Plot Selector")
        self.setGeometry(100, 100, 400, 300)

        # Layout and widgets
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # File selection
        self.label = QLabel("No file selected")
        self.layout.addWidget(self.label)

        self.file_button = QPushButton("Choose File")
        self.file_button.clicked.connect(self.choose_file)
        self.layout.addWidget(self.file_button)

        # Checkboxes
        self.options = ["Temperature", "Wind", "Precipitation"]
        self.checkboxes = []
        for opt in self.options:
            cb = QCheckBox(opt)
            self.layout.addWidget(cb)
            self.checkboxes.append(cb)

        # Plot button
        self.plot_button = QPushButton("Plot Selected")
        self.plot_button.clicked.connect(self.plot_selected)
        self.layout.addWidget(self.plot_button)

        self.file_path = None

    def choose_file(self):
        file_dialog = QFileDialog()
        path, _ = file_dialog.getOpenFileName(self, "Open File", "", "All Files (*)")
        if path:
            self.file_path = path
            self.label.setText(f"Selected: {path.split('/')[-1]}")

    def plot_selected(self):
        if not self.file_path:
            QMessageBox.warning(self, "Error", "Please select a file.")
            return

        selected = [cb.text() for cb in self.checkboxes if cb.isChecked()]
        if not selected:
            QMessageBox.warning(self, "Error", "Please select at least one variable to plot.")
            return

        self.plot_data(self.file_path, selected)

    def create_figure_for_var(self,file_path,opt):

        timeidx = 0

        with Dataset(file_path) as ds:
            mdbz = getvar(ds, "mdbz", timeidx=timeidx)
            #ua  = getvar(ds, "ua", units="kt", timeidx=timeidx)
            #va = getvar(ds, "va", units="kt",timeidx=timeidx)
            #p = getvar(ds, "pressure",timeidx=timeidx)
            #u_500 = interplevel(ua, p, 900)
            #v_500 = interplevel(va, p, 900)

        # Get the latitude and longitude points
        lats, lons = latlon_coords(mdbz)

        # Get the cartopy mapping object
        cart_proj = get_cartopy(mdbz)

        # Create a figure
        fig = plt.figure(figsize=(30,15))

        # Set the GeoAxes to the projection used by WRF
        ax = plt.axes(projection=cart_proj)

        # Download and add the states, lakes and coastlines
        states = NaturalEarthFeature(category="cultural", scale="50m", facecolor="none", name="admin_1_states_provinces")
        ax.add_feature(states, linewidth=.1, edgecolor="black")
        ax.add_feature(cfeature.LAKES.with_scale('50m'),linewidth=1, facecolor="none",  edgecolor="black")
        ax.coastlines('50m', linewidth=1)

        levels = [10, 15, 20, 25, 30, 35, 40, 45,50,55,60]


        # Make the filled countours with specified levels and range
        qcs = plt.contourf(to_np(lons), to_np(lats),mdbz,levels=levels,transform=crs.PlateCarree(),cmap="viridis")

        # Add a color bar
        cbar = plt.colorbar()
        cbar.set_label("dBZ",fontsize=10)

        # Set the map bounds
        ax.set_xlim(cartopy_xlim(mdbz))
        ax.set_ylim(cartopy_ylim(mdbz))

        # Add the gridlines
        gl = ax.gridlines(color="black", linestyle="dotted",draw_labels=True, x_inline=False, y_inline=False)
        gl.xlabel_style = {'rotation': 'horizontal','size': 14,'ha':'center'} # Change 14 to your desired font size
        gl.ylabel_style = {'size': 14}  # Change 14 to your desired font size
        gl.xlines = True
        gl.ylines = True

        gl.top_labels = False  # Disable top labels
        gl.right_labels = False  # Disable right labels
        gl.xpadding = 20

        plt.legend()
        plt.title("")
        plt.show()

    def plot_data(self, file_path, options):
        print(f"Plotting from: {file_path}")

        for opt in options:
            print(f"Plotting: {opt}")
            self.create_figure_for_var(file_path, opt)
                        
       

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PlotApp()
    window.show()
    sys.exit(app.exec_())
