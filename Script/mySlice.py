import numpy as np
import vedo

class Slicer3DPlotterx(vedo.plotter.Plotter):
    def __init__(
            self,
            volume,
            cmaps=("bone", "gist_ncar_r", "hot_r", "bone_r", "jet", "Spectral_r"),
            clamp=True,
            use_slider3d=False,
            show_histo=False,
            at=0,
            **kwargs,
    ):
        ################################
        super().__init__(**kwargs)
        self.at(at)
        ################################

        cx, cy, cz, ch = "dr", "dg", "db", (0.3, 0.3, 0.3)
        if np.sum(self.renderer.GetBackground()) < 1.5:
            cx, cy, cz = "lr", "lg", "lb"
            ch = (0.8, 0.8, 0.8)

        if len(self.renderers) > 1:
            # 2d sliders do not work with multiple renderers
            use_slider3d = True

        self.volume = volume
        box = volume.box().alpha(0.2)
        self.add(box)

        # inits
        la, ld = 0.7, 0.3  # ambient, diffuse
        dims = volume.dimensions()
        data = volume.pointdata[0]
        rmin, rmax = volume.scalar_range()
        if clamp:
            hdata, edg = np.histogram(data, bins=50)
            logdata = np.log(hdata + 1)
            # mean  of the logscale plot
            meanlog = np.sum(np.multiply(edg[:-1], logdata)) / np.sum(logdata)
            rmax = min(rmax, meanlog + (meanlog - rmin) * 0.9)
            rmin = max(rmin, meanlog - (rmax - meanlog) * 0.9)
            # print("scalar range clamped to range: ("
            #       + precision(rmin, 3) + ", " + precision(rmax, 3) + ")")

        self.cmap_slicer = cmaps[0]

        self.current_i = int(dims[0] / 2)

        self.xslice = None

        self.xslice = volume.xslice(self.current_i).lighting("", la, ld, 0)
        self.xslice.name = "XSlice"
        self.xslice.cmap(self.cmap_slicer, vmin=rmin, vmax=rmax)
        self.add(self.xslice)

        #################
        def slider_function_x(widget, event):
            i = int(self.xslider.value)
            if i == self.current_i:
                return
            self.current_i = i
            self.xslice = volume.xslice(i).lighting("", la, ld, 0)
            self.xslice.cmap(self.cmap_slicer, vmin=rmin, vmax=rmax)
            self.xslice.name = "XSlice"
            self.remove("XSlice")  # removes the old one
            if 0 < i < dims[0]-1:
                self.add(self.xslice)
            self.render()

        if not use_slider3d:
            self.xslider = self.add_slider(
                slider_function_x,
                0,
                dims[0],
                title="",
                title_size=0.5,
                pos=[(0.8, 0.12), (0.95, 0.12)],
                value=int(dims[0] / 2),
                show_value=True,
                c=cx,
            )

        #################
        def button_func(obj, ename):
            bu.switch()
            self.cmap_slicer = bu.status()
            for m in self.objects:
                if "Slice" in m.name:
                    m.cmap(self.cmap_slicer, vmin=rmin, vmax=rmax)#改了颜色之后可以3D出现感兴趣区域

        if len(cmaps) > 1:
            bu = self.add_button(
                button_func,
                states=cmaps,
                c=["k9"] * len(cmaps),
                bc=["k1"] * len(cmaps),  # colors of states
                size=16,
                bold=True,
            )
            bu.pos([0.04, 0.01], "bottom-left")

class Slicer3DPlottery(vedo.plotter.Plotter):
    def __init__(
            self,
            volume,
            cmaps=("bone","gist_ncar_r", "hot_r",  "bone_r", "jet", "Spectral_r"),
            clamp=True,
            use_slider3d=False,
            at=0,
            **kwargs,
    ):
        ################################
        super().__init__(**kwargs)
        self.at(at)
        ################################

        cx, cy, cz, ch = "dr", "dg", "db", (0.3, 0.3, 0.3)
        if np.sum(self.renderer.GetBackground()) < 1.5:
            cx, cy, cz = "lr", "lg", "lb"
            ch = (0.8, 0.8, 0.8)

        if len(self.renderers) > 1:
            # 2d sliders do not work with multiple renderers
            use_slider3d = True

        self.volume = volume
        box = volume.box().alpha(0.2)
        self.add(box)

        # inits
        la, ld = 0.7, 0.3  # ambient, diffuse
        dims = volume.dimensions()
        data = volume.pointdata[0]
        rmin, rmax = volume.scalar_range()
        if clamp:
            hdata, edg = np.histogram(data, bins=50)
            logdata = np.log(hdata + 1)
            # mean  of the logscale plot
            meanlog = np.sum(np.multiply(edg[:-1], logdata)) / np.sum(logdata)
            rmax = min(rmax, meanlog + (meanlog - rmin) * 0.9)
            rmin = max(rmin, meanlog - (rmax - meanlog) * 0.9)
            # print("scalar range clamped to range: ("
            #       + precision(rmin, 3) + ", " + precision(rmax, 3) + ")")

        self.cmap_slicer = cmaps[0]

        self.current_j = int(dims[1] / 2)

        self.yslice = None

        self.yslice = volume.yslice(self.current_j).lighting("", la, ld, 0)
        self.yslice.name = "YSlice"
        self.yslice.cmap(self.cmap_slicer, vmin=rmin, vmax=rmax)
        self.add(self.yslice)

        data_reduced = data

        #################
        def slider_function_y(widget, event):
            j = int(self.yslider.value)
            if j == self.current_j:
                return
            self.current_j = j
            self.yslice = volume.yslice(j).lighting("", la, ld, 0)
            self.yslice.cmap(self.cmap_slicer, vmin=rmin, vmax=rmax)
            self.yslice.name = "YSlice"
            self.remove("YSlice")
            if 0 < j < dims[1]:
                self.add(self.yslice)
            self.render()

        if not use_slider3d:
            self.yslider = self.add_slider(
                slider_function_y,
                0,
                dims[1],
                title="",
                title_size=0.5,
                value=int(dims[1] / 2),
                pos=[(0.8, 0.08), (0.95, 0.08)],
                show_value=True,
                c=cy,
            )
        #################
        def button_func(obj, ename):
            bu.switch()
            self.cmap_slicer = bu.status()
            for m in self.objects:
                if "Slice" in m.name:
                    m.cmap(self.cmap_slicer, vmin=rmin, vmax=rmax)

        if len(cmaps) > 1:
            bu = self.add_button(
                button_func,
                states=cmaps,
                c=["k9"] * len(cmaps),
                bc=["k1"] * len(cmaps),  # colors of states
                size=16,
                bold=True,
            )
            bu.pos([0.04, 0.01], "bottom-left")

class Slicer3DPlotterz(vedo.plotter.Plotter):
    def __init__(
            self,
            volume,
            cmaps=( "bone","gist_ncar_r", "hot_r", "bone_r", "jet", "Spectral_r"),
            clamp=True,
            use_slider3d=False,
            show_histo=False,
            at=0,
            **kwargs,
    ):
        ################################
        super().__init__(**kwargs)
        self.at(at)
        ################################

        cx, cy, cz, ch = "dr", "dg", "db", (0.3, 0.3, 0.3)
        if np.sum(self.renderer.GetBackground()) < 1.5:
            cx, cy, cz = "lr", "lg", "lb"
            ch = (0.8, 0.8, 0.8)

        if len(self.renderers) > 1:
            # 2d sliders do not work with multiple renderers
            use_slider3d = True

        self.volume = volume
        box = volume.box().alpha(0.2)
        self.add(box)

        # inits
        la, ld = 0.7, 0.3  # ambient, diffuse
        dims = volume.dimensions()
        data = volume.pointdata[0]
        rmin, rmax = volume.scalar_range()
        if clamp:
            hdata, edg = np.histogram(data, bins=50)
            logdata = np.log(hdata + 1)
            # mean  of the logscale plot
            meanlog = np.sum(np.multiply(edg[:-1], logdata)) / np.sum(logdata)
            rmax = min(rmax, meanlog + (meanlog - rmin) * 0.9)
            rmin = max(rmin, meanlog - (rmax - meanlog) * 0.9)
            # print("scalar range clamped to range: ("
            #       + precision(rmin, 3) + ", " + precision(rmax, 3) + ")")

        self.cmap_slicer = cmaps[0]

        self.current_k = int(dims[2] / 2)

        self.zslice = None

        self.zslice = volume.zslice(self.current_k).lighting("", la, ld, 0)
        self.zslice.name = "ZSlice"
        self.zslice.cmap(self.cmap_slicer, vmin=rmin, vmax=rmax)
        self.add(self.zslice)

        data_reduced = data

        #################
        def slider_function_z(widget, event):
            k = int(self.zslider.value)
            if k == self.current_k:
                return
            self.current_k = k
            self.zslice = volume.zslice(k).lighting("", la, ld, 0)
            self.zslice.cmap(self.cmap_slicer, vmin=rmin, vmax=rmax)
            self.zslice.name = "ZSlice"
            self.remove("ZSlice")
            if 0 < k < dims[2]:
                self.add(self.zslice)
            self.render()

        if not use_slider3d:
            self.zslider = self.add_slider(
                slider_function_z,
                0,
                dims[2],
                title="",
                title_size=0.6,
                value=int(dims[2] / 2),
                pos=[(0.8, 0.04), (0.95, 0.04)],
                show_value=True,
                c=cz,
            )

            #################
            def button_func(obj, ename):
                bu.switch()
                self.cmap_slicer = bu.status()
                for m in self.objects:
                    if "Slice" in m.name:
                        m.cmap(self.cmap_slicer, vmin=rmin, vmax=rmax)

            if len(cmaps) > 1:
                bu = self.add_button(
                    button_func,
                    states=cmaps,
                    c=["k9"] * len(cmaps),
                    bc=["k1"] * len(cmaps),  # colors of states
                    size=16,
                    bold=True,
                )
                bu.pos([0.04, 0.01], "bottom-left")