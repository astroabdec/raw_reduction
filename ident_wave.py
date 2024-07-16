#!/usr/bin/env python3
import os
import re
import sys
import math
import itertools

import numpy as np
import astropy.io.fits as fits
import scipy.interpolate as intp
import scipy.optimize as opt


import matplotlib
matplotlib.use('TkAgg')
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
from matplotlib.figure import Figure

from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)
from matplotlib.backends.backend_agg import FigureCanvasAgg

if sys.version_info[0] < 3:
    import Tkinter as tk
    import ttk
else:
    import tkinter as tk
    import tkinter.ttk as ttk

def search_linelist(linelistname):
    """Search the line list file and load the list.

    Args:
        linelistname (str): Name of the line list file.

    Returns:
        *string*: Path to the line list file
    """

    # first, seach $LINELIST in current working directory
    if os.path.exists(linelistname):
        return linelistname

    # seach $LINELIST.dat in current working directory
    newname = linelistname+'.dat'
    if os.path.exists(newname):
        return newname

    return None

def gaussian(A,center,fwhm,x):
    sigma = fwhm/2.35482
    return A*np.exp(-(x-center)**2/2./sigma**2)
def errfunc(p,x,y):
    return y - gaussian(p[0],p[1],p[2],x)

def gaussian_bkg(A,center,fwhm,bkg,x):
    sigma = fwhm/2.35482
    return bkg + A*np.exp(-(x-center)**2/2./sigma**2)
def errfunc2(p,x,y):
    return y - gaussian_bkg(p[0],p[1],p[2],p[3],x)


def load_linelist(filename):
    """Load standard wavelength line list from a given file.

    Args:
        filename (str): Name of the wavelength standard list file.

    Returns:
        *list*: A list containing (wavelength, species).
    """
    linelist = []
    infile = open(filename)
    for row in infile:
        row = row.strip()
        if len(row)==0 or row[0] in '#%!@':
            continue
        g = row.split()
        wl = float(g[0])
        if len(g)>1:
            species = g[1]
        else:
            species = ''
        linelist.append((wl, species))
    infile.close()
    return linelist


def auto_line_fitting_filter(param, i1, i2):
    """A filter function for fitting of a single calibration line.

    Args:
        param ():
        i1 (int):
        i2 (int):

    Return:
        bool:
    """
    if param[0] <= 0.:
        # line amplitdue too small
        return False
    if param[1] < i1 or param[1] > i2:
        # line center not in the fitting range (i1, i2)
        return False
    if param[2] > 50. or param[2] < 1.0:
        # line too broad or too narrow
        return False
    if param[3] < -0.5*param[0]:
        # background too low
        return False
    return True

def find_local_peak(flux, x, width, figname=None):
    """Find the central pixel of an emission line.

    Args:
        flux (:class:`numpy.ndarray`): Flux array.
        x (int): The approximate coordinate of the peak pixel.
        width (int): Window of profile fitting.

    Returns:
        tuple: A tuple containing:

            * **i1** (*int*) -- Index of the left side.
            * **i2** (*int*) -- Index of the right side.
            * **p1** (*list*) -- List of fitting parameters.
            * **std** (*float*) -- Standard deviation of the fitting.
    """
    width = int(round(width))
    if width%2 != 1:
        width += 1
    half = int((width-1)/2)

    i = int(round(x))

    # find the peak in a narrow range

    i1, i2 = max(0, i-half), min(flux.size, i+half+1)

    if i2 - i1 <= 4:
        # 4 is the number of free parameters in fitting function
        return None

    # find the peak position
    imax = flux[i1:i2].argmax() + i1
    xdata = np.arange(i1,i2)
    ydata = flux[i1:i2]
    # determine the initial parameters for gaussian fitting + background
    p0 = [ydata.max()-ydata.min(), imax, 3., ydata.min()]
    # least square fitting
    #p1,succ = opt.leastsq(errfunc2, p0[:], args=(xdata,ydata))
    p1, cov, info, mesg, ier = opt.leastsq(errfunc2, p0[:],
                                    args=(xdata, ydata), full_output=True)

    res_lst = errfunc2(p1, xdata, ydata)

    if res_lst.size-len(p0)-1 == 0:
        return None

    std = math.sqrt((res_lst**2).sum()/(res_lst.size-len(p0)-1))

    if figname is not None:
        fig = plt.figure()
        ax1 = fig.add_axes([0.1, 0.4, 0.8, 0.5])
        ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.25])
        ax1.plot(xdata, ydata, 'o', ms=4)
        newx = np.arange(xdata[0], xdata[-1], 0.1)
        newy = gaussian_bkg(p1[0], p1[1], p1[2], p1[3], newx)
        ax1.plot(newx, newy, '-', lw=0.6)
        yerr = errfunc2(p1, xdata, ydata)
        ax2.plot(xdata, yerr, 'o', ms=4)
        ax1.set_xlim(xdata[0], xdata[-1])
        ax2.set_xlim(xdata[0], xdata[-1])
        fig.savefig(figname)
        plt.close(fig)

    return i1, i2, p1, std



def save_ident(identlist, coeff, filename):
    """Write the ident line list and coefficients into an ASCII file.
    The existing informations in the ASCII file will not be affected.
    Only the input channel will be overwritten.

    Args:
        identlist (dict): Dict of identified lines.
        coeff (:class:`numpy.ndarray`): Coefficient array.
        result (dict): A dict containing identification results.
        filename (str): Name of the ASCII file.

    See also:
        :func:`load_ident`
    """
    outfile = open(filename, 'w')

    # write identified lines
    fmtstr = ('LINE {:03d} {:10.4f} {:10.4f} {:10.4f} {:12.5e} {:9.5f}'
             '{:1d} {:+10.6f} {:1s}')
    for aper, list1 in sorted(identlist.items()):
        for row in list1:
            pix    = row['pixel']
            wav    = row['wavelength']
            amp    = row['amplitude']
            fwhm   = row['fwhm']
            mask   = int(row['mask'])
            res    = row['residual']
            method = row['method'].decode('ascii')
            outfile.write(fmtstr.format(aper, pix, wav, amp, fwhm,
                mask, res, method)+os.linesep)

    # write coefficients
    for irow in range(coeff.shape[0]):
        string = ' '.join(['{:18.10e}'.format(v) for v in coeff[irow]])
        outfile.write('COEFF {}'.format(string)+os.linesep)

    outfile.close()


def load_ident(filename):
    """Load identified line list from an ASCII file.

    Args:
        filename (str): Name of the identification file.

    Returns:
        tuple: A tuple containing:

            * **identlist** (*dict*) -- Identified lines for all orders.
            * **coeff** (:class:`numpy.ndarray`) -- Coefficients of wavelengths.

    See also:
        :func:`save_ident`
    """
    identlist = {}
    coeff = []

    infile = open(filename)
    for row in infile:
        row = row.strip()
        if len(row)==0 or row[0] in '#$%^@!':
            continue
        g = row.split()

        key = g[0]
        if key == 'LINE':
            aperture    = int(g[1])
            pixel       = float(g[2])
            wavelength  = float(g[3])
            amplitude   = float(g[4])
            fwhm        = float(g[5])
            mask        = bool(g[6])
            residual    = float(g[7])
            method      = g[8].strip()

            item = np.array((aperture, 0, pixel, wavelength, amplitdue, fwhm,
                    0., mask, residual, method),dtype=identlinetype)

            if aperture not in identlist:
                identlist[aperture] = []
            identlist[aperture].append(item)

        elif key == 'COEFF':
            coeff.append([float(v) for v in g[2:]])

        else:
            pass

    infile.close()

    # convert list of every order to numpy structured array
    for aperture, list1 in identlist.items():
        identlist[aperture] = np.array(list1, dtype=identlinetype)

    # convert coeff to numpy array
    coeff = np.array(coeff)

    return identlist, coeff


def get_identlinetype():
    types = [
            ('aperture',    np.int16),
            ('order',       np.int16),
            ('wavelength',  np.float64),
            ('i1',          np.int16),
            ('i2',          np.int16),
            ('pixel',       np.float32),
            ('amplitude',   np.float32),
            ('fwhm',        np.float32),
            ('background',  np.float32),
            ('q',           np.float32),
            ('mask',        np.int16),
            ('residual',    np.float64),
            ('method',      'S1'),
            ]
    names, formats = list(zip(*types))
    return np.dtype({'names': names, 'formats': formats})

identlinetype = get_identlinetype()

def fit_wavelength(identlist, npixel, xorder, yorder, maxiter, clipping,
        fit_filter=None):
    """Fit the wavelength using 2-D polynomial.

    Args:
        identlist (dict): Dict of identification lines for different apertures.
        npixel (int): Number of pixels for each order.
        xorder (int): Order of polynomial along X direction.
        yorder (int): Order of polynomial along Y direction.
        maxiter (int): Maximim number of iterations in the polynomial
            fitting.
        clipping (float): Threshold of sigma-clipping.
        fit_filter (function): Function checking if a pixel/oder combination is
            within the accepted range.

    Returns:
        tuple: A tuple containing:

            * **coeff** (:class:`numpy.ndarray`) -- Coefficients array.
            * **std** (*float*) -- Standard deviation.
            * **k** (*int*) -- *k* in the relationship between aperture
              numbers and diffraction orders: `order = k*aperture + offset`.
            * **offset** (*int*) -- *offset* in the relationship between
              aperture numbers and diffraction orders: `order = k*aperture +
              offset`.
            * **nuse** (*int*) -- Number of lines used in the fitting.
            * **ntot** (*int*) -- Number of lines found.

    See also:
        :func:`get_wavelength`
    """
    # find physical order
    k, offset = find_order(identlist, npixel)

    # parse the fit_filter=None
    if fit_filter is None:
        fit_filter = lambda item: True

    # convert indent_line_lst into fitting inputs
    fit_p_lst = []  # normalized pixel
    fit_o_lst = []  # diffraction order
    fit_w_lst = []  # order*wavelength
    fit_m_lst = []  # initial mask
    # the following list is used to find the position (aperture, no)
    # of each line
    lineid_lst = []
    for aperture, list1 in sorted(identlist.items()):
        order = k*aperture + offset
        #norm_order = 50./order
        #norm_order = order/50.
        list1['order'][:] = order
        for iline, item in enumerate(list1):
            norm_pixel = item['pixel']*2/(npixel-1) - 1
            fit_p_lst.append(norm_pixel)
            fit_o_lst.append(order)
            #fit_o_lst.append(norm_order)
            #fit_w_lst.append(item['wavelength'])
            fit_w_lst.append(item['wavelength']*order)
            fit_m_lst.append(fit_filter(item))
            lineid_lst.append((aperture, iline))
    fit_p_lst = np.array(fit_p_lst)
    fit_o_lst = np.array(fit_o_lst)
    fit_w_lst = np.array(fit_w_lst)
    fit_m_lst = np.array(fit_m_lst)

    mask = fit_m_lst

    for nite in range(maxiter):
        coeff = polyfit2d(fit_p_lst[mask], fit_o_lst[mask], fit_w_lst[mask],
                          xorder=xorder, yorder=yorder)
        res_lst = fit_w_lst - polyval2d(fit_p_lst, fit_o_lst, coeff)
        res_lst = res_lst/fit_o_lst

        mean = res_lst[mask].mean(dtype=np.float64)
        std  = res_lst[mask].std(dtype=np.float64)
        m1 = res_lst > mean - clipping*std
        m2 = res_lst < mean + clipping*std
        new_mask = m1*m2*mask
        if new_mask.sum() == mask.sum():
            break
        else:
            mask = new_mask

    # convert mask back to ident_line_lst
    for lineid, ma, res in zip(lineid_lst, mask, res_lst):
        aperture, iline = lineid
        identlist[aperture][iline]['mask']     = ma
        identlist[aperture][iline]['residual'] = res

    # number of lines and used lines
    nuse = mask.sum()
    ntot = fit_w_lst.size
    return coeff, std, k, offset, nuse, ntot

def polyfit2d(x, y, z, xorder=3, yorder=3, linear=False):
    """Two-dimensional polynomial fit.

    Args:
        x (:class:`numpy.ndarray`): Input X array.
        y (:class:`numpy.ndarray`): Input Y array.
        z (:class:`numpy.ndarray`): Input Z array.
        xorder (int): X order.
        yorder (int): Y order.
        linear (bool): Return linear solution if `True`.
    Returns:
        :class:`numpy.ndarray`: Coefficient array.

    Examples:

        .. code-block:: python
    
           import numpy as np
           numdata = 100
           x = np.random.random(numdata)
           y = np.random.random(numdata)
           z = 6*y**2+8*y-x-9*x*y+10*x*y**2+7+np.random.random(numdata)
           m = polyfit2d(x, y, z, xorder=1, yorder=3)
           # evaluate it on a grid
           nx, ny = 20, 20
           xx, yy = np.meshgrid(np.linspace(x.min(), x.max(), nx),
                                np.linspace(y.min(), y.max(), ny))
           zz = polyval2d(xx, yy, m)
    
           fig1 = plt.figure(figsize=(10,5))
           ax1 = fig1.add_subplot(121,projection='3d')
           ax2 = fig1.add_subplot(122,projection='3d')
           ax1.plot_surface(xx, yy, zz, rstride=1, cstride=1, cmap='jet',
               linewidth=0, antialiased=True, alpha=0.3)
           ax1.set_xlabel('X (pixel)')
           ax1.set_ylabel('Y (pixel)')
           ax1.scatter(x, y, z, linewidth=0)
           ax2.scatter(x, y, z-polyval2d(x,y,m),linewidth=0)
           plt.show()

        if `linear = True`, the fitting only consider linear solutions such as

        .. math::

            z = a(x-x_0)^2 + b(y-y_0)^2 + c
    
        the returned coefficients are organized as an *m* x *n* array, where *m*
        is the order along the y-axis, and *n* is the order along the x-axis::
    
            1   + x     + x^2     + ... + x^n     +
            y   + xy    + x^2*y   + ... + x^n*y   +
            y^2 + x*y^2 + x^2*y^2 + ... + x^n*y^2 +
            ... + ...   + ...     + ... + ...     +
            y^m + x*y^m + x^2*y^m + ... + x^n*y^m

    """
    ncols = (xorder + 1)*(yorder + 1)
    G = np.zeros((x.size, ncols))
    ji = itertools.product(range(yorder+1), range(xorder+1))
    for k, (j,i) in enumerate(ji):
        G[:,k] = x**i * y**j
        if linear & (i != 0) & (j != 0):
            G[:,k] = 0
    coeff, residuals, _, _ = np.linalg.lstsq(G, z, rcond=None)
    coeff = coeff.reshape(yorder+1, xorder+1)
    return coeff

def polyval2d(x, y, m):
    """Get values for the 2-D polynomial values

    Args:
        x (:class:`numpy.ndarray`): Input X array.
        y (:class:`numpy.ndarray`): Input Y array.
        m (:class:`numpy.ndarray`): Coefficients of the 2-D polynomial.
    Returns:
        z (:class:`numpy.ndarray`): Values of the 2-D polynomial.
    """
    yorder = m.shape[0] - 1
    xorder = m.shape[1] - 1
    z = np.zeros_like(x)
    for j,i in itertools.product(range(yorder+1), range(xorder+1)):
        z += m[j,i] * x**i * y**j
    return z

def find_order(identlist, npixel):
    """Find the linear relation between the aperture numbers and diffraction
    orders.
    The relationship is `order = k*aperture + offset`.
    Longer wavelength has lower order number.

    Args:
        identlist (dict): Dict of identified lines.
        npixel (int): Number of pixels along the main dispersion direction.

    Returns:
        tuple: A tuple containing:

            * **k** (*int*) -- Coefficient in the relationship
              `order = k*aperture + offset`.
            * **offset** (*int*) -- Coefficient in the relationship
              `order = k*aperture + offset`.
    """
    aper_lst, wlc_lst = [], []
    for aperture, list1 in sorted(identlist.items()):
        if list1.size<3:
            continue
        less_half = (list1['pixel'] < npixel/2).sum()>0
        more_half = (list1['pixel'] > npixel/2).sum()>0
        #less_half, more_half = False, False
        #for pix, wav in zip(list1['pixel'], list1['wavelength']):
        #    if pix < npixel/2.:
        #        less_half = True
        #    elif pix >= npixel/2.:
        #        more_half = True
        if less_half and more_half:
            if list1['pixel'].size>2:
                deg = 2
            else:
                deg = 1
            c = np.polyfit(list1['pixel'], list1['wavelength'], deg=deg)
            wlc = np.polyval(c, npixel/2.)
            aper_lst.append(aperture)
            wlc_lst.append(wlc)
    aper_lst = np.array(aper_lst)
    wlc_lst  = np.array(wlc_lst)
    if wlc_lst[0] > wlc_lst[-1]:
        k = 1
    else:
        k = -1

    offset_lst = np.arange(-500, 500)
    eva_lst = []
    for offset in offset_lst:
        const = (k*aper_lst + offset)*wlc_lst
        diffconst = np.diff(const)
        eva = (diffconst**2).sum()
        eva_lst.append(eva)
    eva_lst = np.array(eva_lst)
    offset = offset_lst[eva_lst.argmin()]

    return k, offset


def is_identified(wavelength, identlist, aperture):
    """Check if the input wavelength has already been identified.

    Args:
        wavelength (float): Wavelength of the input line.
        identlist (dict): Dict of identified lines.
        aperture (int): Aperture number.

    Returns:
        bool: *True* if **wavelength** and **aperture** in **identlist**.
    """
    if aperture in identlist:
        list1 = identlist[aperture]
        if list1.size==0:
            # has no line in this aperture
            return False
        diff = np.abs(list1['wavelength'] - wavelength)
        if diff.min()<1e-3:
            return True
        else:
            return False
    else:
        return False


def get_wavelength(coeff, npixel, pixel, order):
    """Get wavelength.

    Args:
        coeff (:class:`numpy.ndarray`): 2-D Coefficient array.
        npixel (int): Number of pixels along the main dispersion direction.
        pixel (*int* or :class:`numpy.ndarray`): Pixel coordinates.
        order (*int* or :class:`numpy.ndarray`): Diffraction order number.
            Must have the same length as **pixel**.

    Returns:
        float or :class:`numpy.ndarray`: Wavelength solution of the given pixels.

    See also:
        :func:`fit_wavelength`
    """
    # convert aperture to order
    norm_pixel = pixel*2./(npixel-1) - 1
    #norm_order  = 50./order
    #norm_order  = order/50.
    return polyval2d(norm_pixel, order, coeff)/order

def guess_wavelength(x, aperture, identlist, linelist, param):
    """Guess wavelength according to the identified lines.
    First, try to guess the wavelength from the identified lines in the same
    order (aperture) by fitting polynomials.
    If failed, find the rough wavelength the global wavelength solution.
    Finally, pick up the closet wavelength from the wavelength standards.

    Args:
        x (float): Pixel coordinate.
        aperture (int): Aperture number.
        identlist (dict): Dict of identified lines for different apertures.
        linelist (list): List of wavelength standards.
        param (dict): Parameters of the :class:`CalibWindow`.

    Returns:
        float: Guessed wavelength. If failed, return *None*.
    """
    rough_wl = None

    # guess wavelength from the identified lines in this order
    if aperture in identlist:
        list1 = identlist[aperture]
        if list1.size >= 2:
            fit_order = min(list1.size-1, 2)
            local_coeff = np.polyfit(list1['pixel'], list1['wavelength'],
                            deg=fit_order)
            rough_wl = np.polyval(local_coeff, x)

    # guess wavelength from global wavelength solution
    if rough_wl is None and param['coeff'].size > 0:
        npixel = param['npixel']
        order = aperture*param['k'] + param['offset']
        rough_wl = get_wavelength(param['coeff'], param['npixel'], x, order)

    if rough_wl is None:
        return None
    else:
        # now find the nearest wavelength in linelist
        wave_list = np.array([line[0] for line in linelist])
        iguess = np.abs(wave_list-rough_wl).argmin()
        guess_wl = wave_list[iguess]
        return guess_wl

class CustomToolbar(NavigationToolbar2Tk):
    """Class for customized matplotlib toolbar.

    Args:
        canvas (:class:`matplotlib.backends.backend_agg.FigureCanvasAgg`):
            Canvas object used in :class:`CalibWindow`.
        master (Tkinter widget): Parent widget.
    """

    def __init__(self, canvas, master):
        """Constructor of :class:`CustomToolbar`.
        """
        self.toolitems = (
            ('Home', 'Reset original view', 'home', 'home'),
            ('Back', 'Back to previous view', 'back', 'back'),
            ('Forward', 'Forward to next view', 'forward', 'forward'),
            ('Pan', 'Pan axes with left mouse, zoom with right', 'move','pan'),
            ('Zoom', 'Zoom to rectangle', 'zoom_to_rect','zoom'),
            ('Subplots', 'Configure subplots', 'subplots','configure_subplots'),
            ('Save', 'Save the figure', 'filesave', 'save_figure'),
        )
        NavigationToolbar2Tk.__init__(self, canvas, master)

    def set_message(self, msg):
        """Remove the coordinate displayed in the toolbar.
        """
        pass

class PlotFrame(tk.Frame):
    """The frame for plotting spectrum in the :class:`CalibWindow`.

    Args:
        master (Tkinter widget): Parent widget.
        width (int): Width of frame.
        height (int): Height of frame.
        dpi (int): DPI of figure.
        identlist (dict): Dict of identified lines.
        linelist (list): List of wavelength standards.
    """
    def __init__(self, master, width, height, dpi, identlist, linelist):
        """Constructor of :class:`PlotFrame`.
        """

        tk.Frame.__init__(self, master, width=width, height=height)

        self.fig = CalibFigure(width  = width,
                               height = height,
                               dpi    = dpi,
                               title  = master.param['title'],
                               )
        self.ax1 = self.fig._ax1
        self.ax2 = self.fig._ax2
        self.ax3 = self.fig._ax3

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP)
        self.canvas.mpl_connect('button_press_event', master.on_click)
        self.canvas.mpl_connect('draw_event', master.on_draw)

        aperture = master.param['aperture']
        self.ax1._aperture_text.set_text('Aperture %d'%aperture)

        # add toolbar
        self.toolbar = CustomToolbar(self.canvas, master=self)
        self.toolbar.update()
        self.toolbar.pack(side=tk.LEFT)

        self.pack()

class InfoFrame(tk.Frame):
    """The frame for buttons and tables on the right side of the
    :class:`CalibWindow`.

    Args:
        master (Tkinter widget): Parent widget.
        width (int): Width of the frame.
        height (int): Height of the frame.
        linelist (list): List of wavelength standards.
        identlist (dict): Dict of identified lines.
    """

    def __init__(self, master, width, height, linelist, identlist):
        """Constuctor of :class:`InfoFrame`.
        """

        self.master = master

        title = master.param['title']

        tk.Frame.__init__(self, master, width=width, height=height)

        self.fname_label = tk.Label(master = self,
                                    width  = width,
                                    font   = ('Arial', 14),
                                    text   = title,
                                    )
        self.order_label = tk.Label(master = self,
                                    width  = width,
                                    font   = ('Arial', 10),
                                    text   = '',
                                    )
        self.fname_label.pack(side=tk.TOP,pady=(30,5))
        self.order_label.pack(side=tk.TOP,pady=(5,10))

        button_width = 13

        self.switch_frame = tk.Frame(master=self, width=width, height=30)
        self.prev_button = tk.Button(master  = self.switch_frame,
                                     text    = '◀',
                                     width   = button_width,
                                     font    = ('Arial',10),
                                     command = master.prev_aperture,
                                     )
        self.next_button = tk.Button(master  = self.switch_frame,
                                     text    = '▶',
                                     width   = button_width,
                                     font    = ('Arial',10),
                                     command = master.next_aperture,
                                     )
        self.prev_button.pack(side=tk.LEFT)
        self.next_button.pack(side=tk.RIGHT)
        self.switch_frame.pack(side=tk.TOP, pady=5, padx=10, fill=tk.X)

        # line table area
        self.line_frame = LineTable(master    = self,
                                    width     = width-30,
                                    height    = 400, #900,
                                    identlist = identlist,
                                    linelist  = linelist)
        self.line_frame.pack(side=tk.TOP, padx=10, pady=5)


        # batch operation buttons
        button_width2 = 13
        self.batch_frame = tk.Frame(master=self, width=width, height=30)
        self.recenter_button = tk.Button(master  = self.batch_frame,
                                         text    = 'recenter',
                                         width   = button_width2,
                                         font    = ('Arial',10),
                                         command = master.recenter,
                                         )
        self.clearall_button = tk.Button(master  = self.batch_frame,
                                         text    = 'clear all',
                                         width   = button_width2,
                                         font    = ('Arial',10),
                                         command = master.clearall,
                                         )
        self.recenter_button.pack(side=tk.LEFT)
        self.clearall_button.pack(side=tk.RIGHT)
        self.batch_frame.pack(side=tk.TOP, pady=5, padx=10, fill=tk.X)


        # fit buttons
        self.auto_button = tk.Button(master=self, text='Auto Identify',
                            font = ('Arial', 10), width=25,
                            command = master.auto_identify)
        self.fit_button = tk.Button(master=self, text='Fit',
                            font = ('Arial', 10), width=25,
                            command = master.fit)
        self.switch_button = tk.Button(master=self, text='Plot',
                            font = ('Arial', 10), width=25,
                            command = master.switch)
        # set status
        self.auto_button.config(state=tk.DISABLED)
        self.fit_button.config(state=tk.DISABLED)
        self.switch_button.config(state=tk.DISABLED)

        # Now pack from bottom to top
        self.switch_button.pack(side=tk.BOTTOM, pady=(5,30))
        self.fit_button.pack(side=tk.BOTTOM, pady=5)
        self.auto_button.pack(side=tk.BOTTOM, pady=5)

        self.fitpara_frame = FitparaFrame(master=self, width=width-20, height=35)
        self.fitpara_frame.pack(side=tk.BOTTOM, padx=10, pady=5, fill=tk.X)

        self.pack()

    def update_nav_buttons(self):
        """Update the navigation buttons.
        """
        mode = self.master.param['mode']
        if mode == 'ident':
            aperture = self.master.param['aperture']

            if aperture == 0:
                state = tk.DISABLED
            else:
                state = tk.NORMAL
            self.prev_button.config(state=state)

            if aperture == self.master.spec.shape[0]:
                state = tk.DISABLED
            else:
                state = tk.NORMAL
            self.next_button.config(state=state)
        elif mode == 'fit':
            self.prev_button.config(state=tk.DISABLED)
            self.next_button.config(state=tk.DISABLED)
        else:
            pass

    def update_aperture_label(self):
        """Update the order information to be displayed on the top.
        """
        mode     = self.master.param['mode']
        aperture = self.master.param['aperture']
        k        = self.master.param['k']
        offset   = self.master.param['offset']

        if mode == 'ident':
            if None in (k, offset):
                order = '?'
            else:
                order = str(k*aperture + offset)
            text = 'Order %s (Aperture %d)'%(order, aperture)
            self.order_label.config(text=text)
        elif mode == 'fit':
            self.order_label.config(text='')
        else:
            pass


class LineTable(tk.Frame):
    """A table for the input spectral lines embedded in the :class:`InfoFrame`.

    Args:
        master (Tkinter widget): Parent widget.
        width (int): Width of line table.
        height (int): Height of line table.
        identlist (dict): Dict of identified lines.
        linelist (list): List of wavelength standards.
    """
    def __init__(self, master, width, height, identlist, linelist):
        """Constructor of :class:`LineTable`.
        """
        self.master = master

        font = ('Arial', 10)

        tk.Frame.__init__(self, master=master, width=width, height=height)

        self.tool_frame = tk.Frame(master=self, width=width, height=40)
        self.tool_frame.pack(side=tk.TOP, fill=tk.X, pady=(0,5))

        self.search_text = tk.StringVar()
        self.search_entry = tk.Entry(master=self.tool_frame, width=10,
                                     font=font, textvariable=self.search_text)
        self.search_entry.pack(side=tk.LEFT, fil=tk.Y, padx=0)

        # create 3 buttons
        self.clr_button = tk.Button(master=self.tool_frame, text='Clear',
                                    font=font, width=5,
                                    command=self.on_clear_search)
        self.add_button = tk.Button(master=self.tool_frame, text='Add',
                                    font=font, width=5,
                                    command=master.master.on_add_ident)
        self.del_button = tk.Button(master=self.tool_frame, text='Del',
                                    font=font, width=5,
                                    command=master.master.on_delete_ident)

        # put 3 buttons
        self.del_button.pack(side=tk.RIGHT, padx=(5,0))
        self.add_button.pack(side=tk.RIGHT, padx=(5,0))
        self.clr_button.pack(side=tk.RIGHT, padx=(5,0))

        # update status of 3 buttons
        self.clr_button.config(state=tk.DISABLED)
        self.add_button.config(state=tk.DISABLED)
        self.del_button.config(state=tk.DISABLED)

        # create line table
        self.data_frame = tk.Frame(master=self, width=width)
        self.data_frame.pack(side=tk.TOP, fill=tk.X, pady=5)

        # create line tree
        self.line_tree = ttk.Treeview(master  = self.data_frame,
                                      columns    = ('wl', 'species', 'status'),
                                      show       = 'headings',
                                      style      = 'Treeview',
                                      height     = 13, # 22,
                                      selectmode ='browse')
        self.line_tree.bind('<Button-1>', self.on_click_item)

        self.scrollbar = tk.Scrollbar(master = self.data_frame,
                                      orient = tk.VERTICAL,
                                      width  = 20)

        self.line_tree.column('wl',      width=160)
        self.line_tree.column('species', width=140)
        self.line_tree.column('status',  width=width-160-140-20)
        self.line_tree.heading('wl',      text=u'\u03bb in air (\xc5)')
        self.line_tree.heading('species', text='Species')
        self.line_tree.heading('status',  text='Status')
        self.line_tree.config(yscrollcommand=self.scrollbar.set)

        style = ttk.Style()
        style.configure('Treeview', rowheight=30)
        style.configure('Treeview.Heading', font=('Arial', 10))

        self.scrollbar.config(command=self.line_tree.yview)

        self.item_lst = []
        for line in linelist:
            wl, species = line
            iid = self.line_tree.insert('',tk.END,
                    values=(wl, species, ''), tags='normal')
            self.item_lst.append((iid,  wl))
        self.line_tree.tag_configure('normal', font=('Arial', 10))

        self.line_tree.pack(side=tk.LEFT, fill=tk.Y, expand=True)
        self.scrollbar.pack(side=tk.LEFT, fill=tk.Y)
        self.data_frame.pack(side=tk.TOP, fill=tk.Y)
        self.pack()

    def on_clear_search(self):
        """Clear the search bar.
        """
        # clear the search bar
        self.search_text.set('')

        # clear the identified line
        self.master.master.param['center']    = None
        self.master.master.param['amplitude'] = None
        self.master.master.param['fwhm']      = None

        # de-select the line table
        sel_items = self.line_tree.selection()
        self.line_tree.selection_remove(sel_items)
        #self.line_tree.selection_clear()
        # doesn't work?

        # update the status of 3 button
        self.clr_button.config(state=tk.DISABLED)
        self.add_button.config(state=tk.DISABLED)
        self.del_button.config(state=tk.DISABLED)

        # replot
        self.master.master.plot_aperture()

        # set focus to canvas
        self.master.master.plot_frame.canvas.get_tk_widget().focus()

    def on_click_item(self, event):
        """Event response function for clicking lines.
        """

        identlist = self.master.master.identlist
        aperture = self.master.master.param['aperture']

        # find the clicked item
        item = self.line_tree.identify_row(event.y)
        values = self.line_tree.item(item, 'values')
        # put the wavelength into the search bar
        self.search_text.set(values[0])
        # update status
        self.clr_button.config(state=tk.NORMAL)


        # find if the clicked line is in ident list.
        # if yes, replot the figure with idented line with blue color, and set
        # the delete button to normal. Otherwise, replot the figure with black,
        # and disable the delete button.

        if aperture in identlist:

            list1 = identlist[aperture]

            wl_diff = np.abs(list1['wavelength'] - float(values[0]))
            mindiff = wl_diff.min()
            argmin  = wl_diff.argmin()
            if mindiff < 1e-3:
                # the selected line is in identlist of this aperture
                xpos = list1[argmin]['pixel']
                for line, text in self.master.master.ident_objects:
                    if abs(line.get_xdata()[0] - xpos)<1e-3:
                        plt.setp(line, color='b')
                        plt.setp(text, color='b')
                    else:
                        plt.setp(line, color='k')
                        plt.setp(text, color='k')
                # update the status of del button
                self.del_button.config(state=tk.NORMAL)
            else:
                # the selected line is not in identlist of this aperture
                for line, text in self.master.master.ident_objects:
                    plt.setp(line, color='k')
                    plt.setp(text, color='k')
                # update the status of del button
                self.del_button.config(state=tk.DISABLED)

            self.master.master.plot_frame.canvas.draw()

        else:
            # if the current aperture is not in identlist, do nothing
            pass

class FitparaFrame(tk.Frame):
    """Frame for the fitting parameters embedded in the :class:`InfoFrame`.

    Args:
        master (Tkinter widget): Parent widget.
        width (int): Width of frame.
        height (int): Height of frame.
    """
    def __init__(self, master, width, height):
        """Constructor of :class:`FitparaFrame`.
        """

        self.master = master

        font = ('Arial', 10)

        tk.Frame.__init__(self, master, width=width, height=height)

        # the first row
        self.row1_frame = tk.Frame(master=self, width=width)

        self.xorder_label = tk.Label(master = self.row1_frame,
                                     text   = 'X ord =',
                                     font   = font)

        # spinbox for adjusting xorder
        self.xorder_str = tk.StringVar()
        self.xorder_str.set(master.master.param['xorder'])
        self.xorder_box = tk.Spinbox(master = self.row1_frame,
                                     from_        = 1,
                                     to_          = 10,
                                     font         = font,
                                     width        = 2,
                                     textvariable = self.xorder_str,
                                     command      = self.on_change_xorder)

        self.yorder_label = tk.Label(master = self.row1_frame,
                                     text   = 'Y ord =',
                                     font   = font)
        # spinbox for adjusting yorder
        self.yorder_str = tk.StringVar()
        self.yorder_str.set(master.master.param['yorder'])
        self.yorder_box = tk.Spinbox(master       = self.row1_frame,
                                     from_        = 1,
                                     to_          = 10,
                                     font         = font,
                                     width        = 2,
                                     textvariable = self.yorder_str,
                                     command      = self.on_change_yorder)

        self.maxiter_label  = tk.Label(master = self.row1_frame,
                                       text   = 'N =',
                                       font   = font)
        self.maxiter_str = tk.StringVar()
        self.maxiter_str.set(master.master.param['maxiter'])
        self.maxiter_box = tk.Spinbox(master       = self.row1_frame,
                                      from_        = 1,
                                      to_          = 20,
                                      font         = font,
                                      width        = 2,
                                      textvariable = self.maxiter_str,
                                      command      = self.on_change_maxiter)

        self.xorder_label.pack(side=tk.LEFT)
        self.xorder_box.pack(side=tk.LEFT)
        self.yorder_label.pack(side=tk.LEFT, padx=(10,0))
        self.yorder_box.pack(side=tk.LEFT)
        self.maxiter_label.pack(side=tk.LEFT, padx=(10,0))
        self.maxiter_box.pack(side=tk.LEFT)

        self.row1_frame.pack(side=tk.TOP, fill=tk.X, pady=(0,2))

        # the second row
        self.row2_frame = tk.Frame(master=self, width=width)

        self.clip_label = tk.Label(master = self.row2_frame,
                                   text   = 'Clipping =',
                                   font   = font,
                                   width  = width,
                                   anchor = tk.W)
        self.clip_scale = tk.Scale(master     = self.row2_frame,
                                   from_      = 1.0,
                                   to         = 5.0,
                                   orient     = tk.HORIZONTAL,
                                   resolution = 0.1,
                                   command    = self.on_change_clipping)
        self.clip_scale.set(master.master.param['clipping'])

        self.clip_label.pack(side=tk.TOP)
        self.clip_scale.pack(side=tk.TOP, fill=tk.X)
        self.row2_frame.pack(side=tk.TOP, fill=tk.X)

        self.pack()

    def on_change_xorder(self):
        """Response function of changing order of polynomial along x-axis.
        """
        self.master.master.param['xorder'] = int(self.xorder_box.get())

    def on_change_yorder(self):
        """Response function of changing order of polynomial along y-axis.
        """
        self.master.master.param['yorder'] = int(self.yorder_box.get())

    def on_change_maxiter(self):
        """Response function of changing maximum number of iteration.
        """
        self.master.master.param['maxiter'] = int(self.maxiter_box.get())

    def on_change_clipping(self, value):
        """Response function of changing clipping value.
        """
        self.master.master.param['clipping'] = float(value)


class CalibFigure(Figure):
    """Figure class for wavelength calibration.

    Args:
        width (int): Width of figure.
        height (int): Height of figure.
        dpi (int): DPI of figure.
        filename (str): Filename of input spectra.
        channel (str): Channel name of input spectra.
    """

    def __init__(self, width, height, dpi, title):
        """Constuctor of :class:`CalibFigure`.
        """
        # set figsize and dpi
        figsize = (width/dpi, height/dpi)
        super(CalibFigure, self).__init__(figsize=figsize, dpi=dpi)

        # set background color as light gray
        self.patch.set_facecolor('#d9d9d9')

        # add axes
        self._ax1 = self.add_axes([0.07, 0.07,0.52,0.87])
        self._ax2 = self.add_axes([0.655,0.07,0.32,0.40])
        self._ax3 = self.add_axes([0.655,0.54,0.32,0.40])

        # add title
        self.suptitle(title, fontsize=15)

        #draw the aperture number to the corner of ax1
        bbox = self._ax1.get_position()
        self._ax1._aperture_text = self.text(bbox.x0 + 0.05, bbox.y1-0.1,
                                  '', fontsize=15)

        # draw residual and number of identified lines in ax2
        bbox = self._ax3.get_position()
        self._ax3._residual_text = self.text(bbox.x0 + 0.02, bbox.y1-0.03,
                                  '', fontsize=13)

        # draw fitting parameters in ax3
        bbox = self._ax2.get_position()
        self._ax2._fitpar_text = self.text(bbox.x0 + 0.02, bbox.y1-0.03,
                                  '', fontsize=13)

    def plot_solution(self, identlist, aperture_lst, plot_ax1=False,  **kwargs):
        """Plot the wavelength solution.

        Args:
            identlist (dict): Dict of identified lines.
            aperture_lst (list): List of apertures to be plotted.
            plot_ax1 (bool): Whether to plot the first axes.
            coeff (:class:`numpy.ndarray`): Coefficient array.
            k (int): `k` value in the relationship `order = k*aperture +
                offset`.
            offset (int): `offset` value in the relationship `order =
                k*aperture + offset`.
            npixel (int): Number of pixels along the main dispersion
                direction.
            std (float): Standard deviation of wavelength fitting.
            nuse (int): Number of lines actually used in the wavelength
                fitting.
            ntot (int): Number of lines identified.
        """
        coeff    = kwargs.pop('coeff')
        k        = kwargs.pop('k')
        offset   = kwargs.pop('offset')
        npixel   = kwargs.pop('npixel')
        std      = kwargs.pop('std')
        nuse     = kwargs.pop('nuse')
        ntot     = kwargs.pop('ntot')
        xorder   = kwargs.pop('xorder')
        yorder   = kwargs.pop('yorder')
        clipping = kwargs.pop('clipping')
        maxiter  = kwargs.pop('maxiter')

        label_size = 13  # fontsize for x, y labels
        tick_size  = 12  # fontsize for x, y ticks

        #wave_scale = 'linear'
        wave_scale = 'reciprocal'

        #colors = 'rgbcmyk'

        self._ax2.cla()
        self._ax3.cla()

        if plot_ax1:
            self._ax1.cla()
            x = np.linspace(0, npixel-1, 100, dtype=np.float64)

            # find the maximum and minimum wavelength
            wl_min, wl_max = 1e9,0
            allwave_lst = {}
            for aperture in aperture_lst:
                order = k*aperture + offset
                wave = get_wavelength(coeff, npixel, x, np.repeat(order, x.size))
                allwave_lst[aperture] = wave
                wl_max = max(wl_max, wave.max())
                wl_min = min(wl_min, wave.min())
            # plot maximum and minimum wavelength, to determine the display
            # range of this axes, and the tick positions
            self._ax1.plot([0, 0],[wl_min, wl_max], color='none')
            yticks = self._ax1.get_yticks()
            self._ax1.cla()


        for aperture in aperture_lst:
            order = k*aperture + offset
            color = 'C{}'.format(order%10)

            # plot pixel vs. wavelength
            if plot_ax1:
                wave = allwave_lst[aperture]
                if wave_scale=='reciprocal':
                    self._ax1.plot(x, 1/wave,
                            color=color, ls='-', alpha=0.8, lw=0.8)
                else:
                    self._ax1.plot(x, wave,
                            color=color, ls='-', alpha=0.8, lw=0.8)

            # plot identified lines
            if aperture in identlist:
                list1 = identlist[aperture]
                pix_lst = list1['pixel']
                wav_lst = list1['wavelength']
                mask    = list1['mask'].astype(bool)
                res_lst = list1['residual']

                if plot_ax1:
                    if wave_scale=='reciprocal':
                        self._ax1.scatter(pix_lst[mask],  1/wav_lst[mask],
                                          c=color, s=20, lw=0, alpha=0.8)
                        self._ax1.scatter(pix_lst[~mask], 1/wav_lst[~mask],
                                          c='w', s=16, lw=0.7, alpha=0.8,
                                          edgecolor=color)
                    else:
                        self._ax1.scatter(pix_lst[mask],  wav_lst[mask],
                                          c=color, s=20, lw=0, alpha=0.8)
                        self._ax1.scatter(pix_lst[~mask], wav_lst[~mask],
                                          c='w', s=16, lw=0.7, alpha=0.8,
                                          edgecolor=color)

                repeat_aper_lst = np.repeat(aperture, pix_lst.size)
                self._ax2.scatter(repeat_aper_lst[mask], res_lst[mask],
                                  c=color, s=20, lw=0, alpha=0.8)
                self._ax2.scatter(repeat_aper_lst[~mask], res_lst[~mask],
                                  c='w', s=16, lw=0.7, alpha=0.8, ec=color)
                self._ax3.scatter(pix_lst[mask], res_lst[mask],
                                  c=color, s=20, lw=0, alpha=0.8)
                self._ax3.scatter(pix_lst[~mask], res_lst[~mask],
                                  c='w', s=16, lw=0.7, alpha=0.8, ec=color)

        # refresh texts in the residual panels
        text = 'R.M.S. = {:.5f}, N = {}/{}'.format(std, nuse, ntot)
        self._ax3._residual_text.set_text(text)
        text = u'Xorder = {}, Yorder = {}, clipping = \xb1{:g}, Niter = {}'.format(
                xorder, yorder, clipping, maxiter)
        self._ax2._fitpar_text.set_text(text)

        # adjust layout for ax1
        if plot_ax1:
            self._ax1.set_xlim(0, npixel-1)
            if wave_scale == 'reciprocal':
                _y11, _y22 = self._ax1.get_ylim()
                newtick_lst, newticklabel_lst = [], []
                for tick in yticks:
                    if _y11 < 1/tick < _y22:
                        newtick_lst.append(1/tick)
                        newticklabel_lst.append(tick)
                self._ax1.set_yticks(newtick_lst)
                self._ax1.set_yticklabels(newticklabel_lst)
                self._ax1.set_ylim(_y22, _y11)
            self._ax1.set_xlabel('Pixel', fontsize=label_size)
            self._ax1.set_ylabel(u'\u03bb (\xc5)', fontsize=label_size)
            self._ax1.grid(True, ls=':', color='gray', alpha=1, lw=0.5)
            self._ax1.set_axisbelow(True)
            self._ax1._aperture_text.set_text('')
            for tick in self._ax1.xaxis.get_major_ticks():
                tick.label1.set_fontsize(tick_size)
            for tick in self._ax1.yaxis.get_major_ticks():
                tick.label1.set_fontsize(tick_size)

        # adjust axis layout for ax2 (residual on aperture space)
        self._ax2.axhline(y=0, color='k', ls='--', lw=0.5)
        for i in np.arange(-3,3+0.1):
            self._ax2.axhline(y=i*std, color='k', ls=':', lw=0.5)
        x1, x2 = self._ax2.get_xlim()
        x1 = max(x1,aperture_lst.min())
        x2 = min(x2,aperture_lst.max())
        self._ax2.set_xlim(x1, x2)
        self._ax2.set_ylim(-6*std, 6*std)
        self._ax2.set_xlabel('Aperture', fontsize=label_size)
        self._ax2.set_ylabel(u'Residual on \u03bb (\xc5)', fontsize=label_size)
        for tick in self._ax2.xaxis.get_major_ticks():
            tick.label1.set_fontsize(tick_size)
        for tick in self._ax2.yaxis.get_major_ticks():
            tick.label1.set_fontsize(tick_size)

        ## adjust axis layout for ax3 (residual on pixel space)
        self._ax3.axhline(y=0, color='k', ls='--', lw=0.5)
        for i in np.arange(-3,3+0.1):
            self._ax3.axhline(y=i*std, color='k', ls=':', lw=0.5)
        self._ax3.set_xlim(0, npixel-1)
        self._ax3.set_ylim(-6*std, 6*std)
        self._ax3.set_xlabel('Pixel', fontsize=label_size)
        self._ax3.set_ylabel(u'Residual on \u03bb (\xc5)', fontsize=label_size)
        for tick in self._ax3.xaxis.get_major_ticks():
            tick.label1.set_fontsize(tick_size)
        for tick in self._ax3.yaxis.get_major_ticks():
            tick.label1.set_fontsize(tick_size)


class CalibWindow(tk.Frame):
    """Frame of the wavelength calibration window.

    Args:
        master (:class:`tk.TK`): Tkinter root window.
        width (int): Width of window.
        height (int): Height of window.
        dpi (int): DPI of figure.
        spec (:class:`numpy.dtype`): Spectra data.
        figfilename (str): Filename of the output wavelength calibration
            figure.
        title (str): A string to display as the title of calib figure.
        identlist (dict): Identification line list.
        linelist (list): List of wavelength standards (wavelength, species).
        window_size (int): Size of the window in pixel to search for line
            peaks.
        xorder (int): Degree of polynomial along X direction.
        yorder (int): Degree of polynomial along Y direction.
        maxiter (int): Maximim number of interation in polynomial fitting.
        clipping (float): Threshold of sigma-clipping.
        snr_threshold (float): Minimum S/N of the spectral lines to be accepted
            in the wavelength fitting.
    """
    def __init__(self, master, width, height, dpi, spec, figfilename, title,
            identlist, linelist, window_size, xorder, yorder, maxiter, clipping,
            q_threshold, fit_filter):
        """Constructor of :class:`CalibWindow`.
        """

        self.master    = master
        self.spec      = spec
        self.identlist = identlist
        self.linelist  = linelist

        tk.Frame.__init__(self, master, width=width, height=height)

        self.param = {
            'mode':         'ident',
            'aperture':     0,
            'figfilename':  figfilename,
            'title':        title,
            'aperture_min': 0,
            'aperture_max': self.spec.shape[0]-1,
            'npixel':       self.spec.shape[1],
            # parameters of displaying
            'xlim':         {},
            'ylim':         {},
            # line fitting parameters
            'i1':           None,
            'i2':           None,
            'window_size':  window_size,
            # line fitting results
            'center':       None,
            'amplitude':    None,
            'fwhm':         None,
            'background':   None,
            # parameters of converting aperture and order
            'k':            None,
            'offset':       None,
            # wavelength fitting parameters
            'xorder':       xorder,
            'yorder':       yorder,
            'maxiter':      maxiter,
            'clipping':     clipping,
            'q_threshold':  q_threshold,
            # wavelength fitting results
            'std':          0,
            'coeff':        np.array([]),
            'nuse':         0,
            'ntot':         0,
            'direction':    '',
            'fit_filter':   fit_filter,
            }

        for irow, row in enumerate(self.spec):
            aperture = irow
            self.param['xlim'][aperture] = (0, len(row)-1)
            self.param['ylim'][aperture] = (None, None)

        # determine widget size
        info_width    = 400  # 500
        info_height   = height # - 500
        canvas_width  = width - info_width
        canvas_height = height
        # generate plot frame and info frame
        self.plot_frame = PlotFrame(master    = self,
                                    width     = canvas_width,
                                    height    = canvas_height,
                                    dpi       = dpi,
                                    identlist = self.identlist,
                                    linelist  = self.linelist,
                                    )
        self.info_frame = InfoFrame(master    = self,
                                    width     = info_width,
                                    height    = info_height,
                                    identlist = self.identlist,
                                    linelist  = self.linelist,
                                    )
        # pack plot frame and info frame
        self.plot_frame.pack(side=tk.LEFT, fill=tk.Y)
        self.info_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0,10))

        self.pack()

        self.plot_aperture()

        self.update_fit_buttons()

    def fit(self):
        """Fit the wavelength and plot the solution in the figure.
        """

        coeff, std, k, offset, nuse, ntot = fit_wavelength(
                identlist   = self.identlist,
                npixel      = self.param['npixel'],
                xorder      = self.param['xorder'],
                yorder      = self.param['yorder'],
                maxiter     = self.param['maxiter'],
                clipping    = self.param['clipping'],
                fit_filter  = self.param['fit_filter'],
                )

        self.param['coeff']  = coeff
        self.param['std']    = std
        self.param['k']      = k
        self.param['offset'] = offset
        self.param['nuse']   = nuse
        self.param['ntot']   = ntot

        message = 'Wavelength fitted. std={:.6f}, utot={}, nuse={}'.format(
                    std, ntot, nuse)
        print(message)

        self.plot_wavelength()

        # udpdate the order/aperture string
        aperture = self.param['aperture']
        order = k*aperture + offset
        text = 'Order {} (Aperture {})'.format(order, aperture)
        self.info_frame.order_label.config(text=text)

        self.update_fit_buttons()

    def recenter(self):
        """Relocate the peaks for all the identified lines.
        """
        for aperture, list1 in self.identlist.items():
            flux = self.spec[aperture]
            for row in list1:
                pix = int(round(row['pixel']))
                window_size = self.param['window_size']
                _, _, param, _ = find_local_peak(flux, pix, window_size)
                peak_x = param[1]
                row['pixel'] = peak_x

        # replot
        self.plot_aperture()

    def clearall(self):
        """Clear all the identified lines."""

        self.identlist = {}

        self.param['k']      = None
        self.param['offset'] = None
        self.param['std']    = 0
        self.param['coeff']  = np.array([])
        self.param['nuse']   = 0
        self.param['ntot']   = 0

        info_frame = self.info_frame
        # update the status of 3 buttons
        info_frame.line_frame.clr_button.config(state=tk.DISABLED)
        info_frame.line_frame.add_button.config(state=tk.DISABLED)
        info_frame.line_frame.del_button.config(state=tk.DISABLED)

        # update buttons
        info_frame.recenter_button.config(state=tk.DISABLED)
        info_frame.clearall_button.config(state=tk.DISABLED)
        info_frame.switch_button.config(state=tk.DISABLED)
        info_frame.auto_button.config(state=tk.DISABLED)

        # replot
        self.plot_aperture()

    def auto_identify(self):
        """Identify all lines in the wavelength standard list automatically.
        """
        k       = self.param['k']
        offset  = self.param['offset']
        coeff   = self.param['coeff']
        npixel  = self.param['npixel']
        n_insert = 0
        for aperture in range(self.spec.shape[0]):
            flux = self.spec[aperture]

            # scan every order and find the upper and lower limit of wavelength
            order = k*aperture + offset

            # generated the wavelengths for every pixel in this oirder
            x = np.arange(npixel)
            wl = get_wavelength(coeff, npixel, x, np.repeat(order, x.size))
            w1 = min(wl[0], wl[-1])
            w2 = max(wl[0], wl[-1])

            has_insert = False
            # now scan the linelist
            for line in self.linelist:
                if line[0] < w1:
                    continue
                if line[0] > w2:
                    break

                # wavelength in the range of this order
                # check if this line has already been identified
                if is_identified(line[0], self.identlist, aperture):
                    continue

                # now has not been identified. find peaks for this line
                diff = np.abs(wl - line[0])
                i = diff.argmin()
                i1, i2, param, std = find_local_peak(flux, i,
                                        self.param['window_size'])
                keep = auto_line_fitting_filter(param, i1, i2)
                if not keep:
                    continue

                # unpack the fitted parameters
                amplitude  = param[0]
                center     = param[1]
                fwhm       = param[2]
                background = param[3]

                # q = A/std is a proxy of signal-to-noise ratio.
                q = amplitude/std
                if q < self.param['q_threshold']:
                    continue

                '''
                fig = plt.figure(figsize=(6,4),tight_layout=True)
                ax = fig.gca()
                ax.plot(np.arange(i1,i2), flux[i1:i2], 'ro')
                newx = np.arange(i1, i2, 0.1)
                ax.plot(newx, gaussian_bkg(param[0], param[1], param[2],
                            param[3], newx), 'b-')
                ax.axvline(x=param[1], color='k',ls='--')
                y1,y2 = ax.get_ylim()
                ax.text(0.9*i1+0.1*i2, 0.1*y1+0.9*y2, 'A=%.1f'%param[0])
                ax.text(0.9*i1+0.1*i2, 0.2*y1+0.8*y2, 'BKG=%.1f'%param[3])
                ax.text(0.9*i1+0.1*i2, 0.3*y1+0.7*y2, 'FWHM=%.1f'%param[2])
                ax.text(0.9*i1+0.1*i2, 0.4*y1+0.6*y2, 'q=%.1f'%q)
                ax.set_xlim(i1,i2)
                fig.savefig('tmp/%d-%d-%d.png'%(aperture, i1, i2))
                plt.close(fig)
                '''

                # initialize line table
                if aperture not in self.identlist:
                    self.identlist[aperture] = np.array([], dtype=identlinetype)

                item = np.array((aperture, order, line[0], i1, i2, center,
                                amplitude, fwhm, background, q, True, 0.0,
                                'a'), dtype=identlinetype)

                self.identlist[aperture] = np.append(self.identlist[aperture],
                                            item)
                has_insert = True
                #print('insert', aperture, line[0], peak_x, i)
                n_insert += 1

            # resort this order if there's new line inserted
            if has_insert:
                self.identlist[aperture] = np.sort(self.identlist[aperture],
                                                    order='pixel')

        message = '{} new lines inserted'.format(n_insert)
        print(message)

        self.fit()

    def switch(self):
        """Response funtion of switching between "ident" and "fit" mode.
        """

        if self.param['mode']=='ident':
            # switch to fit mode
            self.param['mode']='fit'

            self.plot_wavelength()

            self.info_frame.switch_button.config(text='Identify')

        elif self.param['mode']=='fit':
            # switch to ident mode
            self.param['mode']='ident'

            self.plot_aperture()

            self.info_frame.switch_button.config(text='Plot')
        else:
            pass

        # update order navigation and aperture label
        self.info_frame.update_nav_buttons()
        self.info_frame.update_aperture_label()

    def next_aperture(self):
        """Response function of pressing the next aperture."""
        if self.param['aperture'] < self.spec.shape[0]:
            self.param['aperture'] += 1
            self.plot_aperture()

    def prev_aperture(self):
        """Response function of pressing the previous aperture."""
        if self.param['aperture'] > 0:
            self.param['aperture'] -= 1
            self.plot_aperture()

    def plot_aperture(self):
        """Plot a specific aperture in the figure.
        """
        aperture = self.param['aperture']
        ydata = self.spec[aperture]
        npoints = len(ydata)
        xdata = np.arange(npoints)

        # redraw spectra in ax1
        ax1 = self.plot_frame.ax1
        fig = self.plot_frame.fig
        ax1.cla()
        ax1.plot(xdata, ydata, '-', c='C3')
        x1, x2 = self.param['xlim'][aperture]
        y1, y2 = self.param['ylim'][aperture]
        if y1 is None:
            y1 = ax1.get_ylim()[0]
        if y2 is None:
            y2 = ax1.get_ylim()[1]

        #x1, x2 = xdata[0], xdata[-1]
        #y1, y2 = ax1.get_ylim()
        y1 = min(y1, 0)
        # plot identified line list
        # calculate ratio = value/pixel
        bbox = ax1.get_window_extent().transformed(
                fig.dpi_scale_trans.inverted())
        axwidth_pixel = bbox.width*fig.dpi
        pix_ratio = abs(x2-x1)/axwidth_pixel

        # plot identified lines with vertial dash lines
        if aperture in self.identlist and len(self.identlist[aperture])>0:
            self.ident_objects = []
            list1 = self.identlist[aperture]
            for item in list1:
                pixel      = item['pixel']
                wavelength = item['wavelength']

                # draw vertial dash line
                line = ax1.axvline(pixel, ls='--', color='k')

                # draw text
                x = pixel+pix_ratio*10
                y = 0.4*y1+0.6*y2
                text = ax1.text(x, y, '{:.4f}'.format(wavelength), color='k',
                                rotation='vertical', fontstyle='italic',
                                fontsize=10)
                self.ident_objects.append((line, text))

        # plot the temporarily identified line
        if self.param['center'] is not None:
            ax1.axvline(self.param['center'], linestyle='--', color='k')

        # update the aperture number
        ax1._aperture_text.set_text('Aperture %d'%aperture)
        ax1.set_ylim(y1, y2)
        ax1.set_xlim(x1, x2)
        ax1.xaxis.set_major_locator(tck.MultipleLocator(1000))
        ax1.xaxis.set_minor_locator(tck.MultipleLocator(100))
        ax1.set_xlabel('Pixel')
        ax1.set_ylabel('Flux')

        self.plot_frame.canvas.draw()

        # update order navigation and aperture label
        self.info_frame.update_nav_buttons()
        self.info_frame.update_aperture_label()

    def plot_wavelength(self):
        """A wrap for plotting the wavelength solution."""

        aperture_lst = np.arange(self.param['aperture_min'],
                                 self.param['aperture_max']+1)

        kwargs = {
                'offset':   self.param['offset'],
                'k':        self.param['k'],
                'coeff':    self.param['coeff'],
                'npixel':   self.param['npixel'],
                'std':      self.param['std'],
                'nuse':     self.param['nuse'],
                'ntot':     self.param['ntot'],
                'xorder':   self.param['xorder'],
                'yorder':   self.param['yorder'],
                'clipping': self.param['clipping'],
                'maxiter':  self.param['maxiter'],
                }

        if self.param['mode']=='fit':
            plot_ax1 = True
        else:
            plot_ax1 = False

        self.plot_frame.fig.plot_solution(self.identlist, aperture_lst,
                                            plot_ax1, **kwargs)

        self.plot_frame.canvas.draw()
        self.plot_frame.fig.savefig(self.param['figfilename'])
        message = 'Wavelength solution plotted in {}'.format(
                    self.param['figfilename'])
        print(message)

    def on_click(self, event):
        """Response function of clicking the axes.

        Double click means find the local peak and prepare to add a new
        identified line.
        """
        # double click on ax1: want to add a new identified line
        if event.inaxes == self.plot_frame.ax1 and event.dblclick:
            fig = self.plot_frame.fig
            ax1 = self.plot_frame.ax1
            line_frame = self.info_frame.line_frame
            aperture = self.param['aperture']
            if aperture in self.identlist:
                list1 = self.identlist[aperture]

            # get width of current ax in pixel
            x1, x2 = ax1.get_xlim()
            y1, y2 = ax1.get_ylim()
            iarray = fig.dpi_scale_trans.inverted()
            bbox = ax1.get_window_extent().transformed(iarray)
            width, height = bbox.width, bbox.height
            axwidth_pixel = width*fig.dpi
            # get physical Values Per screen Pixel (VPP) in x direction
            vpp = abs(x2-x1)/axwidth_pixel

            # check if peak has already been identified
            if aperture in self.identlist:

                dist = np.array([abs(line.get_xdata()[0] - event.xdata)/vpp
                                 for line, text in self.ident_objects])

                if dist.min() < 5:
                    # found. change the color of this line
                    imin = dist.argmin()
                    for i, (line, text) in enumerate(self.ident_objects):
                        if i == imin:
                            plt.setp(line, color='b')
                            plt.setp(text, color='b')
                        else:
                            plt.setp(line, color='k')
                            plt.setp(text, color='k')
                    # redraw the canvas
                    self.plot_frame.canvas.draw()

                    wl = list1[imin]['wavelength']

                    # select this line in the linetable
                    for i, record in enumerate(line_frame.item_lst):
                        item = record[0]
                        wave = record[1]
                        if abs(wl - wave)<1e-3:
                            break

                    line_frame.line_tree.selection_set(item)
                    pos = i/float(len(line_frame.item_lst))
                    line_frame.line_tree.yview_moveto(pos)

                    # put the wavelength in the search bar
                    line_frame.search_text.set(str(wl))

                    # update the status of 3 buttons
                    line_frame.clr_button.config(state=tk.NORMAL)
                    line_frame.add_button.config(state=tk.DISABLED)
                    line_frame.del_button.config(state=tk.NORMAL)

                    # end this function without more actions
                    return True
                else:
                    # not found. all of the colors are normal
                    for line, text in self.ident_objects:
                        plt.setp(line, color='k')
                        plt.setp(text, color='k')

                    # clear the search bar
                    line_frame.search_text.set('')

                    # de-select the line table
                    sel_items = line_frame.line_tree.selection()
                    line_frame.line_tree.selection_remove(sel_items)

                    # update the status of 3 button
                    line_frame.clr_button.config(state=tk.DISABLED)
                    line_frame.add_button.config(state=tk.DISABLED)
                    line_frame.del_button.config(state=tk.DISABLED)

                    # continue to act

            flux = self.spec[aperture]

            # find peak
            # first, find the local maximum around the clicking point
            # the searching window is set to be +-5 pixels
            i0 = int(event.xdata)
            i1 = max(int(round(i0 - 5*vpp)), 0)
            i2 = min(int(round(i0 + 5*vpp)), flux.size)
            local_max = i1 + flux[i1:i2].argmax()
            # now found the local max point
            window_size = self.param['window_size']
            _, _, param, _ = find_local_peak(flux, local_max, window_size,
                                             #figname='fit-{:02d}-{:04d}.png'.format(aperture, i1)
                                             )

            # unpack the fitted parameters
            amplitude  = param[0]
            center     = param[1]
            fwhm       = param[2]
            background = param[3]  ### latest-updated

            self.param['i1']         = i1  ### latest-updates
            self.param['i2']         = i2  ### latest-updates
            self.param['center']     = center
            self.param['amplitude']  = amplitude
            self.param['fwhm']       = fwhm
            self.param['background'] = background  ### latest-updates


            # temporarily plot this line
            self.plot_aperture()

            line_frame.clr_button.config(state=tk.NORMAL)

            # guess the input wavelength
            guess_wl = guess_wavelength(center, aperture, self.identlist,
                                        self.linelist, self.param)

            if guess_wl is None:
                # wavelength guess failed
                #line_frame.search_entry.focus()
                # update buttons
                line_frame.add_button.config(state=tk.NORMAL)
                line_frame.del_button.config(state=tk.DISABLED)
            else:
                # wavelength guess succeed

                # check whether wavelength has already been identified
                if is_identified(guess_wl, self.identlist, aperture):
                    # has been identified, do nothing
                    # update buttons
                    line_frame.add_button.config(state=tk.DISABLED)
                    line_frame.del_button.config(state=tk.NORMAL)
                else:
                    # has not been identified yet
                    # put the wavelength in the search bar
                    line_frame.search_text.set(str(guess_wl))
                    # select this line in the linetable
                    for i, record in enumerate(line_frame.item_lst):
                        iid  = record[0]
                        wave = record[1]
                        if abs(guess_wl - wave)<1e-3:
                            break
                    line_frame.line_tree.selection_set(iid)
                    pos = i/float(len(line_frame.item_lst))
                    line_frame.line_tree.yview_moveto(pos)
                    # update buttons
                    line_frame.add_button.config(state=tk.NORMAL)
                    line_frame.del_button.config(state=tk.DISABLED)
                    # unset focus
                    self.focus()

    def on_add_ident(self):
        """Response function of identifying a new line.
        """
        aperture = self.param['aperture']
        k        = self.param['k']
        offset   = self.param['offset']

        line_frame = self.info_frame.line_frame

        if aperture not in self.identlist:
            self.identlist[aperture] = np.array([], dtype=identlinetype)

        list1 = self.identlist[aperture]

        pixel      = self.param['center']
        amplitude  = self.param['amplitude']
        fwhm       = self.param['fwhm']
        i1         = self.param['i1']
        i2         = self.param['i2']
        background = self.param['background']

        selected_iid_lst = line_frame.line_tree.selection()
        iid = selected_iid_lst[0]
        wavelength = float(line_frame.line_tree.item(iid, 'values')[0])
        line_frame.line_tree.selection_remove(selected_iid_lst)

        # find the insert position
        insert_pos = np.searchsorted(list1['pixel'], pixel)

        if None in [k, offset]:
            order = 0
        else:
            order = k*aperture + offset

        item = np.array((aperture, order, wavelength, i1, i2, pixel,
                        amplitude, fwhm, background, -1., True, 0.0,
                        'm'), dtype=identlinetype)

        # insert into identified line list
        self.identlist[aperture] = np.insert(self.identlist[aperture],
                                             insert_pos, item)

        # reset line fitting parameters
        self.param['i1']         = None
        self.param['i2']         = None
        self.param['center']     = None
        self.param['amplitude']  = None
        self.param['fwhm']       = None
        self.param['background'] = None

        # reset the line table
        line_frame.search_text.set('')

        # update the status of 3 buttons
        line_frame.clr_button.config(state=tk.NORMAL)
        line_frame.add_button.config(state=tk.DISABLED)
        line_frame.del_button.config(state=tk.NORMAL)

        self.update_fit_buttons()

        # replot
        self.plot_aperture()

    def on_delete_ident(self):
        """Response function of deleting an identified line.
        """
        line_frame = self.info_frame.line_frame
        target_wl = float(line_frame.search_text.get())
        aperture = self.param['aperture']
        list1 = self.identlist[aperture]

        wl_diff = np.abs(list1['wavelength'] - target_wl)
        mindiff = wl_diff.min()
        argmin  = wl_diff.argmin()
        if mindiff < 1e-3:
            # delete this line from ident list
            list1 = np.delete(list1, argmin)
            self.identlist[aperture] = list1

            # clear the search bar
            line_frame.search_text.set('')

            # de-select the line table
            sel_items = line_frame.line_tree.selection()
            line_frame.line_tree.selection_remove(sel_items)

            # update the status of 3 buttons
            line_frame.clr_button.config(state=tk.DISABLED)
            line_frame.add_button.config(state=tk.DISABLED)
            line_frame.del_button.config(state=tk.DISABLED)

            # update fit buttons
            self.update_fit_buttons()

            # replot
            self.plot_aperture()

    def on_draw(self, event):
        """Response function of drawing.
        """
        if self.param['mode'] == 'ident':
            ax1 = self.plot_frame.ax1
            aperture = self.param['aperture']
            self.param['xlim'][aperture] = ax1.get_xlim()
            self.param['ylim'][aperture] = ax1.get_ylim()

    def update_fit_buttons(self):
        """Update the status of fitting buttons.
        """
        nident = 0
        for aperture, list1 in self.identlist.items():
            nident += list1.size

        xorder = self.param['xorder']
        yorder = self.param['yorder']

        info_frame = self.info_frame

        if nident > (xorder+1)*(yorder+1) and len(self.identlist) > yorder+1:
            info_frame.fit_button.config(state=tk.NORMAL)
        else:
            info_frame.fit_button.config(state=tk.DISABLED)

        if len(self.param['coeff'])>0:
            info_frame.switch_button.config(state=tk.NORMAL)
            info_frame.auto_button.config(state=tk.NORMAL)
        else:
            info_frame.switch_button.config(state=tk.DISABLED)
            info_frame.auto_button.config(state=tk.DISABLED)

    def update_batch_buttons(self):
        """Update the status of batch buttons (recenter and clearall).
        """
        # count how many identified lines
        nident = 0
        for aperture, list1 in self.identlist.items():
            nident += list1.size

        info_frame = self.info_frame

        if nident > 0:
            info_frame.recenter_button.config(state=tk.NORMAL)
            info_frame.clearall_button.config(state=tk.NORMAL)
        else:
            info_frame.recenter_button.config(state=tk.DISABLED)
            info_frame.clearall_button.config(state=tk.DISABLED)

def ident_wavelength(spec, figfilename, title, linelist, identfilename=None,
    window_size=13, xorder=3, yorder=3, maxiter=10, clipping=3,
    q_threshold=10, fit_filter=None
    ):
    """Identify the wavelengths of emission lines in the spectrum of a
    hollow-cathode lamp.

    Args:
        spec (:class:`numpy.dtype`): 1-D spectra.
        figfilename (str): Name of the output wavelength figure to be saved.
        title (str): A string to display as the title of calib figure.
        linelist (str): Name of wavelength standard file.
        identfilename (str): Name of an ASCII formatted wavelength identification
            file.
        window_size (int): Size of the window in pixel to search for the
            lines.
        xorder (int): Degree of polynomial along X direction.
        yorder (int): Degree of polynomial along Y direction.
        maxiter (int): Maximim number of interation in polynomial fitting.
        clipping (float): Threshold of sigma-clipping.
        q_threshold (float): Minimum *Q*-factor of the spectral lines to be
            accepted in the wavelength fitting.
        fit_filter (function): Function checking if a pixel/oder combination is
            within the accepted range.

    Returns:
        dict: A dict containing:

            * **coeff** (:class:`numpy.ndarray`) – Coefficient array.
            * **npixel** (*int*) – Number of pixels along the main dispersion
              direction.
            * **k** (*int*) – Coefficient in the relationship `order =
              k*aperture + offset`.
            * **offset** (*int*) – Coefficient in the relationship `order =
              k*aperture + offset`.
            * **std** (*float*) – Standard deviation of wavelength fitting in Å.
            * **nuse** (*int*) – Number of lines used in the wavelength
              fitting.
            * **ntot** (*int*) – Number of lines found in the wavelength
              fitting.
            * **identlist** (*dict*) – Dict of identified lines.
            * **window_size** (*int*) – Length of window in searching the
              line centers.
            * **xorder** (*int*) – Order of polynomial along X axis in the
              wavelength fitting.
            * **yorder** (*int*) – Order of polynomial along Y axis in the
              wavelength fitting.
            * **maxiter** (*int*) – Maximum number of iteration in the
              wavelength fitting.
            * **clipping** (*float*) – Clipping value of the wavelength fitting.
            * **q_threshold** (*float*) – Minimum *Q*-factor of the spectral
              lines to be accepted in the wavelength fitting.

    Notes:
        If **identfilename** is given and exist, load the identified wavelengths
        from this ASCII file, and display them in the calibration window. If not
        exist, save the identified list into **identfilename** with ASCII
        format.

    See also:
        :func:`recalib`
    """

    # initialize fitting list
    if identfilename is not None and os.path.exists(identfilename):
        identlist, _ = load_ident(identfilename)
    else:
        identlist = {}

    # load the wavelengths
    linefilename = search_linelist(linelist)
    if linefilename is None:
        print('Error: Cannot find linelist file: %s'%linelist)
        exit()
    line_list = load_linelist(linefilename)

    # display an interactive figure
    # reset keyboard shortcuts
    mpl.rcParams['keymap.pan']        = ''   # reset 'p'
    mpl.rcParams['keymap.fullscreen'] = ''   # reset 'f'
    mpl.rcParams['keymap.back']       = ''   # reset 'c'

    # initialize tkinter window
    master = tk.Tk()
    master.resizable(width=False, height=False)

    screen_width  = master.winfo_screenwidth()
    screen_height = master.winfo_screenheight()

    fig_width  = 1300 #2500
    fig_height = 900 #1500
    fig_dpi    = 150 #150
    if None in [fig_width, fig_height]:
        # detremine window size and position
        window_width = int(screen_width-200)
        window_height = int(screen_height-200)
    else:
        window_width = fig_width + 500
        window_height = fig_height + 34

    x = int((screen_width-window_width)/2.)
    y = int((screen_height-window_height)/2.)
    master.geometry('%dx%d+%d+%d'%(window_width, window_height, x, y))

    # display window
    calibwindow = CalibWindow(master,
                              width       = window_width,
                              height      = window_height-34,
                              dpi         = fig_dpi,
                              spec        = spec,
                              figfilename = figfilename,
                              title       = title,
                              identlist   = identlist,
                              linelist    = line_list,
                              window_size = window_size,
                              xorder      = xorder,
                              yorder      = yorder,
                              maxiter     = maxiter,
                              clipping    = clipping,
                              q_threshold = q_threshold,
                              fit_filter  = fit_filter,
                              )

    master.mainloop()

    coeff  = calibwindow.param['coeff']
    npixel = calibwindow.param['npixel']
    k      = calibwindow.param['k']
    offset = calibwindow.param['offset']

    # find the direction code
    aper = 0
    order = k*aper + offset
    wl = get_wavelength(coeff, npixel, np.arange(npixel), order)
    # refresh the direction code
    if wl[0] < wl[-1]:
        sign = '+'
    else:
        sign = '-'
    new_direction = 'x' + {1:'r', -1:'b'}[k] + sign

    # organize results
    result = {
              'coeff':       coeff,
              'npixel':      npixel,
              'k':           k,
              'offset':      offset,
              'std':         calibwindow.param['std'],
              'nuse':        calibwindow.param['nuse'],
              'ntot':        calibwindow.param['ntot'],
              'identlist':   calibwindow.identlist,
              'window_size': calibwindow.param['window_size'],
              'xorder':      calibwindow.param['xorder'],
              'yorder':      calibwindow.param['yorder'],
              'maxiter':     calibwindow.param['maxiter'],
              'clipping':    calibwindow.param['clipping'],
              'q_threshold': calibwindow.param['q_threshold'],
              'direction':   new_direction,
            }

    # save ident list
    if len(calibwindow.identlist)>0 and \
        identfilename is not None and not os.path.exists(identfilename):
        save_ident(calibwindow.identlist, calibwindow.param['coeff'],
                    identfilename)

    return result

def reference_self_wavelength(spec, calib):
    """Calculate the wavelengths for an one dimensional spectra.

    Args:
        spec (:class:`numpy.dtype`):
        calib (tuple):

    Returns:
        tuple: A tuple containing:
    """

    # calculate the wavelength for each aperture
    order_lst = []
    wavelength_lst = []
    for irow, row in enumerate(spec):
        aperture = irow
        npoints  = row.size
        order = aperture*calib['k'] + calib['offset']
        wavelength = get_wavelength(calib['coeff'], calib['npixel'],
                    np.arange(npoints), np.repeat(order, npoints))
        order_lst.append(order)
        wavelength_lst.append(wavelength)

    # pack the identfied line list
    identlist = []
    for aperture, list1 in calib['identlist'].items():
        for row in list1:
            identlist.append(row)
    identlist = np.array(identlist, dtype=list1.dtype)

    return order_lst, wavelength_lst, identlist



if __name__=='__main__':

    filename = sys.argv[1]
    fileid = filename[0:-5]
    hdulst = fits.open(filename)
    spec = hdulst[0].data
    nx = spec.shape[1]
    hdulst.close()


    figname = 'wlcalib.png'
    wlcalib_fig = os.path.join('./', figname)
    
    title = '{}.fits'.format(fileid)

    # pop up a calibration window and identify
    # the wavelengths manually
    calib = ident_wavelength(spec,
                figfilename = wlcalib_fig,
                title       = title,
                linelist    = 'thar.dat',
                window_size = 17,
                xorder      = 3,
                yorder      = 3,
                maxiter     = 5,
                clipping    = 3.0,
                q_threshold = 10,
                )

    # reference the ThAr spectra
    order_lst, wavelength_lst, identlist = reference_self_wavelength(spec, calib)
    
    newspectype = np.dtype(
            {'names': ['order','wavelength','flux'],
             'formats': [np.int32, (np.float64, nx), (np.float32, nx)],
             })
    newspec = []
    for irow, row in enumerate(spec):
        order = order_lst[irow]
        wave = wavelength_lst[irow]
        newspec.append((order, wave, row))
    newspec = np.array(newspec, dtype=newspectype)

    hdu_lst = fits.HDUList([
                fits.PrimaryHDU(),
                fits.BinTableHDU(newspec),
                fits.BinTableHDU(identlist),
                ])

    newfilename = 'wlcalib_{}.fits'.format(fileid)
    hdu_lst.writeto(newfilename, overwrite=True)
    message = 'Wavelength calibrated spectra written to {}'.format(newfilename)
    print(message)
