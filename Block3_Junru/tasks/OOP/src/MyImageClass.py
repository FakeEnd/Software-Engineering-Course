from re import L
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

#define an image class that has useful functions we can use!

class MyImgClass():
    """
    This is the custom class for our lesson 2.
    It contains all the methods to run the accompanying notebook.
    Many you will be implementing.

    """
    
    def __init__(self, arrImg, intLabel=None):
        """Initalize the MyImgClass object

        :param arrImg: a numpy array that contains the image
        :type arrImg: np.array
        :param intLabel: the label value, defaults to None
        :type intLabel: int, optional
        """
        self.arrImg = arrImg
        self.intLabel = intLabel
        self.shape = self.arrImg.shape
        
    def __add__(self, other):
        """
        operator overloading for the '+' operation 
        element wise addtion of two MyImgClass together.

        :param other: the other instance of arrImgClass
        :type other: arrImgClass
        :return: the addition of the two classes
        :rtype: MyImgClass
        """
        toReturn = MyImgClass(np.add(self.arrImg, other.arrImg), intLabel=None)
        return toReturn
    
    def __sub__(self, other):
        """
        operator overloading for the '-' operation 
        element wise subtraction of two MyImgClass together.

        :param other: the other instance of arrImgClass
        :type other: arrImgClass
        :return: the subtraction of the two classes
        :rtype: MyImgClass
        """
        ### TO IMPLEMENT ###
        raise NotImplementedError('You need to implement this method!')
        
    
    def fPixelwiseSqDif(self, other):
        """Find the square difference between two MyImgClass objects

        :param other: the other instance of arrImgClass
        :type other: arrImgClass
        :return: square difference of each pixel
        :rtype: MyImgClass
        """
        ### TO IMPLEMENT ###
        # Use the overloaded '-' from above
        raise NotImplementedError('You need to implement this method!')
    
    
    
    def fMSE(self, other):
        """Find the mean squared error between two images

        :param other: the other instance of arrImgClass
        :type other: arrImgClass
        :return: MSE
        :rtype: float
        """
        ### TO IMPLEMENT ###
        raise NotImplementedError('You need to implement this method!')

    
    def fPlot(self, ax, show_ticks=False, add_colorbar=False, imshow_kwargs={}):
        """Plotting method for the class

        :param ax: the axis to plot on
        :type ax: matplotlib.pyplot.ax
        :param show_ticks: display the x and y axis ticks/labels, defaults to False
        :type show_ticks: bool, optional
        :param add_colorbar: display a color bar, defaults to False
        :type add_colorbar: bool, optional
        :param imshow_kwargs: additional keywords, defaults to {}
        :type imshow_kwargs: dict, optional
        """
        img = ax.imshow(self.arrImg, interpolation='None', **imshow_kwargs)
        if not self.intLabel is None:
            ax.set_title(f'Label: {self.intLabel}')
        if show_ticks is False:
            ax.axes.xaxis.set_ticks([])
            ax.axes.yaxis.set_ticks([])
        if add_colorbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            plt.colorbar(img, cax=cax, orientation='vertical')        
        
    @staticmethod
    def fComputeMeanAcrossImages(lMyImgClass):
        """Calcualte the mean image from a list of images

        :param lMyImgClass: list of MyImgClass
        :type lMyImgClass: list (or 1D iterable)
        :return: The mean image
        :rtype: MyImgClass
        """
        ### TO IMPLEMENT ###
        raise NotImplementedError('You need to implement this method!')
    
    @staticmethod
    def fComputeStdAcrossImages(lMyImgClass):
        """Calcualte the std image from a list of images

        :param lMyImgClass: list of MyImgClass
        :type lMyImgClass: list (or 1D iterable)
        :return: The std image
        :rtype: MyImgClass
        """
        ### TO IMPLEMENT ###
        raise NotImplementedError('You need to implement this method!')
        

    @staticmethod
    def fMeanMSE(lImg1, lImg2):
        """Calcualte the mean MSE across pairs of images
        e.g. mean(MSE(lImg1[0],lImg2[0]),MSE(lImg1[1],lImg2[1])...)

        :param lImg1: list of MyImgClass
        :type lImg1: list (or 1D iterable)
        :param lImg2: list of MyImgClass
        :type lImg2: list (or 1D iterable)
        :return: mean MSE
        :rtype: float
        """
        ### TO IMPLEMENT ###
        raise NotImplementedError('You need to implement this method!')


    @staticmethod
    def fMSEforEachPairCombination(lImg1, lImg2):
        """Calcaulte the MSE for all pairs between lImg1 and lImg2

        :param lImg1: list of MyImgClass
        :type lImg1: list (or 1D iterable)
        :param lImg2: list of MyImgClass
        :type lImg2: list (or 1D iterable)
        :return: mean MSE
        :rtype: float
        """
        lSE = []
        for imgI in lImg1:
            for imgJ in lImg2:
                if imgI != imgJ:
                    lSE.append(imgI.fMSE(imgJ))
                else:
                    lSE.append(np.nan)
        return lSE

