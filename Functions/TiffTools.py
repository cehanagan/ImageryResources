from osgeo import gdal, ogr
import matplotlib.pyplot as plt
import numpy as np

def plot_tiff(file:str,mask:list=None,cmap:str=None,name:str=None):
    '''
    Plot band 1 of a tiff file. 
    :param file: File name
    :param type: str
    :param mask: Min and Max values for masking the array. e.g. [0,1]
    :param type: list
    :param cmap: Maplotlib colormap for vizualization.
    :param type: str
    :param name: Title
    :param type: str
    :return: matplotlib.pyplot fig and ax
    :rtype: matplotlib.figure.Figure, matplotlib.axes._axes.Axes
    '''
    tiff = gdal.Open(file)
    imarray = tiff.GetRasterBand(1).ReadAsArray()
    if mask == None:
        mask = [np.min(imarray), np.max(imarray)]
    masked = np.clip(imarray, mask[0], mask[1])
    fig, ax = plt.subplots()
    ax.imshow(masked)
    plt.show()
    return fig, ax

def make_tfw(file:str,outprefix:str=None):
    '''
    Takes in a tiff file name and produces the legacy tfw file:

    The tfw file is a 6-line file:

    Line 1: pixel size in the x-direction in map units (GSD).
    Line 2: rotation about y-axis. 
    Line 3: rotation about x-axis.
    Line 4: pixel size in the y-direction in map in map units (GSD).
    Line 5: x-coordinate of the upper left corner of the image.
    Line 6: y-coordinate of the upper left corner of the image.

    :param file: File name

    '''
    if outprefix == None:
        outprefix = file[-4:]
    im = gdal.Open(file)
    gt = im.GetGeoTransform()
    outstr = f'{gt[1]}\n{gt[4]}\n{gt[2]}\n{gt[5]}\n{gt[0]}\n{gt[3]}'
    print(f'writing {outstr} to',outprefix[:-4]+'.tfw')
    f = open(outprefix[:-4]+'.tfw','w')
    f.write(outstr)
    f.close()
    return outstr

def getOutputBounds(image_ds):
    '''Takes a geotransform and conputes outpus bounds (ulx, uly, lrx, lry).'''
    gt = image_ds.GetGeoTransform()
    return [gt[0], gt[3], gt[0] + (gt[1] * image_ds.RasterXSize), gt[3] + (gt[5] * image_ds.RasterYSize)]

def getOverlap(im1, im2):
    '''Takes two geotiff images (in same reference system!) and computes minx, miny, maxx, maxy.'''
    r1 = getOutputBounds(im1)
    r2 = getOutputBounds(im2)
    print('(ulx, uly, lrx, lry)')
    print('\t1 bounding box: %s' % str(r1))
    print('\t2 bounding box: %s' % str(r2))
    # find intersection between bounding boxes
    minx, maxy = max(r1[0], r2[0]), min(r1[1], r2[1])
    maxx, miny = min(r1[2], r2[2]), max(r1[3], r2[3])
    print('minx, miny, maxx, maxy:')
    print(minx, miny, maxx, maxy)
    return [minx, miny, maxx, maxy]

def micmacExport(tiffile, outname=None, srs=None, outres=None, interp=None, a_ullr=None,cutlineDSName=None):
    '''Extracts the 1st band of a tif image and saves as float32 greyscale image for micmac ingest.
       Optional SRS code and bounds [ulx, uly, lrx, lry].'''
    im = gdal.Open(tiffile)
    if im is None:
        print("Error: Unable to open the input file.")
        return None, None

    if outname is None:
        outname = tiffile
    if srs is None:
        srs = im.GetProjection()
    if outres is None:
        outres = [im.GetGeoTransform()[1], im.GetGeoTransform()[-1]]
    if interp is None:
        interp = 'cubic'

    if im.RasterCount >= 3:
        print('Computing Gray from RGB values')
        # Read RGB bands
        R = im.GetRasterBand(1).ReadAsArray()
        G = im.GetRasterBand(2).ReadAsArray()
        B = im.GetRasterBand(3).ReadAsArray()
        # Mask NoData values
        nodata_mask = (R != im.GetRasterBand(1).GetNoDataValue()) & \
                      (G != im.GetRasterBand(2).GetNoDataValue()) & \
                      (B != im.GetRasterBand(3).GetNoDataValue())
        # Create a grayscale version of the bands, considering only non-Nodata pixels
        grayscale_band = 0.2989 * R + 0.5870 * G + 0.1140 * B
        grayscale_band[~nodata_mask] = im.GetRasterBand(1).GetNoDataValue()
    else:
        # Read the 1st band
        grayscale_band = im.GetRasterBand(1).ReadAsArray()
    print('Writing to', outname)
    # Create new dataset and band for writing
    driver = gdal.GetDriverByName('GTiff')
    new_ds = driver.Create(outname, im.RasterXSize, im.RasterYSize, 1, gdal.GDT_Float32)
    new_ds.SetProjection(srs)
    new_ds.SetGeoTransform(im.GetGeoTransform())
    new_band = new_ds.GetRasterBand(1)
    new_band.WriteArray(grayscale_band)

    # Set optional parameters
    if a_ullr is not None:
        bounds = [a_ullr[0], a_ullr[3], a_ullr[2], a_ullr[1]]
    else:
        bounds = None

    # Warp the dataset
    imout = gdal.Warp(outname, new_ds, xRes=outres[0], yRes=outres[1],
                      outputBounds=bounds,cutlineDSName=cutlineDSName, dstSRS=srs, resampleAlg=interp,
                      outputType=gdal.GDT_Float32)
    imout = None
    

def save_geotiff(data, output_path, geotransform, projection):
    # Get the shape of the input data
    rows, cols = data.shape

    # Create a driver
    driver = gdal.GetDriverByName('GTiff')

    # Create the output GeoTIFF file
    out_data = driver.Create(output_path, cols, rows, 1, gdal.GDT_Float32)

    # Write the data to the band
    out_band = out_data.GetRasterBand(1)
    out_band.WriteArray(data)

    # Set the geotransform and projection
    out_data.SetGeoTransform(geotransform)
    out_data.SetProjection(projection)

    # Close the file
    out_data = None


