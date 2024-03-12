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
    tiff = None
    return fig, ax

def make_tfw(file:str,outprefix:str):
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
    :param type: str or list
    :param outputprefix: Output File prefix (eg. MyImage, will be saved as MyImage.tfw)
    :param type: str
    '''
    im = gdal.Open(file)
    gt = im.GetGeoTransform()
    outstr = f'{gt[1]}\n{gt[4]}\n{gt[2]}\n{gt[5]}\n{gt[0]}\n{gt[3]}'
    if outprefix is None:
        outprefix = file[-4:]
    if isinstance(outprefix,str):
        outprefix = [outprefix]
    for string in outprefix:
        print(f'writing {outstr} to',string[:-4]+'.tfw')
        f = open(string[:-4]+'.tfw','w')
        f.write(outstr)
        f.close()
    # close tif
    im = None
    return outstr

def getOutputBounds(image_ds):
    '''Takes a geotransform and conputes outpus bounds (ulx, uly, lrx, lry).'''
    gt = image_ds.GetGeoTransform()
    return [gt[0], gt[3], gt[0] + (gt[1] * image_ds.RasterXSize), gt[3] + (gt[5] * image_ds.RasterYSize)]

def getOverlap(im1, im2):
    '''Takes two open geotiff images (in same reference system!) and computes minx, miny, maxx, maxy.'''
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

def save_geotiff(data, output_path, geotransform, projection,dtype=gdal.GDT_Float32,nodata=-999):
    # Get the shape of the input data
    rows, cols = data.shape

    # Create a driver
    driver = gdal.GetDriverByName('GTiff')

    # Create the output GeoTIFF file
    out_data = driver.Create(output_path, cols, rows, 1, dtype)

    # Write the data to the band
    out_band = out_data.GetRasterBand(1)
    out_band.WriteArray(data)
    out_band.SetNoDataValue(nodata)

    # Set the geotransform and projection
    out_data.SetGeoTransform(geotransform)
    out_data.SetProjection(projection)

    # Close the file
    out_data = None
    return

def micmacExport(tiffile, outname=None, srs=None, outres=None, interp=None, a_ullr=None,cutlineDSName=None,dtype=gdal.GDT_Float32):
    '''Extracts the 1st band of a tif image and saves as float32 greyscale image for micmac ingest.
       Optional SRS code and bounds [ulx, uly, lrx, lry]. Cutline can be used to crop irregular shapes.
       Output no data value is -999.'''
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
                      outputType=dtype,dstNodata=-999)
    # Close files

    imout = None
    new_ds = None
    im = None
    return
  
def micmacPostProcessing(folder:str,
                           prefile:str,
                           outprefix:str=None,
                           dtype:int=gdal.GDT_Float32):
    '''
    Takes a MicMac output folder, and uses gdal to calculate NS and EW displacement tifs, and corresponding correlation files.
    
    :param folder: folder with micmac results.
    :param type: str
    :param prefile: file used as the reference image in the correlation.
    :param type: str
    :param dtype: gdal data type, defaults to gdal.GDT_Float32
    :type dtype: int
    '''
    refim = gdal.Open(prefile)
    gt = refim.GetGeoTransform()
    res = gt[1]
    refimNodata = refim.GetRasterBand(1).GetNoDataValue()
    nodata_mask = (refim != refimNodata)

    if outprefix is None:
        outprefix = folder
    
    # NS
    px2ds = gdal.Open(folder+'Px2_Num5_DeZoom1_LeChantier.tif')
    px2 = px2ds.GetRasterBand(1).ReadAsArray() * 0.05 * -1*res
    # Mask NoData values, considering only non-Nodata pixels
    px2[~nodata_mask] = refimNodata
    # Save in a new, georeferenced file
    save_geotiff(px2, outprefix+'NSmicmac.tif', geotransform=gt, projection=refim.GetProjection(),
                 nodata=refimNodata,dtype=dtype)
    px2ds = None

    # EW
    px1ds = gdal.Open(folder+'Px1_Num5_DeZoom1_LeChantier.tif')
    px1 = px1ds.GetRasterBand(1).ReadAsArray() * 0.05 * res
    # considering only non-Nodata pixels
    px1[~nodata_mask] = refimNodata
    save_geotiff(px1, outprefix+'EWmicmac.tif', geotransform=gt, projection=refim.GetProjection(),
                 nodata=refimNodata,dtype=dtype)
    px1ds = None

    # Correlation file
    correlds = gdal.Open(folder+'Correl_LeChantier_Num_5.tif')
    correl = (correlds.GetRasterBand(1).ReadAsArray()-127.5)/127.5
    # Mask NoData values, considering only non-Nodata pixels
    correl[~nodata_mask] = refimNodata
    save_geotiff(correl, outprefix+'Correlmicmac.tif', geotransform=gt, projection=refim.GetProjection(),
                 nodata=refimNodata,dtype=dtype)
    correlds = None
    refim = None
    return

