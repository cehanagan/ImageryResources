from osgeo import gdal, ogr
from osgeo_utils import gdal_calc
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
import os

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

def save_geotiff(data, output_path, geotransform, projection,nodata=-9999):
    # Get the shape of the input data
    rows, cols = data.shape

    # Create a driver
    driver = gdal.GetDriverByName('GTiff')

    # Create the output GeoTIFF file (path, cols, rows, bands, dtype)
    out_data = driver.Create(output_path, cols, rows, 1, gdal.GDT_Float64)
    # Set the geotransform and projection
    out_data.SetGeoTransform(geotransform)
    out_data.SetProjection(projection)
    out_data.GetRasterBand(1).SetNoDataValue(nodata)

    # Write the data to the band
    out_data.GetRasterBand(1).WriteArray(data)

    # Close the file
    out_data = None
    return 

def curl_2d(vector_fieldx,vector_fieldy):
    """
    Compute the curl of a 2D vector field.

    Parameters:
        vector_field (numpy.ndarray): 2D numpy array representing the vector field.
                                       Each row represents a point in the field, and each column represents a component (x, y).

    Returns:
        numpy.ndarray: 1D numpy array representing the curl of the vector field at each point.
    """
    # Extract components
    u = vector_fieldx  # x component
    v = vector_fieldy  # y component

    # Calculate derivatives
    du_dy, du_dx = np.gradient(u)
    dv_dy, dv_dx = np.gradient(v)

    # Compute curl
    curl = (dv_dx - du_dy)

    return curl

def micmacExport(tiffile, outname=None, srs=None, outres=None, interp=None, a_ullr=None,cutlineDSName=None):
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

    nodata = -9999 if not isinstance(im.GetRasterBand(1).GetNoDataValue(), (int, float, complex)) else im.GetRasterBand(1).GetNoDataValue()

    if im.RasterCount >= 3:
        print('Computing Gray from RGB values')
        # Read RGB bands
        R = im.GetRasterBand(1).ReadAsArray()
        G = im.GetRasterBand(2).ReadAsArray()
        B = im.GetRasterBand(3).ReadAsArray()
        # Mask NoData values
        nodata_mask = (R != nodata) & \
                      (G != nodata) & \
                      (B != nodata)
        # Create a grayscale version of the bands, considering only non-Nodata pixels
        grayscale_band = 0.2989 * R + 0.5870 * G + 0.1140 * B
    else:
        # Read the 1st band
        grayscale_band = im.GetRasterBand(1).ReadAsArray()
        nodata_mask = (grayscale_band != nodata)
    grayscale_band[~nodata_mask] = nodata
    print('Writing to', outname)
    # Create new dataset and band for writing
    driver = gdal.GetDriverByName('GTiff')
    new_ds = driver.Create(outname, im.RasterXSize, im.RasterYSize, 1, gdal.GDT_Float32)
    new_ds.SetProjection(im.GetProjection())
    new_ds.SetGeoTransform(im.GetGeoTransform())
    new_band = new_ds.GetRasterBand(1)
    new_band.WriteArray(grayscale_band)
    
    # Set optional parameters
    if a_ullr is not None:
        bounds = [a_ullr[0], a_ullr[3], a_ullr[2], a_ullr[1]]
    else:
        bounds = None

    if cutlineDSName is None:
        cropToCutline = False
    else:
        cropToCutline = True
    # Warp the dataset
    imout = gdal.Warp(outname, new_ds, xRes=outres[0], yRes=outres[1],
                      outputBounds=bounds,cutlineDSName=cutlineDSName,cropToCutline=cropToCutline,dstSRS=srs, resampleAlg=interp,
                      dstNodata=-9999)
    # Close files
    imout = None
    new_ds = None
    im = None
    return 
  
def micmacPostProcessing(folder:str,
                         prefile:str,
                         outprefix:str=None):
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
    if refimNodata == None:
        print('Setting nodata value to 0, because reference had no specified value.')
        refimNodata = 0
    nodata_mask = (refim.GetRasterBand(1).ReadAsArray() != refimNodata)

    if outprefix is None:
        outprefix = folder
    
    # NS
    px2ds = gdal.Open(folder+'Px2_Num5_DeZoom1_LeChantier.tif')
    px2 = px2ds.GetRasterBand(1).ReadAsArray() * 0.05 * -1*res
    # Mask NoData values, considering only non-Nodata pixels
    px2[~nodata_mask] = refimNodata
    # Save in a new, georeferenced file
    print('Saving',outprefix+'NSmicmac.tif')
    save_geotiff(px2, outprefix+'NSmicmac.tif', geotransform=gt, projection=refim.GetProjection(),
                 nodata=refimNodata)
    px2ds = None

    # EW
    px1ds = gdal.Open(folder+'Px1_Num5_DeZoom1_LeChantier.tif')
    px1 = px1ds.GetRasterBand(1).ReadAsArray() * 0.05 * res
    # considering only non-Nodata pixels
    px1[~nodata_mask] = refimNodata
    print('Saving',outprefix+'EWmicmac.tif')
    save_geotiff(px1, outprefix+'EWmicmac.tif', geotransform=gt, projection=refim.GetProjection(),
                 nodata=refimNodata)
    px1ds = None

    # Correlation file
    correlds = gdal.Open(folder+'Correl_LeChantier_Num_5.tif')
    correl = (correlds.GetRasterBand(1).ReadAsArray()-127.5)/127.5
    # Mask NoData values, considering only non-Nodata pixels
    correl[~nodata_mask] = refimNodata
    print('Saving',outprefix+'Correlmicmac.tif')
    save_geotiff(correl, outprefix+'Correlmicmac.tif', geotransform=gt, projection=refim.GetProjection(),
                 nodata=refimNodata)
    correlds = None
    refim = None
    return 

def micmacStack(infolderlist,outfolder,az=None):
    '''
    Creates NS, EW, Parrallel, and Perpendicular stacked maps, weighted by correlation score. 
    Each folder in infolderlist should contain the EWmicmac.tif, NSmicmac.tif, and Correlmicmac.tif created from micmacPostProcessing.
    Will fail if images are too large. 
    '''
    disp = {'NS':[],'EW':[],'Co':[]}
    # Add all of the tifs together
    for infile in ['NSmicmac.tif','EWmicmac.tif','Correlmicmac.tif']:
        print(f'Working on {infile}')
        baseim = gdal.Open(infolderlist[0]+infile)
        Comb = baseim.GetRasterBand(1).ReadAsArray()
        if infile == 'NSmicmac.tif':
            nodatamask = (baseim.GetRasterBand(1).ReadAsArray() != baseim.GetRasterBand(1).GetNoDataValue())
        print('Adding values from',infolderlist[0])
        disp[infile[:2]] = np.zeros((np.shape(Comb)[0],np.shape(Comb)[1],len(infolderlist)))
        disp[infile[:2]][:,:,0] = Comb
        for i,folder in enumerate(infolderlist[1:]):
            im = gdal.Open(folder+infile)
            disp[infile[:2]][:,:,i+1] = im.GetRasterBand(1).ReadAsArray()
            print('Adding values from',folder)

    
    corrtot = np.sum(disp['Co'],axis=2)
    corrtot[~nodatamask] = baseim.GetRasterBand(1).GetNoDataValue()

    NSdisp = (((disp['NS'][:,:,0]))*disp['Co'][:,:,0]/corrtot + (disp['NS'][:,:,1])*disp['Co'][:,:,1]/corrtot + \
          (disp['NS'][:,:,2])*disp['Co'][:,:,2]/corrtot + (disp['NS'][:,:,3])*disp['Co'][:,:,3]/corrtot)
    NSdisp[~nodatamask] = baseim.GetRasterBand(1).GetNoDataValue()
    save_geotiff(NSdisp,outfolder+'NSDispStacked.tif',baseim.GetGeoTransform(),baseim.GetProjection())

    EWdisp = (((disp['EW'][:,:,0]))*disp['Co'][:,:,0]/corrtot + (disp['EW'][:,:,1])*disp['Co'][:,:,1]/corrtot + \
          (disp['EW'][:,:,2])*disp['Co'][:,:,2]/corrtot + (disp['EW'][:,:,3])*disp['Co'][:,:,3]/corrtot)
    EWdisp[~nodatamask] = baseim.GetRasterBand(1).GetNoDataValue()
    save_geotiff(EWdisp,outfolder+'EWDispStacked.tif',baseim.GetGeoTransform(),baseim.GetProjection())

    save_geotiff(corrtot,outfolder+'CorrelStacked.tif',baseim.GetGeoTransform(),baseim.GetProjection())

    if az is not None:
        par, perp = projectDisp(outfolder+'EWdispStacked.tif',outfolder+'NSdispStacked.tif',az,mask=None,partif=outfolder+'ParallelDispStacked.tif',perptif=outfolder+'PerpendicularDispStacked.tif')

    return 'Done!'

def micmacSimpleStack(infolderlist,outfolder):
    '''
    Creates NS, EW, UD stacked maps, weighted by correlation score. 
    Each folder in infolderlist should contain the EWmicmac.tif, NSmicmac.tif, and Correlmicmac.tif created from micmacPostProcessing.
    Infolderlist should contain 4 folders, for pairwise pre and post stacking.
    '''
    baseim = gdal.Open(infolderlist[0]+'NSmicmac.tif')

    NSdisp = gdal_calc.Calc(calc='A*E/(E+F+G+H)+B*F/(E+F+G+H)+C*G/(E+F+G+H)+D*H/(E+F+G+H)',
               A=infolderlist[0]+'NSmicmac.tif',B=infolderlist[1]+'NSmicmac.tif',C=infolderlist[2]+'NSmicmac.tif',D=infolderlist[3]+'NSmicmac.tif',
               E=infolderlist[0]+'Correlmicmac.tif',F=infolderlist[1]+'Correlmicmac.tif',G=infolderlist[2]+'Correlmicmac.tif',H=infolderlist[3]+'Correlmicmac.tif',
               outfile = outfolder+'NSdispStacked.tif')
    #save_geotiff(NSdisp,outfolder+'NSDispStacked.tif',baseim.GetGeoTransform(),baseim.GetProjection())
    NSdisp = 0

    EWdisp = gdal_calc.Calc(calc='A*E/(E+F+G+H)+B*F/(E+F+G+H)+C*G/(E+F+G+H)+D*H/(E+F+G+H)',
               A=infolderlist[0]+'EWmicmac.tif',B=infolderlist[1]+'EWmicmac.tif',C=infolderlist[2]+'EWmicmac.tif',D=infolderlist[3]+'EWmicmac.tif',
               E=infolderlist[0]+'Correlmicmac.tif',F=infolderlist[1]+'Correlmicmac.tif',G=infolderlist[2]+'Correlmicmac.tif',H=infolderlist[3]+'Correlmicmac.tif',
               outfile = outfolder+'EWdispStacked.tif')
    #save_geotiff(EWdisp,outfolder+'EWDispStacked.tif',baseim.GetGeoTransform(),baseim.GetProjection())
    EWdisp = 0

    UDdisp = gdal_calc.Calc(calc='A*E/(E+F+G+H)+B*F/(E+F+G+H)+C*G/(E+F+G+H)+D*H/(E+F+G+H)',
               A=infolderlist[0]+'UDmicmac.tif',B=infolderlist[1]+'UDmicmac.tif',C=infolderlist[2]+'UDmicmac.tif',D=infolderlist[3]+'UDmicmac.tif',
               E=infolderlist[0]+'Correlmicmac.tif',F=infolderlist[1]+'Correlmicmac.tif',G=infolderlist[2]+'Correlmicmac.tif',H=infolderlist[3]+'Correlmicmac.tif',
               outfile = outfolder+'UDdispStacked.tif')
    #save_geotiff(UDdisp,outfolder+'UDdispStacked.tif',baseim.GetGeoTransform(),baseim.GetProjection())
    UDdisp = 0

    Correldisp = gdal_calc.Calc(calc='A+B+C+D',
               A=infolderlist[0]+'Correlmicmac.tif',B=infolderlist[1]+'Correlmicmac.tif',C=infolderlist[2]+'Correlmicmac.tif',D=infolderlist[3]+'Correlmicmac.tif',
               outfile = outfolder+'CorreldispStacked.tif')
    #save_geotiff(UDdisp,outfolder+'UDdispStacked.tif',baseim.GetGeoTransform(),baseim.GetProjection())
    Correldisp = 0

    return 'Done!'


def micmacStackUD(infolderlist,outfolder):
    '''
    Creates UD stacked map, weighted by correlation score. 
    Each folder in infolderlist should contain the UDmicmac.tif and Correlmicmac.tif created from micmacPostProcessing and veerticalDisp.

    '''
    disp = {'UD':[],'Co':[]}
    # Add all of the tifs together
    for infile in ['UDmicmac.tif','Correlmicmac.tif']:
        print(f'Working on {infile}')
        baseim = gdal.Open(infolderlist[0]+infile)
        Comb = baseim.GetRasterBand(1).ReadAsArray()
        if infile == 'UDmicmac.tif':
            nodatamask = (baseim.GetRasterBand(1).ReadAsArray() != baseim.GetRasterBand(1).GetNoDataValue())
        print('Adding values from',infolderlist[0])
        disp[infile[:2]] = np.zeros((np.shape(Comb)[0],np.shape(Comb)[1],len(infolderlist)))
        disp[infile[:2]][:,:,0] = Comb
        for i,folder in enumerate(infolderlist[1:]):
            im = gdal.Open(folder+infile)
            disp[infile[:2]][:,:,i+1] = im.GetRasterBand(1).ReadAsArray()
            print('Adding values from',folder)

    
    corrtot = np.sum(disp['Co'],axis=2)
    corrtot[~nodatamask] = baseim.GetRasterBand(1).GetNoDataValue()

    UDdisp = (((disp['UD'][:,:,0]))*disp['Co'][:,:,0]/corrtot + (disp['UD'][:,:,1])*disp['Co'][:,:,1]/corrtot + \
          (disp['UD'][:,:,2])*disp['Co'][:,:,2]/corrtot + (disp['UD'][:,:,3])*disp['Co'][:,:,3]/corrtot)
    UDdisp[~nodatamask] = baseim.GetRasterBand(1).GetNoDataValue()
    save_geotiff(UDdisp,outfolder+'UDDispStacked.tif',baseim.GetGeoTransform(),baseim.GetProjection())

    return 'Done!'

def projectDisp(ewtif,nstif,azimuth,mask=None,partif='ParallelDisp.tif',perptif='PerpendicularDisp.tif'):
    ewds = gdal.Open(ewtif)

    if ewds.GetRasterBand(1).GetNoDataValue() is None:
        nodata = -9999
    else:
        nodata = ewds.GetRasterBand(1).GetNoDataValue()

    if mask is None:
        nodata_mask = (ewds.GetRasterBand(1).ReadAsArray() != nodata)
    else:
        nodata_mask = mask
    ew = ewds.GetRasterBand(1).ReadAsArray()

    nsds = gdal.Open(nstif)
    ns = nsds.GetRasterBand(1).ReadAsArray()

    # Rotation from en (xy) to fault parallel and perp, must convert azimuth
    theta = (90-azimuth)*np.pi/180
    par = ns*np.sin(theta)+ew*np.cos(theta)
    par[~nodata_mask] = nodata
    perp = -1*ns*np.cos(theta)+ew*np.sin(theta)
    perp[~nodata_mask] = nodata

    save_geotiff(par,partif, ewds.GetGeoTransform(), ewds.GetProjection(),nodata=nodata)
    save_geotiff(perp,perptif, ewds.GetGeoTransform(), ewds.GetProjection(),nodata=nodata)

    ewds = None
    nsds = None
    return par, perp

def verticalDisp(dem1file,dem2file,nsfile,ewfile,outf='VerticalDisp.tif'):
    '''Takes in two DEMs and two NS/EW displacment maps (as tifs), and creates a vertical displacement map.
    Note that it operates on a nearest neighbor assumption!!'''

    # Read displacement  files
    nsf = gdal.Open(nsfile)
    ns = nsf.GetRasterBand(1).ReadAsArray()
    ewf = gdal.Open(ewfile)
    ew = ewf.GetRasterBand(1).ReadAsArray()


    rasterSize = np.shape(ns)
    resolution = nsf.GetGeoTransform()[1]
    nodata = nsf.GetRasterBand(1).GetNoDataValue()
    
    # Precompute the movements and the target indices
    movY = np.round(ns / resolution).astype(int)
    movX = np.round(ew / resolution).astype(int)
    

    # Create boolean masks for invalid conditions
    invalid_mask = (
        np.isnan(ns) | np.isnan(ew) |
        np.isnan(movY) | np.isnan(movX) |
        (movY + np.arange(rasterSize[0]).reshape(-1, 1) >= rasterSize[0]) |
        (movX + np.arange(rasterSize[1]) >= rasterSize[1]) |
        (movY + np.arange(rasterSize[0]).reshape(-1, 1) < 0) |
        (movX + np.arange(rasterSize[1]) < 0)
    )
    ns, ew = 0, 0

    # Calculate the target indices, taking care of boundaries
    targetY = np.clip(np.arange(rasterSize[0]).reshape(-1, 1) + movY, 0, rasterSize[0] - 1)
    targetX = np.clip(np.arange(rasterSize[1]) + movX, 0, rasterSize[1] - 1)

    movX, movY = 0, 0

    dem1 = gdal.Open(dem1file)
    u1 = dem1.GetRasterBand(1).ReadAsArray()
    dem2 = gdal.Open(dem2file)
    u2 = dem2.GetRasterBand(1).ReadAsArray()
    
    # Perform the computation for valid indices
    d1 = u1
    d2 = u2[targetY, targetX]

    u1, u2 = 0, 0
    
    U = d2 - d1
    U[invalid_mask] = nodata
    
    save_geotiff(U,outf, dem1.GetGeoTransform(), dem1.GetProjection(),nodata=nodata)

    return U

def reproject_raster_to_match(ref_raster, raster_file):
    with rasterio.open(ref_raster) as ref_src, rasterio.open(raster_file) as src:
        # Read the source raster data
        data = src.read(1)

        # Reproject raster data to match reference raster's transform and CRS
        aligned_data = np.empty((ref_src.height, ref_src.width), dtype=src.dtypes[0])
        reproject(
            source=data,
            destination=aligned_data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_src.transform,
            dst_crs=ref_src.crs,
            resampling=Resampling.bilinear
        )
        return aligned_data, ref_src.transform
    
def verticalDispExtentAgnostic(dem1file, dem2file, nsfile, ewfile, outf='VerticalDisp.tif'):
    """
    Takes in two DEMs and two NS/EW displacement maps (as tifs), and creates a vertical displacement map.
    Operates on a nearest-neighbor assumption.
    """
    # Clip and align all rasters to the first DEM's grid
    dem1_clipped, transform = reproject_raster_to_match(nsfile, dem1file)
    dem2_clipped, _ = reproject_raster_to_match(nsfile, dem2file)
    ns_clipped, _ = reproject_raster_to_match(nsfile, nsfile)
    ew_clipped, _ = reproject_raster_to_match(nsfile, ewfile)

    # Read metadata from one of the displacement files for reference
    with rasterio.open(nsfile) as nsf:
        resolution = nsf.transform[0]
        nodata = nsf.nodata
    raster_size = ns_clipped.shape
    # Precompute the movements and the target indices
    movY = np.round(ns_clipped / resolution).astype(int)
    movX = np.round(ew_clipped / resolution).astype(int)
    # Create boolean masks for invalid conditions
    invalid_mask = (
        np.isnan(ns_clipped) | np.isnan(ew_clipped) |
        (movY + np.arange(raster_size[0]).reshape(-1, 1) >= raster_size[0]) |
        (movX + np.arange(raster_size[1]) >= raster_size[1]) |
        (movY + np.arange(raster_size[0]).reshape(-1, 1) < 0) |
        (movX + np.arange(raster_size[1]) < 0)
    )
    # Calculate the target indices, taking care of boundaries
    targetY = np.clip(np.arange(raster_size[0]).reshape(-1, 1) + movY, 0, raster_size[0] - 1)
    targetX = np.clip(np.arange(raster_size[1]) + movX, 0, raster_size[1] - 1)
    # Perform the computation for valid indices
    d1 = dem1_clipped
    d2 = dem2_clipped[targetY, targetX]
    # Compute vertical displacement
    U = d2 - d1
    U[invalid_mask] = nodata
    # Save result as a GeoTIFF
    with rasterio.open(
        outf,
        "w",
        driver="GTiff",
        height=U.shape[0],
        width=U.shape[1],
        count=1,
        dtype=U.dtype,
        crs=rasterio.open(dem1file).crs,
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(U, 1)

    return U

from scipy.ndimage import map_coordinates

def verticalDispExtentAgnosticBilinInterp(dem1file, dem2file, nsfile, ewfile, outf='VerticalDisp.tif'):
    """
    Computes vertical displacement using two DEMs and NS/EW displacement maps.
    Replaces nearest-neighbor indexing with bilinear interpolation.
    """

    # Clip and align all rasters to the first DEM's grid
    dem1_clipped, transform = reproject_raster_to_match(nsfile, dem1file)
    dem2_clipped, _ = reproject_raster_to_match(nsfile, dem2file)
    ns_clipped, _ = reproject_raster_to_match(nsfile, nsfile)
    ew_clipped, _ = reproject_raster_to_match(nsfile, ewfile)

    # Read metadata from one of the displacement files for reference
    with rasterio.open(nsfile) as nsf:
        resolution = nsf.transform[0]
        nodata = nsf.nodata

    # Calculate displacement in pixel coordinates
    movY = ns_clipped / resolution
    movX = ew_clipped / resolution

    # Get grid indices for interpolation
    y_indices, x_indices = np.meshgrid(np.arange(dem2_clipped.shape[0]),
                                       np.arange(dem2_clipped.shape[1]),
                                       indexing='ij')
    targetY = y_indices + movY
    targetX = x_indices + movX

    # Mask invalid indices
    invalid_mask = (
        np.isnan(ns_clipped) | np.isnan(ew_clipped) |
        (targetY < 0) | (targetY >= dem2_clipped.shape[0]) |
        (targetX < 0) | (targetX >= dem2_clipped.shape[1])
    )

    # Interpolate the second DEM to the displaced locations
    d2_interp = map_coordinates(dem2_clipped, [targetY, targetX], order=1, mode='nearest')
    
    # Compute vertical displacement
    U = d2_interp - dem1_clipped
    U[invalid_mask] = nodata

    # Save result as a GeoTIFF
    with rasterio.open(
        outf,
        "w",
        driver="GTiff",
        height=U.shape[0],
        width=U.shape[1],
        count=1,
        dtype=U.dtype,
        crs=rasterio.open(dem1file).crs,
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(U, 1)

    return U
