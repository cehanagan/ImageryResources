from osgeo import gdal, ogr
from osgeo_utils import gdal_calc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling
import os

from scipy import ndimage
from scipy.special import erf
from scipy.optimize import curve_fit

from pyproj import Proj, Geod
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon

def projectParPerp(ns, ew, az):
    theta = (az)*np.pi/180
    par = ns*np.cos(theta)+ew*np.sin(theta)
    perp = -1*ns*np.sin(theta)+ew*np.cos(theta)
    return par.flatten(), perp.flatten()

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
    
def save_geotiff(data, output_path, geotransform, projection, nodata=-9999):
    # Get the shape of the input data
    rows, cols = data.shape

    # Create a driver
    driver = gdal.GetDriverByName('GTiff')

    # Define creation options
    creation_options = [
        'COMPRESS=LZW',      # LZW compression
        'ZLEVEL=9',          # Maximum compression level
        'BIGTIFF=YES'        # Enable BigTIFF support
    ]

    # Create the output GeoTIFF file with creation options
    out_data = driver.Create(output_path, cols, rows, 1, gdal.GDT_Float32, options=creation_options)

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

def dilation_curl_2d(vector_fieldx, vector_fieldy):
    """
    Compute the dilation (divergence) and curl of a 2D vector field.

    Parameters:
        vector_fieldx (numpy.ndarray): 2D array of the x-component of the vector field.
        vector_fieldy (numpy.ndarray): 2D array of the y-component of the vector field.

    Returns:
        dilation (numpy.ndarray): 2D array of the dilation (∂ux/∂x + ∂uy/∂y).
        curl (numpy.ndarray): 2D array of the curl (∂uy/∂x − ∂ux/∂y).
    """
    # Extract components
    ux = vector_fieldx  # x component
    uy = vector_fieldy  # y component

    # Calculate spatial derivatives
    dux_dy, dux_dx = np.gradient(ux)
    duy_dy, duy_dx = np.gradient(uy)

    # Compute dilation (divergence)
    dilation = dux_dx + duy_dy

    # Compute curl (out-of-plane scalar curl)
    curl = duy_dx - dux_dy

    return dilation, curl

def micmacExport(tiffile, outname=None, srs=None, outres=None, interp=None, a_ullr=None,cutlineDSName=None,nodata=None):
    '''Extracts the 1st band of a tif image and saves as float32 greyscale image for micmac ingest.
       Optional SRS code and bounds [ulx, uly, lrx, lry]. Cutline can be used to crop irregular shapes.
       Output no data value is -9999.'''
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
        interp = 'near'
    if nodata is None:
        nodata = -9999 if not isinstance(im.GetRasterBand(1).GetNoDataValue(), (int, float, complex)) else im.GetRasterBand(1).GetNoDataValue()

    if im.RasterCount >= 3:
        print('Computing Gray from RGB values')
        # Read RGB bands
        R = im.GetRasterBand(1).ReadAsArray()
        G = im.GetRasterBand(2).ReadAsArray()
        B = im.GetRasterBand(3).ReadAsArray()
        # Mask NoData values
        nodata_mask = (R != nodata) | \
                      (G != nodata) | \
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
                         prefiles:str,
                         outprefix:str=None):
    '''
    Takes a MicMac output folder, and uses gdal to calculate NS and EW displacement tifs, and corresponding correlation files.
    
    :param folder: folder with micmac results.
    :param type: str
    :param prefiles: files used in the correlation.
    :param type: str
    :param dtype: gdal data type, defaults to gdal.GDT_Float32
    :type dtype: int
    '''
    refim1 = gdal.Open(prefiles[0])
    refim2 = gdal.Open(prefiles[1])
    gt = refim1.GetGeoTransform()
    res = gt[1]
    refimNodata = refim1.GetRasterBand(1).GetNoDataValue()
    if refimNodata == None:
        print('Setting nodata value to -9999, because reference had no specified value.')
        refimNodata = -9999
    nodata_mask = ((refim1.GetRasterBand(1).ReadAsArray() == refimNodata) | (refim2.GetRasterBand(1).ReadAsArray() == refimNodata))
    print('Nodata value for mask:',refimNodata)
    if outprefix is None:
        outprefix = folder
    print(f'Setting nodata value to -9999')
    # NS
    px2ds = gdal.Open(folder+'Px2_Num5_DeZoom1_LeChantier.tif')
    px2 = px2ds.GetRasterBand(1).ReadAsArray() * 0.05 * -1*res
    # Mask NoData values, considering only non-Nodata pixels
    px2[nodata_mask] = -9999
    # Save in a new, georeferenced file
    print('Saving',outprefix+'NSmicmac.tif')
    save_geotiff(px2, outprefix+'NSmicmac.tif', geotransform=gt, projection=refim1.GetProjection(),
                 nodata=-9999)
    px2ds = None

    # EW
    px1ds = gdal.Open(folder+'Px1_Num5_DeZoom1_LeChantier.tif')
    px1 = px1ds.GetRasterBand(1).ReadAsArray() * 0.05 * res
    # considering only non-Nodata pixels
    px1[nodata_mask] = -9999
    print('Saving',outprefix+'EWmicmac.tif')
    save_geotiff(px1, outprefix+'EWmicmac.tif', geotransform=gt, projection=refim1.GetProjection(),
                 nodata=-9999)
    px1ds = None

    # Correlation file
    correlds = gdal.Open(folder+'Correl_LeChantier_Num_5.tif')
    correl = (correlds.GetRasterBand(1).ReadAsArray()-127.5)/127.5
    # Mask NoData values, considering only non-Nodata pixels
    correl[nodata_mask] = -9999
    print('Saving',outprefix+'Correlmicmac.tif')
    save_geotiff(correl, outprefix+'Correlmicmac.tif', geotransform=gt, projection=refim1.GetProjection(),
                 nodata=-9999)
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


def micmacAnyStack(infolderlist, outfolder):
    """
    Creates NS, EW stacked maps, weighted by correlation score, while handling NoData values.
    
    Each folder in infolderlist should contain:
      - EWmicmac.tif
      - NSmicmac.tif
      - Correlmicmac.tif
    
    Parameters:
        - infolderlist: List of folders containing input displacement maps
        - outfolder: Output directory for stacked maps
        - nodata_value: NoData value to use (if None, it will be inferred from the first file)
    """
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)
    
    nodata_value = -9999  # NoData value 

    print(f"Using NoData value: {nodata_value}")

    # Define displacement components
    components = ["NS", "EW"]

    for comp in components:
        # Construct file lists dynamically
        disp_files = [os.path.join(folder, f"{comp}micmac.tif") for folder in infolderlist]
        corr_files = [os.path.join(folder, "Correlmicmac.tif") for folder in infolderlist]

        # Build weighted sum formula
        # eg for 4 input: 'A*E/(E+F+G+H)+B*F/(E+F+G+H)+C*G/(E+F+G+H)+D*H/(E+F+G+H)'
        calc_expr = "+".join([f"(A{i}*C{i})/({"+".join([f"C{j}" for j in range(len(infolderlist))])})" for i in range(len(infolderlist))])
        print(calc_expr)

        # Prepare argument list for gdal_calc.py
        calc_args = {f"A{i}": disp_files[i] for i in range(len(infolderlist))}
        calc_args.update({f"C{i}": corr_files[i] for i in range(len(infolderlist))})
        calc_args["calc"] = calc_expr
        calc_args["outfile"] = os.path.join(outfolder, f"{comp}dispStacked.tif")
        calc_args["NoDataValue"] = nodata_value

        # Run gdal_calc
        gdal_calc.Calc(**calc_args)

        print(calc_args)

        # nodata is reset to another value, change:
        ds = gdal.Open(os.path.join(outfolder, f"{comp}dispStacked.tif"), gdal.GA_Update)
        band = ds.GetRasterBand(1)
        band.SetNoDataValue(-9999)
        band.FlushCache()
        ds = None

    # Compute sum of correlation scores, ignoring NoData values
    calc_expr_corr = "+".join([f"A{i}" for i in range(len(infolderlist))])
    gdal_calc.Calc(
        calc=calc_expr_corr,
        **{f"A{i}": corr_files[i] for i in range(len(infolderlist))},
        outfile=os.path.join(outfolder, "CorreldispStacked.tif"),
        NoDataValue=nodata_value
    )

    # nodata is reset to another value, change:
    ds = gdal.Open(os.path.join(outfolder, "CorreldispStacked.tif"), gdal.GA_Update)
    band = ds.GetRasterBand(1)
    band.SetNoDataValue(-9999)
    band.FlushCache()
    ds = None


    return "Done!"



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
    theta = (azimuth)*np.pi/180
    #par = ns*np.cos(theta)-ew*np.sin(theta)
    par = ns*np.cos(theta)+ew*np.sin(theta)
    par[~nodata_mask] = nodata
    #perp = ns*np.sin(theta)+ew*np.cos(theta)
    perp = -1*ns*np.sin(theta)+ew*np.cos(theta)
    perp[~nodata_mask] = nodata

    save_geotiff(par,partif, ewds.GetGeoTransform(), ewds.GetProjection(),nodata=nodata)
    save_geotiff(perp,perptif, ewds.GetGeoTransform(), ewds.GetProjection(),nodata=nodata)
    ewds = None
    nsds = None
    return par, perp


def verticalDispNear(dem1file,dem2file,nsfile,ewfile,outf='VerticalDisp.tif'):
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



def verticalDispBilin(dem1file, dem2file, nsfile, ewfile, outf='VerticalDisp.tif'):
    """Compute vertical displacement using bilinear interpolation of DEM2."""
    
    # --- Read displacement fields ---
    ns_ds = gdal.Open(nsfile)
    ew_ds = gdal.Open(ewfile)
    ns = ns_ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
    ew = ew_ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
    nodata = ns_ds.GetRasterBand(1).GetNoDataValue()

    # Mask invalid values
    ns[ns == nodata] = np.nan
    ew[ew == nodata] = np.nan

    geotransform = ns_ds.GetGeoTransform()
    resolution = geotransform[1]
    ns_ds, ew_ds = None, None  # Close displacement datasets

    # --- Read DEMs ---
    dem1 = gdal.Open(dem1file)
    dem2 = gdal.Open(dem2file)
    u1 = dem1.GetRasterBand(1).ReadAsArray().astype(np.float32)
    u2 = dem2.GetRasterBand(1).ReadAsArray().astype(np.float32)
    u1[u1 == dem1.GetRasterBand(1).GetNoDataValue()] = np.nan
    u2[u2 == dem2.GetRasterBand(1).GetNoDataValue()] = np.nan

    print("Raster shapes:", ns.shape, ew.shape, u1.shape, u2.shape)
    
    # --- Compute pixel displacements ---
    movY = ns / resolution
    movX = ew / resolution
    ns, ew = None, None  # Free memory

    rows, cols = u1.shape
    y_grid, x_grid = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')
    targetY = y_grid + movY
    targetX = x_grid + movX
    movY, movX = None, None  # Free memory

    # --- Interpolate u2 at displaced positions ---
    interp_u2 = ndimage.map_coordinates(u2, [targetY, targetX], order=1, mode='nearest')

    # Mask invalid values (NaNs or out-of-bounds)
    invalid_mask = (
        np.isnan(targetX) | np.isnan(targetY) |
        (targetX < 0) | (targetX >= cols - 1) |
        (targetY < 0) | (targetY >= rows - 1)
    )
    interp_u2[invalid_mask] = np.nan
    interp_u2[np.isnan(u1)] = np.nan

    # --- Compute vertical displacement ---
    vertical_disp = interp_u2 - u1
    vertical_disp[np.isnan(vertical_disp)] = nodata

    # --- Save output ---
    save_geotiff(vertical_disp, outf, dem1.GetGeoTransform(), dem1.GetProjection(), nodata)
    print(f"Saved vertical displacement to: {outf}")

    return vertical_disp


### Profile tools
def calcSlopeFrom2Pts(x1,y1,x2,y2):
    '''calculate the slope of a line from two points and the strike of that line in degrees from X AXIS
    returns slope, strike'''
    slope = (y2-y1)/(x2-x1)
    strike = np.degrees(np.arctan(slope)) #this is not from NORTH!! this would be from the x-axis (math, not geology)
    return slope, strike

def calcAzimuthFromNorth(x1,y1,x2,y2):
    '''calculate the azimuth (strike) of a line from North using two points'''
    dx = x2 - x1
    dy = y2 - y1

    # make sure dx and dy are not zero. 
    if (dx == 0.0):
        dx = 0.000001
    if (dy == 0.0):
        dy = 0.000001

    #Calculate angle
    angle = np.degrees(np.abs(np.arctan(dy/dx))) #* (180/np.pi)))
    
    #first quadrant
    if (x2>=x1) and (y2>y1):
        azimuth = 90.0 - angle
        # print('first q')
    #second quadrant
    elif (x2>x1) and (y2<=y1):
        azimuth = angle + 90.0
        # print('second q')
    #third quadrant
    elif (x2<=x1) and (y2<y1):
        azimuth = 270.0 - angle
        # print('third q')
    #fourth quadrant
    else:
        azimuth = 270.0 + angle
        # print('fourth q')

    return azimuth


def calcPerpProfile(orig_strike, profile_length, profile_width, center_point): 
    '''calculate a profile perpendicular to a line from a fault strike and a center point.
    return the profile_line as LineString and profile_swath as Polygon
    orig_strike of fault in degrees
    profile_length in meters = TOTAL length
    profile_width in meters = TOTAL width
    center_point as (x1,y1)
    returns profile_line, profile_swath'''
    
    
    
    profile_strike = orig_strike + 90
    profile_angle = np.radians(profile_strike)
    
    # calc perp_line start point, profile_length/2 away from center point
    x = center_point[0] + (profile_length/2) * np.cos(profile_angle)
    y = center_point[1] + (profile_length/2) * np.sin(profile_angle)
    startpt = Point(x,y)
    
    # calc perp_line end point, profile_length/2 away from center point
    x = center_point[0] - (profile_length/2) * np.cos(profile_angle)
    y = center_point[1] - (profile_length/2) * np.sin(profile_angle)
    endpt = Point(x,y)
    
    profile_line = LineString([startpt,endpt])
    profile_swath = profile_line.buffer(profile_width/2,16,cap_style=3)
    return profile_line, profile_swath


def generateProfiles(shapefile_lines_path, profile_length, profile_width, profile_spacing,plot=True,verbose=False,save=False,prefix='all',folder='./'): #save_shp=False):
    '''
    generates profiles perpendicular to lines in a shapefile.
    optionally plots input shapefile lines and output profile swaths. default is plot=True. 
    
    inputs are:
    shapefile_lines_path - path location of the shapefile. shapefile must be lines. best if in a UTM coordinate system.
    profile_length - TOTAL length of the profiles in the same units as the shapefile coordinate system (i.e., m if in UTM)
    profile_width - TOTAL width of the profiles in the same units as the shapefile coordinate system (i.e., m if in UTM)
    profile_spacing - distance between each profile in the same units as the shapefile coordinate system (i.e., m if in UTM)
    
    outputs are:
    profile_swaths = geopandas geodataframe of POLYGON features, the profile swath.
    profile_lines = geopandas geodataframe of LINE features, the centerline of each profile perpendicular to the shapefile lines.
    profile_centerpts = geopandas geodataframe of POINT features, the midpoint of each profile line along the shapefile lines.
    
    exports:
    swaths, profiles, centerpts in geoJSON format
    strikes, n_profiles in np format
    
    '''
    
    print('profiles are',profile_width,'m wide,',profile_length,'m long, and',profile_spacing,'m apart')
    
    # read in shapefile lines
    lines = gpd.read_file(shapefile_lines_path)
    # Currently, index_parts defaults to True, but in the future, it will default to False to be consistent with Pandas. Use `index_parts=True` to keep the current behavior and True/False to silence the warning.
    lines = lines.explode(index_parts=True) # this turns multipart lines into single part lines so that profiles remain perpendicular to each line segment if it changes strike...but often doesn't work in Python and might need to do in QGIS first and use the exploded lines shapefile as input file
    lines = lines.reset_index(drop=True) # reset the index because we just added a bunch of lines
    print('CRS of shapefile:', lines.crs)
    
    # initialize geodataframes to hold ALL swath, profile, center_pt, strike info
    ## starting with empty dataframes because we don't know the total size yet
    # this throws a filterwarning that i'm ignoring that you can't initialize a gdf without a CRS...will need to fix eventually
    all_swaths = gpd.GeoDataFrame(geometry=[], crs=lines.crs)#geometry=gpd.points_from_xy(x,y)) # where x and y = np.zeros(len(lines))...but if start this way will have a bunch of nonesense/empty lines to remove later, but it didn't like it when I tried to start with one point
    all_profiles = gpd.GeoDataFrame(geometry=[], crs=lines.crs)#geometry=gpd.points_from_xy(x,y))
    all_centerpts = gpd.GeoDataFrame(geometry=[], crs=lines.crs)#geometry=gpd.points_from_xy(x,y))
    all_strikes = np.ones(len(lines),dtype=float)
    all_swath_strikes = np.array([],dtype=float)
    n_profiles= np.ones(len(lines),dtype=int)
    all_azimuths = np.ones(len(lines),dtype=float)
    
    for i, row in lines.iterrows():
        line_geom = row.geometry
        coords = list(line_geom.coords)
    
        # initialize per-line storage
        swaths_list = []
        profiles_list = []
        centerpts_list = []
        strikes_list = []
        azimuths_list = []
    
        for j in range(len(coords) - 1):
            x1, y1 = coords[j]
            x2, y2 = coords[j + 1]
            seg_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if seg_length == 0:
                continue
            
            dx = (x2 - x1) / seg_length
            dy = (y2 - y1) / seg_length
            
            # Ensure at least one profile, even on short segments
            if seg_length < profile_spacing:
                dists = [seg_length / 2]
            else:
                dists = np.arange(profile_spacing / 2, seg_length, profile_spacing)
            
            for dist_along in dists:
                cx = x1 + dx * dist_along
                cy = y1 + dy * dist_along
            
                slope, strike = calcSlopeFrom2Pts(x1, y1, x2, y2)
                azimuth = calcAzimuthFromNorth(x1, y1, x2, y2)
            
                profile_line, profile_swath = calcPerpProfile(strike, profile_length, profile_width, (cx, cy))
            
                swaths_list.append(profile_swath)
                profiles_list.append(profile_line)
                centerpts_list.append(Point(cx, cy))
                strikes_list.append(strike)
                azimuths_list.append(azimuth)


    
        # convert lists to GeoDataFrames
        swaths = gpd.GeoDataFrame(geometry=swaths_list, crs=lines.crs)
        profiles = gpd.GeoDataFrame(geometry=profiles_list, crs=lines.crs)
        center_points = gpd.GeoDataFrame(geometry=centerpts_list, crs=lines.crs)
        center_points['fault_azimuth'] = azimuths_list  # last segment azimuth; optional: store full list
    
        all_swaths = pd.concat([all_swaths, swaths])
        all_profiles = pd.concat([all_profiles, profiles])
        all_centerpts = pd.concat([all_centerpts, center_points])
        all_swath_strikes = np.concatenate([all_swath_strikes, np.array(strikes_list)])
        all_strikes[i] = np.mean(strikes_list)
        all_azimuths[i] = np.mean(azimuths_list)
        n_profiles[i] = len(strikes_list)
    
        # reset index because otherwise the index is duplicated for each line in the shapefile
        all_swaths = all_swaths.reset_index(drop=True)
        all_profiles = all_profiles.reset_index(drop=True)
        all_centerpts = all_centerpts.reset_index(drop=True)
    
        # ensure profiles have same CRS as input shapefile
        all_swaths = all_swaths.set_crs(lines.crs)
        all_profiles = all_profiles.set_crs(lines.crs)
        all_centerpts = all_centerpts.set_crs(lines.crs)
    
    # to plot or not to plot
    if plot==True:
        fig,ax = plt.subplots(figsize=(8,8))
        all_swaths.plot(ax=ax,alpha=.2,color='k')
        all_profiles.plot(ax=ax,color='k',lw=.5)
        all_centerpts.plot(ax=ax,color='b')
        lines.plot(ax=ax,color='r',lw=.75)
        plt.show()


    if save==True:
        all_swaths.to_file(folder+'%s_swaths_%sprofiles.geojson' %(prefix,sum(n_profiles)), driver='GeoJSON') 
        all_profiles.to_file(folder+'%s_profiles_%sprofiles.geojson' %(prefix,sum(n_profiles)), driver='GeoJSON')  
        all_centerpts.to_file(folder+'%s_centerpts_%sprofiles.geojson' %(prefix,sum(n_profiles)), driver='GeoJSON') 
        np.save(folder+'%s_swath_strikes_%sprofiles.npy' %(prefix,sum(n_profiles)),all_swath_strikes, allow_pickle=True,)
        np.save(folder+'%s_strikes_%sprofiles.npy' %(prefix,sum(n_profiles)),all_strikes,allow_pickle=True,)
        np.save(folder+'%s_azimuths_%sprofiles.npy' %(prefix,sum(n_profiles)),all_azimuths,allow_pickle=True,)
        np.save(folder+'%s_n_profiles_%sprofiles.npy' %(prefix,sum(n_profiles)),n_profiles,allow_pickle=True,)

    return all_swaths, all_swath_strikes, all_profiles, all_centerpts, all_strikes, n_profiles, all_azimuths

# Raster Sampler



def sample_swath(raster, fault_point_x, fault_point_y, fault_az_deg, profile_length, swath_width, resolution,crs):
    '''
    Raster must be opened by rasterio.open(path,masked=True).
    '''
    P = Proj(crs)
    G = Geod(ellps='WGS84')

    lon, lat = P(fault_point_x,
                 fault_point_y, inverse=True)
    
    lon1, lat1, _ = G.fwd(
        lon, lat, fault_az_deg - 90,
        profile_length / 2
    )
    lon2, lat2, _ = G.fwd(
        lon, lat, fault_az_deg + 90,
        profile_length / 2
    )

    tmppts = np.array(
        G.npts(
            lon1=lon1, lat1=lat1,
            lon2=lon2, lat2=lat2,
            npts=profile_length/resolution
        )
    )

    starts = G.fwd(
        tmppts[:, 0], tmppts[:, 1],
        [fault_az_deg] * len(tmppts),
        [swath_width / 2] * len(tmppts)
    )
    ends = G.fwd(
        tmppts[:, 0], tmppts[:, 1],
        [fault_az_deg + 180] * len(tmppts),
        [swath_width / 2] * len(tmppts)
    )

    map_object = np.array(list(map(
        lambda lon1, lat1, lon2, lat2:
        G.npts(lon1, lat1, lon2, lat2, swath_width / resolution),
        starts[0], starts[1], ends[0], ends[1]
    )))

    pts = map_object.reshape(
        len(starts[0]) * int(swath_width / resolution), 2
    )
    pts = P(pts[:, 0], pts[:, 1])
    pts = np.column_stack(pts)

    dist = np.arange(-profile_length / 2,
                     profile_length / 2, resolution)
    dists = np.repeat(dist, int(swath_width / resolution))

    raster_samps = np.array([x for x in raster.sample(pts)])
    raster_samps[raster_samps == -9999] = np.nan
    return raster_samps, pts, dists

def projectParPerp(ns, ew, az):
    theta = (az)*np.pi/180
    par = ns*np.cos(theta)+ew*np.sin(theta)
    perp = -1*ns*np.sin(theta)+ew*np.cos(theta)
    return par.flatten(), perp.flatten()

def erf_function_noslope(x, a, b, c, ws):
    '''From Milliner et al., 2021
    The solved parameters include the intercept (a), the total fault displacement (b), 
    the fault location (c), the shear width (ws)'''
    return a+b/2*erf((x-c)/(ws*np.sqrt(2)))

def erf_function(x, a, b, c, ws, m):
    '''From Milliner et al., 2021
    The solved parameters include the intercept (a), the total fault displacement (b), 
    the fault location (c), the shear width (ws), and the slope (m),'''
    return a+b/2*erf((x-c)/(ws*np.sqrt(2)))+m*x

def erf_function_twoslope(x, a, b, c, ws, m1, m2):
    '''From Milliner et al., 2021
    The solved parameters include the intercept (a), the total fault displacement (b), 
    the fault location (c), the shear width (ws), and the slope (m),'''
    return a+b/2*erf((x-c)/(ws*np.sqrt(2)))+ np.where(x > c, m2 * (x - c), m1 * (x - c)) 

# arctan function with independent slopes for upper and lower asymptotes

def fit_arctan_independent_slopes(x, a, b, c, m1, m2, d): 
    '''
    ( m_1 ) is the slope of the upper asymptote,
    ( m_2 ) is the slope of the lower asymptote
    ( a ) is the amplitude,
    ( b ) controls the steepness of the curve,
    ( c ) is the x-value of the midpoint,
    ( d ) is the vertical shift
    '''
    return a * np.arctan(b * (x - c)) + np.where(x >= c, m1 * (x - c), m2 * (x - c)) + d


def erf_curve_fit(samps, dists, bounds=None):
    if bounds is None:
        bounds = ((-np.inf,np.nanmin(samps)*2,dists.min(),0,-np.inf),(np.inf,np.nanmax(samps)*2,dists.max(),2*dists.max(),np.inf))
    # Try fit
    try:
        popt, pcov = curve_fit(erf_function, dists[~np.isnan(samps)], samps[~np.isnan(samps)], maxfev=10000,bounds=bounds)
        intercept = popt[0]
        total_offset = popt[1]
        fault_loc = popt[2]
        shear_width = popt[3]
        total_offset_sig = np.sqrt(pcov[1, 1])
        fault_loc_sig = np.sqrt(pcov[2, 2])
        shear_width_sig = np.sqrt(pcov[3, 3])
        slope = popt[4]
        slope_sig = np.sqrt(pcov[4, 4])
    except:
        print('Fit failed.')
        return (np.nan,) * 9
    return (intercept,total_offset, fault_loc, shear_width, total_offset_sig, fault_loc_sig, shear_width_sig, slope, slope_sig)

def erf_curve_fit_noslope(samps, dists, bounds=None):
    if bounds is None:
        bounds = ((-np.inf,np.nanmin(samps)*2,dists.min(),0),(np.inf,np.nanmax(samps)*2,dists.max(),2*dists.max()))
    # Try fit
    try:
        popt, pcov = curve_fit(erf_function_noslope, dists[~np.isnan(samps)], samps[~np.isnan(samps)], maxfev=10000,bounds=bounds)
        intercept = popt[0]
        total_offset = popt[1]
        fault_loc = popt[2]
        shear_width = popt[3]
        total_offset_sig = np.sqrt(pcov[1, 1])
        fault_loc_sig = np.sqrt(pcov[2, 2])
        shear_width_sig = np.sqrt(pcov[3, 3])
    except:
        print('Fit failed.')
        return (np.nan,) * 7
    return (intercept,total_offset, fault_loc, shear_width, total_offset_sig, fault_loc_sig, shear_width_sig)

def erf_curve_fit_twoslope(samps, dists, bounds=None):
    if bounds is None:
        max_diff = np.nanmax(samps)-np.nanmin(samps)
        max_width = (dists.max() if dists.max() < 5000 else 5000)
        # a, b, c, ws, m1, m2
        bounds = ((-np.inf,-max_diff,dists.min()/2,0,-max_diff/np.nanmax(dists),-max_diff/np.nanmax(dists)),
                  (np.inf,max_diff,dists.max()/2,max_width,max_diff/np.nanmax(dists),max_diff/np.nanmax(dists)))
    # Try fit
    try:
        popt, pcov = curve_fit(erf_function_twoslope, dists[~np.isnan(samps)], samps[~np.isnan(samps)], maxfev=10000,bounds=bounds)
        intercept = popt[0]
        fault_loc_offset = popt[1]
        fault_loc = popt[2]
        shear_width = popt[3]
        total_offset_sig = np.sqrt(pcov[1, 1])
        fault_loc_sig = np.sqrt(pcov[2, 2])
        shear_width_sig = np.sqrt(pcov[3, 3])
        slope1 = popt[4]
        slope1_sig = np.sqrt(pcov[4, 4])
        slope2 = popt[5]
        slope2_sig = np.sqrt(pcov[5, 5])
        of1, of2 = erf_function_twoslope([fault_loc+shear_width*2,fault_loc-shear_width*2], *popt)
        fzw_offset = of2 - of1
        full_erf = erf_function_twoslope(dists, *popt)
        max_offset = full_erf.max() - full_erf.min()
        ep_offset = full_erf[1] - full_erf[-1]
        print('Returning intercept, offset at fault location, shifted fault location, 1/2 1 std shear width (4 shear width is 2 sigma \
              fault zone width), uncertainties for those parameters, then lhs slope and uncertainty, and rhs slope and uncertainty). \
              Other derived parameters are the offset at the edges of the 2 sigma shear zone, the max offset, and the endpoint offset.\
                The latter two are from left to right across profile.')
    except:
        print('Fit failed.')
        return (np.nan,) * 14
    return (intercept,fault_loc_offset, fault_loc, shear_width, total_offset_sig, fault_loc_sig, shear_width_sig, slope1, slope1_sig, slope2, slope2_sig, fzw_offset, max_offset, ep_offset)
