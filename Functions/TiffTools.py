from osgeo import gdal, ogr
from osgeo_utils import gdal_calc
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
import os

from scipy import ndimage

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


    

