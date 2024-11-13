from osgeo import gdal, ogr
from osgeo_utils import gdal_calc
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

def createMicmacParamFile(im1,im2,out_folder='./',results_directory='MEC/',SzW=4,CorrelMin=0.5,SzW_base=2):
    
    string = f'''<ParamMICMAC>
      <DicoLoc>
            <Symb>Im1=XXXX</Symb>
            <Symb>Im2=XXXX</Symb>
            <Symb>Masq=XXXX</Symb>
            <Symb>DirMEC=MEC/</Symb>
            <Symb>Pyr=Pyram/</Symb>
            <Symb>Inc=2.0</Symb>
            <Symb>Pas=0.2</Symb>
            <Symb>Teta0=0</Symb>
            <Symb>UseMasq=false</Symb>
            <Symb>UseTeta=false</Symb>
            <Symb>RegulBase=0.3</Symb>
            <Symb>UseDequant=false</Symb>
            <Symb>SzW=4</Symb>
            <Symb>CorrelMin=0.5</Symb>
            <Symb>GammaCorrel=2</Symb>
            <Symb>PdsF=0.1</Symb>
            <Symb>NbDir=7</Symb>
            <Symb>VPds=[1,0.411765,0.259259,0.189189,0.148936,0.122807,0.104478,0.090909,0.104478,0.090909,0.148936,0.189189,0.259259,0.411765]</Symb>
            <Symb>SsResolOpt=4</Symb>
            <Symb>Px1Moy=0</Symb>
            <Symb>Px2Moy=0</Symb>
            <Symb>Interpolateur=eInterpolSinCard</Symb>
            <Symb>SurEchWCor=1</Symb>
            <Symb>ZoomInit=1</Symb>
            <eSymb>P0= / 0.1 +   0.1 / 0 7</eSymb>
            <eSymb>P1= / 0.1 +   0.1 / 1 7</eSymb>
            <eSymb>P2= / 0.1 +   0.1 / 2 7</eSymb>
            <eSymb>P3= / 0.1 +   0.1 / 3 7</eSymb>
            <eSymb>P4= / 0.1 +   0.1 / 4 7</eSymb>
            <eSymb>P5= / 0.1 +   0.1 / 5 7</eSymb>
            <eSymb>P6= / 0.1 +   0.1 / 6 7</eSymb>
            <eSymb>P7= / 0.1 +   0.1 / 7 7</eSymb>
            <eSymb>NbDirTot=* 2 7</eSymb>
            <eSymb>Regul=* 0.300000  ? false 3 1</eSymb>
            <eSymb>SsResolOptInterm1=* 4 4</eSymb>
            <eSymb>SsResolOptInterm2=* 2 4</eSymb>
            <eSymb>WithZ4= SupEq 1 4</eSymb>
            <eSymb>WithZ2= SupEq 1 2</eSymb>
      </DicoLoc>
      <Section_Terrain>
            <IntervalPaxIsProportion>false</IntervalPaxIsProportion>
            <EstimPxPrefZ2Prof>false</EstimPxPrefZ2Prof>
            <IntervParalaxe>
                  <Px1Moy>0</Px1Moy>
                  <Px2Moy>0</Px2Moy>
                  <Px1IncCalc>2</Px1IncCalc>
                  <Px1PropProf>0</Px1PropProf>
                  <Px2IncCalc>2</Px2IncCalc>
            </IntervParalaxe>
            <Planimetrie>
                  <FilterEstimTerrain>.*</FilterEstimTerrain>
            </Planimetrie>
      </Section_Terrain>
      <Section_PriseDeVue>
            <BordImage>5</BordImage>
            <PrefixMasqImRes>MasqIm</PrefixMasqImRes>
            <DirMasqueImages></DirMasqueImages>
            <GeomImages>eGeomImage_Epip</GeomImages>
            <Images>
                  <Im1>{im1}</Im1>
                  <Im2>{im2}</Im2>
            </Images>
            <NomsGeometrieImage>
                  <UseIt>true</UseIt>
                  <PatternSel>.*</PatternSel>
                  <PatNameGeom>GridDistId</PatNameGeom>
                  <AddNumToNameGeom>false</AddNumToNameGeom>
            </NomsGeometrieImage>
            <SingulariteInCorresp_I1I2>false</SingulariteInCorresp_I1I2>
      </Section_PriseDeVue>
      <Section_MEC>
            <PasIsInPixel>false</PasIsInPixel>
            <ClipMecIsProp>false</ClipMecIsProp>
            <ZoomClipMEC>1</ZoomClipMEC>
            <NbMinImagesVisibles>2</NbMinImagesVisibles>
            <OneDefCorAllPxDefCor>false</OneDefCorAllPxDefCor>
            <ZoomBeginODC_APDC>4</ZoomBeginODC_APDC>
            <DefCorrelation>-0.0123400000000000003</DefCorrelation>
            <ReprojPixelNoVal>true</ReprojPixelNoVal>
            <EpsilonCorrelation>1.00000000000000008e-05</EpsilonCorrelation>
            <ChantierFullImage1>true</ChantierFullImage1>
            <ChantierFullMaskImage1>true</ChantierFullMaskImage1>
            <ExportForMultiplePointsHomologues>false</ExportForMultiplePointsHomologues>
            <EtapeMEC>
                  <DeZoom>-1</DeZoom>
                  <DynamiqueCorrel>eCoeffGamma</DynamiqueCorrel>
                  <CorrelMin>{CorrelMin}</CorrelMin>
                  <GammaCorrel>2</GammaCorrel>
                  <SzW>{SzW}</SzW>
                  <SurEchWCor>1</SurEchWCor>
                  <AlgoRegul>eAlgo2PrgDyn</AlgoRegul>
                  <ExportZAbs>false</ExportZAbs>
                  <ModulationProgDyn>
                        <EtapeProgDyn>
                              <NbDir>14</NbDir>
                              <ModeAgreg>ePrgDAgrSomme</ModeAgreg>
                              <Teta0>0</Teta0>
                        </EtapeProgDyn>
                        <Px1PenteMax>0.400000000000000022</Px1PenteMax>
                        <Px2PenteMax>0.400000000000000022</Px2PenteMax>
                  </ModulationProgDyn>
                  <SsResolOptim>4</SsResolOptim>
                  <ModeInterpolation>eInterpolSinCard</ModeInterpolation>
                  <SzSinCard>5</SzSinCard>
                  <SzAppodSinCard>5</SzAppodSinCard>
                  <TailleFenetreSinusCardinal>3</TailleFenetreSinusCardinal>
                  <ApodisationSinusCardinal>false</ApodisationSinusCardinal>
                  <Px1Regul>0.299999999999999989</Px1Regul>
                  <Px1Pas>0.200000000000000011</Px1Pas>
                  <Px1DilatAlti>10</Px1DilatAlti>
                  <Px1DilatPlani>{SzW_base}</Px1DilatPlani>
                  <Px2Regul>0.299999999999999989</Px2Regul>
                  <Px2Pas>0.200000000000000011</Px2Pas>
                  <Px2DilatAlti>2</Px2DilatAlti>
                  <Px2DilatPlani>{SzW_base}</Px2DilatPlani>
                  <GenImagesCorrel>true</GenImagesCorrel>
            </EtapeMEC>
            <EtapeMEC>
                  <DeZoom>1</DeZoom>
                  <DynamiqueCorrel>eCoeffGamma</DynamiqueCorrel>
                  <CorrelMin>{CorrelMin}</CorrelMin>
                  <GammaCorrel>2</GammaCorrel>
                  <AggregCorr>eAggregSymetrique</AggregCorr>
                  <SzW>{SzW}</SzW>
                  <SurEchWCor>1</SurEchWCor>
                  <AlgoRegul>eAlgo2PrgDyn</AlgoRegul>
                  <ExportZAbs>false</ExportZAbs>
                  <ModulationProgDyn>
                        <EtapeProgDyn>
                              <NbDir>14</NbDir>
                              <ModeAgreg>ePrgDAgrSomme</ModeAgreg>
                              <Teta0>0</Teta0>
                        </EtapeProgDyn>
                        <Px1PenteMax>0.400000000000000022</Px1PenteMax>
                        <Px2PenteMax>0.400000000000000022</Px2PenteMax>
                  </ModulationProgDyn>
                  <SsResolOptim>16</SsResolOptim>
                  <ModeInterpolation>eInterpolSinCard</ModeInterpolation>
                  <SzSinCard>5</SzSinCard>
                  <SzAppodSinCard>5</SzAppodSinCard>
                  <TailleFenetreSinusCardinal>3</TailleFenetreSinusCardinal>
                  <ApodisationSinusCardinal>false</ApodisationSinusCardinal>
                  <Px1Regul>2</Px1Regul>
                  <Px1Pas>0.800000000000000044</Px1Pas>
                  <Px1DilatAlti>2</Px1DilatAlti>
                  <Px1DilatPlani>{SzW_base}</Px1DilatPlani>
                  <Px2Regul>2</Px2Regul>
                  <Px2Pas>0.800000000000000044</Px2Pas>
                  <Px2DilatAlti>2</Px2DilatAlti>
                  <Px2DilatPlani>{SzW_base}</Px2DilatPlani>
                  <GenImagesCorrel>true</GenImagesCorrel>
            </EtapeMEC>
            <EtapeMEC>
                  <DeZoom>1</DeZoom>
                  <DynamiqueCorrel>eCoeffGamma</DynamiqueCorrel>
                  <CorrelMin>{CorrelMin}</CorrelMin>
                  <GammaCorrel>2</GammaCorrel>
                  <AggregCorr>eAggregSymetrique</AggregCorr>
                  <SzW>{SzW}</SzW>
                  <SurEchWCor>1</SurEchWCor>
                  <AlgoRegul>eAlgo2PrgDyn</AlgoRegul>
                  <ExportZAbs>false</ExportZAbs>
                  <ModulationProgDyn>
                        <EtapeProgDyn>
                              <NbDir>14</NbDir>
                              <ModeAgreg>ePrgDAgrSomme</ModeAgreg>
                              <Teta0>0</Teta0>
                        </EtapeProgDyn>
                        <Px1PenteMax>0.400000000000000022</Px1PenteMax>
                        <Px2PenteMax>0.400000000000000022</Px2PenteMax>
                  </ModulationProgDyn>
                  <SsResolOptim>8</SsResolOptim>
                  <ModeInterpolation>eInterpolSinCard</ModeInterpolation>
                  <SzSinCard>5</SzSinCard>
                  <SzAppodSinCard>5</SzAppodSinCard>
                  <TailleFenetreSinusCardinal>3</TailleFenetreSinusCardinal>
                  <ApodisationSinusCardinal>false</ApodisationSinusCardinal>
                  <Px1Regul>2</Px1Regul>
                  <Px1Pas>0.400000000000000022</Px1Pas>
                  <Px1DilatAlti>2</Px1DilatAlti>
                  <Px1DilatPlani>{SzW_base}</Px1DilatPlani>
                  <Px2Regul>2</Px2Regul>
                  <Px2Pas>0.400000000000000022</Px2Pas>
                  <Px2DilatAlti>2</Px2DilatAlti>
                  <Px2DilatPlani>{SzW_base}</Px2DilatPlani>
                  <GenImagesCorrel>true</GenImagesCorrel>
            </EtapeMEC>
            <EtapeMEC>
                  <DeZoom>1</DeZoom>
                  <DynamiqueCorrel>eCoeffGamma</DynamiqueCorrel>
                  <CorrelMin>{CorrelMin}</CorrelMin>
                  <GammaCorrel>2</GammaCorrel>
                  <AggregCorr>eAggregSymetrique</AggregCorr>
                  <SzW>{SzW}</SzW>
                  <SurEchWCor>1</SurEchWCor>
                  <AlgoRegul>eAlgo2PrgDyn</AlgoRegul>
                  <ExportZAbs>false</ExportZAbs>
                  <ModulationProgDyn>
                        <EtapeProgDyn>
                              <NbDir>14</NbDir>
                              <ModeAgreg>ePrgDAgrSomme</ModeAgreg>
                              <Teta0>0</Teta0>
                        </EtapeProgDyn>
                        <Px1PenteMax>0.400000000000000022</Px1PenteMax>
                        <Px2PenteMax>0.400000000000000022</Px2PenteMax>
                  </ModulationProgDyn>
                  <SsResolOptim>8</SsResolOptim>
                  <ModeInterpolation>eInterpolSinCard</ModeInterpolation>
                  <SzSinCard>5</SzSinCard>
                  <SzAppodSinCard>5</SzAppodSinCard>
                  <TailleFenetreSinusCardinal>3</TailleFenetreSinusCardinal>
                  <ApodisationSinusCardinal>false</ApodisationSinusCardinal>
                  <Px1Regul>1</Px1Regul>
                  <Px1Pas>0.200000000000000011</Px1Pas>
                  <Px1DilatAlti>2</Px1DilatAlti>
                  <Px1DilatPlani>{SzW_base}</Px1DilatPlani>
                  <Px2Regul>1</Px2Regul>
                  <Px2Pas>0.200000000000000011</Px2Pas>
                  <Px2DilatAlti>2</Px2DilatAlti>
                  <Px2DilatPlani>{SzW_base}</Px2DilatPlani>
                  <GenImagesCorrel>true</GenImagesCorrel>
            </EtapeMEC>
            <EtapeMEC>
                  <DeZoom>1</DeZoom>
                  <DynamiqueCorrel>eCoeffGamma</DynamiqueCorrel>
                  <CorrelMin>{CorrelMin}</CorrelMin>
                  <GammaCorrel>2</GammaCorrel>
                  <AggregCorr>eAggregSymetrique</AggregCorr>
                  <SzW>{SzW}</SzW>
                  <SurEchWCor>1</SurEchWCor>
                  <AlgoRegul>eAlgo2PrgDyn</AlgoRegul>
                  <ExportZAbs>false</ExportZAbs>
                  <ModulationProgDyn>
                        <EtapeProgDyn>
                              <NbDir>14</NbDir>
                              <ModeAgreg>ePrgDAgrSomme</ModeAgreg>
                              <Teta0>0</Teta0>
                        </EtapeProgDyn>
                        <Px1PenteMax>0.400000000000000022</Px1PenteMax>
                        <Px2PenteMax>0.400000000000000022</Px2PenteMax>
                  </ModulationProgDyn>
                  <SsResolOptim>4</SsResolOptim>
                  <ModeInterpolation>eInterpolSinCard</ModeInterpolation>
                  <SzSinCard>5</SzSinCard>
                  <SzAppodSinCard>5</SzAppodSinCard>
                  <TailleFenetreSinusCardinal>3</TailleFenetreSinusCardinal>
                  <ApodisationSinusCardinal>false</ApodisationSinusCardinal>
                  <Px1Regul>1</Px1Regul>
                  <Px1Pas>0.100000000000000006</Px1Pas>
                  <Px1DilatAlti>2</Px1DilatAlti>
                  <Px1DilatPlani>{SzW_base}</Px1DilatPlani>
                  <Px2Regul>1</Px2Regul>
                  <Px2Pas>0.100000000000000006</Px2Pas>
                  <Px2DilatAlti>2</Px2DilatAlti>
                  <Px2DilatPlani>{SzW_base}</Px2DilatPlani>
                  <GenImagesCorrel>true</GenImagesCorrel>
            </EtapeMEC>
            <EtapeMEC>
                  <DeZoom>1</DeZoom>
                  <DynamiqueCorrel>eCoeffGamma</DynamiqueCorrel>
                  <CorrelMin>{CorrelMin}</CorrelMin>
                  <GammaCorrel>2</GammaCorrel>
                  <AggregCorr>eAggregSymetrique</AggregCorr>
                  <SzW>{SzW}</SzW>
                  <SurEchWCor>1</SurEchWCor>
                  <AlgoRegul>eAlgo2PrgDyn</AlgoRegul>
                  <ExportZAbs>false</ExportZAbs>
                  <ModulationProgDyn>
                        <EtapeProgDyn>
                              <NbDir>14</NbDir>
                              <ModeAgreg>ePrgDAgrSomme</ModeAgreg>
                              <Teta0>0</Teta0>
                        </EtapeProgDyn>
                        <Px1PenteMax>0.400000000000000022</Px1PenteMax>
                        <Px2PenteMax>0.400000000000000022</Px2PenteMax>
                  </ModulationProgDyn>
                  <SsResolOptim>4</SsResolOptim>
                  <ModeInterpolation>eInterpolSinCard</ModeInterpolation>
                  <SzSinCard>5</SzSinCard>
                  <SzAppodSinCard>5</SzAppodSinCard>
                  <TailleFenetreSinusCardinal>3</TailleFenetreSinusCardinal>
                  <ApodisationSinusCardinal>false</ApodisationSinusCardinal>
                  <Px1Regul>0.299999999999999989</Px1Regul>
                  <Px1Pas>0.0500000000000000028</Px1Pas>
                  <Px1DilatAlti>2</Px1DilatAlti>
                  <Px1DilatPlani>{SzW_base}</Px1DilatPlani>
                  <Px2Regul>0.299999999999999989</Px2Regul>
                  <Px2Pas>0.0500000000000000028</Px2Pas>
                  <Px2DilatAlti>2</Px2DilatAlti>
                  <Px2DilatPlani>{SzW_base}</Px2DilatPlani>
                  <GenImagesCorrel>true</GenImagesCorrel>
            </EtapeMEC>
            <HighPrecPyrIm>true</HighPrecPyrIm>
      </Section_MEC>
      <Section_Results>
            <Use_MM_EtatAvancement>false</Use_MM_EtatAvancement>
            <X_DirPlanInterFaisceau>0</X_DirPlanInterFaisceau>
            <Y_DirPlanInterFaisceau>0</Y_DirPlanInterFaisceau>
            <Z_DirPlanInterFaisceau>0</Z_DirPlanInterFaisceau>
            <GeomMNT>eGeomPxBiDim</GeomMNT>
            <Prio2OwnAltisolForEmprise>false</Prio2OwnAltisolForEmprise>
            <TagRepereCorrel>RepereCartesien</TagRepereCorrel>
            <DoMEC>true</DoMEC>
            <DoFDC>false</DoFDC>
            <GenereXMLComp>true</GenereXMLComp>
            <SaturationTA>50</SaturationTA>
            <OrthoTA>false</OrthoTA>
            <LazyZoomMaskTerrain>false</LazyZoomMaskTerrain>
            <MakeImCptTA>false</MakeImCptTA>
            <GammaVisu>1</GammaVisu>
            <ZoomVisuLiaison>-1</ZoomVisuLiaison>
            <TolerancePointHomInImage>0</TolerancePointHomInImage>
            <FiltragePointHomInImage>0</FiltragePointHomInImage>
            <BaseCodeRetourMicmacErreur>100</BaseCodeRetourMicmacErreur>
      </Section_Results>
      <Section_WorkSpace>
            <UseProfInVertLoc>true</UseProfInVertLoc>
            <WorkDir>./</WorkDir>
            <TmpMEC>{results_directory}</TmpMEC>
            <TmpPyr>Pyram/</TmpPyr>
            <TmpGeom></TmpGeom>
            <TmpResult>{results_directory}</TmpResult>
            <CalledByProcess>false</CalledByProcess>
            <IdMasterProcess>-1</IdMasterProcess>
            <CreateGrayFileAtBegin>false</CreateGrayFileAtBegin>
            <Visu>false</Visu>
            <ByProcess>128</ByProcess>
            <StopOnEchecFils>true</StopOnEchecFils>
            <AvalaibleMemory>128</AvalaibleMemory>
            <SzRecouvrtDalles>50</SzRecouvrtDalles>
            <SzDalleMin>500</SzDalleMin>
            <SzDalleMax>800</SzDalleMax>
            <NbCelluleMax>80000000</NbCelluleMax>
            <SzMinDecomposCalc>10</SzMinDecomposCalc>
            <DefTileFile>100000</DefTileFile>
            <NbPixDefFilesAux>30000000</NbPixDefFilesAux>
            <DeZoomDefMinFileAux>4</DeZoomDefMinFileAux>
            <FirstEtapeMEC>0</FirstEtapeMEC>
            <LastEtapeMEC>10000</LastEtapeMEC>
            <FirstBoiteMEC>0</FirstBoiteMEC>
            <NbBoitesMEC>100000000</NbBoitesMEC>
            <NomChantier>LeChantier</NomChantier>
            <PatternSelPyr>(.*)@(.*)</PatternSelPyr>
            <PatternNomPyr>$1DeZoom$2.tif</PatternNomPyr>
            <SeparateurPyr>@</SeparateurPyr>
            <KeyCalNamePyr>Key-Assoc-Pyram-MM</KeyCalNamePyr>
            <ActivePurge>false</ActivePurge>
            <PurgeMECResultBefore>false</PurgeMECResultBefore>
            <UseChantierNameDescripteur>false</UseChantierNameDescripteur>
            <ComprMasque>eComprTiff_FAX4</ComprMasque>
            <TypeMasque>eTN_Bits1MSBF</TypeMasque>
      </Section_WorkSpace>
      <Section_Vrac>
            <DebugMM>false</DebugMM>
            <SL_XSzW>1000</SL_XSzW>
            <SL_YSzW>900</SL_YSzW>
            <SL_Epip>false</SL_Epip>
            <SL_YDecEpip>0</SL_YDecEpip>
            <SL_PackHom0></SL_PackHom0>
            <SL_RedrOnCur>false</SL_RedrOnCur>
            <SL_NewRedrCur>false</SL_NewRedrCur>
            <SL_L2Estim>true</SL_L2Estim>
            <SL_TJS_FILTER>false</SL_TJS_FILTER>
            <SL_Step_Grid>10</SL_Step_Grid>
            <SL_Name_Grid_Exp>GridMap_%I_To_%J</SL_Name_Grid_Exp>
            <VSG_DynImRed>5</VSG_DynImRed>
            <VSG_DeZoomContr>16</VSG_DeZoomContr>
            <DumpNappesEnglob>false</DumpNappesEnglob>
            <InterditAccelerationCorrSpec>false</InterditAccelerationCorrSpec>
            <InterditCorrelRapide>false</InterditCorrelRapide>
            <ForceCorrelationByRect>false</ForceCorrelationByRect>
            <WithMessage>false</WithMessage>
            <ShowLoadedImage>false</ShowLoadedImage>
      </Section_Vrac>
</ParamMICMAC>'''

    f = open(out_folder+'param_LeChantier_Compl.xml','w')
    f.write(string)
    f.close()
    return f'{out_folder} param_LeChantier_Compl.xml written.'