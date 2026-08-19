import numpy as np
import vtk
from vtk.util import numpy_support
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

import xml.etree.ElementTree as ET



"""
Read vtk files (AMITEX output)
"""
def vtkFieldReader(vtk_name, fieldName='tomo_Volume'):
    reader = vtk.vtkStructuredPointsReader()
    reader.SetFileName(vtk_name)
    reader.Update()
    data = reader.GetOutput()
    dim = data.GetDimensions()
    siz = list(dim)
    siz = [i - 1 for i in siz]
    orig = data.GetOrigin()
    spacing = data.GetSpacing()
    mesh = vtk_to_numpy(data.GetCellData().GetArray(fieldName))
    return mesh.reshape(siz, order='F'), orig, spacing


'''
Extract parameters/coefficents from mat*.xml file
    Inputs
    ------
        > fname : file name
        >   mID : (default 1) material ID for the porous solid region
        >   idx : (default 1) indices to be extracted, can be a 1D array or scalar
    Outputs
    -------
        > coeff : coefficients to be extracted
'''
def extract_mat(fname, mID=1, idx=1):
    
    tree = ET.parse(fname)
    root = tree.getroot()
    
    #print(' Material ID: ' + root[mID].attrib['numM'])
    #print(' --- coeff indices: ' + str(idx))
    
    if np.isscalar(idx):
        coeff = float(root[mID][idx].attrib['Value'])
    else:
        coeff = list()
        for i in idx:
            coeff.append(float(root[mID][i].attrib['Value']))
    
    coeff = np.array(coeff)
    
    return coeff
    


'''
Extract parameters from algo*.xml file
    Inputs
    ------
        > fname : file name
        >   tag : e.g. "Non_local_algorithm"
        >   idx : indices to be extracted, can be a 1D array or scalar
    Outputs
    -------
        > param : parameters to be extracted
'''
def extract_algo(fname, tag, idx):

    tree = ET.parse(fname)
    root = tree.getroot()

    for child in root:
        if child.tag==tag:
            if np.isscalar(idx):
                param = child[idx].attrib['Value']
            else:
                param = list()
                for i in idx:
                    param.append(float(child[i].attrib['Value']))
                param = np.array(param)
                
    return param



"""
#file name *.vti
#volume data (3D, 4D array, shape: [x,y,z] or [x,y,z,c])
#volume name, e.g. velocity, phase, etc.
#origin
#spacing in x,y,z direction, (3x1 array)
#number of voxels in x,y,z direction, (3x1 array)
"""
def saveField2VTK(fileout, vdata, vname, origin=[0,0,0], spacing=[1,1,1], Legacy=None):
    
    #
    dx, dy, dz = spacing
    x0, y0, z0 = origin
    vcomponents = np.array(vdata[0,0,0]).size
    
    #dimension + swap components in case of vector input    
    if vcomponents==1:
        nx, ny, nz = np.shape(vdata)
    elif vcomponents==3:
        nx, ny, nz = np.shape(vdata[:,:,:,0])
    
    #swap x and z axes
    vdata = np.swapaxes(vdata, 0, 2)
    
    #data type
    vtype = vtk.util.numpy_support.get_vtk_array_type(vdata.dtype)
    
    #create vtk image object
    imageData = vtk.vtkImageData()
    imageData.SetSpacing(dx, dy, dz)
    imageData.SetOrigin(x0, y0, z0)
    imageData.SetDimensions(nx, ny, nz)
    imageData.AllocateScalars(vtype, vcomponents)
    
    vtk_data_array = numpy_to_vtk(num_array=vdata.ravel(), deep=True, array_type=vtype)
    vtk_data_array.SetNumberOfComponents(vcomponents)
    vtk_data_array.SetName(vname)
    imageData.GetPointData().SetScalars(vtk_data_array)
    
    if Legacy==None or Legacy==False:
        writer = vtk.vtkXMLImageDataWriter()
    elif Legacy==True:
        writer = vtk.vtkStructuredPointsWriter()
        writer.SetFileTypeToBinary()
        
    
    writer.SetInputData(imageData)
    writer.SetFileName(fileout)
    writer.Write()


'''
Save mesh image to VTK file specified by AMITEX
'''
def saveMesh2VTK_amitex(fileout, vdata, vname, origin=[0,0,0], spacing=[1,1,1]):
    x0, y0, z0 = origin
    dx, dy, dz = spacing
    nx, ny, nz = np.shape(vdata)
    
    if vdata.dtype == 'uint8':
        vtktype = 'unsigned_char'
    elif vdata.dtype == 'uint16':
        vtktype = 'unsigned_short'
    else:
        raise TypeError("data type not supported! (needs to be uint8 or uint16)")
    
    with open(fileout, 'w') as f:
        f.write("# vtk DataFile Version 4.2\n")
        f.write("mesh_grid\n")
        f.write("BINARY\n")
        f.write("DATASET {}\n".format("STRUCTURED_POINTS"))
        f.write("DIMENSIONS {:d} {:d} {:d}\n".format(nx+1, ny+1, nz+1))
        f.write("ORIGIN {:e} {:e} {:e}\n".format(x0, y0, z0))
        f.write("SPACING {:e} {:e} {:e}\n".format(dx, dy, dz))
        f.write("CELL_DATA {:d}\n".format(nx*ny*nz))
        f.write("SCALARS {} {}\n".format(vname, vtktype))
        f.write("LOOKUP_TABLE {}\n".format("default"))
    with open(fileout, 'ab') as f:
        f.write(vdata)


'''
Use this function to properly format the XML file
'''
def xml_indent(elem, level=0):
#code from internet (not verified)
    i = "\n" + level*"    "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "    "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            xml_indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


'''
Write material parameters into XML file
    - preparation for AMITEX
'''
def write_AMITEX_xml_mat(xmlname, Lambda0, Mu0, matlib, matlaw, coeffs):
    
    root = ET.Element("Materials")
    
    nMAT = coeffs.shape[0]
    ncoeffs = coeffs.shape[1]
    
    # Reference material
    refmat = ET.Element("Reference_Material")
    refmat.set("Lambda0", str(Lambda0))
    refmat.set("Mu0", str(Mu0))
    root.append(refmat)
    
    # Material
    for imat in range(nMAT):
        
        # material attributes (num, Lib, Law)
        child = ET.Element("Material")
        child.set("numM", str(imat+1))
        child.set("Lib", matlib)
        child.set("Law", matlaw)
        
        # coefficients
        for icoeff in range(ncoeffs):
            cchild = ET.Element("Coeff")
            cchild.set("Index", str(icoeff+1))
            cchild.set("Type", "Constant")
            cchild.set("Value", str(coeffs[imat, icoeff]))
            child.append(cchild)
            
        # Coefficient index for the nonlocal model
        cchild = ET.Element("IndexCoeffNloc")
        cchild.set("NLocMod_num", "1")
        cchild.text = ' '.join([str(x) for x in np.linspace(1, ncoeffs, ncoeffs).astype('int')])
        child.append(cchild)
        
        # Variable index for the nonlocal model
        cchild = ET.Element("IndexVarNloc")
        cchild.set("NLocMod_num", "1")
        cchild.text = " "
        child.append(cchild)
        
        root.append(child)
    
    # Nonlocal model setup
    child = ET.Element("Non_local_modeling")
    child.set("NLocMod_num", "1")
    child.set("Modelname", "user_nloc1")
    child.set("Nnloc", "0")
    child.set("Ngnloc", "0")
    child.set("Ncoeff_nloc", str(ncoeffs))
    cchild = ET.Element("numM")
    cchild.set("Nmat", str(nMAT))
    cchild.text = ' '.join([str(x) for x in np.linspace(1, nMAT, nMAT).astype('int')])
    child.append(cchild)
    root.append(child)
    
    # write
    xml_indent(root)
    tree = ET.ElementTree(root)
    with open(xmlname, "wb") as f:
        tree.write(f, encoding="utf-8", xml_declaration=True)
    
    #
    print('  --> AMITEX material file: '+xmlname)



'''
Write load parameters into XML file
    - preparation for AMITEX
'''
def write_AMITEX_xml_load(xmlname, outVTKstrain=0, outVTKstress=0):
    
    root = ET.Element("Loading_Output")
    
    # vtk fields to be print (stress/strain/intvar)
    child = ET.Element("Output")
    cchild = ET.Element("vtk_StressStrain")
    cchild.set("Strain", str(outVTKstrain))
    cchild.set("Stress", str(outVTKstress))
    child.append(cchild)
    root.append(child)
    
    # User field initialisation (TODO)
    
    # User sub-field initialisation (TODO)
    
    # successive loading (hard-coded, no user-defined param yet)
    child = ET.Element("Loading")
    child.set("Tag", "1")
    
    cchild = ET.Element("Time_Discretization")
    cchild.set("Discretization", "linear")
    cchild.set("Nincr", "1")
    cchild.set("Tfinal", "1")
    child.append(cchild)
    
    cchild = ET.Element("Output_vtkList")
    cchild.text = "1"
    child.append(cchild)
    
    cchild = ET.Element("xx")
    cchild.set("Driving", "Strain")
    cchild.set("Evolution", "Linear")
    cchild.set("Value", "0")
    child.append(cchild)
    
    cchild = ET.Element("yy")
    cchild.set("Driving", "Strain")
    cchild.set("Evolution", "Linear")
    cchild.set("Value", "0")
    child.append(cchild)
    
    cchild = ET.Element("zz")
    cchild.set("Driving", "Strain")
    cchild.set("Evolution", "Linear")
    cchild.set("Value", "0")
    child.append(cchild)
    
    cchild = ET.Element("xy")
    cchild.set("Driving", "Strain")
    cchild.set("Evolution", "Linear")
    cchild.set("Value", "0")
    child.append(cchild)
    
    cchild = ET.Element("xz")
    cchild.set("Driving", "Strain")
    cchild.set("Evolution", "Linear")
    cchild.set("Value", "0")
    child.append(cchild)
    
    cchild = ET.Element("yz")
    cchild.set("Driving", "Strain")
    cchild.set("Evolution", "Linear")
    cchild.set("Value", "0")
    child.append(cchild)
    
    root.append(child)
    
    # write
    xml_indent(root)
    tree = ET.ElementTree(root)
    with open(xmlname, "wb") as f:
        tree.write(f, encoding="utf-8", xml_declaration=True)
    
    #
    print('  --> AMITEX load file: '+xmlname)



'''
Write algorithm parameters into XML file
    - preparation for AMITEX
'''
def write_AMITEX_xml_algo(xmlname, params):
    
    nparams = len(params)
    
    root = ET.Element("Algorithm_Parameters")
    
    # Algorithm (amitex core algo -- hard coded)
    child = ET.Element("Algorithm")
    child.set("Type", "Basic_Scheme")
    cchild = ET.Element("Convergence_Criterion")
    cchild.set("Value", "Default")
    child.append(cchild)
    cchild = ET.Element("Convergence_Acceleration")
    cchild.set("Value", "False")
    child.append(cchild)
    cchild = ET.Element("Nitermin")
    cchild.set("Value", "0")
    child.append(cchild)    
    root.append(child)
    
    # Mechanics (hard coded)
    child = ET.Element("Mechanics")
    cchild = ET.Element("Filter")
    cchild.set("Type", "Default")
    child.append(cchild)
    cchild = ET.Element("Small_Perturbations")
    cchild.set("Value", "True")
    child.append(cchild)
    root.append(child)
    
    # Nonlocal model
    child = ET.Element("Non_local_algorithm")
    child.set("NLocMod_num", "1")
    child.set("Algo", "explicit")
    
    for iparam in range(nparams):
        cchild = ET.Element("P_real")
        cchild.set("Index", str(iparam+1))
        cchild.set("Value", str(params[iparam]))
        child.append(cchild)
    
    root.append(child)
    
    # write
    xml_indent(root)
    tree = ET.ElementTree(root)
    with open(xmlname, "wb") as f:
        tree.write(f, encoding="utf-8", xml_declaration=True)
    
    #
    print('  --> AMITEX algo file: '+xmlname)


'''
Write the launcher script for AMITEX
'''
def write_AMITEX_launcher(scriptname, prefix, env_amitex="../../../../amitex_fftp/env_amitex.sh",\
                                              exe_amitex="../../src/amitex_fftp", \
                                              prefix_load="res", \
                                              np=1):
    with open(scriptname, 'w') as f:
        f.write("#!/bin/bash\n")
        
        f.write("source {}\n".format(env_amitex))
        f.write("AMITEX0={}\n".format('"'+exe_amitex+'"'))
        
        f.write("MESH={}\n".format('"'+'../micro/'+prefix+'_mesh.vtk'+'"'))
        f.write("ALGO={}\n".format('"'+'../algo/'+prefix+'_algo.xml'+'"'))
        f.write("MATE={}\n".format('"'+'../mate/'+prefix+'_mat.xml'+'"'))
        f.write("LOAD={}\n".format('"'+'../load/'+prefix+'_load.xml'+'"'))
        
        f.write("mkdir ../results\n")
        f.write("mkdir ../results/{}\n".format(prefix))
        
        f.write("mpirun -np {:d} $AMITEX0 -nm $MESH -m $MATE -c $LOAD -a $ALGO -s ../results/{}/{}\n" \
                .format(np, prefix, prefix_load))


'''
Save the collated information (vfield and macroscopic props) into XML file
'''
def saveBrinkman2XML(prefix_output, meshname, vmacro, W, H, idx):
    
    # read algorithm parameters
    mu = extract_algo(prefix_output+'_algo.xml', 'Non_local_algorithm', idx['mu'])
    mue = extract_algo(prefix_output+'_algo.xml', 'Non_local_algorithm', idx['mue'])
    crit = extract_algo(prefix_output+'_algo.xml', 'Non_local_algorithm', idx['crit'])
    FDscheme = extract_algo(prefix_output+'_algo.xml', 'Non_local_algorithm', idx['FD'])
    ACV = extract_algo(prefix_output+'_algo.xml', 'Non_local_algorithm', idx['ACV'])
    modACV = extract_algo(prefix_output+'_algo.xml', 'Non_local_algorithm', idx['modACV'])
    
    #############
    # extract the computation performance (TODO)
    logfile = prefix_output + '.log'
    #############
    
    # write the collated info into an xml file
    root = ET.Element("AMITEX_Brinkman")
    
    # simulation setting
    child = ET.Element("Setting")
    
    cchild = ET.Element("Mesh")
    cchild.set("File", meshname)
    child.append(cchild)
    
    cchild = ET.Element("Viscosity")
    cchild.set("mu", mu)
    cchild.set("mue", mue)
    child.append(cchild)
    
    cchild = ET.Element("FiniteDifferenceScheme")
    cchild.set("Tag", FDscheme)
    child.append(cchild)
    
    cchild = ET.Element("ConvAccelaration")
    cchild.set("activate", ACV)
    cchild.set("increment", modACV)
    child.append(cchild)
    
    cchild = ET.Element("ConvTolerence")
    cchild.set("Value", crit)
    child.append(cchild)
    
    root.append(child)
    
    # simulation result
    child = ET.Element("Result")
    
    cchild = ET.Element("MacroVelocity")
    cchild.set("vx", str(vmacro[0]))
    cchild.set("vy", str(vmacro[1]))
    cchild.set("vz", str(vmacro[2]))
    child.append(cchild)
    
    cchild = ET.Element("MacroPressureGradient")
    cchild.set("Gx", str(W[0]))
    cchild.set("Gy", str(W[1]))
    cchild.set("Gz", str(W[2]))
    child.append(cchild)
    
    cchild = ET.Element("MacroResistivity")
    cchild.set("Hx", str(H[0]))
    cchild.set("Hy", str(H[1]))
    cchild.set("Hz", str(H[2]))
    child.append(cchild)
    
    # computation performance
    cchild = ET.Element("Performance")
    cchild.set("nIters", "")
    cchild.set("wallT", "")
    child.append(cchild)
    
    root.append(child)
    
    # write
    xml_indent(root)
    tree = ET.ElementTree(root)
    with open(prefix_output+'_collated.xml', "wb") as f:
        tree.write(f, encoding="utf-8", xml_declaration=True)

