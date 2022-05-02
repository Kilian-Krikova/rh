import xdrlib
import numpy as np
import os
import warnings
import datetime
import numpy as np
import xarray as xr
import h5py
import netCDF4
from io import StringIO
from astropy import units

import rhanalyze.rhgeometry
from rhanalyze.rhtools import read_farray, read_string, write_farray

class element:

    def __init__(self, up):
        self.read_element(up)
        
    def read_element(self, up):
        self.ID     = read_string(up)
        self.weight = float(up.unpack_double())
        self.abund  = float(up.unpack_double())
    
class atmos:
    
    def __init__(self, geometry, filename='atmos.out'):
        self.filename = filename
        self.read(geometry)
         
    def read(self, geometry):

        f  = open(self.filename, 'rb')
        up = xdrlib.Unpacker(f.read())
        f.close()

        self.NHydr  = up.unpack_int()
        self.Nelem  = up.unpack_int()
        self.moving = up.unpack_int()

        if geometry.type == "ONE_D_PLANE":           
            dim1 = [geometry.Ndep]         
            dim2 = [geometry.Ndep, self.NHydr]
            
        elif geometry.type == 'SPHERICAL_SYMMETRIC':
            dim1 = [geometry.Nradius]         
            dim2 = [geometry.Nradius, self.NHydr]
            
        elif geometry.type == 'TWO_D_PLANE':
            dim1 = [geometry.Nx, geometry.Nz]
            dim2 = [geometry.Nx, geometry.Nz, self.NHydr]

        elif geometry.type == 'THREE_D_PLANE':
            dim1 = [geometry.Nx, geometry.Ny, geometry.Nz]
            dim2 = [geometry.Nx, geometry.Ny, geometry.Nz, self.NHydr]


        self.T      = read_farray(dim1, up, "double")
        self.n_elec = read_farray(dim1, up, "double")
        self.vturb  = read_farray(dim1, up, "double")

        self.nH = read_farray(dim2, up, "double")       
        self.ID = read_string(up)

        
        self.elements = {}
        for n in range(self.Nelem):
            self.elements[n] = element(up)

        if geometry.type != 'SPHERICAL_SYMMETRIC':
            try:
                stokes = up.unpack_int()
            except EOFError or IOError:
                self.stokes = False
                return
            else:
                self.stokes = True

                self.B       = read_farray(dim1, up, "double")
                self.gamma_B = read_farray(dim1, up, "double")
                self.chi_B   = read_farray(dim1, up, "double")
                
        up.done()

class input_atmos:
    def __init__(self, geometrytype, atmosfile, Bfile=None, data_format='XDR'):
        if data_format == 'XDR':
            self.read(geometrytype, atmosfile, Bfile)

        elif data_format == 'HDF5':
            self.read_hdf5(geometrytype, atmosfile)
    
    def read_hdf5(self, geometrytype, atmosfile):
        
        self.type = geometrytype
        if self.type == "THREE_D_PLANE":
            self.file = read_hdf5(self, atmosfile)

            # Set dimensions
            self.Nx = self.nx.shape[0]
            self.Ny = self.nx.shape[0]
            self.Nz = self.nz.shape[0]
            self.NHydr = self.nhydr.shape[0]

            # Vertical boundary values 
            self.boundary = np.array([1,2])

            # Grid dist
            self.dx = np.mean(np.diff(self.x[:]))/1e3 # In km, seems like only equal distance possible
            self.dy = np.mean(np.diff(self.y[:]))/1e3
            self.z = self.z[0,:]/1e3 # Convert to km, read_hdf5 already sets self.z 

            # Atmospheric quantities
            self.T = self.temperature[0,:,:,:]
            self.n_elec = self.electron_density[0,:,:,:]
            self.vturb = np.zeros((self.Nx, self.Ny, self.Nz))
            self.vx = self.velocity_x[0,:,:,:]/1e3 # to km/s
            self.vy = self.velocity_y[0,:,:,:]/1e3
            self.vz = self.velocity_z[0,:,:,:]/1e3
            self.nH = np.moveaxis(self.hydrogen_populations[0,:,:,:,:],0,-1) # Level population axis should be at the last dimension

            # Close atmosphere file
            #self.file.close()
        else:
            raise ValueError('Only 3D geometry implemented')


    def decrease_resolution(self, x_step, y_step):

        if self.type == "THREE_D_PLANE":

            # Decrease x and y dimension
            self.Nx = int(self.nx.shape[0]/x_step)
            self.Ny = int(self.nx.shape[0]/y_step)

            # Take every x and y step location 
            self.T = self.T[0:-1:x_step, 0:-1:y_step, :]
            self.n_elec = self.n_elec[0:-1:x_step, 0:-1:y_step, :]
            self.vturb = self.vturb[0:-1:x_step, 0:-1:y_step, :]
            self.vx = self.vx[0:-1:x_step, 0:-1:y_step, :]
            self.vy = self.vy[0:-1:x_step, 0:-1:y_step, :]
            self.vz = self.vz[0:-1:x_step, 0:-1:y_step, :]
            self.nH = self.nH[0:-1:x_step, 0:-1:y_step, :, :]

        else:
            raise ValueError('Only 3D geometry implemented')
    
    def cut_box(self, x_center, y_center, width, height):

        if self.type == "THREE_D_PLANE":
            # Decrease x and y dimension
            self.Nx = int(width*2 +1)
            self.Ny = int(height*2 +1)

            # Quantities inside the box with inpit width and height around center
            # Width takes x locations with x_center - width to x_center + width
            # Same for height in y-direction
            self.T = self.T[x_center-width:x_center+width+1, y_center-height:y_center+height+1, :]
            self.n_elec = self.n_elec[x_center-width:x_center+width+1, y_center-height:y_center+height+1, :]
            self.vturb = self.vturb[x_center-width:x_center+width+1, y_center-height:y_center+height+1, :]
            self.vx = self.vx[x_center-width:x_center+width+1, y_center-height:y_center+height+1, :]
            self.vy = self.vy[x_center-width:x_center+width+1, y_center-height:y_center+height+1, :]
            self.vz = self.vz[x_center-width:x_center+width+1, y_center-height:y_center+height+1, :]
            self.nH = self.nH[x_center-width:x_center+width+1, y_center-height:y_center+height+1, :, :]

        else:
            raise ValueError('Only 3D geometry implemented')

    def cut_slice(self, y_pos, width, x_pos=False):

        if self.type == "THREE_D_PLANE":
            if x_pos:
                print('Slice along a x position')
                print('Not Implemented')
            else:
                print('Slice along a y position')
                self.Ny = int(width*2 +1)

                # Quantities along a slice at y-position
                # Width gets more y-slices with y_pos +- width
                self.T = self.T[:, y_pos-width:y_pos+width+1, :]
                self.n_elec = self.n_elec[:, y_pos-width:y_pos+width+1, :]
                self.vturb = self.vturb[:, y_pos-width:y_pos+width+1, :]
                self.vx = self.vx[:, y_pos-width:y_pos+width+1, :]
                self.vy = self.vy[:, y_pos-width:y_pos+width+1, :]
                self.vz = self.vz[:, y_pos-width:y_pos+width+1 :]
                self.nH = self.nH[:, y_pos-width:y_pos+width+1, :, :]

        else:
            raise ValueError('Only 3D geometry implemented')

    def read(self, geometrytype, atmosfile, Bfile):
        self.type = geometrytype
       
        if self.type == "ONE_D_PLANE" or self.type == "SPHERICAL_SYMMETRIC":

            CM_TO_M = 1.0E-2
            G_TO_KG = 1.0E-3

            data = []
            with open(atmosfile, 'r') as file:
                for line in file:
                    if line.startswith('*'):
                        continue
                    data.append(line.strip())
            
            self.ID    = data[0]
            scale      = data[1][0]
            
            if self.type == "ONE_D_PLANE":
                self.grav = float(data[2])
                self.Ndep = int(data[3])
                Nd = self.Ndep
            else:
                self.grav, self.radius = [float(x) for x in data[2].split()]
                self.Nradius, self.Ncore, self.Ninter = \
                    [int(x) for x in data[3].split()]
                Nd = self.Nradius

            self.grav  = np.power(10.0, self.grav) * CM_TO_M
            self.NHydr = 6

            hscale      = np.array(range(Nd), dtype="float")
            self.T      = np.array(range(Nd), dtype="float")
            self.n_elec = np.array(range(Nd), dtype="float")
            self.v      = np.array(range(Nd), dtype="float")
            self.vturb  = np.array(range(Nd), dtype="float")

            for n in range(Nd):
                hscale[n], self.T[n],\
                    self.n_elec[n], self.v[n], self.vturb[n] =\
                        [float(x) for x in data[n+4].split()]
            
            if scale == 'M':
                self.scale  = 'MASS_SCALE'
                self.cmass  = np.power(10.0, hscale)
                self.cmass *= G_TO_KG / CM_TO_M**2

            elif scale == 'T':
                self.scale  = 'TAU500_SCALE'
                self.tau500 = np.power(10.0, hscale)
            elif scale == 'H':
                self.scale  = 'TAU500_SCALE'
                self.height = hscale

            if len(data) > (4 + Nd):
                self.HLTE = False
                self.nH = np.array(range(Nd * self.NHydr),\
                                   dtype="float").reshape([Nd,\
                                                           self.NHydr],\
                                                          order='F')
                for n in range(Nd):
                    self.nH[n,:] =\
                        [float(x) for x in data[n+4+Nd].split()]
            else:
                self.HLTE = True
                                  
            self.nH     /= CM_TO_M**3
            self.n_elec /= CM_TO_M**3

            dim1 = [Nd]

        elif self.type == "TWO_D_PLANE" or self.type == "THREE_D_PLANE":
            
            f = open(atmosfile, 'rb')
            up = xdrlib.Unpacker(f.read())
            f.close()
             
            if self.type == "TWO_D_PLANE":
                self.Nx = up.unpack_int()
                self.Nz = up.unpack_int()
                self.NHydr = up.unpack_int()

                self.boundary = read_farray([3], up, "int")

                self.dx = read_farray([self.Nx], up, "double")
                self.z  = read_farray([self.Nz], up, "double")

                dim1 = [self.Nx, self.Nz]
                dim2 = [self.Nx, self.Nz, self.NHydr]
                
            elif self.type == "THREE_D_PLANE":
                self.Nx = up.unpack_int()
                self.Ny = up.unpack_int()
                self.Nz = up.unpack_int()
                self.NHydr = up.unpack_int()

                self.boundary = read_farray([2], up, "int")

                self.dx = up.unpack_double()
                self.dy = up.unpack_double()
                self.z  = read_farray([self.Nz], up, "double")

                dim1 = [self.Nx, self.Ny, self.Nz]
                dim2 = [self.Nx, self.Ny, self.Nz, self.NHydr]
                
            self.T      = read_farray(dim1, up, "double")
            self.n_elec = read_farray(dim1, up, "double")
            self.vturb  = read_farray(dim1, up, "double")
            self.vx     = read_farray(dim1, up, "double")
            
            if self.type == "THREE_D_PLANE":
                self.vy     = read_farray(dim1, up, "double")
                
            self.vz     = read_farray(dim1, up, "double")

            self.nH     = read_farray(dim2, up, "double")
                
            up.done()
            
        else:
            print("Not a valid input atmosphere type: {0}".format(self.type))
            return
            
        if Bfile != None:
            
            f = open(Bfile, 'rb')
            up = xdrlib.Unpacker(f.read())
            f.close()

            self.B     = read_farray(dim1, up, "double")
            self.gamma = read_farray(dim1, up, "double")
            self.chi   = read_farray(dim1, up, "double")
            
            up.done()

    def write(self, outfile, Bfile=None):

        if self.type == "ONE_D_PLANE" or self.type == "SPHERICAL_SYMMETRIC":

            CM_TO_M = 1.0E-2
            G_TO_KG = 1.0E-3

            nH     = self.nH.copy() * CM_TO_M**3
            n_elec = self.n_elec.copy() * CM_TO_M**3

            data = []

            data.append("* Model atmosphere written by " \
                        "rhatmos.input_atmos.write()\n")
            data.append("*\n")
            data.append("  {0}\n".format(self.ID))

            if self.scale == "MASS_SCALE":
                hscale = np.log10(self.cmass / (G_TO_KG / CM_TO_M**2))
                data.append("  Mass scale\n")
            elif self.scale == "TAU500_SCALE":
                hscale = np.log10(self.tau500)
                data.append("  Tau500 scale\n")
            elif self.scale == "GEOMETRIC_SCALE":
                hscale = self.height
                data.append("  Height scale\n")

            data.append('*\n')

            grav = np.log10(self.grav / CM_TO_M)
    
            if self.type == "ONE_D_PLANE":
                data.append("* lg g [cm s^-2]\n")
                data.append('     {:5.2f}\n'.format(grav))
                data.append("* Ndep\n")   
                data.append('   {:4d}\n'.format(self.Ndep))
                
                Nd = self.Ndep
            else:
                data.append("* lg g [cm s^-2]      Radius [km]\n")
                data.append('     {:5.2f}          '\
                            '{:7.2E}\n'.format(grav, self.radius))
                data.append("* Nradius   Ncore   Ninter\n")
                fmt = 3 * '    {:4d}' + "\n"
                data.append(fmt.format(self.Nradius, self.Ncore, self.Ninter))
                            
                Nd = self.Nradius

            data.append("*\n")
            data.append("*  lg column Mass   Temperature    "\
                        "Ne             V              Vturb\n")

            fmt = '  {: 12.8E}' + 4 * '  {: 10.6E}' + "\n"
            for k in range(Nd):
                data.append(fmt.format(hscale[k], self.T[k], n_elec[k],\
                                       self.v[k], self.vturb[k]))

            data.append("*\n")
            
            if not self.HLTE:
                data.append("* NLTE Hydrogen populations\n")
                data.append("*  nh[1]        nh[2]        nh[3]        "\
                            "nh[4]        nh[5]        np\n")

                fmt = self.NHydr * '   {:8.4E}' + "\n"
                for k in range(Nd):
                    data.append(fmt.format(*nH[k, :]))
                    
            f = open(outfile, 'w')
            for line in data:
                f.write(line)
            f.close()

             
        elif self.type == "TWO_D_PLANE" or self.type == "THREE_D_PLANE":

            pck = xdrlib.Packer()

            if self.type == "TWO_D_PLANE":
                write_farray(np.array([self.Nx, self.Nz, self.NHydr]),\
                             pck, "int")
                write_farray(np.array(self.boundary), pck, "int")
                write_farray(self.dx, pck, "double")
               
            elif self.type == "THREE_D_PLANE":
                write_farray(np.array([self.Nx, self.Ny,\
                                       self.Nz, self.NHydr]),\
                             pck, "int")
                write_farray(np.array(self.boundary), pck, "int")
                pck.pack_double(self.dx)
                pck.pack_double(self.dy)

            write_farray(self.z, pck, "double")
            write_farray(self.T, pck, "double")
            write_farray(self.n_elec, pck, "double")
            write_farray(self.vturb, pck, "double")
            write_farray(self.vx, pck, "double")

            if self.type == "THREE_D_PLANE":
                write_farray(self.vy, pck, "double")
            
            write_farray(self.vz, pck, "double")
            write_farray(self.nH, pck, "double")
 
            f = open(outfile, 'wb')
            f.write(pck.get_buffer())
            f.close()
            pck.reset()
            
        else:
            print("Not a valid input atmosphere type: {0}".format(self.type))
            return

        if Bfile != None:

            pck = xdrlib.Packer()

            write_farray(self.B, pck, "double")
            write_farray(self.gamma, pck, "double")
            write_farray(self.chi, pck, "double")

            f = open(Bfile, 'wb')
            f.write(pck.get_buffer())
            f.close()
            pck.reset()

def read_hdf5(inclass, infile):
    """
    Reads HDF5/netCDF4 file into inclass, instance of any class.
    Variables are read into class attributes, dimensions and attributes
    are read into params dictionary.
    """
    if not os.path.isfile(infile):
        raise IOError('read_hdf5: File %s not found' % infile)
    f = h5py.File(infile, mode='r')
    if 'params' not in dir(inclass):
        inclass.params = {}
    # add attributes
    attrs = [a for a in f.attrs]
    for att in f.attrs:
        try:
            inclass.params[att] = f.attrs[att]
        except OSError:  # catch errors where h5py cannot read UTF-8 strings
            pass
    # add variables and groups
    for element in f:
        name = element.replace(' ', '_')    # sanitise string for spaces
        if type(f[element]) == h5py._hl.dataset.Dataset:
            setattr(inclass, name, f[element])
            # special case for netCDF dimensions, add them to param list
            if 'NAME' in f[element].attrs:
                if f[element].attrs['NAME'][:20] == b'This is a netCDF dim':
                    inclass.params[element] = f[element].shape[0]
        if type(f[element]) == h5py._hl.group.Group:
            setattr(inclass, name, DataHolder())
            cur_class = getattr(inclass, name)
            cur_class.params = {}
            for variable in f[element]:   # add group variables
                vname = variable.replace(' ', '_')
                setattr(cur_class, vname, f[element][variable])
            for att in f[element].attrs:  # add group attributes
                cur_class.params[att] = f[element].attrs[att]
    return f