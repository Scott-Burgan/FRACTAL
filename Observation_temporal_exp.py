"""Script to compare observational data products over Lusaka across
various temporal resolutions. Datasets include CHIRPS(6hr), MSWEP(3hr) and CMORPH(1hr).
 We will focus on 2009 where Lusaka experienced the worst flooding in 40 years. - Scott Burgan"""

import matplotlib as mpl
from matplotlib import gridspec
import time
import os
import matplotlib.pyplot as plt
import iris
import iris.analysis.cartography
import iris.coord_categorisation as iccat
import iris.util as util
import iris.plot as iplt
import cartopy.crs as ccrs  # coordinate reference system
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.feature as cfeature
import numpy as np
import collections as col
import cf_units
import matplotlib.colors as colors
import iris.quickplot as qplt
import datetime

def subdomain_constraint(cube, subdomain):
    """Get subdomain of cube

    Arguments:
        cube (iris.Cube): Input cube to extract subdomain from
        subdomain (list): List of coordinates [min lon, max lon, min lat, max lat]

    Returns:
        Iris cube

    """
    if isinstance(subdomain, list):
        # Extract subdomain for rectangular domains
        subcube = cube.intersection(longitude=subdomain[:2],
                                    latitude=subdomain[2:])
        return subcube
    elif isinstance(subdomain, str):
        # Extract subdomain for irregular areas based on shapefiles
        return cube
    else:
        # Raise error if other types of subdomain are passed.
        raise ValueError("Subdomain is type {}.  Should be list or str.".format(type(subdomain)))

def add_day(cube, coord, name="day"):
    """Add a categorical calendar-year coordinate."""
    add_categorised_coord(
        cube, name, coord, lambda coord, x: _pt_date(coord, x).day
    )

def add_categorised_coord(cube, name, from_coord, category_function, units="1"):
    """
    Add a new coordinate to a cube, by categorising an existing one.

    Make a new :class:`iris.coords.AuxCoord` from mapped values, and add
    it to the cube.

    Args:

    * cube (:class:`iris.cube.Cube`):
        the cube containing 'from_coord'.  The new coord will be added into it.
    * name (string):
        name of the created coordinate
    * from_coord (:class:`iris.coords.Coord` or string):
        coordinate in 'cube', or the name of one
    * category_function (callable):
        function(coordinate, value), returning a category value for a
        coordinate point-value

    Kwargs:

    * units:
        units of the category value, typically 'no_unit' or '1'.
    """
    # Interpret coord, if given as a name
    if isinstance(from_coord, str):
        from_coord = cube.coord(from_coord)

    if len(cube.coords(name)) > 0:
        msg = 'A coordinate "%s" already exists in the cube.' % name
        raise ValueError(msg)

    # Construct new coordinate by mapping values, using numpy.vectorize to
    # support multi-dimensional coords.
    # Test whether the result contains strings. If it does we must manually
    # force the dtype because of a numpy bug (see numpy #3270 on GitHub).
    result = category_function(from_coord, from_coord.points.ravel()[0])
    if isinstance(result, str):
        str_vectorised_fn = np.vectorize(category_function, otypes=[object])

        def vectorised_fn(*args):
            # Use a common type for string arrays (N.B. limited to 64 chars).
            return str_vectorised_fn(*args).astype("|U64")

    else:
        vectorised_fn = np.vectorize(category_function)
    new_coord = iris.coords.AuxCoord(
        vectorised_fn(from_coord, from_coord.points),
        units=units,
        attributes=from_coord.attributes.copy(),
    )
    new_coord.rename(name)

    # Add into the cube
    cube.add_aux_coord(new_coord, cube.coord_dims(from_coord))

def _pt_date(coord, time):
    """
    Return the datetime of a time-coordinate point.

    Args:

    * coord (Coord):
        coordinate (must be Time-type)
    * time (float):
        value of a coordinate point

    Returns:
        cftime.datetime

    """
    # NOTE: All of the currently defined categorisation functions are
    # calendar operations on Time coordinates.
    return coord.units.num2date(time)

def calc_plot_data(cube_data, season, var):
    '''
    Calculate data to plot for given season.
    Essentially, extract just what we need and collapse time dimension to mean
    :param cube_data: iris cube of source data
    :param season: season as string
    :return: seasonal mean of data as iris cube
    '''

    assert not cube_data is None
    # extract season
    # the season membership approach is used because iris doens't like adding a season co-ordinate that
    # includes single month seasons, or doesn't span the whole year
    plot_data = cube_data

    # Units conversion if necessary
    if var == 'MSLP':
        plot_data.convert_units('hPa')
    elif var == 'pr':
        if plot_data.units == 'mm day-1':
            plot_data.units = 'mm day -1'
        elif plot_data.units != 'mm month -1':
            #convert_units can't handle mm/month.. but we at least check we are converting what we think we are here
            assert plot_data.units == cf_units.Unit('kg m-2 s-1'), 'Expect units kg m-2 s-1, got {}'.format(
                plot_data.units)
            plot_data = plot_data * 86400
            plot_data.units = 'mm day -1'
    # Collapse time dimension to mean
    if type(cube_data) == iris.cube.CubeList:
        for i in range(len(cube_data)):
            plot_data[i] = plot_data[i].collapsed('time', iris.analysis.MAX)

    plot_data = plot_data.collapsed('time', iris.analysis.MAX)

    return plot_data

years = [2009, 2009]
time_con = iris.Constraint(time=lambda t: years[0] <= t.point.year <= years[-1])
cs = iris.coord_systems.GeogCS(iris.fileformats.pp.EARTH_RADIUS)

DOM_LAT = [-15.5, -15]
DOM_LON = [28, 28.5]
DOM_NAME = 'Lusaka'
SEASONS = ['JFM']

CHIRPS = f"/project/ciid/obs_datasets/africa/CHIRPS/v2.0/daily/pr/*nc"
MSWEP= f"/project/applied/Data/MSWEP/v2.80/mswep.pr.day.p1d.1979-2020.nc"
CMORPH_hr=f"/scratch/sburgan/obs/CMORPH_V1.0_ADJ_8km-1hr_2009.nc"


# load data and then concatenate
print('\33[93m  loading pr observations\33[00m')
data = {}
# data['CHIRPS'] = iris.load(CHIRPS, time_con)
# data['MSWEP'] = iris.load_cube(MSWEP, time_con)
data['CMORPH_hr'] = iris.load_cube(CMORPH_hr)
# data['CMORPH_day'] = iris.load_cube(CMORPH_hr)


for i in data:
    print(f'\33[93m  loading {i} observations\33[00m')
    if i == 'CHIRPS':
        util.equalise_attributes(data[i])
        data[i] = data[i].concatenate_cube()

    data[i].coord("latitude").coord_system = cs
    data[i].coord("longitude").coord_system = cs
    data[i] = subdomain_constraint(data[i], [28, 28.5, -16, -15])

    if i == 'CMORPH_day':
        iccat.add_day_of_year(data['CMORPH_day'], 'time')
        data['CMORPH_day'] = data['CMORPH_day'].aggregated_by('day_of_year', iris.analysis.SUM)

    #Extract Nov
    months = [1,1]
    data[i] = data[i].extract(iris.Constraint(time=lambda t: months[0] <= t.point.month <= months[1]))
    days = [20,20]
    data[i] = data[i].extract(iris.Constraint(time=lambda t: days[0] <= t.point.day <= days[-1]))
    iccat.add_day_of_year(data['CMORPH_hr'], 'time')
    iccat.add_hour(data['CMORPH_hr'], 'time')
    #data[i] = data[i].collapsed(['latitude','longitude'], iris.analysis.MAX)




for i in data:
    qplt.plot(data[i], marker = ".")
plt.legend(['CHIRPS', 'MSWEP', 'CMORPH_hr','CMORPH_day', 'CMORPH_day_T'])
